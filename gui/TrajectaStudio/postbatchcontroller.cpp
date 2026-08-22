#include "postbatchcontroller.h"

#include "coherence.h"
#include "routecompare.h"

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTimer>

#include <utility>

PostBatchController::PostBatchController(QObject *parent)
    : QObject(parent)
    , m_runner(new TrajectaRunner(this))
{
    connect(m_runner, &TrajectaRunner::consoleOutput,
            this, &PostBatchController::consoleOutput);
    connect(m_runner, &TrajectaRunner::consoleErrorLine,
            this, &PostBatchController::consoleErrorLine);
    connect(m_runner, &TrajectaRunner::statusChanged,
            this, &PostBatchController::statusChanged);
    connect(m_runner, &TrajectaRunner::pauseStateChanged,
            this, &PostBatchController::pauseStateChanged);
    connect(m_runner, &TrajectaRunner::progressChanged, this, [this](double pct) {
        if (m_current >= 0 && m_current < m_results.size()) {
            m_results[m_current].percent = pct;
            emit rowProgress(m_current, pct);
        }
    });
    connect(m_runner, &TrajectaRunner::finished, this,
            [this](TrajectaRunner::Outcome outcome, const QString &report) {
        switch (outcome) {
        case TrajectaRunner::Outcome::Success:
            finishRow(RowState::Done, QString());
            break;
        case TrajectaRunner::Outcome::Cancelled:
            finishRow(RowState::Cancelled,
                      m_stopRequested ? tr("Batch stopped by the user.")
                                      : tr("Row skipped by the user."));
            break;
        case TrajectaRunner::Outcome::Failed:
            finishRow(RowState::Failed, report);
            break;
        }
    });
}

bool PostBatchController::start(const PostBatch::Job &job,
                                const TrajectaRunner::Parameters &env, QString *error)
{
    return start(job, env, error, Resume());
}

bool PostBatchController::start(const PostBatch::Job &job,
                                const TrajectaRunner::Parameters &env, QString *error,
                                const Resume &resume)
{
    if (m_running)
        return false;

    m_job = job;
    m_env = env;
    m_results.clear();
    m_current = -1;
    m_stopRequested = false;
    m_skipRequested = false;
    m_paused = false;

    for (int i = 0; i < job.chunks.size(); ++i)
        m_results.append(RowResult{i, RowState::Pending, QString(), 0.0, 0});

    const QList<PostBatch::Issue> issues = PostBatch::validate(job);
    QString fatal;
    for (const PostBatch::Issue &issue : issues) {
        if (issue.chunk < 0) {
            if (!fatal.isEmpty())
                fatal += QLatin1Char('\n');
            fatal += issue.message;
        }
    }
    if (!fatal.isEmpty()) {
        if (error)
            *error = fatal;
        return false;
    }
    for (const PostBatch::Issue &issue : issues) {
        if (issue.chunk < 0 || issue.chunk >= m_results.size())
            continue;
        if (m_results[issue.chunk].state == RowState::Pending) {
            m_results[issue.chunk].state = RowState::Invalid;
            m_results[issue.chunk].message = issue.message;
        }
    }

    if (resume.isValid() && resume.chunkIndex < m_results.size()) {
        for (int i = 0; i < resume.chunkIndex; ++i) {
            if (m_results[i].state == RowState::Pending) {
                m_results[i].state = RowState::Done;
                m_results[i].percent = 100.0;
                m_results[i].message = tr("Completed before the interruption.");
            }
        }
        m_current = resume.chunkIndex - 1;
    }

    m_running = true;
    int alreadyDone = 0;
    for (const RowResult &r : std::as_const(m_results))
        if (r.state != RowState::Pending && r.state != RowState::Running)
            ++alreadyDone;
    emit batchProgress(alreadyDone, m_results.size());
    QTimer::singleShot(0, this, &PostBatchController::startNext);
    return true;
}

void PostBatchController::startNext()
{
    if (!m_running)
        return;
    if (m_stopRequested) {
        m_running = false;
        emit batchFinished(buildReport());
        return;
    }
    // A pause requested while an in-process chunk was running (NNI stays
    // frozen through its own subprocess and never reaches here while paused)
    // holds the queue open without ending the batch.
    if (m_paused) {
        QTimer::singleShot(150, this, &PostBatchController::startNext);
        return;
    }

    for (;;) {
        ++m_current;
        if (m_current >= m_results.size()) {
            m_running = false;
            emit batchFinished(buildReport());
            return;
        }
        if (m_results.at(m_current).state != RowState::Invalid)
            break;
        emit rowFinished(m_current, RowState::Invalid, m_results.at(m_current).message);
        int done = 0;
        for (const RowResult &r : std::as_const(m_results))
            if (r.state != RowState::Pending && r.state != RowState::Running)
                ++done;
        emit batchProgress(done, m_results.size());
    }

    const PostBatch::Chunk &chunk = m_job.chunks.at(m_current);
    m_results[m_current].state = RowState::Running;
    m_results[m_current].percent = 0.0;
    m_rowClock.start();
    emit rowStarted(m_current);

    if (m_job.mode == PostBatch::Mode::Nni)
        runNniChunk(chunk);
    else
        runInProcessChunk(chunk);
}

void PostBatchController::runNniChunk(const PostBatch::Chunk &chunk)
{
    TrajectaRunner::Parameters p = PostBatch::toParameters(m_job, chunk, m_env);
    if (!QDir().mkpath(p.outputDir)) {
        finishRow(RowState::Failed,
                  tr("Cannot create the output folder: %1").arg(p.outputDir));
        return;
    }
    m_runner->start(p);
}

// Compare and Coherence run synchronously, on this thread — the same thing
// the single-run buttons do (MainWindow::runRouteComparison() /
// runCoherence()). Events are pumped with user input excluded, exactly as
// those do, so the window keeps repainting and the log keeps filling in
// without opening a door for a second batch to start from inside this one.
void PostBatchController::runInProcessChunk(const PostBatch::Chunk &chunk)
{
    emit statusChanged(m_job.mode == PostBatch::Mode::Compare
                           ? tr("Comparing…")
                           : tr("Scoring…"));

    auto pump = [](const QString &line) {
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        return line;
    };

    if (m_job.mode == PostBatch::Mode::Compare) {
        const RouteCompare::Result res = RouteCompare::compare(
            chunk.cmpComputedPath, chunk.cmpKnownPath, chunk.cmpTolerance,
            [this, pump](const QString &line) {
                emit consoleOutput(pump(line) + QLatin1Char('\n'));
            });
        if (m_skipRequested) {
            m_skipRequested = false;
            finishRow(RowState::Cancelled, tr("Row skipped by the user."));
            return;
        }
        if (!res.ok) {
            finishRow(RowState::Failed, res.error);
            return;
        }
        emit consoleOutput(res.report() + QLatin1Char('\n'));
        finishRow(RowState::Done, QString());
        return;
    }

    // Coherence
    Coherence::Params p;
    p.rasterPath = chunk.cohRasterPath;
    p.pointsPath = chunk.cohPointsPath;
    p.radiusMetres = chunk.cohRadius;
    switch (chunk.cohThresholdMode) {
    case 1: p.thresholdMode = Coherence::ThresholdMode::Otsu; break;
    case 2: p.thresholdMode = Coherence::ThresholdMode::Absolute; break;
    default: p.thresholdMode = Coherence::ThresholdMode::TopPercent; break;
    }
    p.thresholdValue = chunk.cohThresholdValue;
    p.nullModel = chunk.cohNullModel;
    p.nullMode = chunk.cohNullMode == 1 ? Coherence::NullMode::Uniform
                                       : Coherence::NullMode::RandomShift;
    p.nullReplicates = chunk.cohNullReplicates;
    for (const QString &part : chunk.cohEcdfDistances.split(
             QRegularExpression(QStringLiteral("[,;\\s]+")), Qt::SkipEmptyParts)) {
        bool okNum = false;
        const double d = part.toDouble(&okNum);
        if (okNum && d >= 0.0)
            p.ecdfDistances << d;
    }
    p.sensitivity = chunk.cohSensitivity;
    if (p.sensitivity) {
        const QStringList parts = chunk.cohSensitivityRadii.split(
            QRegularExpression(QStringLiteral("[,;\\s]+")), Qt::SkipEmptyParts);
        for (const QString &s : parts) {
            bool okNum = false;
            const double r = s.toDouble(&okNum);
            if (okNum && r > 0.0)
                p.sensitivityRadii << r;
        }
    }
    p.edgeGuard = chunk.cohEdgeGuard;
    p.writeVector = true;
    p.vectorAsGeoPackage = chunk.cohVectorAsGeoPackage;
    p.writeDistanceRaster = chunk.cohWriteDistanceRaster;
    p.writeHistogramScript = chunk.cohWriteHistogramScript;
    p.outputPrefix = chunk.cohPrefix.trimmed();
    p.outputDir = chunk.cohOutputDir.trimmed();
    if (p.outputDir.isEmpty())
        p.outputDir = QFileInfo(chunk.cohRasterPath).absolutePath();

    const Coherence::Result res = Coherence::run(
        p, [this, pump](const QString &line) {
            emit consoleOutput(pump(line) + QLatin1Char('\n'));
        });
    if (m_skipRequested) {
        m_skipRequested = false;
        finishRow(RowState::Cancelled, tr("Row skipped by the user."));
        return;
    }
    if (!res.ok) {
        finishRow(RowState::Failed, res.error);
        return;
    }
    emit consoleOutput(res.report() + QLatin1Char('\n'));
    finishRow(RowState::Done, QString());
}

void PostBatchController::finishRow(RowState state, const QString &message)
{
    if (m_current >= 0 && m_current < m_results.size()) {
        m_results[m_current].state = state;
        m_results[m_current].message = message;
        m_results[m_current].elapsedMs = m_rowClock.isValid() ? m_rowClock.elapsed() : 0;
        if (state == RowState::Done) {
            m_results[m_current].percent = 100.0;
            emit chunkLayersReady(m_current);
        }
        emit rowFinished(m_current, state, message);
    }

    int done = 0;
    for (const RowResult &r : std::as_const(m_results))
        if (r.state != RowState::Pending && r.state != RowState::Running)
            ++done;
    emit batchProgress(done, m_results.size());

    // Through the event loop for the same reason as BatchController: a
    // subprocess's finished() signal must be allowed to unwind before start()
    // tears the QProcess down, and for the in-process chunks this keeps both
    // execution paths going through the identical hand-off.
    QTimer::singleShot(0, this, &PostBatchController::startNext);
}

void PostBatchController::skipCurrentRow()
{
    if (!m_running)
        return;
    if (m_job.mode == PostBatch::Mode::Nni) {
        if (m_runner->isRunning())
            m_runner->cancel();
        return;
    }
    // The in-process chunk currently running cannot be interrupted (see the
    // class comment); the request is honoured the moment it returns.
    m_skipRequested = true;
}

void PostBatchController::stopBatch()
{
    if (!m_running)
        return;
    m_stopRequested = true;
    if (m_job.mode == PostBatch::Mode::Nni && m_runner->isRunning()) {
        m_runner->cancel();
    } else if (m_job.mode == PostBatch::Mode::Nni) {
        m_running = false;
        emit batchFinished(buildReport());
    }
    // For an in-process job the running chunk finishes on its own and
    // finishRow() -> startNext() sees m_stopRequested and ends the batch then.
}

void PostBatchController::pause()
{
    if (m_job.mode == PostBatch::Mode::Nni) {
        m_runner->pause();
        return;
    }
    m_paused = true;
    emit pauseStateChanged(true);
}

void PostBatchController::resume()
{
    if (m_job.mode == PostBatch::Mode::Nni) {
        m_runner->resume();
        return;
    }
    m_paused = false;
    emit pauseStateChanged(false);
}

bool PostBatchController::isPaused() const
{
    return m_job.mode == PostBatch::Mode::Nni ? m_runner->isPaused() : m_paused;
}

bool PostBatchController::isRunning() const
{
    return m_running;
}

QString PostBatchController::buildReport() const
{
    int done = 0, failed = 0, cancelled = 0, invalid = 0;
    for (const RowResult &r : m_results) {
        switch (r.state) {
        case RowState::Done:      ++done; break;
        case RowState::Failed:    ++failed; break;
        case RowState::Cancelled: ++cancelled; break;
        case RowState::Invalid:   ++invalid; break;
        default: break;
        }
    }
    QStringList parts;
    parts << tr("%1 of %2 chunks completed.").arg(done).arg(m_results.size());
    if (failed)
        parts << tr("%1 failed.").arg(failed);
    if (cancelled)
        parts << tr("%1 cancelled.").arg(cancelled);
    if (invalid)
        parts << tr("%1 rejected before starting.").arg(invalid);
    return parts.join(QLatin1Char(' '));
}
