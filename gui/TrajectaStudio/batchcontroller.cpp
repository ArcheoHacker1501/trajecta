#include "batchcontroller.h"

#include <QDir>
#include <QTimer>

#include <utility>

BatchController::BatchController(QObject *parent)
    : QObject(parent)
    , m_runner(new TrajectaRunner(this))
{
    connect(m_runner, &TrajectaRunner::consoleOutput,
            this, &BatchController::consoleOutput);
    connect(m_runner, &TrajectaRunner::consoleErrorLine,
            this, &BatchController::consoleErrorLine);
    connect(m_runner, &TrajectaRunner::statusChanged,
            this, &BatchController::statusChanged);
    connect(m_runner, &TrajectaRunner::pauseStateChanged,
            this, &BatchController::pauseStateChanged);
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

bool BatchController::start(const Batch::Job &job,
                            const TrajectaRunner::Parameters &env, QString *error)
{
    return start(job, env, error, Resume());
}

bool BatchController::start(const Batch::Job &job,
                            const TrajectaRunner::Parameters &env, QString *error,
                            const Resume &resume)
{
    if (m_running)
        return false;

    m_job = job;
    m_env = env;
    m_queue = Batch::flatten(job);
    m_results.clear();
    m_current = -1;
    m_stopRequested = false;
    m_pendingResumeCheckpoint.clear();

    for (const Batch::Index &at : std::as_const(m_queue))
        m_results.append(RowResult{at, RowState::Pending, QString(), 0.0, 0});

    // Pre-flight. A problem that is not tied to one row (an empty batch, a
    // chunk whose cost modifier file is gone) would make every row fail the
    // same way, so it stops the batch before anything starts. A problem in one
    // row only takes that row out.
    const QList<Batch::Issue> issues = Batch::validate(job);
    QString fatal;
    for (const Batch::Issue &issue : issues) {
        if (issue.where.row < 0) {
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
    for (const Batch::Issue &issue : issues) {
        if (!issue.where.isValid())
            continue;
        const int q = m_queue.indexOf(issue.where);
        if (q >= 0 && m_results.at(q).state == RowState::Pending) {
            m_results[q].state = RowState::Invalid;
            m_results[q].message = issue.message;
        }
    }

    // Resuming: the rows before the interrupted one already produced their
    // outputs, so they are reported as done and never launched again.
    if (resume.isValid() && resume.queueIndex < m_queue.size()) {
        for (int i = 0; i < resume.queueIndex; ++i) {
            if (m_results.at(i).state == RowState::Pending) {
                m_results[i].state = RowState::Done;
                m_results[i].percent = 100.0;
                m_results[i].message = tr("Completed before the interruption.");
            }
        }
        m_current = resume.queueIndex - 1;
        // Only the interrupted row has engine state to pick up, and only when
        // there is any; if the pre-flight rejected that row, startNext walks
        // past it and the checkpoint is simply never used.
        if (m_results.at(resume.queueIndex).state == RowState::Pending)
            m_pendingResumeCheckpoint = resume.checkpointPath;
    }

    m_running = true;
    // Not always zero: a resumed batch starts with the rows that finished
    // before the interruption already counted.
    int alreadyDone = 0;
    for (const RowResult &r : std::as_const(m_results))
        if (r.state != RowState::Pending && r.state != RowState::Running)
            ++alreadyDone;
    emit batchProgress(alreadyDone, m_queue.size());
    // Through the event loop, so a caller reacting to the first signals is not
    // re-entered from inside start().
    QTimer::singleShot(0, this, &BatchController::startNext);
    return true;
}

void BatchController::startNext()
{
    // stopBatch() between two rows finishes the batch on the spot, while the
    // hop through the event loop scheduled by the previous row is still
    // pending. Without this guard that pending call would report the batch
    // finished a second time.
    if (!m_running)
        return;
    if (m_stopRequested) {
        m_running = false;
        emit batchFinished(buildReport());
        return;
    }

    // Walk past everything the pre-flight already rejected, reporting each so
    // the table turns red immediately instead of at the end.
    for (;;) {
        ++m_current;
        if (m_current >= m_queue.size()) {
            m_running = false;
            emit batchFinished(buildReport());
            return;
        }
        if (m_results.at(m_current).state != RowState::Invalid)
            break;
        emit rowFinished(m_current, RowState::Invalid,
                         m_results.at(m_current).message);
        int done = 0;
        for (const RowResult &r : std::as_const(m_results))
            if (r.state != RowState::Pending && r.state != RowState::Running)
                ++done;
        emit batchProgress(done, m_queue.size());
    }

    const Batch::Index at = m_queue.at(m_current);
    const Batch::Chunk &chunk = m_job.chunks.at(at.chunk);
    const Batch::Row &row = chunk.rows.at(at.row);

    m_results[m_current].state = RowState::Running;
    m_results[m_current].percent = 0.0;
    m_rowClock.start();
    emit rowStarted(m_current);

    TrajectaRunner::Parameters p = Batch::toParameters(m_job, chunk, row, m_env);
    // Used once, by the row that was running when the batch was interrupted.
    if (!m_pendingResumeCheckpoint.isEmpty()) {
        p.resumeCheckpoint = m_pendingResumeCheckpoint;
        m_pendingResumeCheckpoint.clear();
        // A row that generates its own sample points must not generate them
        // again on the way back in. The engine fingerprints the points file it
        // was given — size and timestamp included — and refuses a checkpoint
        // taken against a different one; rewriting the layer would therefore
        // make the row reject its own state and fail outright. The file is
        // already on disk from the first attempt, and toParameters has already
        // pointed pointsPath at it, so importing it is both correct and what
        // Checkpoint::toJson does for a single run.
        p.generatePoints = false;
    }

    // The engine creates its own output folder, but not reliably for every
    // mode, and a per-row subfolder may not exist yet. Doing it here also
    // turns "the drive is read-only" into a clean row failure.
    if (!QDir().mkpath(p.outputDir)) {
        finishRow(RowState::Failed,
                  tr("Cannot create the output folder: %1").arg(p.outputDir));
        return;
    }

    m_runner->start(p);
}

void BatchController::finishRow(RowState state, const QString &message)
{
    if (m_current >= 0 && m_current < m_results.size()) {
        m_results[m_current].state = state;
        m_results[m_current].message = message;
        m_results[m_current].elapsedMs = m_rowClock.isValid() ? m_rowClock.elapsed() : 0;
        if (state == RowState::Done)
            m_results[m_current].percent = 100.0;
        emit rowFinished(m_current, state, message);
    }

    int done = 0;
    for (const RowResult &r : std::as_const(m_results))
        if (r.state != RowState::Pending && r.state != RowState::Running)
            ++done;
    emit batchProgress(done, m_queue.size());

    // Never call the runner's start() from inside its own finished() signal:
    // start() deletes the QProcess whose signal is still being emitted. Going
    // through the event loop lets that emission unwind first.
    QTimer::singleShot(0, this, &BatchController::startNext);
}

void BatchController::skipCurrentRow()
{
    if (!m_running)
        return;
    if (m_runner->isRunning())
        m_runner->cancel();
}

void BatchController::stopBatch()
{
    if (!m_running)
        return;
    m_stopRequested = true;
    if (m_runner->isRunning()) {
        m_runner->cancel();
    } else {
        // Between two rows: nothing to kill, just do not start the next one.
        m_running = false;
        emit batchFinished(buildReport());
    }
}

void BatchController::pause()
{
    m_runner->pause();
}

void BatchController::resume()
{
    m_runner->resume();
}

bool BatchController::isPaused() const
{
    return m_runner->isPaused();
}

bool BatchController::isRunning() const
{
    return m_running;
}

QString BatchController::buildReport() const
{
    int done = 0, failed = 0, invalid = 0, cancelled = 0, pending = 0;
    qint64 totalMs = 0;
    for (const RowResult &r : m_results) {
        totalMs += r.elapsedMs;
        switch (r.state) {
        case RowState::Done:      ++done; break;
        case RowState::Failed:    ++failed; break;
        case RowState::Invalid:   ++invalid; break;
        case RowState::Cancelled: ++cancelled; break;
        default:                  ++pending; break;
        }
    }

    QString out;
    out += tr("%1 of %2 rows completed.").arg(done).arg(m_results.size());
    out += QLatin1Char('\n');
    if (failed)
        out += tr("%1 failed.").arg(failed) + QLatin1Char('\n');
    if (invalid)
        out += tr("%1 rejected before starting.").arg(invalid) + QLatin1Char('\n');
    if (cancelled)
        out += tr("%1 cancelled.").arg(cancelled) + QLatin1Char('\n');
    if (pending)
        out += tr("%1 never started.").arg(pending) + QLatin1Char('\n');
    out += tr("Total time: %1").arg(QStringLiteral("%1:%2:%3")
                                        .arg(totalMs / 3600000, 2, 10, QLatin1Char('0'))
                                        .arg((totalMs / 60000) % 60, 2, 10, QLatin1Char('0'))
                                        .arg((totalMs / 1000) % 60, 2, 10, QLatin1Char('0')));

    // Then the rows that did not simply work, with the reason: this is what
    // the user actually needs after leaving the batch running overnight.
    QString problems;
    for (const RowResult &r : m_results) {
        if (r.state == RowState::Done || r.state == RowState::Pending)
            continue;
        problems += QLatin1Char('\n')
                    + tr("Chunk %1 row %2: %3")
                          .arg(r.where.chunk + 1)
                          .arg(r.where.row + 1)
                          .arg(r.message.isEmpty() ? tr("not run") : r.message.trimmed());
    }
    if (!problems.isEmpty())
        out += QLatin1Char('\n') + tr("Problems:") + problems;
    return out;
}
