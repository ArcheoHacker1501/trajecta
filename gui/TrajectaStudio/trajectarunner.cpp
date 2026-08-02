#include "trajectarunner.h"

#include <QDir>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <QRegularExpression>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

#ifdef Q_OS_WIN
// NtSuspendProcess/NtResumeProcess freeze or thaw every thread of a process
// in one call. They live in ntdll and are not in the SDK headers, but have
// been stable for decades (Task Manager and Process Explorer rely on them).
bool callNtProcessFn(const char *name, qint64 pid)
{
    using NtProcessFn = LONG(NTAPI *)(HANDLE);
    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (!ntdll)
        return false;
    const auto fn = reinterpret_cast<NtProcessFn>(
        reinterpret_cast<void *>(GetProcAddress(ntdll, name)));
    if (!fn)
        return false;
    const HANDLE process = OpenProcess(PROCESS_SUSPEND_RESUME, FALSE, DWORD(pid));
    if (!process)
        return false;
    const bool ok = fn(process) >= 0;  // NTSTATUS: >= 0 means success
    CloseHandle(process);
    return ok;
}
#endif

// Matches trajecta's progress bar lines in the *raw* (unstripped) stream:
// print_progress() always emits "\r\033[K\033[1m 45.3%\033[0m ..."
const QRegularExpression kProgressRe(QStringLiteral("\\r\\x1B\\[K\\x1B\\[1m\\s*(\\d{1,3}(?:\\.\\d)?)%"));

// Any ANSI CSI escape sequence (colors, clear-line, cursor movement).
const QRegularExpression kAnsiRe(QStringLiteral("\\x1B\\[[0-9;?]*[A-Za-z]"));

// Activity lines worth surfacing as a status message while trajecta works.
const QRegularExpression kStatusRe(QStringLiteral(
    "^(Loading|Reading|Computing|Calculating|Rasterizing|Writing|Validating|"
    "Processing|Building|Extracting|Applying|Saving|Running)\\b"));

bool looksLikePrompt(const QString &strippedBuffer)
{
    // Every trajecta question ends with "> " printed at the start of a line.
    // Trim trailing spaces (but not newlines) and check the tail.
    int end = strippedBuffer.size();
    while (end > 0 && (strippedBuffer.at(end - 1) == QLatin1Char(' ')
                       || strippedBuffer.at(end - 1) == QLatin1Char('\t')))
        --end;
    if (end == 0 || strippedBuffer.at(end - 1) != QLatin1Char('>'))
        return false;
    return end == 1 || strippedBuffer.at(end - 2) == QLatin1Char('\n')
           || strippedBuffer.at(end - 2) == QLatin1Char('\r');
}

} // namespace

TrajectaRunner::TrajectaRunner(QObject *parent)
    : QObject(parent)
{
}

bool TrajectaRunner::isRunning() const
{
    return m_process && m_process->state() != QProcess::NotRunning;
}

QString TrajectaRunner::stripAnsi(QString text)
{
    return text.remove(kAnsiRe);
}

void TrajectaRunner::buildRules()
{
    m_rules.clear();

    auto add = [this](const char *key, std::function<QString()> answer) {
        m_rules.append({QString::fromLatin1(key), std::move(answer)});
    };

    // --- Startup ---
    add("Choose computation mode",
        [this] { return QString::number(static_cast<int>(m_params.mode)); });
    add("Enable detailed debug output",
        [this] { return m_params.verbose ? QStringLiteral("yes") : QStringLiteral("no"); });

    // --- Performance ---
    add("Enter maximum CPU threads",
        [this] { return QString::number(m_params.maxThreads); });
    add("Enter maximum RAM to allocate",
        [this] { return QString::number(m_params.maxRamMb); });
    add("Use large memory pages",
        [this] { return m_params.largePages ? QStringLiteral("yes") : QStringLiteral("no"); });

    // --- Input data ---
    add("Enter path to DEM file",
        [this] { return QDir::toNativeSeparators(m_params.demPath); });
    add("Sample points source",
        [this] { return m_params.generatePoints ? QStringLiteral("2") : QStringLiteral("1"); });
    add("Enter path to sample points file",
        [this] { return QDir::toNativeSeparators(m_params.pointsPath); });

    // Sample point generation. These questions are only ever asked when the
    // answer to "Sample points source" was 2, so in import mode none of these
    // rules can fire.
    add("express the point density",
        [this] { return m_params.genByTargetCount ? QStringLiteral("2") : QStringLiteral("1"); });
    add("Enter point spacing in cells",
        [this] { return QString::number(m_params.genSpacing); });
    add("Enter the target number of points",
        [this] { return QString::number(m_params.genTargetCount); });
    add("Select point arrangement",
        [this] { return m_params.genRandom ? QStringLiteral("2") : QStringLiteral("1"); });
    add("Enter the random seed",
        [this] { return QString::number(m_params.genSeed); });
    add("Enter edge buffer in cells",
        [this] { return QString::number(m_params.genEdgeBuffer); });
    add("Enter filename for the generated points layer",
        [this] { return m_params.genLayerName; });

    add("Enter path to ORIGIN file",
        [this] { return QDir::toNativeSeparators(m_params.originPath); });
    add("Enter path to DESTINATIONS file",
        [this] { return QDir::toNativeSeparators(m_params.destinationsPath); });
    add("Enter output directory",
        [this] { return QDir::toNativeSeparators(m_params.outputDir); });

    // --- Cost modifiers ---
    add("Do you want to add additional cost modifiers",
        [this] { return m_params.useCostModifiers ? QStringLiteral("yes") : QStringLiteral("no"); });
    add("Enter path to cost modifiers vector file",
        [this] { return QDir::toNativeSeparators(m_params.costVectorPath); });  // empty = skip
    add("for polyline rasterization",
        [this] { return QString::number(m_params.polylineBufferRadius); });
    add("Enter path to cost modifiers raster",
        [this] { return QDir::toNativeSeparators(m_params.costRasterPath); });  // empty = skip
    add("Treat extreme cost multipliers as impassable barriers",
        [this] { return QString::number(m_params.barrierThreshold, 'f', 1); });

    // --- NNI post-processing (mode 3) ---
    add("Enter path to density raster",
        [this] { return QDir::toNativeSeparators(m_params.interpInputRaster); });
    add("minimum density value",
        [this] { return QString::number(m_params.interpThreshold); });
    add("sample spacing",
        [this] { return QString::number(m_params.interpSampleSpacing); });
    add("maximum search radius",
        [this] { return QString::number(m_params.interpMaxRadius); });
    add("Enter interpolated raster filename",
        [this] { return m_params.interpOutputName; });

    // --- Algorithm ---
    add("Select number of neighbours", [this] {
        switch (m_params.neighbours) {
        case 8:  return QStringLiteral("1");
        case 24: return QStringLiteral("3");
        case 32: return QStringLiteral("4");
        case 64: return QStringLiteral("5");
        case 16:
        default: return QStringLiteral("2");
        }
    });
    add("Select cost function",
        [this] { return QString::number(m_params.costFunction); });
    add("for path smoothing",
        [this] { return QString::number(m_params.smoothingBufferRadius); });

    // --- Output file names ---
    add("Enter slope raster filename", [this] { return m_params.slopeName; });
    add("Enter base cost surface raster filename", [this] { return m_params.costName; });
    add("Enter additional cost surface raster filename", [this] { return m_params.additionalCostName; });
    add("Enter total cost surface raster filename", [this] { return m_params.totalCostName; });
    add("Enter output density raster filename", [this] { return m_params.densityName; });
    add("Enter path raster filename", [this] { return m_params.pathRasterName; });
    add("Enter path lines shapefile filename", [this] { return m_params.pathLinesName; });

    // --- Session end: run once, then quit cleanly ---
    add("Run another", [] { return QStringLiteral("no"); });
    add("Generate another", [] { return QStringLiteral("no"); });
    add("Exit program?", [] { return QStringLiteral("yes"); });
}

void TrajectaRunner::start(const Parameters &params)
{
    if (isRunning())
        return;

    m_params = params;
    m_askCount.clear();
    m_pending.clear();
    m_fullLog.clear();
    m_errBuffer.clear();
    m_outDecoder.resetState();
    m_errDecoder.resetState();
    m_cancelled = false;
    m_aborted = false;
    m_paused = false;
    m_abortReason.clear();
    buildRules();

    delete m_process;
    m_process = new QProcess(this);
    // Keep stderr separate: GDAL/PROJ write unbuffered error lines that would
    // otherwise interleave mid-line with stdout and corrupt the Unicode
    // charts and progress bars drawn by the engine.
    m_process->setProcessChannelMode(QProcess::SeparateChannels);

    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    if (!m_params.gdalBinDir.isEmpty()) {
        env.insert(QStringLiteral("PATH"),
                   QDir::toNativeSeparators(m_params.gdalBinDir) + QLatin1Char(';')
                       + env.value(QStringLiteral("PATH")));
    }
    // PROJ/GDAL need their data files (proj.db, coordinate tables). Export the
    // detected locations, but never override what the user configured.
    // PROJ >= 9.1 reads PROJ_DATA and only falls back to PROJ_LIB, so both are
    // set: leaving PROJ_DATA unset lets the library keep resolving proj.db
    // through its built-in path, which on a multi-version OSGeo4W root is the
    // stale copy next to the oldest DLLs.
    if (!m_params.projDataDir.isEmpty()
        && env.value(QStringLiteral("PROJ_LIB")).isEmpty()
        && env.value(QStringLiteral("PROJ_DATA")).isEmpty()) {
        const QString projDir = QDir::toNativeSeparators(m_params.projDataDir);
        env.insert(QStringLiteral("PROJ_DATA"), projDir);
        env.insert(QStringLiteral("PROJ_LIB"), projDir);
    }
    if (!m_params.gdalDataDir.isEmpty()
        && env.value(QStringLiteral("GDAL_DATA")).isEmpty()) {
        env.insert(QStringLiteral("GDAL_DATA"),
                   QDir::toNativeSeparators(m_params.gdalDataDir));
    }
    m_process->setProcessEnvironment(env);

    if (!m_params.workingDir.isEmpty())
        m_process->setWorkingDirectory(m_params.workingDir);

    connect(m_process, &QProcess::readyReadStandardOutput,
            this, &TrajectaRunner::onReadyRead);
    connect(m_process, &QProcess::readyReadStandardError,
            this, &TrajectaRunner::onReadyReadError);
    connect(m_process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this, &TrajectaRunner::onProcessFinished);
    connect(m_process, &QProcess::errorOccurred,
            this, &TrajectaRunner::onProcessError);

    emit statusChanged(tr("Starting the Trajecta engine..."));
    m_process->start(m_params.exePath, QStringList());
}

void TrajectaRunner::cancel()
{
    if (!isRunning())
        return;
    m_cancelled = true;
    resume();  // thaw first so the kill is delivered and reported promptly
    m_process->kill();
}

void TrajectaRunner::pause()
{
    if (!isRunning() || m_paused)
        return;
#ifdef Q_OS_WIN
    if (callNtProcessFn("NtSuspendProcess", m_process->processId())) {
        m_paused = true;
        emit pauseStateChanged(true);
    }
#endif
}

void TrajectaRunner::resume()
{
    if (!m_paused)
        return;
#ifdef Q_OS_WIN
    if (isRunning())
        callNtProcessFn("NtResumeProcess", m_process->processId());
#endif
    m_paused = false;
    emit pauseStateChanged(false);
}

void TrajectaRunner::onReadyRead()
{
    const QByteArray bytes = m_process->readAllStandardOutput();
    // trajecta prints UTF-8 (Unicode progress bars, charts, middle dots).
    const QString raw = m_outDecoder(bytes);

    emit consoleOutput(raw);

    // Progress: use the last progress-bar match in this chunk.
    QRegularExpressionMatchIterator it = kProgressRe.globalMatch(raw);
    QRegularExpressionMatch lastMatch;
    while (it.hasNext())
        lastMatch = it.next();
    if (lastMatch.hasMatch())
        emit progressChanged(lastMatch.captured(1).toDouble());

    QString stripped = stripAnsi(raw);
    stripped.replace(QLatin1Char('\r'), QLatin1Char('\n'));
    m_pending += stripped;
    m_fullLog += stripped;
    // Multi-hour verbose runs would otherwise grow the transcript without
    // bound; the report extraction only ever looks at the tail.
    if (m_fullLog.size() > 4 * 1024 * 1024)
        m_fullLog = m_fullLog.right(2 * 1024 * 1024);

    // Status: surface the last recognisable activity line of this chunk.
    const QStringList lines = stripped.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    for (auto rit = lines.crbegin(); rit != lines.crend(); ++rit) {
        const QString line = rit->trimmed();
        if (kStatusRe.match(line).hasMatch()) {
            emit statusChanged(line);
            break;
        }
    }

    handlePendingOutput();
}

void TrajectaRunner::onReadyReadError()
{
    m_errBuffer += m_errDecoder(m_process->readAllStandardError());

    // Emit only complete lines: GDAL/PROJ messages then reach the console as
    // whole blocks instead of fragments spliced into the engine's charts.
    int nl;
    while ((nl = m_errBuffer.indexOf(QLatin1Char('\n'))) >= 0) {
        QString line = m_errBuffer.left(nl);
        m_errBuffer.remove(0, nl + 1);
        if (line.endsWith(QLatin1Char('\r')))
            line.chop(1);
        line = stripAnsi(line);
        m_fullLog += line + QLatin1Char('\n');
        if (!line.trimmed().isEmpty())
            emit consoleErrorLine(line);
    }
}

void TrajectaRunner::handlePendingOutput()
{
    if (m_aborted || m_cancelled)
        return;
    if (looksLikePrompt(m_pending))
        answerPrompt();
}

void TrajectaRunner::answerPrompt()
{
    // Identify which question is being asked. m_pending holds everything that
    // was printed since our previous answer, so exactly one question is in it;
    // if several keys match (e.g. banner text), take the one closest to the
    // prompt, i.e. with the highest position in the buffer.
    int bestPos = -1;
    const PromptRule *bestRule = nullptr;
    for (const PromptRule &rule : m_rules) {
        const int pos = m_pending.lastIndexOf(rule.key);
        if (pos > bestPos) {
            bestPos = pos;
            bestRule = &rule;
        }
    }

    if (!bestRule) {
        abortRun(tr("Trajecta asked an unexpected question that the interface "
                    "does not know how to answer:\n\n%1")
                     .arg(m_pending.right(400).trimmed()));
        return;
    }

    // The same question asked twice means trajecta rejected our previous
    // answer (its internal validation failed): stop and report the error.
    const int timesAsked = ++m_askCount[bestRule->key];
    if (timesAsked > 1) {
        QString detail = errorLinesFromLog();
        if (detail.isEmpty())
            detail = tr("Trajecta did not accept one of the provided inputs "
                        "(question repeated: \"%1...\").").arg(bestRule->key);
        abortRun(detail);
        return;
    }

    const QString answer = bestRule->makeAnswer();
    m_pending.clear();

    // toLocal8Bit matches how the console version receives typed paths.
    m_process->write(answer.toLocal8Bit() + '\n');
    emit answerSent(answer.isEmpty() ? tr("(blank — skip)") : answer);
}

void TrajectaRunner::abortRun(const QString &reason)
{
    m_aborted = true;
    m_abortReason = reason;
    if (isRunning()) {
        resume();
        m_process->kill();
    }
}

QString TrajectaRunner::errorLinesFromLog() const
{
    // ERROR lines describe the failure; WARNING lines are context at best.
    // Report warnings only when there is no error at all, so an abort message
    // never leads with an unrelated warning.
    QStringList errors;
    QStringList warnings;
    const QStringList lines = m_fullLog.split(QLatin1Char('\n'));
    for (const QString &line : lines) {
        const QString t = line.trimmed();
        if (t.startsWith(QLatin1String("ERROR"), Qt::CaseInsensitive)
            || t.contains(QLatin1String("Please correct"))) {
            errors << t;
        } else if (t.startsWith(QLatin1String("WARNING"), Qt::CaseInsensitive)) {
            warnings << t;
        }
    }
    QStringList &found = errors.isEmpty() ? warnings : errors;
    // Keep the last few: the most recent ones describe the actual failure.
    while (found.size() > 6)
        found.removeFirst();
    return found.join(QLatin1Char('\n'));
}

QString TrajectaRunner::extractResultReport() const
{
    // Grab everything from "Output Summary:" up to the question that closes the
    // session ("Run another..." after an analysis, "Generate another..." after
    // a points-only run).
    const int start = m_fullLog.lastIndexOf(QLatin1String("Output Summary:"));
    if (start < 0)
        return QString();
    int end = m_fullLog.indexOf(QLatin1String("Run another"), start);
    if (end < 0)
        end = m_fullLog.indexOf(QLatin1String("Generate another"), start);
    if (end < 0)
        end = m_fullLog.size();
    QString report = m_fullLog.mid(start, end - start).trimmed();
    report.replace(QRegularExpression(QStringLiteral("\\n{2,}")),
                   QStringLiteral("\n"));
    return report;
}

void TrajectaRunner::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    // The process can die while flagged paused (e.g. killed from outside):
    // make sure the UI never stays stuck in the paused state.
    if (m_paused) {
        m_paused = false;
        emit pauseStateChanged(false);
    }

    // Flush a trailing stderr fragment that never got its newline.
    if (!m_errBuffer.isEmpty()) {
        const QString line = stripAnsi(m_errBuffer);
        m_errBuffer.clear();
        m_fullLog += line + QLatin1Char('\n');
        if (!line.trimmed().isEmpty())
            emit consoleErrorLine(line);
    }

    if (m_cancelled) {
        emit finished(Outcome::Cancelled, tr("The analysis was cancelled."));
        return;
    }
    if (m_aborted) {
        emit finished(Outcome::Failed, m_abortReason);
        return;
    }

    // "...successfully computed!" from an analysis, "...successfully
    // generated!" from a points-only run.
    const bool computed = m_fullLog.contains(QLatin1String("successfully computed!"))
                          || m_fullLog.contains(QLatin1String("successfully generated!"));
    if (computed) {
        emit progressChanged(100.0);
        QString report = extractResultReport();
        if (report.isEmpty())
            report = tr("The analysis completed successfully.");
        emit finished(Outcome::Success, report);
        return;
    }

    // No success marker: figure out the most helpful explanation.
    QString reason = errorLinesFromLog();
    if (reason.isEmpty()) {
        if (exitStatus == QProcess::CrashExit || exitCode != 0) {
            if (quint32(exitCode) == 0xC0000135u) {
                reason = tr("trajecta.exe could not start because the GDAL "
                            "libraries were not found.\n\nInstall GDAL through "
                            "OSGeo4W (see the Guide page) or point the interface "
                            "to your OSGeo4W\\bin folder from the sidebar.");
            } else {
                reason = tr("The Trajecta engine stopped unexpectedly "
                            "(exit code %1).").arg(exitCode);
            }
        } else {
            reason = tr("The analysis ended without producing a result. "
                        "Check the console log for details.");
        }
    }
    emit finished(Outcome::Failed, reason);
}

void TrajectaRunner::onProcessError(QProcess::ProcessError error)
{
    if (error == QProcess::FailedToStart) {
        emit finished(Outcome::Failed,
                      tr("Could not start the Trajecta engine:\n%1\n\n"
                         "Use \"Locate engine\" in the sidebar to select "
                         "trajecta.exe manually.")
                          .arg(QDir::toNativeSeparators(m_params.exePath)));
    }
    // Crashed/killed cases are reported by onProcessFinished.
}
