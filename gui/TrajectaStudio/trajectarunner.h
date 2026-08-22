#pragma once

#include <QHash>
#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringConverter>
#include <QStringList>

#include <functional>

// TrajectaRunner drives the interactive trajecta.exe console program.
//
// trajecta.exe asks its questions one at a time on stdout and reads the
// answers from stdin. Because std::cin is tied to std::cout, the full text of
// every question (ending with the "> " prompt) is guaranteed to be flushed
// through the pipe before the process blocks waiting for input. The runner
// therefore accumulates stdout, waits until the stream ends with a "> "
// prompt, recognises which question was asked from its distinctive wording,
// and writes back the answer derived from the Parameters set by the GUI.
//
// A question that is asked twice means trajecta rejected the previous answer
// (its own validation failed). In that case the run is aborted and the ERROR
// lines captured from the transcript are reported to the user.
class TrajectaRunner : public QObject
{
    Q_OBJECT

public:
    // Points writes a sample-points layer and stops, with no analysis: the
    // same code path FETE uses to generate its own, so the layer produced can
    // be looked at first and then consumed by a run unchanged.
    enum class Mode { Fete = 1, Lcpa = 2, Interp = 3, Points = 4 };

    // How a run ended. A typed outcome instead of string-matching the report
    // text, which broke as soon as the two translations diverged.
    enum class Outcome { Success, Cancelled, Failed };

    struct Parameters {
        Mode mode = Mode::Fete;
        bool verbose = false;

        // Performance
        int maxThreads = 1;
        int maxRamMb = 8192;
        // 2 MB pages for the engine's per-thread buffers. Opt-in: the engine
        // falls back to normal pages by itself when they are unavailable.
        bool largePages = false;

        // Write a text record of the run next to its results. On by default:
        // it costs seconds and answers, months later, which DEM and which
        // settings produced the raster somebody is looking at.
        bool writeManifest = true;

        // Input data
        QString demPath;
        QString pointsPath;        // FETE only
        QString originPath;        // LCPA only
        QString destinationsPath;  // LCPA only
        QString outputDir;

        // Sample points generated from the DEM instead of imported (FETE only).
        // While generatePoints is false the engine never asks any of the
        // questions below and pointsPath is used exactly as before.
        bool generatePoints = false;
        bool genByTargetCount = true;    // false: genSpacing is used directly
        int genSpacing = 10;             // one point every N cells, both axes
        int genTargetCount = 5000;       // desired count (genByTargetCount)
        bool genRandom = false;          // false: regular grid, true: stratified random
        int genSeed = 1;
        int genEdgeBuffer = 0;   // cells kept clear along each DEM border
        QString genLayerName = QStringLiteral("sample_points");

        // Optional cost modifiers
        bool useCostModifiers = false;
        QString costVectorPath;            // polylines with a 'cost' field (may be empty)
        int polylineBufferRadius = 2;      // cells per side used when rasterizing polylines
        QString costRasterPath;            // multiplier raster (may be empty)
        double barrierThreshold = 1000.0;  // multipliers >= threshold are impassable, 0 = disabled

        // Algorithm
        int neighbours = 16;           // 8, 16, 24, 32 or 64
        // 1 Tobler-White 2015, 2 Marquez-Perez 2017, 3 Irmischer-Clarke 2017,
        // 4 Herzog 2013 (energy, kJ/kg), 5/6 Campbell 2019 5th/50th percentile
        int costFunction = 1;
        // Moves steeper than these are refused outright rather than priced.
        // Off by default: a limit nobody chose would change every result.
        bool slopeCutoffEnabled = false;
        int maxSlopeUpDeg = 30;
        int maxSlopeDownDeg = 30;
        int smoothingBufferRadius = 0; // cells per side around computed paths

        // NNI post-processing (Interp mode only)
        QString interpInputRaster;      // density raster to interpolate
        double interpThreshold = 1.0;   // cells >= threshold become samples
        int interpSampleSpacing = 4;
        // Also keep each block's maximum, so peaks survive subsampling.
        bool interpPreservePeaks = false;    // sample every Nth cell (1 = every cell)
        int interpMaxRadius = 0;        // cells, 0 = unlimited
        QString interpOutputName = QStringLiteral("FETE_density_NNI");

        // Output file names (entered without extension)
        QString slopeName = QStringLiteral("slope");
        QString costName = QStringLiteral("cost_surface");
        QString additionalCostName = QStringLiteral("cost_surface_additional");
        QString totalCostName = QStringLiteral("cost_surface_total");
        QString densityName = QStringLiteral("FETE_density");    // FETE
        QString pathRasterName = QStringLiteral("raster_lcps");  // LCPA
        QString pathLinesName = QStringLiteral("LCPS_vectors");  // LCPA
        // Cost corridor, LCPA only. Off by default: it costs an extra
        // search per destination.
        bool costCorridor = false;
        double corridorWidthPercent = 10.0;
        QString corridorName = QStringLiteral("cost_corridor");

        // Automatic saving of the FETE propagation phase. Off by default and
        // passed to the engine as environment variables, so a run that does not
        // want it behaves exactly as it always did.
        bool checkpointEnabled = false;
        // Fractional on purpose: the interface only ever offers whole minutes,
        // but the test harness needs an interval short enough to fire twice in
        // a run that lasts seconds.
        double checkpointMinutes = 30.0;
        QString checkpointDir;   // empty: the engine's own AppData location
        // Non-empty: resume this checkpoint instead of starting from source 0.
        QString resumeCheckpoint;

        // Environment
        QString exePath;      // full path to trajecta.exe
        QString gdalBinDir;   // if non-empty, prepended to PATH of the child process
        QString projDataDir;  // if non-empty, exported as PROJ_LIB (unless already set)
        QString gdalDataDir;  // if non-empty, exported as GDAL_DATA (unless already set)
        QString workingDir;   // where trajecta writes fete_config.txt / lcpa_config.txt
    };

    explicit TrajectaRunner(QObject *parent = nullptr);

    void start(const Parameters &params);
    void cancel();
    bool isRunning() const;

    // Freeze/thaw the whole engine process (Windows: NtSuspendProcess).
    // Pausing releases the CPU immediately; the engine's RAM stays allocated
    // and its state survives system sleep/hibernation, but not a shutdown.
    void pause();
    void resume();
    bool isPaused() const { return m_paused; }

signals:
    void consoleOutput(const QString &rawText);  // raw stdout, ANSI escapes included
    void consoleErrorLine(const QString &line);  // one complete stderr line (GDAL/PROJ messages)
    void pauseStateChanged(bool paused);         // engine process frozen / thawed
    void answerSent(const QString &answer);      // what the runner typed (for transparency)
    void progressChanged(double percent);        // 0..100, from trajecta's progress bar
    void statusChanged(const QString &status);   // last meaningful activity line
    void finished(TrajectaRunner::Outcome outcome, const QString &report);

private slots:
    void onReadyRead();
    void onReadyReadError();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);

private:
    struct PromptRule {
        QString key;                          // distinctive substring of the question text
        std::function<QString()> makeAnswer;  // supplies the answer when the question appears
    };

    void buildRules();
    void handlePendingOutput();
    void answerPrompt();
    void abortRun(const QString &reason);
    QString errorLinesFromLog() const;
    QString extractResultReport() const;
    static QString stripAnsi(QString text);

    QProcess *m_process = nullptr;
    Parameters m_params;
    QList<PromptRule> m_rules;
    QHash<QString, int> m_askCount;
    QString m_pending;  // ANSI-stripped output accumulated since the last answer
    QString m_fullLog;  // ANSI-stripped transcript of the whole run
    // Stateful decoders: a multi-byte UTF-8 character split across two pipe
    // chunks must not decode to U+FFFD replacement characters.
    QStringDecoder m_outDecoder{QStringDecoder::Utf8};
    QStringDecoder m_errDecoder{QStringDecoder::Utf8};
    QString m_errBuffer;  // stderr accumulated until a complete line is available
    bool m_cancelled = false;
    bool m_aborted = false;
    bool m_paused = false;
    QString m_abortReason;
};
