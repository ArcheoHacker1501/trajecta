#pragma once

#include <QList>
#include <QString>

#include "trajectarunner.h"

class QJsonObject;

// The data behind the post-processing batch, and nothing else: no widgets, no
// engine process. Mirrors batchmodel.h's Batch:: namespace, but simpler,
// because the three post-processing tools do not group naturally into rows
// that share settings the way FETE and LCPA runs do — see PostBatch::Chunk.
namespace PostBatch {

// Which of the three post-processing tools this batch runs. One job runs one
// tool throughout, the same way a Processing batch runs one of FETE or LCPA
// throughout — switching tool mid-batch would mean every chunk carrying
// fields for a tool it does not use, which is exactly the confusion the
// "simpler chunks" request was asking to avoid.
enum class Mode { Nni = 1, Compare = 2, Coherence = 3 };

QString modeLabel(Mode mode);

// One chunk = one analysis. Unlike Batch::Chunk, which groups many rows under
// one algorithm, a post-processing chunk already is the smallest unit: an NNI
// interpolation, a route comparison or a coherence score is not naturally
// split into an algorithm shared by several inputs. Fields for all three
// tools live side by side, and only the ones the job's mode reads are used —
// the same reasoning as Batch::Row keeping FETE and LCPA fields apart:
// switching mode must not throw away what was already typed.
struct Chunk {
    // --- NNI ---
    QString interpInputRaster;
    QString interpOutputDir;
    double interpThreshold = 1.0;
    int interpSampleSpacing = 4;
    bool interpPreservePeaks = false;
    int interpMaxRadius = 0;
    QString interpOutputName = QStringLiteral("FETE_density_NNI");

    // --- Compare with a known route ---
    QString cmpComputedPath;
    QString cmpKnownPath;
    double cmpTolerance = 100.0;

    // --- Site-corridor coherence ---
    QString cohRasterPath;
    QString cohPointsPath;
    double cohRadius = 250.0;
    int cohThresholdMode = 0;      // Coherence::ThresholdMode, kept as int here
                                   // so this header stays free of coherence.h
    double cohThresholdValue = 1.0;
    bool cohNullModel = true;
    int cohNullMode = 0;           // Coherence::NullMode
    int cohNullReplicates = 999;
    bool cohSensitivity = false;
    QString cohSensitivityRadii = QStringLiteral("100, 250, 500, 1000");
    // The "share of sites within X metres" ladder. Fixed metres, so two
    // chunks can be compared row for row whatever radius each was given.
    QString cohEcdfDistances = QStringLiteral("0, 100, 250, 500, 1000, 2500");
    bool cohEdgeGuard = true;
    bool cohWriteHistogramScript = true;
    QString cohOutputDir;
    QString cohPrefix = QStringLiteral("coherence");
    bool cohVectorAsGeoPackage = true;
    bool cohWriteDistanceRaster = true;

    // Registers this chunk's result with the Viewer once it finishes. On by
    // default here — unlike Processing's batch, where dozens of FETE/LCPA
    // rows would pile up hundreds of layers — because a post-processing batch
    // is small by nature and looking at each result is usually the point. The
    // one option a row keeps in this simpler batch: everything else that
    // would normally vary per row (hardware, whether to keep extra rasters,
    // a shared folder) either does not apply here or is chosen once for the
    // whole job.
    bool loadInViewer = true;

    bool collapsed = false;
};

// A whole post-processing batch.
struct Job {
    Mode mode = Mode::Nni;

    // Hardware, NNI only: Compare and Coherence run in the interface and
    // never see these. Kept on the job regardless of mode, exactly like
    // Batch::Job, so switching mode does not lose what was chosen.
    int maxThreads = 1;
    int maxRamMb = 8192;
    bool largePages = false;
    bool writeManifest = true;

    QList<Chunk> chunks;
};

// A problem found before anything is launched. A chunk-level issue only takes
// that chunk out of the run; a job-level one (chunk < 0) stops the batch.
struct Issue {
    int chunk = -1;
    QString message;
};

// Checks that do not need GDAL: files present, names sane, no two NNI chunks
// writing the same output. Anything requiring the CRS or the raster extent is
// left to RouteCompare::compare() / Coherence::run(), which already validate
// it and report per chunk.
QList<Issue> validate(const Job &job);

// The run parameters for one NNI chunk. `env` supplies exePath, gdalBinDir,
// projDataDir, gdalDataDir and workingDir; hardware and the interpolation
// fields come from the job and the chunk. Meaningless for Compare and
// Coherence chunks, which never build a TrajectaRunner::Parameters — those
// two call RouteCompare::compare() / Coherence::run() directly, in-process.
TrajectaRunner::Parameters toParameters(const Job &job, const Chunk &chunk,
                                        const TrajectaRunner::Parameters &env);

QJsonObject toJson(const Job &job);
bool fromJson(const QJsonObject &obj, Job *job, QString *error);

} // namespace PostBatch
