#pragma once

#include <QList>
#include <QString>

#include "trajectarunner.h"

class QJsonObject;

// The data behind batch processing, and nothing else: no widgets, no QProcess.
//
// A batch is a list of chunks, each a list of rows. A **row** is one engine
// run — the smallest unit that can succeed or fail. A **chunk** groups rows
// that share an algorithm and a set of cost modifiers, which is exactly the
// pair of settings a user changes rarely and would otherwise have to repeat on
// every row. Everything above that (mode, hardware, where the output goes) is
// fixed for the whole batch.
//
// The point of keeping this layer free of Qt widgets is `toParameters()`: it
// turns one row into the very same `TrajectaRunner::Parameters` the single-run
// form produces, so the batch reuses the engine driver unchanged and can be
// tested head-less. If that function is right, the batch is right.
namespace Batch {

// One engine run. Which fields matter depends on Job::mode; the others are
// carried along untouched so switching mode back and forth does not lose what
// was already typed.
struct Row {
    // --- Input, FETE ---
    QString demPath;
    bool generatePoints = false;  // false: pointsPath is imported as-is
    QString pointsPath;
    bool genByTargetCount = true;  // false: genSpacing is used directly
    int genSpacing = 10;
    int genTargetCount = 5000;
    bool genRandom = false;  // false: regular grid, true: stratified random
    int genSeed = 1;
    int genEdgeBuffer = 0;
    QString genLayerName = QStringLiteral("sample_points");

    // --- Input, LCPA ---
    QString originPath;
    QString destinationsPath;

    // --- Output ---
    QString outputDir;
    // FETE: the density raster. LCPA: the paths raster, which may be left
    // empty as long as pathLinesName is not.
    QString outputName;
    QString pathLinesName = QStringLiteral("LCPS_vectors");  // LCPA only
};

// Rows sharing an algorithm and a set of cost modifiers.
struct Chunk {
    int neighbours = 16;            // 8, 16, 24, 32 or 64
    // 1 Tobler-White, 2 Marquez-Perez, 3 Irmischer-Clarke,
    // 4 Herzog (energy, kJ/kg), 5/6 Campbell 2019 5th/50th percentile
    int costFunction = 1;
    // Moves steeper than these are refused outright rather than priced.
    // Off by default, like everywhere else.
    bool slopeCutoffEnabled = false;
    int maxSlopeUpDeg = 30;
    int maxSlopeDownDeg = 30;
    // Cost corridor, LCPA rows only. Off by default: it costs an extra
    // search per destination, and a batch multiplies that by every row.
    bool costCorridor = false;
    double corridorWidthPercent = 10.0;
    int smoothingBufferRadius = 0;  // cells per side around computed paths

    bool useCostModifiers = false;
    QString costVectorPath;
    int polylineBufferRadius = 2;
    QString costRasterPath;
    bool barrierEnabled = true;
    double barrierThreshold = 1000.0;

    // Slope and the cost surfaces. Off by default: a batch is normally run for
    // its main result only, and the engine skips whatever is left unnamed. It
    // belongs to the chunk rather than the row because it is a decision about
    // this group of runs, not about one input.
    bool keepExtraRasters = false;

    // Whether the results of these rows are registered with the Viewer as they
    // finish. Off by default in a batch: dozens of rows would pile up hundreds
    // of layers, and each one keeps a file handle open.
    bool loadRastersInViewer = false;
    bool loadVectorsInViewer = false;

    // Purely presentational, but part of the document: a batch of six chunks is
    // built by folding away the ones already finished with.
    bool collapsed = false;

    QList<Row> rows;
};

// A whole batch.
struct Job {
    TrajectaRunner::Mode mode = TrajectaRunner::Mode::Fete;

    // Hardware is chosen once: the rows run one after another, so there is
    // never more than one engine process to size.
    int maxThreads = 1;
    int maxRamMb = 8192;
    bool largePages = false;
    // On by default here, unlike the single-run form: a batch is left running
    // unattended, so the detailed transcript is usually the only account of
    // what happened. Still the user's choice.
    bool verbose = true;

    // A run manifest written next to every row's results. On by default, and a
    // batch is where it earns the most: twenty rows differing by one setting
    // are exactly the situation in which nobody remembers, a month later,
    // which folder was which.
    bool writeManifest = true;

    // Each row writes into a subfolder of its output directory, named after
    // its output name. On by default, and deliberately a single switch for the
    // whole batch: the outputs of two rows sharing a folder would overwrite
    // each other, since slope and the cost surfaces have fixed names.
    bool folderPerRow = true;

    QList<Chunk> chunks;
};

// Where a row actually writes, once folderPerRow has been applied.
QString outputDirFor(const Job &job, const Row &row);

// The run parameters for one row. `env` supplies the fields the batch knows
// nothing about — exePath, gdalBinDir, projDataDir, gdalDataDir, workingDir —
// and every other field is overwritten from the job, chunk and row.
//
// Verbose is forced on: a batch runs unattended, so the detailed log is the
// only record of what happened, and it costs nothing to produce.
TrajectaRunner::Parameters toParameters(const Job &job, const Chunk &chunk,
                                        const Row &row,
                                        const TrajectaRunner::Parameters &env);

// Position of a row inside the job. The queue is a flat list of these.
struct Index {
    int chunk = -1;
    int row = -1;

    bool isValid() const { return chunk >= 0 && row >= 0; }
    bool operator==(const Index &o) const { return chunk == o.chunk && row == o.row; }
};

// Every row, in execution order.
QList<Index> flatten(const Job &job);

int rowCount(const Job &job);

// A problem found before anything is launched. A row-level issue only takes
// that row out of the run; a batch-level one (chunk < 0) stops the batch,
// because nothing would succeed anyway.
struct Issue {
    Index where;       // invalid chunk index means the whole batch
    QString message;
};

// Checks that do not need GDAL: files present, names sane, no two rows writing
// the same file. Anything requiring the CRS or the DEM extent is left to the
// engine, which already validates it and reports per row.
QList<Issue> validate(const Job &job);

// Persistence. The batch is a working document — dozens of rows typed by hand
// — so it lives in a file the user can keep, not in the registry.
QJsonObject toJson(const Job &job);
bool fromJson(const QJsonObject &obj, Job *job, QString *error);

} // namespace Batch
