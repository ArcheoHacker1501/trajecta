#include "batchmodel.h"

#include <QDir>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QRegularExpression>

namespace Batch {

namespace {

// What the row is called when one short name is needed for it: the main output
// name, falling back to the shapefile for an LCPA row that only saves vectors.
QString label(const Row &row)
{
    const QString main = row.outputName.trimmed();
    return main.isEmpty() ? row.pathLinesName.trimmed() : main;
}

bool nameHasBadChars(const QString &name)
{
    static const QRegularExpression bad(QStringLiteral("[\\\\/:*?\"<>|]"));
    return bad.match(name).hasMatch();
}

// The engine mangles non-ASCII paths, so they are rejected before the run
// rather than failing halfway through it.
bool isAscii(const QString &s)
{
    for (const QChar &c : s)
        if (c.unicode() > 127)
            return false;
    return true;
}

bool fileMissing(const QString &path)
{
    return path.trimmed().isEmpty() || !QFileInfo::exists(path);
}

// Every file a row will write, absolute and normalised, so two rows aiming at
// the same one can be spotted before either runs.
QStringList plannedFiles(const Job &job, const Chunk &chunk, const Row &row)
{
    const QDir dir(outputDirFor(job, row));
    QStringList files;
    auto add = [&](const QString &name, const QString &ext) {
        if (!name.trimmed().isEmpty())
            files << QDir::cleanPath(dir.absoluteFilePath(name.trimmed() + ext));
    };

    if (job.mode == TrajectaRunner::Mode::Fete) {
        add(row.outputName, QStringLiteral(".tif"));
        if (row.generatePoints)
            add(row.genLayerName, QStringLiteral(".shp"));
    } else {
        add(row.outputName, QStringLiteral(".tif"));
        add(row.pathLinesName, QStringLiteral(".shp"));
    }

    // The intermediate rasters have fixed names, so they are exactly what
    // collides when two rows share a folder.
    if (chunk.keepExtraRasters) {
        add(QStringLiteral("slope"), QStringLiteral(".tif"));
        add(QStringLiteral("cost_surface"), QStringLiteral(".tif"));
        if (chunk.useCostModifiers) {
            add(QStringLiteral("cost_surface_additional"), QStringLiteral(".tif"));
            add(QStringLiteral("cost_surface_total"), QStringLiteral(".tif"));
        }
    }
    return files;
}

} // namespace

QString outputDirFor(const Job &job, const Row &row)
{
    const QString name = label(row);
    if (!job.folderPerRow || name.isEmpty() || row.outputDir.trimmed().isEmpty())
        return row.outputDir;
    return QDir(row.outputDir).filePath(name);
}

TrajectaRunner::Parameters toParameters(const Job &job, const Chunk &chunk,
                                        const Row &row,
                                        const TrajectaRunner::Parameters &env)
{
    TrajectaRunner::Parameters p = env;  // exePath, GDAL/PROJ dirs, working dir

    // Cleared unconditionally: exactly one row of a resumed batch — the one
    // that was interrupted — may pick up engine state, and BatchController is
    // what decides which. Inheriting it from the environment would hand the
    // same half-finished density to every row of the batch.
    p.resumeCheckpoint.clear();

    p.mode = job.mode;
    p.verbose = job.verbose;
    p.writeManifest = job.writeManifest;
    p.maxThreads = job.maxThreads;
    p.maxRamMb = job.maxRamMb;
    p.largePages = job.largePages;

    p.demPath = row.demPath;
    p.outputDir = outputDirFor(job, row);

    const bool fete = job.mode == TrajectaRunner::Mode::Fete;
    if (fete) {
        p.generatePoints = row.generatePoints;
        if (row.generatePoints) {
            p.genByTargetCount = row.genByTargetCount;
            p.genSpacing = row.genSpacing;
            p.genTargetCount = row.genTargetCount;
            p.genRandom = row.genRandom;
            p.genSeed = row.genSeed;
            p.genEdgeBuffer = row.genEdgeBuffer;
            p.genLayerName = row.genLayerName;
            // The engine writes the layer into the output folder and reads it
            // straight back; this is where it will land.
            p.pointsPath = QDir(p.outputDir).filePath(row.genLayerName.trimmed()
                                                      + QStringLiteral(".shp"));
        } else {
            p.pointsPath = row.pointsPath;
        }
        p.originPath.clear();
        p.destinationsPath.clear();
    } else {
        p.generatePoints = false;
        p.pointsPath.clear();
        p.originPath = row.originPath;
        p.destinationsPath = row.destinationsPath;
    }

    p.useCostModifiers = chunk.useCostModifiers;
    p.costVectorPath = chunk.useCostModifiers ? chunk.costVectorPath : QString();
    p.costRasterPath = chunk.useCostModifiers ? chunk.costRasterPath : QString();
    p.polylineBufferRadius = chunk.polylineBufferRadius;
    // 0 is how the engine is told to treat extreme multipliers as soft costs.
    p.barrierThreshold = chunk.barrierEnabled ? chunk.barrierThreshold : 0.0;

    p.neighbours = chunk.neighbours;
    p.costFunction = chunk.costFunction;
    p.costCorridor = chunk.costCorridor;
    p.corridorWidthPercent = chunk.corridorWidthPercent;
    // Named after the row rather than fixed, so two rows sharing an
    // output folder cannot overwrite each other's corridor.
    p.corridorName = row.outputName.trimmed().isEmpty()
                         ? QStringLiteral("cost_corridor")
                         : row.outputName.trimmed() + QStringLiteral("_corridor");
    p.slopeCutoffEnabled = chunk.slopeCutoffEnabled;
    p.maxSlopeUpDeg = chunk.maxSlopeUpDeg;
    p.maxSlopeDownDeg = chunk.maxSlopeDownDeg;
    p.smoothingBufferRadius = chunk.smoothingBufferRadius;

    // An empty name means the engine does not write that file at all.
    if (chunk.keepExtraRasters) {
        p.slopeName = QStringLiteral("slope");
        p.costName = QStringLiteral("cost_surface");
        p.additionalCostName = QStringLiteral("cost_surface_additional");
        p.totalCostName = QStringLiteral("cost_surface_total");
    } else {
        p.slopeName.clear();
        p.costName.clear();
        p.additionalCostName.clear();
        p.totalCostName.clear();
    }
    if (fete) {
        p.densityName = row.outputName.trimmed();
    } else {
        p.pathRasterName = row.outputName.trimmed();
        p.pathLinesName = row.pathLinesName.trimmed();
    }
    return p;
}

QList<Index> flatten(const Job &job)
{
    QList<Index> out;
    for (int c = 0; c < job.chunks.size(); ++c)
        for (int r = 0; r < job.chunks.at(c).rows.size(); ++r)
            out.append(Index{c, r});
    return out;
}

int rowCount(const Job &job)
{
    int n = 0;
    for (const Chunk &c : job.chunks)
        n += c.rows.size();
    return n;
}

QList<Issue> validate(const Job &job)
{
    QList<Issue> issues;
    const bool fete = job.mode == TrajectaRunner::Mode::Fete;

    if (rowCount(job) == 0) {
        issues.append({Index{},
                       QObject::tr("The batch is empty: add at least one chunk "
                                   "with one row.")});
        return issues;
    }

    // chunk-level
    for (int c = 0; c < job.chunks.size(); ++c) {
        const Chunk &chunk = job.chunks.at(c);
        const Index at{c, -1};
        if (chunk.rows.isEmpty())
            issues.append({at, QObject::tr("Chunk %1 has no rows.").arg(c + 1)});
        if (chunk.useCostModifiers) {
            if (!chunk.costVectorPath.trimmed().isEmpty()
                && fileMissing(chunk.costVectorPath))
                issues.append({at, QObject::tr("Chunk %1: the cost modifiers vector "
                                               "file does not exist.").arg(c + 1)});
            if (!chunk.costRasterPath.trimmed().isEmpty()
                && fileMissing(chunk.costRasterPath))
                issues.append({at, QObject::tr("Chunk %1: the cost modifiers raster "
                                               "file does not exist.").arg(c + 1)});
        }
    }

    // row-level
    QHash<QString, Index> claimed;  // output file -> first row that writes it
    for (int c = 0; c < job.chunks.size(); ++c) {
        const Chunk &chunk = job.chunks.at(c);
        for (int r = 0; r < chunk.rows.size(); ++r) {
            const Row &row = chunk.rows.at(r);
            const Index at{c, r};
            auto fail = [&](const QString &m) { issues.append({at, m}); };

            if (fileMissing(row.demPath))
                fail(QObject::tr("The DEM does not exist."));

            if (fete) {
                if (!row.generatePoints && fileMissing(row.pointsPath))
                    fail(QObject::tr("The sample points file does not exist."));
                if (row.generatePoints && row.genLayerName.trimmed().isEmpty())
                    fail(QObject::tr("The generated points layer needs a name."));
                if (row.outputName.trimmed().isEmpty())
                    fail(QObject::tr("The density raster name is required."));
            } else {
                if (fileMissing(row.originPath))
                    fail(QObject::tr("The origin file does not exist."));
                if (fileMissing(row.destinationsPath))
                    fail(QObject::tr("The destinations file does not exist."));
                if (row.outputName.trimmed().isEmpty()
                    && row.pathLinesName.trimmed().isEmpty())
                    fail(QObject::tr("Name at least one of the paths raster and "
                                     "the paths shapefile."));
            }

            if (row.outputDir.trimmed().isEmpty())
                fail(QObject::tr("The output folder is required."));

            for (const QString &name : {row.outputName, row.pathLinesName,
                                        row.genLayerName}) {
                if (!name.trimmed().isEmpty() && nameHasBadChars(name)) {
                    fail(QObject::tr("Output names cannot contain "
                                     "\\ / : * ? \" < > | characters."));
                    break;
                }
            }

            for (const QString &p : {row.demPath, row.pointsPath, row.originPath,
                                     row.destinationsPath, row.outputDir}) {
                if (!p.trimmed().isEmpty() && !isAscii(p)) {
                    fail(QObject::tr("Paths must not contain accented or non-Latin "
                                     "characters: the engine cannot handle them."));
                    break;
                }
            }

            // Two rows writing the same file: the second would silently
            // overwrite the first, hours after the mistake was made.
            for (const QString &f : plannedFiles(job, chunk, row)) {
                const auto it = claimed.constFind(f);
                if (it != claimed.constEnd()) {
                    fail(QObject::tr("Writes the same file as chunk %1 row %2 (%3). "
                                     "Change the output name, or turn on the "
                                     "per-row folders.")
                             .arg(it->chunk + 1)
                             .arg(it->row + 1)
                             .arg(QFileInfo(f).fileName()));
                    break;
                }
                claimed.insert(f, at);
            }
        }
    }
    return issues;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

namespace {

QJsonObject rowToJson(const Row &r)
{
    QJsonObject o;
    o["demPath"] = r.demPath;
    o["generatePoints"] = r.generatePoints;
    o["pointsPath"] = r.pointsPath;
    o["genByTargetCount"] = r.genByTargetCount;
    o["genSpacing"] = r.genSpacing;
    o["genTargetCount"] = r.genTargetCount;
    o["genRandom"] = r.genRandom;
    o["genSeed"] = r.genSeed;
    o["genEdgeBuffer"] = r.genEdgeBuffer;
    o["genLayerName"] = r.genLayerName;
    o["originPath"] = r.originPath;
    o["destinationsPath"] = r.destinationsPath;
    o["outputDir"] = r.outputDir;
    o["outputName"] = r.outputName;
    o["pathLinesName"] = r.pathLinesName;
    return o;
}

Row rowFromJson(const QJsonObject &o)
{
    Row r;
    r.demPath = o["demPath"].toString();
    r.generatePoints = o["generatePoints"].toBool();
    r.pointsPath = o["pointsPath"].toString();
    r.genByTargetCount = o["genByTargetCount"].toBool(true);
    r.genSpacing = o["genSpacing"].toInt(10);
    r.genTargetCount = o["genTargetCount"].toInt(5000);
    r.genRandom = o["genRandom"].toBool();
    r.genSeed = o["genSeed"].toInt(1);
    r.genEdgeBuffer = o["genEdgeBuffer"].toInt(0);
    r.genLayerName = o["genLayerName"].toString(QStringLiteral("sample_points"));
    r.originPath = o["originPath"].toString();
    r.destinationsPath = o["destinationsPath"].toString();
    r.outputDir = o["outputDir"].toString();
    r.outputName = o["outputName"].toString();
    r.pathLinesName = o["pathLinesName"].toString(QStringLiteral("LCPS_vectors"));
    return r;
}

QJsonObject chunkToJson(const Chunk &c)
{
    QJsonObject o;
    o["neighbours"] = c.neighbours;
    o["costFunction"] = c.costFunction;
    o["smoothingBufferRadius"] = c.smoothingBufferRadius;
    o["useCostModifiers"] = c.useCostModifiers;
    o["costVectorPath"] = c.costVectorPath;
    o["polylineBufferRadius"] = c.polylineBufferRadius;
    o["costRasterPath"] = c.costRasterPath;
    o["barrierEnabled"] = c.barrierEnabled;
    o["barrierThreshold"] = c.barrierThreshold;
    o["keepExtraRasters"] = c.keepExtraRasters;
    o["loadRastersInViewer"] = c.loadRastersInViewer;
    o["loadVectorsInViewer"] = c.loadVectorsInViewer;
    o["slopeCutoffEnabled"] = c.slopeCutoffEnabled;
    o["maxSlopeUpDeg"] = c.maxSlopeUpDeg;
    o["maxSlopeDownDeg"] = c.maxSlopeDownDeg;
    o["costCorridor"] = c.costCorridor;
    o["corridorWidthPercent"] = c.corridorWidthPercent;
    o["collapsed"] = c.collapsed;
    QJsonArray rows;
    for (const Row &r : c.rows)
        rows.append(rowToJson(r));
    o["rows"] = rows;
    return o;
}

Chunk chunkFromJson(const QJsonObject &o)
{
    Chunk c;
    c.neighbours = o["neighbours"].toInt(16);
    // Defaults match a batch written before these existed, so an older
    // .trjbatch loads with the options simply off.
    c.slopeCutoffEnabled = o["slopeCutoffEnabled"].toBool(false);
    c.maxSlopeUpDeg = o["maxSlopeUpDeg"].toInt(30);
    c.maxSlopeDownDeg = o["maxSlopeDownDeg"].toInt(30);
    c.costCorridor = o["costCorridor"].toBool(false);
    c.corridorWidthPercent = o["corridorWidthPercent"].toDouble(10.0);
    c.costFunction = o["costFunction"].toInt(1);
    c.smoothingBufferRadius = o["smoothingBufferRadius"].toInt(0);
    c.useCostModifiers = o["useCostModifiers"].toBool();
    c.costVectorPath = o["costVectorPath"].toString();
    c.polylineBufferRadius = o["polylineBufferRadius"].toInt(2);
    c.costRasterPath = o["costRasterPath"].toString();
    c.barrierEnabled = o["barrierEnabled"].toBool(true);
    c.barrierThreshold = o["barrierThreshold"].toDouble(1000.0);
    c.keepExtraRasters = o["keepExtraRasters"].toBool();
    c.loadRastersInViewer = o["loadRastersInViewer"].toBool();
    c.loadVectorsInViewer = o["loadVectorsInViewer"].toBool();
    c.collapsed = o["collapsed"].toBool();
    const QJsonArray rows = o["rows"].toArray();
    for (const QJsonValue &v : rows)
        c.rows.append(rowFromJson(v.toObject()));
    // Batch files written before the switch moved to the chunk carried it on
    // every row; any row that had it on turns it on for the whole chunk.
    if (!o.contains(QStringLiteral("keepExtraRasters"))) {
        for (const QJsonValue &v : rows) {
            if (v.toObject()[QStringLiteral("keepExtraRasters")].toBool()) {
                c.keepExtraRasters = true;
                break;
            }
        }
    }
    return c;
}

} // namespace

QJsonObject toJson(const Job &job)
{
    QJsonObject o;
    // Written so a future reader can refuse a file it does not understand
    // instead of silently loading half of it.
    o["format"] = QStringLiteral("trajecta-batch");
    o["version"] = 1;
    o["mode"] = job.mode == TrajectaRunner::Mode::Lcpa ? QStringLiteral("lcpa")
                                                       : QStringLiteral("fete");
    o["maxThreads"] = job.maxThreads;
    o["maxRamMb"] = job.maxRamMb;
    o["largePages"] = job.largePages;
    o["verbose"] = job.verbose;
    o["writeManifest"] = job.writeManifest;
    o["folderPerRow"] = job.folderPerRow;
    QJsonArray chunks;
    for (const Chunk &c : job.chunks)
        chunks.append(chunkToJson(c));
    o["chunks"] = chunks;
    return o;
}

bool fromJson(const QJsonObject &o, Job *job, QString *error)
{
    if (!job)
        return false;
    if (o["format"].toString() != QLatin1String("trajecta-batch")) {
        if (error)
            *error = QObject::tr("This is not a Trajecta batch file.");
        return false;
    }
    if (o["version"].toInt() > 1) {
        if (error)
            *error = QObject::tr("This batch file was written by a newer version "
                                 "of Trajecta Studio.");
        return false;
    }

    Job loaded;
    loaded.mode = o["mode"].toString() == QLatin1String("lcpa")
                      ? TrajectaRunner::Mode::Lcpa
                      : TrajectaRunner::Mode::Fete;
    loaded.maxThreads = o["maxThreads"].toInt(1);
    loaded.maxRamMb = o["maxRamMb"].toInt(8192);
    loaded.largePages = o["largePages"].toBool();
    // Batch files written before the option existed were all run verbose.
    loaded.verbose = o["verbose"].toBool(true);
    // Same reasoning in the other direction: a file written before manifests
    // existed says nothing about them, and the default is to write them.
    loaded.writeManifest = o["writeManifest"].toBool(true);
    loaded.folderPerRow = o["folderPerRow"].toBool(true);
    for (const QJsonValue &v : o["chunks"].toArray())
        loaded.chunks.append(chunkFromJson(v.toObject()));

    *job = loaded;
    return true;
}

} // namespace Batch
