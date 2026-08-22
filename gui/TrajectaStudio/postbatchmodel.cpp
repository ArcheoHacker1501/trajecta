#include "postbatchmodel.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QRegularExpression>

namespace PostBatch {

namespace {

bool nameHasBadChars(const QString &name)
{
    static const QRegularExpression bad(QStringLiteral("[\\\\/:*?\"<>|]"));
    return bad.match(name).hasMatch();
}

bool fileMissing(const QString &path)
{
    return path.trimmed().isEmpty() || !QFileInfo::exists(path);
}

QJsonObject chunkToJson(const Chunk &c)
{
    QJsonObject o;
    o["interpInputRaster"] = c.interpInputRaster;
    o["interpOutputDir"] = c.interpOutputDir;
    o["interpThreshold"] = c.interpThreshold;
    o["interpSampleSpacing"] = c.interpSampleSpacing;
    o["interpPreservePeaks"] = c.interpPreservePeaks;
    o["interpMaxRadius"] = c.interpMaxRadius;
    o["interpOutputName"] = c.interpOutputName;

    o["cmpComputedPath"] = c.cmpComputedPath;
    o["cmpKnownPath"] = c.cmpKnownPath;
    o["cmpTolerance"] = c.cmpTolerance;

    o["cohRasterPath"] = c.cohRasterPath;
    o["cohPointsPath"] = c.cohPointsPath;
    o["cohRadius"] = c.cohRadius;
    o["cohThresholdMode"] = c.cohThresholdMode;
    o["cohThresholdValue"] = c.cohThresholdValue;
    o["cohNullModel"] = c.cohNullModel;
    o["cohNullMode"] = c.cohNullMode;
    o["cohNullReplicates"] = c.cohNullReplicates;
    o["cohSensitivity"] = c.cohSensitivity;
    o["cohSensitivityRadii"] = c.cohSensitivityRadii;
    o["cohEcdfDistances"] = c.cohEcdfDistances;
    o["cohEdgeGuard"] = c.cohEdgeGuard;
    o["cohWriteHistogramScript"] = c.cohWriteHistogramScript;
    o["cohOutputDir"] = c.cohOutputDir;
    o["cohPrefix"] = c.cohPrefix;
    o["cohVectorAsGeoPackage"] = c.cohVectorAsGeoPackage;
    o["cohWriteDistanceRaster"] = c.cohWriteDistanceRaster;

    o["loadInViewer"] = c.loadInViewer;
    o["collapsed"] = c.collapsed;
    return o;
}

Chunk chunkFromJson(const QJsonObject &o)
{
    Chunk c;
    c.interpInputRaster = o["interpInputRaster"].toString();
    c.interpOutputDir = o["interpOutputDir"].toString();
    c.interpThreshold = o["interpThreshold"].toDouble(1.0);
    c.interpSampleSpacing = o["interpSampleSpacing"].toInt(4);
    c.interpPreservePeaks = o["interpPreservePeaks"].toBool();
    c.interpMaxRadius = o["interpMaxRadius"].toInt(0);
    c.interpOutputName = o["interpOutputName"].toString(
        QStringLiteral("FETE_density_NNI"));

    c.cmpComputedPath = o["cmpComputedPath"].toString();
    c.cmpKnownPath = o["cmpKnownPath"].toString();
    c.cmpTolerance = o["cmpTolerance"].toDouble(100.0);

    c.cohRasterPath = o["cohRasterPath"].toString();
    c.cohPointsPath = o["cohPointsPath"].toString();
    c.cohRadius = o["cohRadius"].toDouble(250.0);
    c.cohThresholdMode = o["cohThresholdMode"].toInt(0);
    c.cohThresholdValue = o["cohThresholdValue"].toDouble(1.0);
    c.cohNullModel = o["cohNullModel"].toBool(true);
    c.cohNullMode = o["cohNullMode"].toInt(0);
    c.cohNullReplicates = o["cohNullReplicates"].toInt(999);
    c.cohSensitivity = o["cohSensitivity"].toBool();
    c.cohSensitivityRadii = o["cohSensitivityRadii"].toString(
        QStringLiteral("100, 250, 500, 1000"));
    // A file written before distance bands existed has no such key, and
    // must get the default ladder rather than an empty one - which the
    // tool would read as "no table at all".
    c.cohEcdfDistances = o["cohEcdfDistances"].toString(
        QStringLiteral("0, 100, 250, 500, 1000, 2500"));
    c.cohEdgeGuard = o["cohEdgeGuard"].toBool(true);
    c.cohWriteHistogramScript = o["cohWriteHistogramScript"].toBool(true);
    c.cohOutputDir = o["cohOutputDir"].toString();
    c.cohPrefix = o["cohPrefix"].toString(QStringLiteral("coherence"));
    c.cohVectorAsGeoPackage = o["cohVectorAsGeoPackage"].toBool(true);
    c.cohWriteDistanceRaster = o["cohWriteDistanceRaster"].toBool(true);

    c.loadInViewer = o["loadInViewer"].toBool(true);
    c.collapsed = o["collapsed"].toBool();
    return c;
}

} // namespace

QString modeLabel(Mode mode)
{
    switch (mode) {
    case Mode::Compare:   return QCoreApplication::translate("PostBatch", "Compare with a known route");
    case Mode::Coherence: return QCoreApplication::translate("PostBatch", "Site-corridor coherence");
    case Mode::Nni:
    default:              return QCoreApplication::translate("PostBatch", "NNI");
    }
}

QList<Issue> validate(const Job &job)
{
    QList<Issue> issues;
    if (job.chunks.isEmpty()) {
        issues.append({-1, QObject::tr("The batch is empty: add at least one chunk.")});
        return issues;
    }

    for (int i = 0; i < job.chunks.size(); ++i) {
        const Chunk &c = job.chunks.at(i);
        auto fail = [&](const QString &m) { issues.append({i, m}); };

        switch (job.mode) {
        case Mode::Nni:
            if (fileMissing(c.interpInputRaster))
                fail(QObject::tr("The density raster does not exist."));
            if (c.interpOutputDir.trimmed().isEmpty())
                fail(QObject::tr("The output folder is required."));
            if (c.interpOutputName.trimmed().isEmpty())
                fail(QObject::tr("The output raster needs a name."));
            else if (nameHasBadChars(c.interpOutputName))
                fail(QObject::tr("The output name contains \\ / : * ? \" < > | "
                                 "characters."));
            break;
        case Mode::Compare:
            if (fileMissing(c.cmpComputedPath))
                fail(QObject::tr("The computed routes layer does not exist."));
            if (fileMissing(c.cmpKnownPath))
                fail(QObject::tr("The known route layer does not exist."));
            if (c.cmpTolerance <= 0.0)
                fail(QObject::tr("The tolerance must be greater than zero."));
            break;
        case Mode::Coherence:
            if (fileMissing(c.cohRasterPath))
                fail(QObject::tr("The FETE surface does not exist."));
            if (fileMissing(c.cohPointsPath))
                fail(QObject::tr("The sites layer does not exist."));
            if (c.cohRadius <= 0.0)
                fail(QObject::tr("The radius must be greater than zero."));
            break;
        }
    }

    // NNI chunks that would write over each other: same folder, same name.
    if (job.mode == Mode::Nni) {
        QHash<QString, int> claimed;
        for (int i = 0; i < job.chunks.size(); ++i) {
            const Chunk &c = job.chunks.at(i);
            const QString key = QDir::cleanPath(
                QDir(c.interpOutputDir).absoluteFilePath(c.interpOutputName.trimmed()));
            if (claimed.contains(key)) {
                issues.append({i, QObject::tr("Chunk %1 writes the same output as "
                                              "chunk %2.")
                                      .arg(i + 1).arg(claimed[key] + 1)});
            } else {
                claimed.insert(key, i);
            }
        }
    }

    return issues;
}

TrajectaRunner::Parameters toParameters(const Job &job, const Chunk &chunk,
                                        const TrajectaRunner::Parameters &env)
{
    TrajectaRunner::Parameters p;
    p.mode = TrajectaRunner::Mode::Interp;
    p.maxThreads = job.maxThreads;
    p.maxRamMb = job.maxRamMb;
    p.largePages = job.largePages;
    p.writeManifest = job.writeManifest;
    // NNI has no card for it in the single-run form either — see
    // MainWindow::startInterpRun() — so a batch of NNI chunks stays quiet too.
    p.verbose = false;

    p.outputDir = chunk.interpOutputDir;
    p.interpInputRaster = chunk.interpInputRaster;
    p.interpThreshold = chunk.interpThreshold;
    p.interpSampleSpacing = chunk.interpSampleSpacing;
    p.interpPreservePeaks = chunk.interpPreservePeaks;
    p.interpMaxRadius = chunk.interpMaxRadius;
    p.interpOutputName = chunk.interpOutputName.trimmed();

    p.exePath = env.exePath;
    p.gdalBinDir = env.gdalBinDir;
    p.projDataDir = env.projDataDir;
    p.gdalDataDir = env.gdalDataDir;
    p.workingDir = env.workingDir;
    return p;
}

QJsonObject toJson(const Job &job)
{
    QJsonObject o;
    o["format"] = QStringLiteral("trajecta-postbatch");
    o["version"] = 1;
    o["mode"] = int(job.mode);
    o["maxThreads"] = job.maxThreads;
    o["maxRamMb"] = job.maxRamMb;
    o["largePages"] = job.largePages;
    o["writeManifest"] = job.writeManifest;
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
    if (o["format"].toString() != QLatin1String("trajecta-postbatch")) {
        if (error)
            *error = QObject::tr("This is not a Trajecta post-processing batch file.");
        return false;
    }
    if (o["version"].toInt() > 1) {
        if (error)
            *error = QObject::tr("This batch file was written by a newer version "
                                 "of Trajecta Studio.");
        return false;
    }

    Job loaded;
    const int m = o["mode"].toInt(int(Mode::Nni));
    loaded.mode = (m == int(Mode::Compare)) ? Mode::Compare
                 : (m == int(Mode::Coherence)) ? Mode::Coherence
                 : Mode::Nni;
    loaded.maxThreads = o["maxThreads"].toInt(1);
    loaded.maxRamMb = o["maxRamMb"].toInt(8192);
    loaded.largePages = o["largePages"].toBool();
    loaded.writeManifest = o["writeManifest"].toBool(true);
    for (const QJsonValue &v : o["chunks"].toArray())
        loaded.chunks.append(chunkFromJson(v.toObject()));

    *job = loaded;
    return true;
}

} // namespace PostBatch
