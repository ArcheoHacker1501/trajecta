#include "checkpointstore.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QLocale>
#include <QObject>
#include <QSettings>
#include <QStandardPaths>

// The engine's own header, included rather than reimplemented: the binary
// layout of a checkpoint has exactly one definition.
#include "checkpoint.h"

namespace Checkpoint {

namespace {

QString settingsKey(const char *leaf)
{
    return QLatin1String("checkpoint/") + QLatin1String(leaf);
}

} // namespace

QString defaultDir()
{
    // Must agree with ckpt::defaultDir() in src/checkpoint.h, which the engine
    // uses to decide where to write. The engine has no Qt, so it spells the
    // paths out by hand; the standard location that matches what it builds is
    // not the same one on every platform:
    //
    //   Windows  %LOCALAPPDATA%                == GenericConfigLocation
    //   macOS    ~/Library/Application Support == GenericDataLocation
    //   Linux    ~/.local/share                == GenericDataLocation
    //
    // (On Windows GenericDataLocation is the roaming AppData folder, which is
    // *not* what the engine picks — hence the split rather than one call.)
#ifdef Q_OS_WIN
    const QString base =
        QStandardPaths::writableLocation(QStandardPaths::GenericConfigLocation);
#else
    const QString base =
        QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation);
#endif
    if (base.isEmpty())
        return QString();
    return QDir(base).filePath(QStringLiteral("Trajecta/checkpoints"));
}

Settings settings()
{
    QSettings s;
    Settings out;
    out.enabled = s.value(settingsKey("enabled"), true).toBool();
    out.minutes = s.value(settingsKey("minutes"), 30).toInt();
    if (out.minutes < 1)
        out.minutes = 1;
    out.dir = s.value(settingsKey("dir")).toString();
    return out;
}

void setSettings(const Settings &v)
{
    QSettings s;
    s.setValue(settingsKey("enabled"), v.enabled);
    s.setValue(settingsKey("minutes"), qMax(1, v.minutes));
    s.setValue(settingsKey("dir"), v.dir);
}

QString activeDir()
{
    const Settings s = settings();
    return s.dir.isEmpty() ? defaultDir() : s.dir;
}

Info latest(const QString &dir)
{
    Info info;
    if (dir.isEmpty())
        return info;
    ckpt::Header header;
    const std::string found =
        ckpt::findLatest(QDir::toNativeSeparators(dir).toStdString(), &header);
    if (found.empty())
        return info;
    info.found = true;
    info.path = QString::fromStdString(found);
    info.nextSource = header.nextSource;
    info.sources = header.fingerprint.sources;
    const QFileInfo fi(info.path);
    info.sizeBytes = fi.size();
    info.modified = QLocale().toString(fi.lastModified(), QLocale::ShortFormat);
    return info;
}

QString sessionPath(const QString &dir)
{
    return QDir(dir).filePath(QStringLiteral("session.json"));
}

bool writeSession(const QString &dir, const Session &s)
{
    if (dir.isEmpty())
        return false;
    QDir().mkpath(dir);
    QJsonObject o;
    o[QStringLiteral("format")] = QStringLiteral("trajecta-session");
    o[QStringLiteral("version")] = 1;
    o[QStringLiteral("batch")] = s.batch;
    o[QStringLiteral("params")] = s.params;
    if (s.batch) {
        o[QStringLiteral("isPostBatch")] = s.isPostBatch;
        o[QStringLiteral("job")] = s.job;
        o[QStringLiteral("queueIndex")] = s.queueIndex;
    }
    o[QStringLiteral("startedAt")] = s.startedAt.isEmpty()
                                        ? QDateTime::currentDateTime().toString(Qt::ISODate)
                                        : s.startedAt;
    o[QStringLiteral("label")] = s.label;
    o[QStringLiteral("deliberate")] = s.deliberate;

    QFile f(sessionPath(dir));
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return false;
    f.write(QJsonDocument(o).toJson(QJsonDocument::Indented));
    return true;
}

Session readSession(const QString &dir)
{
    Session s;
    if (dir.isEmpty())
        return s;
    QFile f(sessionPath(dir));
    if (!f.open(QIODevice::ReadOnly))
        return s;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
    if (!doc.isObject())
        return s;
    const QJsonObject o = doc.object();
    if (o[QStringLiteral("format")].toString() != QLatin1String("trajecta-session"))
        return s;
    s.valid = true;
    s.batch = o[QStringLiteral("batch")].toBool();
    s.isPostBatch = o[QStringLiteral("isPostBatch")].toBool();
    s.params = o[QStringLiteral("params")].toObject();
    s.job = o[QStringLiteral("job")].toObject();
    s.queueIndex = o[QStringLiteral("queueIndex")].toInt(-1);
    s.startedAt = o[QStringLiteral("startedAt")].toString();
    s.label = o[QStringLiteral("label")].toString();
    s.deliberate = o[QStringLiteral("deliberate")].toBool();
    return s;
}

void clearSession(const QString &dir)
{
    if (dir.isEmpty())
        return;
    QFile::remove(sessionPath(dir));
}

void discard(const QString &dir)
{
    if (dir.isEmpty())
        return;
    ckpt::discard(QDir::toNativeSeparators(dir).toStdString());
    clearSession(dir);
}

// `copiedNames`, when given, is filled with the files that actually made it
// across — exportTo() removes exactly those and nothing else.
bool copyTo(const QString &dir, const QString &targetDir, QString *error,
            QStringList *copiedNames)
{
    if (dir.isEmpty() || targetDir.isEmpty()) {
        if (error)
            *error = QObject::tr("No folder was given.");
        return false;
    }
    if (!QDir().mkpath(targetDir)) {
        if (error)
            *error = QObject::tr("Cannot create %1").arg(targetDir);
        return false;
    }
    if (QFileInfo(dir).canonicalFilePath() == QFileInfo(targetDir).canonicalFilePath()) {
        if (error) {
            *error = QObject::tr("That is the folder the checkpoint is already "
                                 "kept in. Choose another one.");
        }
        return false;
    }
    const QDir from(dir);
    const QStringList names = from.entryList(
        {QStringLiteral("*.tckpt"), QStringLiteral("session.json")}, QDir::Files);
    if (names.isEmpty()) {
        if (error)
            *error = QObject::tr("There is nothing to save: the checkpoint is gone.");
        return false;
    }
    int copied = 0;
    for (const QString &name : names) {
        const QString target = QDir(targetDir).filePath(name);
        QFile::remove(target);
        if (QFile::copy(from.filePath(name), target)) {
            ++copied;
            if (copiedNames)
                copiedNames->append(name);
            continue;
        }
        // A file can legitimately disappear underneath a copy taken during a
        // run: the engine keeps two checkpoints and deletes the older one as
        // soon as the newer is safely in place. That is only a failure if it
        // leaves nothing behind.
        if (QFileInfo::exists(from.filePath(name))) {
            if (error)
                *error = QObject::tr("Cannot copy %1").arg(name);
            return false;
        }
    }
    if (copied == 0) {
        if (error) {
            *error = QObject::tr("The checkpoint was replaced while it was being "
                                 "copied. Try again.");
        }
        return false;
    }
    return true;
}

bool exportTo(const QString &dir, const QString &targetDir, QString *error)
{
    // Copy first, remove after: an interrupted export must not be able to
    // destroy the only copy. And remove exactly what was copied, rather than
    // whatever the folder holds by then: re-reading it would delete a
    // checkpoint written after the copy was taken, which is a file nobody has
    // a second copy of.
    QStringList copied;
    if (!copyTo(dir, targetDir, error, &copied))
        return false;
    const QDir from(dir);
    for (const QString &name : copied)
        QFile::remove(from.filePath(name));
    return true;
}

bool importFrom(const QString &sourceDir, const QString &dir, QString *error)
{
    if (sourceDir.isEmpty() || dir.isEmpty()) {
        if (error)
            *error = QObject::tr("No folder was given.");
        return false;
    }
    const QDir from(sourceDir);
    const QStringList names = from.entryList(
        {QStringLiteral("*.tckpt"), QStringLiteral("session.json")}, QDir::Files);
    if (names.isEmpty()) {
        if (error)
            *error = QObject::tr("That folder holds no saved process.");
        return false;
    }
    QDir().mkpath(dir);
    for (const QString &name : names) {
        const QString target = QDir(dir).filePath(name);
        QFile::remove(target);
        if (!QFile::copy(from.filePath(name), target)) {
            if (error)
                *error = QObject::tr("Cannot copy %1").arg(name);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Parameters <-> JSON
// ---------------------------------------------------------------------------

QJsonObject toJson(const TrajectaRunner::Parameters &p)
{
    QJsonObject o;
    o["mode"] = int(p.mode);
    o["verbose"] = p.verbose;
    o["writeManifest"] = p.writeManifest;
    o["maxThreads"] = p.maxThreads;
    o["maxRamMb"] = p.maxRamMb;
    o["largePages"] = p.largePages;

    o["demPath"] = p.demPath;
    o["pointsPath"] = p.pointsPath;
    o["originPath"] = p.originPath;
    o["destinationsPath"] = p.destinationsPath;
    o["outputDir"] = p.outputDir;

    // Deliberately stored as false: on a resume the layer already exists on
    // disk and is read as an ordinary input. Regenerating it would be harmless
    // with a fixed seed and wrong with a random one.
    o["generatePoints"] = false;
    o["genByTargetCount"] = p.genByTargetCount;
    o["genSpacing"] = p.genSpacing;
    o["genTargetCount"] = p.genTargetCount;
    o["genRandom"] = p.genRandom;
    o["genSeed"] = p.genSeed;
    o["genEdgeBuffer"] = p.genEdgeBuffer;
    o["genLayerName"] = p.genLayerName;

    o["useCostModifiers"] = p.useCostModifiers;
    o["costVectorPath"] = p.costVectorPath;
    o["polylineBufferRadius"] = p.polylineBufferRadius;
    o["costRasterPath"] = p.costRasterPath;
    o["barrierThreshold"] = p.barrierThreshold;

    o["neighbours"] = p.neighbours;
    o["costFunction"] = p.costFunction;
    o["smoothingBufferRadius"] = p.smoothingBufferRadius;

    o["slopeName"] = p.slopeName;
    o["costName"] = p.costName;
    o["additionalCostName"] = p.additionalCostName;
    o["totalCostName"] = p.totalCostName;
    o["densityName"] = p.densityName;
    o["pathRasterName"] = p.pathRasterName;
    o["pathLinesName"] = p.pathLinesName;

    o["checkpointEnabled"] = p.checkpointEnabled;
    o["checkpointMinutes"] = double(p.checkpointMinutes);
    o["checkpointDir"] = p.checkpointDir;
    return o;
}

TrajectaRunner::Parameters fromJson(const QJsonObject &o)
{
    TrajectaRunner::Parameters p;
    p.mode = TrajectaRunner::Mode(o["mode"].toInt(int(TrajectaRunner::Mode::Fete)));
    p.verbose = o["verbose"].toBool(true);
    p.writeManifest = o["writeManifest"].toBool(true);
    p.maxThreads = o["maxThreads"].toInt(1);
    p.maxRamMb = o["maxRamMb"].toInt(8192);
    p.largePages = o["largePages"].toBool();

    p.demPath = o["demPath"].toString();
    p.pointsPath = o["pointsPath"].toString();
    p.originPath = o["originPath"].toString();
    p.destinationsPath = o["destinationsPath"].toString();
    p.outputDir = o["outputDir"].toString();

    p.generatePoints = o["generatePoints"].toBool(false);
    p.genByTargetCount = o["genByTargetCount"].toBool(true);
    p.genSpacing = o["genSpacing"].toInt(10);
    p.genTargetCount = o["genTargetCount"].toInt(5000);
    p.genRandom = o["genRandom"].toBool();
    p.genSeed = o["genSeed"].toInt(1);
    p.genEdgeBuffer = o["genEdgeBuffer"].toInt(0);
    p.genLayerName = o["genLayerName"].toString(QStringLiteral("sample_points"));

    p.useCostModifiers = o["useCostModifiers"].toBool();
    p.costVectorPath = o["costVectorPath"].toString();
    p.polylineBufferRadius = o["polylineBufferRadius"].toInt(2);
    p.costRasterPath = o["costRasterPath"].toString();
    p.barrierThreshold = o["barrierThreshold"].toDouble(1000.0);

    p.neighbours = o["neighbours"].toInt(16);
    p.costFunction = o["costFunction"].toInt(1);
    p.smoothingBufferRadius = o["smoothingBufferRadius"].toInt(0);

    // No default here: an empty name means "do not write that file", which is
    // a decision the original run made and the resume must repeat.
    p.slopeName = o["slopeName"].toString();
    p.costName = o["costName"].toString();
    p.additionalCostName = o["additionalCostName"].toString();
    p.totalCostName = o["totalCostName"].toString();
    p.densityName = o["densityName"].toString();
    p.pathRasterName = o["pathRasterName"].toString();
    p.pathLinesName = o["pathLinesName"].toString();

    p.checkpointEnabled = o["checkpointEnabled"].toBool();
    p.checkpointMinutes = o["checkpointMinutes"].toDouble(30.0);
    p.checkpointDir = o["checkpointDir"].toString();
    return p;
}

} // namespace Checkpoint
