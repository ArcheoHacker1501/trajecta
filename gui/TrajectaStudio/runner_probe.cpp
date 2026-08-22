// Headless test harness for TrajectaRunner: drives a full analysis without
// the GUI, so the prompt/answer state machine can be exercised end-to-end
// from the command line. Not installed; built only with
// -DTRAJECTA_STUDIO_BUILD_TESTTOOL=ON.

#include "batchcontroller.h"
#include "batchmodel.h"
#include "checkpointstore.h"
#include "trajectarunner.h"

#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTextStream>

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QTextStream out(stdout);

    QCommandLineParser parser;
    parser.setApplicationDescription(QStringLiteral("TrajectaRunner probe"));
    parser.addHelpOption();
    parser.addOptions({
        {QStringLiteral("exe"), QStringLiteral("Path to trajecta.exe"), QStringLiteral("path")},
        {QStringLiteral("mode"), QStringLiteral("fete or lcpa"), QStringLiteral("mode"), QStringLiteral("fete")},
        {QStringLiteral("dem"), QStringLiteral("DEM path"), QStringLiteral("path")},
        {QStringLiteral("points"), QStringLiteral("Sample points path (FETE)"), QStringLiteral("path")},
        {QStringLiteral("origin"), QStringLiteral("Origin path (LCPA)"), QStringLiteral("path")},
        {QStringLiteral("destinations"), QStringLiteral("Destinations path (LCPA)"), QStringLiteral("path")},
        {QStringLiteral("out"), QStringLiteral("Output directory"), QStringLiteral("path")},
        {QStringLiteral("gen-points"), QStringLiteral("Generate the FETE sample points from the DEM")},
        {QStringLiteral("gen-spacing"), QStringLiteral("Point spacing in cells"), QStringLiteral("n")},
        {QStringLiteral("gen-target"), QStringLiteral("Target number of points"), QStringLiteral("n")},
        {QStringLiteral("gen-random"), QStringLiteral("Stratified random instead of a regular grid")},
        {QStringLiteral("gen-seed"), QStringLiteral("Random seed"), QStringLiteral("n"), QStringLiteral("1")},
        {QStringLiteral("gen-edge"), QStringLiteral("Edge buffer in cells"), QStringLiteral("n"), QStringLiteral("0")},
        {QStringLiteral("gen-name"), QStringLiteral("Generated layer name"), QStringLiteral("name"),
         QStringLiteral("sample_points")},
        {QStringLiteral("cost-vector"), QStringLiteral("Cost modifiers vector"), QStringLiteral("path")},
        {QStringLiteral("cost-raster"), QStringLiteral("Cost modifiers raster"), QStringLiteral("path")},
        {QStringLiteral("threads"), QStringLiteral("Max CPU threads"), QStringLiteral("n"), QStringLiteral("4")},
        {QStringLiteral("ram"), QStringLiteral("Max RAM MB"), QStringLiteral("mb"), QStringLiteral("2048")},
        {QStringLiteral("gdal"), QStringLiteral("GDAL bin dir to prepend to PATH"), QStringLiteral("path")},
        {QStringLiteral("projlib"), QStringLiteral("PROJ data dir (proj.db)"), QStringLiteral("path")},
        {QStringLiteral("gdaldata"), QStringLiteral("GDAL_DATA dir"), QStringLiteral("path")},
        {QStringLiteral("echo"), QStringLiteral("Echo raw engine output")},
        // Output names. An empty value means "do not save this output", which
        // is what the batch runner uses; --no-extra is the shorthand for
        // clearing all four intermediate rasters at once.
        {QStringLiteral("slope-name"), QStringLiteral("Slope raster name"), QStringLiteral("name"),
         QStringLiteral("slope")},
        {QStringLiteral("cost-name"), QStringLiteral("Base cost surface name"), QStringLiteral("name"),
         QStringLiteral("cost_surface")},
        {QStringLiteral("additional-name"), QStringLiteral("Additional cost surface name"),
         QStringLiteral("name"), QStringLiteral("cost_surface_additional")},
        {QStringLiteral("total-name"), QStringLiteral("Total cost surface name"),
         QStringLiteral("name"), QStringLiteral("cost_surface_total")},
        {QStringLiteral("density-name"), QStringLiteral("FETE density raster name"),
         QStringLiteral("name"), QStringLiteral("FETE_density")},
        {QStringLiteral("path-raster-name"), QStringLiteral("LCPA path raster name"),
         QStringLiteral("name"), QStringLiteral("raster_lcps")},
        {QStringLiteral("path-lines-name"), QStringLiteral("LCPA path lines shapefile name"),
         QStringLiteral("name"), QStringLiteral("LCPS_vectors")},
        {QStringLiteral("no-extra"),
         QStringLiteral("Skip slope, base/additional/total cost surfaces")},
        {QStringLiteral("no-manifest"),
         QStringLiteral("Do not write the run manifest (it is written by default)")},
        // Checkpointing, so the resume path can be exercised head-less.
        {QStringLiteral("checkpoint-dir"), QStringLiteral("Write checkpoints here"),
         QStringLiteral("path")},
        {QStringLiteral("checkpoint-minutes"), QStringLiteral("Minutes between checkpoints"),
         QStringLiteral("n"), QStringLiteral("30")},
        {QStringLiteral("resume"), QStringLiteral("Resume from this checkpoint file"),
         QStringLiteral("path")},
        // Reports what the recovery prompt would find in a checkpoint folder,
        // without putting a modal dialog on screen.
        {QStringLiteral("checkpoint-info"),
         QStringLiteral("Describe the checkpoint and session in this folder"),
         QStringLiteral("path")},
        {QStringLiteral("batch-resume-at"),
         QStringLiteral("Resume a batch at this queue index (0-based)"),
         QStringLiteral("n")},
        // Batch mode: run a whole .trjbatch file through BatchController,
        // which is the same object the GUI drives.
        {QStringLiteral("batch"), QStringLiteral("Run a batch JSON file"),
         QStringLiteral("path")},
    });
    parser.process(app);

    if (parser.isSet("checkpoint-info")) {
        const QString dir = parser.value("checkpoint-info");
        const Checkpoint::Info info = Checkpoint::latest(dir);
        const Checkpoint::Session session = Checkpoint::readSession(dir);
        out << "[probe] dir           : " << QDir::toNativeSeparators(dir) << "\n";
        out << "[probe] default dir   : "
            << QDir::toNativeSeparators(Checkpoint::defaultDir()) << "\n";
        out << "[probe] checkpoint    : " << (info.found ? "yes" : "no") << "\n";
        if (info.found) {
            out << "[probe]   file        : " << QFileInfo(info.path).fileName() << "\n";
            out << "[probe]   nextSource  : " << info.nextSource << "\n";
            out << "[probe]   sources     : " << info.sources << "\n";
            out << "[probe]   modified    : " << info.modified << "\n";
        }
        out << "[probe] session       : " << (session.valid ? "yes" : "no") << "\n";
        if (session.valid) {
            out << "[probe]   batch       : " << (session.batch ? "yes" : "no") << "\n";
            out << "[probe]   queueIndex  : " << session.queueIndex << "\n";
            out << "[probe]   deliberate  : " << (session.deliberate ? "yes" : "no") << "\n";
            out << "[probe]   label       : " << session.label << "\n";
            const TrajectaRunner::Parameters p = Checkpoint::fromJson(session.params);
            out << "[probe]   dem         : " << QDir::toNativeSeparators(p.demPath) << "\n";
            out << "[probe]   outputDir   : " << QDir::toNativeSeparators(p.outputDir) << "\n";
            out << "[probe]   densityName : " << p.densityName << "\n";
            out << "[probe]   neighbours  : " << p.neighbours << "\n";
        }
        // What the prompt would do: both present is the only case that offers a
        // resume, and that is exactly the condition MainWindow tests.
        out << "[probe] would offer resume: "
            << ((info.found && session.valid) ? "yes" : "no") << "\n";
        out.flush();
        return 0;
    }

    if (parser.isSet("batch")) {
        const QString path = parser.value("batch");
        QFile f(path);
        if (!f.open(QIODevice::ReadOnly)) {
            out << "[probe] cannot open " << path << "\n";
            return 2;
        }
        QJsonParseError perr;
        const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &perr);
        if (perr.error != QJsonParseError::NoError) {
            out << "[probe] bad JSON: " << perr.errorString() << "\n";
            return 2;
        }
        Batch::Job job;
        QString err;
        if (!Batch::fromJson(doc.object(), &job, &err)) {
            out << "[probe] " << err << "\n";
            return 2;
        }

        TrajectaRunner::Parameters env;
        env.exePath = parser.value("exe");
        env.gdalBinDir = parser.value("gdal");
        env.projDataDir = parser.value("projlib");
        env.gdalDataDir = parser.value("gdaldata");
        env.workingDir = QDir::tempPath();
        // The same fields BatchPage::startBatch puts on its environment. Without
        // them the rows of a batch run with checkpointing off, and the one thing
        // this harness exists to prove — that a batch interrupted inside a row
        // picks that row up again — cannot be set up at all.
        if (parser.isSet("checkpoint-dir")) {
            env.checkpointEnabled = true;
            env.checkpointDir = parser.value("checkpoint-dir");
            env.checkpointMinutes = parser.value("checkpoint-minutes").toDouble();
        }

        auto *ctl = new BatchController(&app);
        const bool echo = parser.isSet("echo");
        QObject::connect(ctl, &BatchController::consoleOutput, [&](const QString &t) {
            if (echo) { out << t; out.flush(); }
        });
        QObject::connect(ctl, &BatchController::rowStarted, [&](int i) {
            const Batch::Index at = ctl->results().at(i).where;
            out << "[probe] row " << (i + 1) << "/" << ctl->total()
                << " started (chunk " << (at.chunk + 1) << " row " << (at.row + 1) << ")\n";
            out.flush();
        });
        QObject::connect(ctl, &BatchController::rowFinished,
                         [&](int i, BatchController::RowState st, const QString &msg) {
            const char *name = "?";
            switch (st) {
            case BatchController::RowState::Done:      name = "DONE"; break;
            case BatchController::RowState::Failed:    name = "FAILED"; break;
            case BatchController::RowState::Invalid:   name = "INVALID"; break;
            case BatchController::RowState::Cancelled: name = "CANCELLED"; break;
            default: break;
            }
            out << "[probe] row " << (i + 1) << " " << name;
            if (!msg.isEmpty())
                out << " -- " << msg.trimmed().left(300);
            out << "\n";
            out.flush();
        });
        QObject::connect(ctl, &BatchController::batchFinished, [&](const QString &report) {
            out << "\n[probe] ===== batch finished =====\n" << report << "\n";
            out.flush();
            int bad = 0;
            for (const auto &r : ctl->results())
                if (r.state != BatchController::RowState::Done)
                    ++bad;
            app.exit(bad == 0 ? 0 : 1);
        });

        // Picking a batch up after a crash: the rows before --batch-resume-at
        // are taken as done, and that row itself continues from the checkpoint
        // given by --resume (which may be omitted, for a batch stopped between
        // two rows).
        BatchController::Resume resume;
        if (parser.isSet("batch-resume-at")) {
            resume.queueIndex = parser.value("batch-resume-at").toInt();
            resume.checkpointPath = parser.value("resume");
        }

        QString startError;
        if (!ctl->start(job, env, &startError, resume)) {
            out << "[probe] batch refused: " << startError << "\n";
            return 2;
        }
        return app.exec();
    }

    TrajectaRunner::Parameters p;
    p.mode = parser.value("mode").toLower() == QLatin1String("lcpa")
                 ? TrajectaRunner::Mode::Lcpa
                 : TrajectaRunner::Mode::Fete;
    p.exePath = parser.value("exe");
    p.demPath = parser.value("dem");
    p.pointsPath = parser.value("points");
    p.originPath = parser.value("origin");
    p.destinationsPath = parser.value("destinations");
    p.outputDir = parser.value("out");
    p.maxThreads = parser.value("threads").toInt();
    p.maxRamMb = parser.value("ram").toInt();
    p.gdalBinDir = parser.value("gdal");
    p.projDataDir = parser.value("projlib");
    p.gdalDataDir = parser.value("gdaldata");
    p.workingDir = QDir::tempPath();
    if (parser.isSet("checkpoint-dir")) {
        p.checkpointEnabled = true;
        p.checkpointDir = parser.value("checkpoint-dir");
        p.checkpointMinutes = parser.value("checkpoint-minutes").toDouble();
    }
    p.resumeCheckpoint = parser.value("resume");
    if (parser.isSet("gen-points")) {
        p.generatePoints = true;
        p.genByTargetCount = parser.isSet("gen-target");
        if (parser.isSet("gen-spacing"))
            p.genSpacing = parser.value("gen-spacing").toInt();
        if (parser.isSet("gen-target"))
            p.genTargetCount = parser.value("gen-target").toInt();
        p.genRandom = parser.isSet("gen-random");
        p.genSeed = parser.value("gen-seed").toInt();
        p.genEdgeBuffer = parser.value("gen-edge").toInt();
        p.genLayerName = parser.value("gen-name");
        p.pointsPath = QDir(p.outputDir).filePath(p.genLayerName + QStringLiteral(".shp"));
    }
    if (!parser.value("cost-vector").isEmpty() || !parser.value("cost-raster").isEmpty()) {
        p.useCostModifiers = true;
        p.costVectorPath = parser.value("cost-vector");
        p.costRasterPath = parser.value("cost-raster");
    }

    p.slopeName = parser.value("slope-name");
    p.costName = parser.value("cost-name");
    p.additionalCostName = parser.value("additional-name");
    p.totalCostName = parser.value("total-name");
    p.densityName = parser.value("density-name");
    p.pathRasterName = parser.value("path-raster-name");
    p.pathLinesName = parser.value("path-lines-name");
    if (parser.isSet("no-extra")) {
        p.slopeName.clear();
        p.costName.clear();
        p.additionalCostName.clear();
        p.totalCostName.clear();
    }
    if (parser.isSet("no-manifest"))
        p.writeManifest = false;

    TrajectaRunner runner;
    const bool echo = parser.isSet("echo");

    QObject::connect(&runner, &TrajectaRunner::consoleOutput, [&](const QString &t) {
        if (echo) {
            out << t;
            out.flush();
        }
    });
    QObject::connect(&runner, &TrajectaRunner::answerSent, [&](const QString &a) {
        out << "\n[probe] answered: " << a << "\n";
        out.flush();
    });
    QObject::connect(&runner, &TrajectaRunner::progressChanged, [&](double pct) {
        out << "[probe] progress: " << QString::number(pct, 'f', 1) << "%\r";
        out.flush();
    });
    QObject::connect(&runner, &TrajectaRunner::statusChanged, [&](const QString &s) {
        out << "\n[probe] status: " << s << "\n";
        out.flush();
    });
    QObject::connect(&runner, &TrajectaRunner::finished,
                     [&](TrajectaRunner::Outcome outcome, const QString &report) {
        const bool ok = outcome == TrajectaRunner::Outcome::Success;
        const char *label = ok ? "SUCCESS"
                               : (outcome == TrajectaRunner::Outcome::Cancelled
                                      ? "CANCELLED" : "FAILURE");
        out << "\n[probe] ===== finished: " << label << " =====\n";
        out << report << "\n";
        out.flush();
        app.exit(ok ? 0 : 1);
    });

    runner.start(p);
    return app.exec();
}
