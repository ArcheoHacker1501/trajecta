// Headless test harness for TrajectaRunner: drives a full analysis without
// the GUI, so the prompt/answer state machine can be exercised end-to-end
// from the command line. Not installed; built only with
// -DTRAJECTA_STUDIO_BUILD_TESTTOOL=ON.

#include "trajectarunner.h"

#include <QCoreApplication>
#include <QCommandLineParser>
#include <QDir>
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
    });
    parser.process(app);

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
