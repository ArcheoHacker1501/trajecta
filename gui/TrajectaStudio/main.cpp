#include "coherence.h"
#include "confirmdialog.h"
#include "gdalapi.h"
#include "largepages.h"
#include "mainwindow.h"
#include "reportform.h"
#include "routecompare.h"
#include "thememanager.h"

#include <QDir>
#include <QProcessEnvironment>
#include <QTextStream>

#include <QApplication>
#include <QDropEvent>
#include <QFile>
#include <QFont>
#include <QMimeData>
#include <QIcon>
#include <QLabel>
#include <QMouseEvent>
#include <QPushButton>
#include <QScreen>
#include <QSettings>
#include <QStyleFactory>
#include <QTimer>

#include <cstdlib>

int main(int argc, char *argv[])
{
    // The identity has to be set before the first QSettings is constructed.
    // These are static setters on QCoreApplication and are legal this early.
    QCoreApplication::setOrganizationName(QStringLiteral("Trajecta"));
    QCoreApplication::setApplicationName(QStringLiteral("TrajectaStudio"));

    QApplication app(argc, argv);

    QApplication::setOrganizationName(QStringLiteral("Trajecta"));
    QApplication::setApplicationName(QStringLiteral("TrajectaStudio"));
    QApplication::setApplicationDisplayName(QStringLiteral("Trajecta Studio"));
    QApplication::setApplicationVersion(QStringLiteral(TRAJECTA_STUDIO_VERSION));
    QApplication::setWindowIcon(QIcon(QStringLiteral(":/assets/trajecta.ico")));

    QApplication::setStyle(QStyleFactory::create(QStringLiteral("Fusion")));

    QFont font(QStringLiteral("Segoe UI"), 11);
    QApplication::setFont(font);

    // Stylesheet and palette both come from the chosen colour theme. A first
    // run has no stored choice and falls back to ThemeManager::defaultIndex(),
    // which is Light — not index 0, which is reserved for the palette the
    // stylesheet is written in.
    const int saved = ThemeManager::indexOfId(
        QSettings().value(QStringLiteral("ui/theme")).toString());
    ThemeManager::apply(saved >= 0 ? saved : ThemeManager::defaultIndex());

    const QStringList args = app.arguments();

    // --grant-large-pages: grant SeLockMemoryPrivilege to this account and
    // exit. This is what the "Set up…" button launches, elevated, through the
    // UAC prompt; it builds no window and touches nothing else, so the whole
    // application never has to run as administrator.
    //
    // The exit code is the answer: 0 only when the right was added *and* read
    // back from the policy afterwards.
    if (args.contains(QStringLiteral("--grant-large-pages"))) {
        QString error;
        const bool ok = grantLockMemoryPrivilege(&error);
        if (!ok) {
            QTextStream(stderr) << error << Qt::endl;
            return 1;
        }
        return 0;
    }

    // --compare-routes <computed> <known> [tolerance_m]: run the known-route
    // comparison, print the report and exit, with no window at all. Handled
    // before MainWindow is built because it needs neither the interface nor a
    // display: it makes the comparison scriptable, which is how it is tested
    // and how a batch of comparisons can be run from a shell.
    // Both headless switches read layers through GDAL, so the library has to be
    // found before either runs. Done here rather than by building a MainWindow:
    // neither switch needs an interface, and the same candidate folders the
    // viewer uses are enough.
    const auto loadGdalHeadless = [] {
        QStringList dirs;
        const QString overrideDir =
            QSettings().value(QStringLiteral("env/gdalBinDir")).toString();
        if (!overrideDir.isEmpty())
            dirs << overrideDir;
        // The usual homes of the GDAL runtime on each platform, then whatever
        // is on PATH. Guarded rather than duplicated per copy: this file is
        // meant to read the same on Windows and on macOS.
#ifdef Q_OS_WIN
        dirs << QStringLiteral("C:/OSGeo4W/bin") << QStringLiteral("C:/OSGeo4W64/bin")
             << QStringLiteral("D:/OSGeo4W/bin") << QStringLiteral("D:/OSGeo4W64/bin");
#elif defined(Q_OS_MAC)
        // Homebrew on Apple silicon and on Intel, then a QGIS.app install, then
        // the self-contained bundle's own Frameworks folder.
        dirs << QStringLiteral("/opt/homebrew/lib") << QStringLiteral("/usr/local/lib")
             << QStringLiteral("/Applications/QGIS.app/Contents/Frameworks")
             << QCoreApplication::applicationDirPath() + QStringLiteral("/../Frameworks");
#else
        dirs << QStringLiteral("/usr/lib") << QStringLiteral("/usr/local/lib");
#endif
        for (const QString &dir : QProcessEnvironment::systemEnvironment()
                                     .value(QStringLiteral("PATH"))
                                     .split(QDir::listSeparator(), Qt::SkipEmptyParts)) {
            dirs << dir;
        }
        dirs.removeDuplicates();
        GdalApi::instance().load(dirs, QString(), QString());
    };

    const int cmpIdx = args.indexOf(QStringLiteral("--compare-routes"));
    if (cmpIdx >= 0 && cmpIdx + 2 < args.size()) {
        const double tol = (cmpIdx + 3 < args.size())
                               ? args.at(cmpIdx + 3).toDouble()
                               : 100.0;
        QTextStream out(stdout);
        loadGdalHeadless();

        // The running commentary goes to stderr, the report to stdout: a script
        // that only wants the numbers keeps parsing exactly what it did before,
        // and one that wants to see what was read redirects the other stream.
        QTextStream err(stderr);
        const RouteCompare::Result r =
            RouteCompare::compare(args.at(cmpIdx + 1), args.at(cmpIdx + 2),
                                  tol > 0.0 ? tol : 100.0,
                                  [&err](const QString &line) { err << line << Qt::endl; });
        out << r.report() << Qt::endl;
        return r.ok ? 0 : 1;
    }

    // --coherence <raster> <points> [options]: the site-corridor coherence,
    // printed and gone. Same reasoning as --compare-routes, and the same use:
    // it is how the numbers are checked against an independent implementation,
    // and it makes a study of twelve periods a shell loop rather than twelve
    // sittings at the interface.
    //
    //   --radius <m>            default 250
    //   --top <percent>         corridor = the top q% of the surface (default 1)
    //   --otsu                  corridor = Otsu's split on log values
    //   --absolute <value>      corridor = cells at or above a raw value
    //   --no-null               skip the random point sets
    //   --uniform-null          scatter the null points instead of shifting them
    //   --replicates <n>        default 999
    //   --sensitivity <r,r,...> also report these radii
    //   --ecdf <m,m,...>        the "share of sites within X metres" bands
    //   --out <folder> [--prefix <name>]   write the table, layer and raster
    const int cohIdx = args.indexOf(QStringLiteral("--coherence"));
    if (cohIdx >= 0 && cohIdx + 2 < args.size()) {
        Coherence::Params p;
        p.rasterPath = args.at(cohIdx + 1);
        p.pointsPath = args.at(cohIdx + 2);
        const auto value = [&args](const QString &flag, const QString &fallback) {
            const int i = args.indexOf(flag);
            return (i >= 0 && i + 1 < args.size()) ? args.at(i + 1) : fallback;
        };
        p.radiusMetres = value(QStringLiteral("--radius"), QStringLiteral("250")).toDouble();
        p.thresholdValue = value(QStringLiteral("--top"), QStringLiteral("1")).toDouble();
        if (args.contains(QStringLiteral("--otsu"))) {
            p.thresholdMode = Coherence::ThresholdMode::Otsu;
        } else if (args.contains(QStringLiteral("--absolute"))) {
            p.thresholdMode = Coherence::ThresholdMode::Absolute;
            p.thresholdValue =
                value(QStringLiteral("--absolute"), QStringLiteral("0")).toDouble();
        }
        p.nullModel = !args.contains(QStringLiteral("--no-null"));
        if (args.contains(QStringLiteral("--uniform-null")))
            p.nullMode = Coherence::NullMode::Uniform;
        p.nullReplicates =
            value(QStringLiteral("--replicates"), QStringLiteral("999")).toInt();
        const QString radii = value(QStringLiteral("--sensitivity"), QString());
        if (!radii.isEmpty()) {
            p.sensitivity = true;
            for (const QString &r : radii.split(QLatin1Char(','), Qt::SkipEmptyParts))
                p.sensitivityRadii << r.toDouble();
        }
        const QString bands = value(QStringLiteral("--ecdf"), QString());
        for (const QString &b : bands.split(QLatin1Char(','), Qt::SkipEmptyParts))
            p.ecdfDistances << b.toDouble();
        p.outputDir = value(QStringLiteral("--out"), QString());
        p.outputPrefix = value(QStringLiteral("--prefix"), QStringLiteral("coherence"));

        loadGdalHeadless();
        QTextStream out(stdout), err(stderr);
        const Coherence::Result r = Coherence::run(
            p, [&err](const QString &line) { err << line << Qt::endl; });
        out << r.report() << Qt::endl;
        return r.ok ? 0 : 1;
    }

    MainWindow window;
    window.show();

    // Hidden testing hooks: open a specific page / start the stored analysis.
    const int pageIdx = args.indexOf(QStringLiteral("--page"));
    if (pageIdx >= 0 && pageIdx + 1 < args.size()) {
        // "run" is kept as an alias for "setup": the live run panel is the tail
        // of that page now, and old scripts should not break over it. "about"
        // is kept the same way — that page is now its own Guide section, last
        // in the sidebar, and is opened directly rather than left for the
        // Overview it used to be folded into.
        const QStringList names = {QStringLiteral("setup"), QStringLiteral("post"),
                                   QStringLiteral("viewer"), QStringLiteral("guide")};
        const QString wanted = args.at(pageIdx + 1).toLower();
        const int page = (wanted == QLatin1String("run"))     ? 0
                         : (wanted == QLatin1String("about")) ? 3
                                                              : names.indexOf(wanted);
        if (page >= 0)
            window.showPage(page);
        if (wanted == QLatin1String("about") && window.guidePageCount() > 0)
            window.showGuideSectionForTest(window.guidePageCount() - 1);
    }
    // --guide-page <n>: the Guide is a dozen pages now, and a screenshot run
    // cannot click the one it wants out of the list on the left. 0 is Overview.
    const int guidePageIdx = args.indexOf(QStringLiteral("--guide-page"));
    if (guidePageIdx >= 0 && guidePageIdx + 1 < args.size()) {
        window.showPage(3);
        window.showGuideSectionForTest(args.at(guidePageIdx + 1).toInt());
    }
    // --mode <fete|lcpa|batch>: pick the analysis mode card. Batch mode is
    // otherwise only reachable by clicking, which a screenshot run cannot do.
    const int modeIdx = args.indexOf(QStringLiteral("--mode"));
    if (modeIdx >= 0 && modeIdx + 1 < args.size())
        window.selectMode(args.at(modeIdx + 1).toLower());
    // --post-mode <nni|compare>: pick the post-processing tool, for the same
    // reason --mode exists.
    const int postModeIdx = args.indexOf(QStringLiteral("--post-mode"));
    if (postModeIdx >= 0 && postModeIdx + 1 < args.size())
        window.selectPostMode(args.at(postModeIdx + 1));
    // --autorun-compare <computed> <known> [tolerance_m]: run the comparison
    // inside the interface, so the result panel and its log can be photographed
    // in the state a user would see. --compare-routes above is the headless
    // form; this one is about the widgets.
    const int autoCmpIdx = args.indexOf(QStringLiteral("--autorun-compare"));
    if (autoCmpIdx >= 0 && autoCmpIdx + 2 < args.size()) {
        const double tol = (autoCmpIdx + 3 < args.size())
                               ? args.at(autoCmpIdx + 3).toDouble()
                               : 0.0;
        window.triggerRouteComparison(args.at(autoCmpIdx + 1), args.at(autoCmpIdx + 2),
                                      tol);
    }
    // --autorun-coherence <raster> <points> [radius]: the same, for the
    // site-corridor tool. The headless form is --coherence; this one goes
    // through the pickers and the button, which is the path a user takes and
    // the only one that tests the panel, the log and the Viewer hand-off.
    const int autoCohIdx = args.indexOf(QStringLiteral("--autorun-coherence"));
    if (autoCohIdx >= 0 && autoCohIdx + 2 < args.size()) {
        const double radius = (autoCohIdx + 3 < args.size())
                                  ? args.at(autoCohIdx + 3).toDouble()
                                  : 0.0;
        window.triggerCoherence(args.at(autoCohIdx + 1), args.at(autoCohIdx + 2), radius);
    }
    // --drop-log <file>: write one line for every drag event the application
    // receives, with the class of the widget it was delivered to. A drag that
    // is refused and a drag that never arrives look identical on screen — the
    // same "no entry" cursor — and this is what tells them apart.
    const int dropLogIdx = args.indexOf(QStringLiteral("--drop-log"));
    if (dropLogIdx >= 0 && dropLogIdx + 1 < args.size()) {
        static QString logPath = args.at(dropLogIdx + 1);
        class DragSpy : public QObject
        {
        public:
            using QObject::QObject;
        protected:
            bool eventFilter(QObject *watched, QEvent *event) override
            {
                const char *name = nullptr;
                switch (event->type()) {
                case QEvent::DragEnter: name = "DragEnter"; break;
                case QEvent::DragMove:  name = "DragMove";  break;
                case QEvent::DragLeave: name = "DragLeave"; break;
                case QEvent::Drop:      name = "Drop";      break;
                default: break;
                }
                if (name) {
                    QFile f(logPath);
                    if (f.open(QIODevice::Append | QIODevice::Text)) {
                        QTextStream out(&f);
                        out << name << " -> "
                            << (watched ? watched->metaObject()->className() : "null")
                            << " (" << (watched ? watched->objectName() : QString())
                            << ")";
                        // What the drag is actually carrying. "It works for one
                        // file type and not another" can only be answered here:
                        // every handler downstream decides on the URLs, so if a
                        // format arrives without them the refusal happened in
                        // the conversion, before any of our code ran.
                        if (auto *de = dynamic_cast<QDropEvent *>(event)) {
                            const QMimeData *m = de->mimeData();
                            if (!m) {
                                out << "  [no mime data]";
                            } else {
                                out << "  formats=[" << m->formats().join(QLatin1Char('|'))
                                    << "] hasUrls=" << (m->hasUrls() ? "yes" : "no");
                                const QList<QUrl> urls = m->urls();
                                out << " urls=" << urls.size();
                                for (const QUrl &u : urls) {
                                    out << "\n    url: " << u.toString()
                                        << "  local='" << u.toLocalFile() << "'";
                                }
                                out << "\n    accepted=" << (de->isAccepted() ? "yes" : "no");
                            }
                        }
                        out << Qt::endl;
                    }
                }
                return QObject::eventFilter(watched, event);
            }
        };
        app.installEventFilter(new DragSpy(&app));
        QFile::remove(logPath);
    }
    // --pick-demo [n]: click the n-th point of the first overlay on the map, so
    // the feature panel it opens can be photographed.
    const int pickIdx = args.indexOf(QStringLiteral("--pick-demo"));
    if (pickIdx >= 0) {
        const int n = (pickIdx + 1 < args.size() && !args.at(pickIdx + 1).startsWith(QLatin1String("--")))
                          ? args.at(pickIdx + 1).toInt()
                          : 0;
        QTimer::singleShot(2500, &window, [&window, n] { window.pickFeatureForTest(n); });
    }
    // --wheel-demo [n]: open the overlay colour wheel, which is otherwise only
    // reachable by right-clicking a row.
    const int wheelIdx = args.indexOf(QStringLiteral("--wheel-demo"));
    if (wheelIdx >= 0) {
        const int n = (wheelIdx + 1 < args.size()
                       && !args.at(wheelIdx + 1).startsWith(QLatin1String("--")))
                          ? args.at(wheelIdx + 1).toInt()
                          : 0;
        QTimer::singleShot(2500, &window, [&window, n] { window.pickColourForTest(n); });
    }
    // --wheel-size-demo <index> <percent>: sets one overlay's size directly,
    // without opening the wheel — the way the slider inside it would, so two
    // overlays can be screenshotted at different sizes to confirm a change
    // never leaks onto the layer that was not right-clicked.
    const int wheelSizeIdx = args.indexOf(QStringLiteral("--wheel-size-demo"));
    if (wheelSizeIdx >= 0 && wheelSizeIdx + 2 < args.size()) {
        const int idx = args.at(wheelSizeIdx + 1).toInt();
        const int pct = args.at(wheelSizeIdx + 2).toInt();
        QTimer::singleShot(2500, &window,
                           [&window, idx, pct] { window.setOverlaySizeForTest(idx, pct); });
    }
    // --open-log: unfold every log canvas. They are collapsed by default and
    // opening one is a click, which a screenshot run cannot make.
    if (args.contains(QStringLiteral("--open-log")))
        window.openAllLogs();
    // --tour [--tour-step N]: start the guided walkthrough, optionally at a
    // given screen (1-based), so any of them can be photographed without
    // clicking through the ones before it.
    if (args.contains(QStringLiteral("--tour"))) {
        int step = 0;
        const int stepIdx = args.indexOf(QStringLiteral("--tour-step"));
        if (stepIdx >= 0 && stepIdx + 1 < args.size())
            step = qMax(0, args.at(stepIdx + 1).toInt() - 1);
        // After the window is on screen: the overlay measures widgets, and a
        // widget that has never been laid out has no geometry to measure.
        QTimer::singleShot(400, &window, [&window, step] { window.startWalkthrough(step); });
        // --tour-autoclose <ms>: shut the tour again after a while, so a
        // screenshot taken later shows what the application looks like once it
        // is over — which is how the "put everything back" rule gets checked.
        const int closeIdx = args.indexOf(QStringLiteral("--tour-autoclose"));
        if (closeIdx >= 0 && closeIdx + 1 < args.size()) {
            const int after = qMax(100, args.at(closeIdx + 1).toInt());
            QTimer::singleShot(400 + after, &window, [&window] { window.closeWalkthrough(); });
        }
    }
    // --batch-load <file.trjbatch> fills the batch page; adding --batch-run
    // starts it, which is how the batch is exercised end to end from the GUI.
    const int batchIdx = args.indexOf(QStringLiteral("--batch-load"));
    if (batchIdx >= 0 && batchIdx + 1 < args.size()) {
        const QString file = args.at(batchIdx + 1);
        const bool run = args.contains(QStringLiteral("--batch-run"));
        QTimer::singleShot(0, &window, [&window, file, run] {
            window.loadBatchFile(file);
            if (run)
                window.triggerBatchRun();
        });
    }
    // The same two hooks, for the post-processing batch page.
    const int postBatchIdx = args.indexOf(QStringLiteral("--post-batch-load"));
    if (postBatchIdx >= 0 && postBatchIdx + 1 < args.size()) {
        const QString file = args.at(postBatchIdx + 1);
        const bool run = args.contains(QStringLiteral("--post-batch-run"));
        QTimer::singleShot(0, &window, [&window, file, run] {
            window.loadPostBatchFile(file);
            if (run)
                window.triggerPostBatchRun();
        });
    }
    // --viewer-load may be repeated: a raster plus the vector overlays to
    // draw over it.
    for (int i = 0; i + 1 < args.size(); ++i) {
        if (args.at(i) != QStringLiteral("--viewer-load"))
            continue;
        const QString file = args.at(i + 1);
        QTimer::singleShot(0, &window, [&window, file] {
            window.viewerLoadFile(file);
        });
    }
    // --size <W>x<H>: force a window size, for checking layouts that follow
    // the window (the Guide figures) at a chosen width.
    const int sizeIdx = args.indexOf(QStringLiteral("--size"));
    if (sizeIdx >= 0 && sizeIdx + 1 < args.size()) {
        const QStringList wh = args.at(sizeIdx + 1).split(QLatin1Char('x'));
        if (wh.size() == 2) {
            const int w = wh.at(0).toInt();
            const int h = wh.at(1).toInt();
            if (w > 200 && h > 200)
                window.resize(w, h);
        }
    }
    // --scroll-end [fraction]: scroll the current long page (default: bottom).
    const int scrollIdx = args.indexOf(QStringLiteral("--scroll-end"));
    if (scrollIdx >= 0) {
        double fraction = 1.0;
        if (scrollIdx + 1 < args.size()) {
            bool ok = false;
            const double v = args.at(scrollIdx + 1).toDouble(&ok);
            if (ok)
                fraction = v;
        }
        // Late enough that the page has been laid out and its figures sized.
        QTimer::singleShot(800, &window,
                           [&window, fraction] { window.scrollSetupToEnd(fraction); });
    }
    // --autorun-points generates the sample points; adding --autorun chains the
    // analysis onto them instead of starting a second, independent run.
    if (args.contains(QStringLiteral("--autorun-points"))) {
        const bool chain = args.contains(QStringLiteral("--autorun"));
        QTimer::singleShot(0, &window, [&window, chain] {
            window.triggerPointsRun(chain);
        });
    } else if (args.contains(QStringLiteral("--autorun"))) {
        QTimer::singleShot(0, &window, &MainWindow::triggerRun);
    }
    if (args.contains(QStringLiteral("--autorun-interp")))
        QTimer::singleShot(0, &window, &MainWindow::triggerInterpRun);
    // --open-combo <n>: drop open the n-th combo of the current page, so its
    // popup can be inspected in a screenshot.
    const int comboIdx = args.indexOf(QStringLiteral("--open-combo"));
    if (comboIdx >= 0 && comboIdx + 1 < args.size()) {
        const int which = args.at(comboIdx + 1).toInt();
        QTimer::singleShot(600, &window, [&window, which] {
            window.openComboForTest(which);
        });
    }
    // --report-form: open the Guide page's report form. Here for the same
    // reason as --open-combo: the only way into it is a click on a link inside
    // a paragraph, which a screenshot run cannot make.
    if (args.contains(QStringLiteral("--report-form"))) {
        QTimer::singleShot(500, &window,
                           [&window] { TrajectaUi::showReportForm(&window); });
    }
    // --report-selftest: the same form, but it fills itself in, attaches a
    // generated image and presses Send. The only way to exercise the send path
    // without a person typing, and the join it crosses — Qt's multipart writer
    // against the relay's parser — is where a mistake would otherwise sit
    // undiscovered until a user's first real report vanished.
    if (args.contains(QStringLiteral("--report-selftest"))) {
        QTimer::singleShot(500, &window, [&window] {
            TrajectaUi::showReportForm(
                &window,
                QStringLiteral("Automated self-test of the report form. "
                               "Nothing is wrong; this message was sent by "
                               "--report-selftest and can be discarded."));
        });
    }
    // --advanced-settings: opens the dialog behind the gear menu's "Advanced
    // settings…" entry, so it can be screenshotted — it is modal, and nothing
    // else on the CLI can reach the menu item that opens it.
    if (args.contains(QStringLiteral("--advanced-settings"))) {
        QTimer::singleShot(500, &window, [&window] {
            window.showAdvancedSettingsForTest();
        });
    }
    // --progress <0-100> [--paused]: park the run panel's bar and the status
    // bar's ticker at a percentage, with no engine running, so the travelling
    // highlight and the two ticker states can be photographed.
    const int progIdx = args.indexOf(QStringLiteral("--progress"));
    if (progIdx >= 0 && progIdx + 1 < args.size()) {
        const int percent = args.at(progIdx + 1).toInt();
        const bool paused = args.contains(QStringLiteral("--paused"));
        const bool drawer = args.contains(QStringLiteral("--ticker-drawer"));
        QTimer::singleShot(200, &window, [&window, percent, paused, drawer] {
            window.setProgressForTest(percent, paused);
            if (drawer)
                window.openTickerDrawerForTest();
        });
    }
    // --help-demo: click the first "?" badge on the page in view, so the panel
    // it opens can be photographed. A real press through the real code path —
    // the badge's own mousePressEvent — not a shortcut into the popup.
    if (args.contains(QStringLiteral("--help-demo"))) {
        QTimer::singleShot(900, &window, [&window] {
            const QList<QLabel *> labels = window.findChildren<QLabel *>();
            for (QLabel *dot : labels) {
                if (dot->objectName() != QLatin1String("HelpDot") || !dot->isVisible())
                    continue;
                const QPointF local(8, 8);
                QMouseEvent press(QEvent::MouseButtonPress, local,
                                  dot->mapToGlobal(local), Qt::LeftButton,
                                  Qt::LeftButton, Qt::NoModifier);
                QApplication::sendEvent(dot, &press);
                break;
            }
        });
    }
    // --largepages-demo: press the real "Set up…" button, so the confirmation
    // is raised the way the user raises it — from the option's own widget, not
    // from here. --confirm-demo builds the same box with the window as its
    // parent, and the two did not look the same: this is the one that tells the
    // truth. Nothing is granted, because the elevated half only starts once the
    // question is answered.
    if (args.contains(QStringLiteral("--largepages-demo"))) {
        QTimer::singleShot(900, &window, [&window] {
            const QList<QPushButton *> buttons = window.findChildren<QPushButton *>();
            for (QPushButton *b : buttons) {
                if (b->text().startsWith(QLatin1String("Set up"))) {
                    b->click();
                    break;
                }
            }
        });
    }
    // --drop-demo <file> [<file> ...]: drop files on the Viewer the way a file
    // manager would, up to the next switch.
    const int dropIdx = args.indexOf(QStringLiteral("--drop-demo"));
    if (dropIdx >= 0) {
        QStringList files;
        for (int i = dropIdx + 1; i < args.size(); ++i) {
            if (args.at(i).startsWith(QLatin1String("--")))
                break;
            files << args.at(i);
        }
        if (!files.isEmpty()) {
            QTimer::singleShot(700, &window,
                               [&window, files] { window.dropOnViewerForTest(files); });
        }
    }
    // --drop-again <file> ...: a SECOND drop, four seconds after the first.
    // "The first one works and nothing works afterwards" cannot be reproduced
    // with one drop, and one drop is all this could do.
    const int dropAgainIdx = args.indexOf(QStringLiteral("--drop-again"));
    if (dropAgainIdx >= 0) {
        QStringList files;
        for (int i = dropAgainIdx + 1; i < args.size(); ++i) {
            if (args.at(i).startsWith(QLatin1String("--")))
                break;
            files << args.at(i);
        }
        if (!files.isEmpty()) {
            QTimer::singleShot(4000, &window,
                               [&window, files] { window.dropOnViewerForTest(files); });
        }
    }
    // --confirm-demo: raise the two-colour confirmation (the one the large
    // pages setup asks) so its buttons can be photographed in any theme
    // without granting a Windows privilege to do it.
    if (args.contains(QStringLiteral("--confirm-demo"))) {
        QTimer::singleShot(600, &window, [&window] {
            TrajectaUi::confirm(
                &window, QObject::tr("Enable large memory pages"),
                QObject::tr("Windows will ask for administrator approval.\n\n"
                            "The \"Lock pages in memory\" privilege will be granted "
                            "to your account. You then have to sign out of Windows "
                            "and back in before it takes effect.\n\nContinue?"),
                QString(), QString(), 70, TrajectaUi::Fill::AcceptDanger);
        });
    }
    // --theme <id>: apply a palette for this launch only. The choice is put
    // back immediately, because applying a theme is also what saves it, and a
    // screenshot run must not change what the user opens tomorrow.
    const int themeIdx = args.indexOf(QStringLiteral("--theme"));
    if (themeIdx >= 0 && themeIdx + 1 < args.size()) {
        const int wanted = ThemeManager::indexOfId(args.at(themeIdx + 1));
        if (wanted >= 0) {
            const QString before = QSettings().value(QStringLiteral("ui/theme")).toString();
            window.applyThemeForTest(wanted);
            QSettings().setValue(QStringLiteral("ui/theme"), before);
        }
    }
    const int shotIdx = args.indexOf(QStringLiteral("--screenshot"));
    if (shotIdx >= 0 && shotIdx + 1 < args.size()) {
        const QString file = args.at(shotIdx + 1);
        int delayMs = 700;
        const int delayIdx = args.indexOf(QStringLiteral("--screenshot-delay"));
        if (delayIdx >= 0 && delayIdx + 1 < args.size())
            delayMs = qMax(100, args.at(delayIdx + 1).toInt());
        QTimer::singleShot(delayMs, &window, [&window, file] {
            // A modal dialog is a whole top-level window, so it can be grabbed
            // on its own. That is better than photographing the screen: the
            // screen shows whichever window happens to be in front, and a
            // freshly launched application does not always win that contest
            // from Windows — the shot then comes back showing something else
            // entirely, and whatever else the user had open with it.
            //
            // A combo popup is not the same case: it is a fragment that only
            // means anything drawn over the control it belongs to, so that one
            // still takes the screen.
            if (QWidget *modal = QApplication::activeModalWidget()) {
                modal->grab().save(file);
            } else if (QApplication::activePopupWidget()) {
                if (QScreen *screen = window.screen())
                    screen->grabWindow(0).save(file);
            } else {
                window.grab().save(file);
            }
            // Hard exit, not quit(): a modal dialog runs its own nested event
            // loop, which quit() does not unwind, and the screenshot process
            // would then sit on screen forever waiting for someone to click it.
            // It also skips closeEvent(), and therefore saveSettings() — which
            // is what we want here: taking a screenshot must not overwrite the
            // preferences of the copy the user is actually working in.
            std::_Exit(0);
        });
    }

    return app.exec();
}
