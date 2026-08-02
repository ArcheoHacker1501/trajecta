#include "mainwindow.h"
#include "thememanager.h"

#include <QApplication>
#include <QFile>
#include <QFont>
#include <QIcon>
#include <QScreen>
#include <QSettings>
#include <QStyleFactory>
#include <QTimer>

int main(int argc, char *argv[])
{
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

    MainWindow window;
    window.show();

    // Hidden testing hooks: open a specific page / start the stored analysis.
    const QStringList args = app.arguments();
    const int pageIdx = args.indexOf(QStringLiteral("--page"));
    if (pageIdx >= 0 && pageIdx + 1 < args.size()) {
        const QStringList names = {QStringLiteral("setup"), QStringLiteral("run"),
                                   QStringLiteral("post"), QStringLiteral("viewer"),
                                   QStringLiteral("guide"), QStringLiteral("about")};
        const int page = names.indexOf(args.at(pageIdx + 1).toLower());
        if (page >= 0)
            window.showPage(page);
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
    const int shotIdx = args.indexOf(QStringLiteral("--screenshot"));
    if (shotIdx >= 0 && shotIdx + 1 < args.size()) {
        const QString file = args.at(shotIdx + 1);
        int delayMs = 700;
        const int delayIdx = args.indexOf(QStringLiteral("--screenshot-delay"));
        if (delayIdx >= 0 && delayIdx + 1 < args.size())
            delayMs = qMax(100, args.at(delayIdx + 1).toInt());
        QTimer::singleShot(delayMs, &window, [&window, file] {
            // A combo popup is its own top-level window: grabbing the main
            // window alone would miss it.
            if (QApplication::activePopupWidget()) {
                if (QScreen *screen = window.screen())
                    screen->grabWindow(0).save(file);
            } else {
                window.grab().save(file);
            }
            QApplication::quit();
        });
    }

    return app.exec();
}
