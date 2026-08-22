#include "thememanager.h"

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QPalette>
#include <QRegularExpression>
#include <QSettings>
#include <QStandardPaths>

namespace {

// Writes (once per colour) the spin-box triangle in `ink` and returns a path
// usable inside a stylesheet url(). Empty on failure, which leaves the
// %ARROW_*% token in place and simply shows no arrow rather than crashing.
QString arrowSvgPath(const QString &direction, const QString &ink)
{
    const QString dir =
        QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    if (dir.isEmpty() || !QDir().mkpath(dir))
        return QStringLiteral(":/assets/arrow_%1.svg").arg(direction);

    // Colour in the name: themes then share a cache instead of overwriting it.
    QString hex = ink;
    hex.remove(QLatin1Char('#'));
    const QString path = QStringLiteral("%1/arrow_%2_%3.svg").arg(dir, direction, hex);

    if (!QFile::exists(path)) {
        const QString shape = direction == QLatin1String("up")
            ? QStringLiteral("M4.5 0.5 L8.5 5.5 L0.5 5.5 Z")
            : QStringLiteral("M0.5 0.5 L8.5 0.5 L4.5 5.5 Z");
        QFile out(path);
        if (!out.open(QFile::WriteOnly | QFile::Truncate))
            return QStringLiteral(":/assets/arrow_%1.svg").arg(direction);
        out.write(QStringLiteral(
                      "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"9\" "
                      "height=\"6\" viewBox=\"0 0 9 6\">\n"
                      "  <path d=\"%1\" fill=\"%2\"/>\n</svg>\n")
                      .arg(shape, ink)
                      .toUtf8());
        out.close();
    }
    // Stylesheet url() wants forward slashes even on Windows.
    return QDir::fromNativeSeparators(path);
}

// Every colour literal in theme.qss, grouped by the role it plays. A palette
// supplies one replacement per entry, in this order — same length, same
// meaning — so adding a colour to the stylesheet without giving the palettes a
// value for it is caught by the assertion in themes() rather than by a stray
// dark patch in a light theme.
const char *const kDarkColors[] = {
    // Surfaces
    "#14171c", "#1a1e24", "#1b1f26", "#171b21", "#0f1216", "#0d0f13",
    "#22272f", "#272d36", "#1c2128", "#191d23", "#1b2027", "#262c34",
    // Borders
    "#262b33", "#282d36", "#2a2f38", "#333a44", "#3a414b", "#3f4753",
    "#21262e", "#242932", "#2e343d",
    // Text, brightest to faintest
    "#eef1f5", "#e4e7ec", "#d3dae2", "#b6bec9", "#99a1ac", "#8a929d",
    "#79818c", "#69717b", "#5c646e", "#4a525c",
    // Accent
    "#7ea8a0", "#93bcb3", "#6b968f", "#5c847d", "#a8d0c8", "#d7e6e2",
    "#1e2a28", "#1a2422", "#12211e", "#10201d", "#2f3a38", "#6b7d79",
    // Status: running, paused, success, failed, danger button
    "#2f2a17", "#d3a25e", "#1e2733", "#7f9cc4", "#1e2a24", "#7fb08a",
    "#2a2022", "#cf7f7f", "#5c3a3e", "#cf9a9a", "#33262a", "#1e1a1c",
    "#2c2427",
    // Mode cards, selected: LCPA and batch. FETE keeps the accent, so only two
    // values are needed. Picked theme by theme to stand well apart from that
    // theme's accent and from each other — which is the whole point of the
    // colour — and to carry the same "text on the accent" as the accent does,
    // so they invert with it on the paper palettes.
    "#c9905a", "#a58bd0",
};
constexpr int kColorCount = int(std::size(kDarkColors));

// Nordic, Ember, Parchment and Indigo lived here: cool blue-grey, warm
// copper charcoal, warm paper, and blue-violet, four more turns on the same
// dark-with-a-map-and-cards design. Withdrawn together to leave the menu
// short enough to read at a glance — four palettes that each say something
// the others do not (a working dark theme, a light one, a terminal, an
// artwork) rather than eight that mostly differed in hue. Daylight had gone
// the same way earlier, for the same reason relative to Liquid Glass.

QVector<QPair<QString, QString>> buildMap(const char *const *values)
{
    QVector<QPair<QString, QString>> map;
    map.reserve(kColorCount);
    for (int i = 0; i < kColorCount; ++i)
        map.append({QString::fromLatin1(kDarkColors[i]), QString::fromLatin1(values[i])});
    return map;
}

int g_current = 0;

} // namespace

// --- Liquid Glass: the iOS frosted-panel look. Qt cannot blur what is behind a
// widget, so the impression is built the way a flat design would: near-white
// cool panels, borders that are barely there, generous radii and a single
// saturated blue doing all the signalling. ---
const char *const kGlass[kColorCount] = {
    "#eef2f7", "#f7fafc", "#ffffff", "#f2f6fa", "#ffffff", "#e8eef5",
    "#e9eff6", "#dfe8f2", "#d3dfec", "#f1f5f9", "#eaf0f6", "#e0e8f1",
    "#dde5ee", "#d6e0ea", "#cfdae6", "#c3d1e0", "#b3c4d6", "#a3b7cc",
    "#e3eaf2", "#dce4ed", "#c9d6e3",
    "#0f172a", "#1e293b", "#334155", "#475569", "#64748b", "#708096",
    "#7d8ba1", "#94a3b8", "#a8b4c4", "#bcc6d4",
    // The last two of this row are text drawn ON the accent background. In the
    // dark theme they are light because that background is dark; on a light
    // theme the background inverts, so these must invert too or the selected
    // mode card ends up pale-on-pale.
    "#3b82f6", "#60a5fa", "#2563eb", "#1d4ed8", "#1e40af", "#1e3a8a",
    "#eff6ff", "#e0edff", "#ffffff", "#ffffff", "#dbe4ef", "#8fa7c4",
    "#fff7e6", "#b7791f", "#eef4ff", "#3b6fb5", "#eafaf0", "#2f8f55",
    "#fdeaea", "#c23b3b", "#f0b4b4", "#a83232", "#fce9e9", "#f7f0f0",
    "#f0e2e2",
    "#c2610a", "#17916b",   // mode cards: LCPA, batch
};

// --- Neon Circuit: near-black violet with a cyan that does the work of a
// backlight, magenta reserved for failure. Corners tighten and labels gain
// letter spacing, which is most of what reads as "terminal". ---
const char *const kCyber[kColorCount] = {
    "#0a0812", "#100c1c", "#140f22", "#0e0a18", "#06040c", "#050308",
    "#1c1430", "#251a3e", "#150f26", "#120d1e", "#1a1230", "#221838",
    "#2a1f45", "#33254f", "#3d2b5e", "#4d3673", "#614589", "#7a58a8",
    "#241a3c", "#1f1634", "#382a54",
    "#f0e6ff", "#ddd0f5", "#c4b3e6", "#a892d1", "#8f76bd", "#7d64ab",
    "#6b5399", "#5a4487", "#4a3775", "#3b2b60",
    "#00e5ff", "#5cf2ff", "#00b8cc", "#0091a3", "#7df6ff", "#c9fbff",
    "#0a2b33", "#08222a", "#061a20", "#04121a", "#1b3a42", "#4d8e99",
    "#3a2410", "#ffb454", "#16213f", "#6ea8ff", "#0d2e1c", "#4ef08a",
    "#3a0f2a", "#ff3d8b", "#6b1d47", "#ff77b0", "#2c0d20", "#200a18",
    "#2a0c1e",
    "#ffb454", "#c77dff",   // mode cards: LCPA, batch
};

// --- Washi: sumi ink on rice paper. Warm off-white surfaces, a text ramp that
// behaves like diluted ink rather than grey, and the accent taken from the
// vermillion of a hanko seal. Serif type and hairline borders. ---
const char *const kWashi[kColorCount] = {
    "#f3ede0", "#f7f2e7", "#fbf7ee", "#f0e9db", "#fdfaf3", "#e8e0cf",
    "#ece4d4", "#e3d9c5", "#d9cdb5", "#f0ebe0", "#eee7d8", "#e5dcc9",
    "#ded3bd", "#d6cab2", "#cec1a7", "#bfae90", "#ab9877", "#97835f",
    "#e4dbc8", "#ded5c2", "#c6b79c",
    "#1c1814", "#2b251e", "#3d352b", "#52483b", "#6b5f4e", "#7a6d5a",
    "#897b66", "#998a74", "#a89882", "#b8a892",
    // Same inversion as Liquid Glass: these two are ink on the accent wash.
    "#b8452f", "#cf5a41", "#9c3521", "#82271a", "#8f3020", "#6b2415",
    "#f7e9e4", "#f2ded7", "#fdfaf3", "#fdfaf3", "#e0d0c6", "#a08578",
    "#f7edd6", "#8a6a1c", "#e6ecf2", "#3f5f80", "#e6f0e2", "#4a7a42",
    "#f7e3df", "#a83a28", "#d4a396", "#8f3020", "#f2e0db", "#ede2dd",
    "#e8d8d2",
    // Vermillion accent: both of these stay well away from red.
    "#4a6b53", "#4a5a8a",   // mode cards: LCPA, batch
};

// Shape overrides, appended after the colour-mapped sheet. Kept to properties
// that cannot break a layout — radii, type, spacing, border weight — so a
// palette can still never move a widget.
const char *const kGlassQss = R"(
QMainWindow, QWidget { font-family: "Segoe UI"; }
QFrame#Card { border-radius: 18px; border: 1px solid rgba(255,255,255,0.7); }
/* The map frame is a card in every way that matters, so it follows the same
   radius and the same barely-there border — and the walkthrough, which cuts
   its light in the card's shape, has one number to trust rather than two.
   Without the border line here it kept the base sheet's #282d36, a visibly
   stronger line than every other panel on the page. */
QFrame#CanvasHolder { border-radius: 18px; border: 1px solid rgba(255,255,255,0.7); }
/* The Guide's pages are cards too — its two widget-based ones (Overview,
   About) already get the two lines above via #Card, but its document-based
   ones do not carry that object name, so without this they keep the base
   sheet's 12px/#282d36 and sit at a visibly different radius with a visibly
   stronger border right next to pages that used the barely-there Glass one. */
QTextBrowser#GuideBrowser { border-radius: 18px; border: 1px solid rgba(255,255,255,0.7); }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { border-radius: 12px; padding: 9px 12px; }
QPushButton { border-radius: 12px; }
QPushButton#RunButton, QPushButton#SecondaryRunButton { border-radius: 14px; }
QComboBox QAbstractItemView { border-radius: 14px; }
QDialog#ConfirmDialog { border-radius: 16px; }
QLabel#CardTitle { font-weight: 600; letter-spacing: 0px; }
QToolButton#WindowButton, QToolButton#WindowCloseButton { border-radius: 10px; }
)";

const char *const kCyberQss = R"(
QFrame#Card { border-radius: 3px; }
QFrame#CanvasHolder { border-radius: 3px; }
/* Same reasoning as Glass's matching line: the Guide's document-based pages
   do not carry the #Card object name, so they need their own copy of this
   override or they sit at the base sheet's 12px next to the 3px cards. */
QTextBrowser#GuideBrowser { border-radius: 3px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { border-radius: 2px; }
QPushButton { border-radius: 2px; letter-spacing: 1px; }
QPushButton#RunButton, QPushButton#SecondaryRunButton { border-radius: 3px; letter-spacing: 2px; }
QComboBox QAbstractItemView { border-radius: 3px; }
QDialog#ConfirmDialog { border-radius: 3px; }
QLabel#CardTitle { letter-spacing: 3px; text-transform: uppercase; }
QPushButton#TabButton { border-radius: 2px; letter-spacing: 1px; }
QToolButton#WindowButton, QToolButton#WindowCloseButton { border-radius: 2px; }
)";

// Washi is the only theme with a picture behind it. Three things have to line
// up for that to be visible at all:
//
//  1. border-image, not background-image. QSS has no "background-size: cover",
//     so background-image would show a 1:1 crop of the middle of a 3840 px
//     picture; border-image scales the whole thing to the window. The source
//     is oversampled for any real display, so downscaling keeps it sharp.
//  2. The chain of containers between the window and the cards — central
//     widget, stacked pages, scroll area and its viewport — all inherit the
//     opaque background that theme.qss gives every QWidget, and would hide the
//     picture entirely. They are cleared here.
//  3. The cards themselves stay paper, but slightly translucent, so the
//     artwork reads behind them without costing text contrast. 0.90 keeps body
//     text well above the contrast floor; lower values start to hurt.
const char *const kWashiQss = R"(
QMainWindow, QWidget { font-family: "Georgia", "Cambria", serif; }

QMainWindow {
    border-image: url(:/assets/background/japanese_art.jpg) 0 0 0 0 stretch stretch;
}
QWidget#CentralArea { background: transparent; }
QStackedWidget, QStackedWidget > QWidget { background: transparent; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }

QFrame#TopBar { background-color: rgba(247, 242, 231, 0.92); }
QFrame#StatusBar { background-color: rgba(247, 242, 231, 0.92); }

/* Loose labels would each paint their own opaque rectangle over the picture,
   turning a page into a stack of unrelated strips. */
QLabel { background: transparent; }

/* Rounded like the Guide's panel, which is the reference for every section. */
QFrame#Card {
    background-color: rgba(251, 247, 238, 0.90);
    border-radius: 12px;
    border: 1px solid #cec1a7;
}
QTextBrowser#GuideBrowser { background-color: rgba(251, 247, 238, 0.90); }

/* The map canvas is a card too, and on this theme it says so: paper over the
   artwork rather than the one solid slab on the page. The view inside paints
   nothing of its own — see MapView, which drops its background brush when the
   theme asks for this — so what shows through the empty parts of the map is the
   picture, at the same strength as under every other panel. */
QFrame#CanvasHolder {
    background-color: rgba(251, 247, 238, 0.90);
    border-radius: 12px;
    border: 1px solid #cec1a7;
}
QGraphicsView#ViewerCanvas { background: transparent; }

/* The Guide's two side columns are transparent by default like everything
   else on this theme, which here means the wave print at full strength
   right behind the text. The same paper backing every other panel gets
   fixes it; the dark slate divider line becomes the same warm border
   Card/CanvasHolder use, since a cool grey line looked stray on paper.
   Same radius too, so all three panels on the page read as the one shape. */
QListWidget#GuideSidebar {
    background-color: rgba(251, 247, 238, 0.90);
    border-right: 1px solid #cec1a7;
    border-radius: 12px;
}
QFrame#GuideTocPanel {
    background-color: rgba(251, 247, 238, 0.90);
    border-left: 1px solid #cec1a7;
    border-radius: 12px;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { border-radius: 4px; }
QPushButton { border-radius: 4px; }
QPushButton#RunButton, QPushButton#SecondaryRunButton { border-radius: 6px; font-weight: 600; }
QComboBox QAbstractItemView { border-radius: 6px; }
QDialog#ConfirmDialog { border-radius: 8px; }
QLabel#CardTitle { font-weight: 600; }
QTextBrowser { font-family: "Consolas", monospace; }
)";

const QVector<ThemeManager::Theme> &ThemeManager::themes()
{
    static const QVector<Theme> list = [] {
        QVector<Theme> t;
        // Order here is the order in the menu. Nothing depends on a particular
        // index: Midnight is the palette theme.qss is literally written in, and
        // it stays the identity because its map is *empty*, not because of
        // where it sits.
        t.append({QStringLiteral("glass"), QObject::tr("Default"), true,
                  QColor(0x1d, 0x4e, 0xd8), buildMap(kGlass),
                  QString::fromLatin1(kGlassQss)});
        t.append({QStringLiteral("midnight"), QObject::tr("Midnight"), false,
                  QColor(0xd3, 0xa2, 0x5e), {}});
        // These two also override shape, not just colour (see extraQss).
        t.append({QStringLiteral("cyber"), QObject::tr("Neon Circuit"), false,
                  QColor(0xff, 0x3d, 0x8b), buildMap(kCyber),
                  QString::fromLatin1(kCyberQss)});
        // The only theme with a picture behind it, and so the only one where a
        // translucent canvas shows anything: hence the trailing true.
        t.append({QStringLiteral("washi"), QObject::tr("Washi"), true,
                  QColor(0xb8, 0x45, 0x2f), buildMap(kWashi),
                  QString::fromLatin1(kWashiQss), true});
        return t;
    }();
    return list;
}

const ThemeManager::Theme &ThemeManager::theme(int index)
{
    const QVector<Theme> &list = themes();
    if (index < 0 || index >= list.size())
        index = 0;
    return list.at(index);
}

// All of these ship with Windows 10/11, so none of them can fail to resolve and
// silently fall back to something arbitrary.
const QVector<ThemeManager::FontChoice> &ThemeManager::fonts()
{
    static const QVector<FontChoice> list = [] {
        QVector<FontChoice> f;
        f.append({QStringLiteral("theme"),    QObject::tr("Theme default"), QString()});
        f.append({QStringLiteral("segoe"),    QStringLiteral("Segoe UI"),   QStringLiteral("Segoe UI")});
        f.append({QStringLiteral("calibri"),  QStringLiteral("Calibri"),    QStringLiteral("Calibri")});
        f.append({QStringLiteral("verdana"),  QStringLiteral("Verdana"),    QStringLiteral("Verdana")});
        f.append({QStringLiteral("georgia"),  QStringLiteral("Georgia"),    QStringLiteral("Georgia")});
        f.append({QStringLiteral("cambria"),  QStringLiteral("Cambria"),    QStringLiteral("Cambria")});
        f.append({QStringLiteral("consolas"), QStringLiteral("Consolas"),   QStringLiteral("Consolas")});
        return f;
    }();
    return list;
}

int ThemeManager::currentFont()
{
    const QString id = QSettings().value(QStringLiteral("ui/font")).toString();
    const QVector<FontChoice> &list = fonts();
    for (int i = 0; i < list.size(); ++i) {
        if (list.at(i).id == id)
            return i;
    }
    // Nothing chosen, or something no longer offered: the platform's neutral
    // sans, which is entry 1 by construction — the list is built as "Theme
    // default" followed by that face. Not entry 0: a theme is free to ask for a
    // face that suits its idea (Liquid Glass asks for a monospace, Washi for a
    // serif), and that is a fine thing to be able to choose, but it is not what
    // an interface should look like before anyone has chosen anything.
    return list.size() > 1 ? 1 : 0;
}

void ThemeManager::setFont(int index)
{
    const QVector<FontChoice> &list = fonts();
    if (index < 0 || index >= list.size())
        index = 0;
    QSettings().setValue(QStringLiteral("ui/font"), list.at(index).id);
    apply(current());   // the font is part of the sheet, so rebuild it
}

// Midnight. The palette in theme.qss is a dark one and every other theme is a
// colour map laid over it, so the dark theme is the one nothing has to be
// re-checked against — and this is a program whose users sit in front of a
// raster for hours. Only for someone who has never chosen: a saved choice wins,
// which is what the `saved >= 0` test in main() is for.
int ThemeManager::defaultIndex()
{
    const int i = indexOfId(QStringLiteral("midnight"));
    return i >= 0 ? i : 0;
}

int ThemeManager::indexOfId(const QString &id)
{
    const QVector<Theme> &list = themes();
    for (int i = 0; i < list.size(); ++i) {
        if (list.at(i).id == id)
            return i;
    }
    return -1;
}

int ThemeManager::current()
{
    return g_current;
}

QColor ThemeManager::mapped(const char *darkHex)
{
    const Theme &t = theme(g_current);
    for (const auto &pair : t.map) {
        if (pair.first == QLatin1String(darkHex))
            return QColor(pair.second);
    }
    return QColor(QString::fromLatin1(darkHex));
}

bool ThemeManager::isLight()
{
    return theme(g_current).light;
}

int ThemeManager::cardRadius()
{
    // theme.qss says 12; the two themes that restate it in extraQss are the
    // exceptions. Kept beside them on purpose — if either number is edited up
    // there, this is the line that has to follow, and a comment in one place is
    // easier to obey than a rule spread over two files.
    const QString id = theme(g_current).id;
    if (id == QLatin1String("glass"))
        return 18;
    if (id == QLatin1String("cyber"))
        return 3;
    return 12;
}

void ThemeManager::apply(int index)
{
    if (index < 0 || index >= themes().size())
        index = 0;
    const Theme &t = theme(index);
    g_current = index;

    QFile qss(QStringLiteral(":/theme.qss"));
    if (!qss.open(QFile::ReadOnly | QFile::Text))
        return;
    QString sheet = QString::fromUtf8(qss.readAll());

    if (!t.map.isEmpty()) {
        QHash<QString, QString> lookup;
        for (const auto &pair : t.map)
            lookup.insert(pair.first, pair.second);
        // One pass over the sheet: a sequence of replace() calls could rewrite
        // a colour a previous replacement had just produced.
        static const QRegularExpression hex(QStringLiteral("#[0-9a-fA-F]{6}"));
        QString out;
        out.reserve(sheet.size());
        int pos = 0;
        auto it = hex.globalMatch(sheet);
        while (it.hasNext()) {
            const QRegularExpressionMatch m = it.next();
            out += sheet.mid(pos, m.capturedStart() - pos);
            out += lookup.value(m.captured().toLower(), m.captured());
            pos = m.capturedEnd();
        }
        out += sheet.mid(pos);
        sheet = out;
    }

    // Fusion draws combo popups, menus, message boxes and spin arrows from the
    // QPalette, not from the stylesheet: keep the two in step.
    const auto col = [&t](const char *darkHex) {
        for (const auto &pair : t.map) {
            if (pair.first == QLatin1String(darkHex))
                return QColor(pair.second);
        }
        return QColor(QString::fromLatin1(darkHex));
    };
    // Spin-box arrows sit on the accent background, so their glyph has to be
    // the accent's foreground — which every palette defines separately (dark
    // ink on the dark theme's light teal, paper on Parchment's sienna). A
    // resource SVG carries a baked fill that the colour map above cannot
    // reach, so the two triangles are emitted per theme and the stylesheet
    // points at those instead. Falls back to the bundled greys if the cache
    // directory is not writable.
    const QString arrowInk = col("#10201d").name();
    sheet.replace(QLatin1String("%ARROW_UP%"),
                  arrowSvgPath(QStringLiteral("up"), arrowInk));
    sheet.replace(QLatin1String("%ARROW_DOWN%"),
                  arrowSvgPath(QStringLiteral("down"), arrowInk));

    // Shape overrides come last so they win over the base rules they restate.
    // The colour map has already run, so any hex written here is a literal and
    // must be one this theme actually wants.
    if (!t.extraQss.isEmpty())
        sheet += QLatin1String("\n/* ---- theme shape overrides ---- */\n") + t.extraQss;

    // Last of all, so an explicit font choice beats the one a theme asked for
    // (Washi's serif). "Theme default" adds nothing and leaves the theme in
    // charge. The monospace console keeps its own family either way.
    const FontChoice &fc = fonts().at(currentFont());
    if (!fc.family.isEmpty()) {
        sheet += QStringLiteral("\n/* ---- user font ---- */\n"
                                "QMainWindow, QWidget, QMenu, QDialog "
                                "{ font-family: \"%1\"; }\n").arg(fc.family);
    }

    // Absolutely last: the brand keeps one face and one spacing whatever the
    // theme or the user font says. A wider face or extra letter spacing made
    // the layout clip it to "TRAJECTA STU".
    sheet += QLatin1String(
        "\n/* ---- brand, never restyled ---- */\n"
        "QLabel#TopBarTitle { font-family: \"Segoe UI\"; font-size: 22px; "
        "font-weight: 700; letter-spacing: 3px; }\n");

    QPalette p;
    p.setColor(QPalette::Window, col("#14171c"));
    p.setColor(QPalette::WindowText, col("#e4e7ec"));
    p.setColor(QPalette::Base, col("#0f1216"));
    p.setColor(QPalette::AlternateBase, col("#1b1f26"));
    p.setColor(QPalette::Text, col("#e4e7ec"));
    p.setColor(QPalette::Button, col("#22272f"));
    p.setColor(QPalette::ButtonText, col("#e4e7ec"));
    p.setColor(QPalette::BrightText, col("#eef1f5"));
    p.setColor(QPalette::Highlight, col("#7ea8a0"));
    p.setColor(QPalette::HighlightedText, col("#12211e"));
    p.setColor(QPalette::Link, col("#7f9cc4"));
    p.setColor(QPalette::ToolTipBase, col("#22272f"));
    p.setColor(QPalette::ToolTipText, col("#e4e7ec"));
    p.setColor(QPalette::PlaceholderText, col("#5c646e"));
    p.setColor(QPalette::Disabled, QPalette::Text, col("#5c646e"));
    p.setColor(QPalette::Disabled, QPalette::WindowText, col("#5c646e"));
    p.setColor(QPalette::Disabled, QPalette::ButtonText, col("#5c646e"));
    QApplication::setPalette(p);
    qApp->setStyleSheet(sheet);

    QSettings().setValue(QStringLiteral("ui/theme"), t.id);
}
