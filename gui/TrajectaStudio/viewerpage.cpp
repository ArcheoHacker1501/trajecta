#include "viewerpage.h"

#include "confirmdialog.h"
#include "gdalapi.h"
#include "rangeslider.h"
#include "smoothcombobox.h"
#include "thememanager.h"
#include "uiwidgets.h"

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QCursor>
#include <QDesktopServices>
#include <QDialog>
#include <QDir>
#include <QDialogButtonBox>
#include <QStandardItemModel>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGraphicsPathItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QHBoxLayout>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDropEvent>
#include <QImage>
#include <QLabel>
#include <QMimeData>
#include <QMouseEvent>
#include <QNetworkAccessManager>
#include <QNetworkDiskCache>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPainter>
#include <QPainterPath>
#include <QRegion>
#include <QScrollBar>
#include <QSettings>
#include <QSlider>
#include <QListView>
#include <QMessageBox>
#include <QStackedLayout>
#include <QStandardPaths>
#include <QStyle>
#include <QStyledItemDelegate>
#include <QScrollArea>
#include <QTemporaryDir>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>


namespace {

// Shown on the empty canvas. One place, because it is set from three.
QString kEmptyViewerHint()
{
    return QObject::tr("No layer loaded.");
}

// Parented to the view rather than to its viewport, like the legend and for the
// same reason: dragging the map must not drag the panel with it.
// Wide enough for a name and a number side by side, short enough to leave the
// map visible on any window this application is usable in at all. Declared here
// because the scalebar needs the width too — it is drawn on the opposite corner
// and must stop short of this one.
constexpr int kFeaturePanelWidth = 270;
// Grown twice over the original 270: a feature with more than four or five
// attributes filled the old cap and started scrolling immediately, and the
// shorter scalebar beside it (see the "Scalebar, bottom-left" block) freed up
// more of the corner to grow into.
// Fixed, not just a cap: set as both the minimum and the maximum below, so
// the panel is the same size whether the clicked feature has two fields or
// twenty. Left to shrink-to-content it jumped around from click to click,
// which read as broken rather than as a feature.
constexpr int kFeaturePanelMaxHeight = 400;
// The panel is anchored to its bottom edge (see positionFeaturePanel()), so
// taller alone only reaches further up. This claws back part of the growth
// for the bottom edge too — roughly a third of it — rather than leaving the
// extra room felt only as "more space over the fields".
constexpr int kFeaturePanelExtraBelow = 43;

// The pointer shown when a feature is under the cursor: an ordinary arrow with
// a small "i" beside it, as desktop GIS has drawn "identify" for twenty years.
//
// It replaces the open hand deliberately. In this viewer the hand means "drag
// the map", which is what every press does; if the hand also appeared over the
// one place where a press does something else, the cursor would be saying the
// same thing about two different gestures. The arrow says "this one is not a
// drag", and the badge says what it is instead.
QCursor identifyCursor()
{
    static QCursor cached;
    static bool built = false;
    if (built)
        return cached;
    built = true;

    const qreal dpr = qApp ? qApp->devicePixelRatio() : 1.0;
    QPixmap pm(int(32 * dpr), int(32 * dpr));
    pm.setDevicePixelRatio(dpr);
    pm.fill(Qt::transparent);

    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing, true);

    // The classic arrow: black body, white outline, so it survives both a dark
    // raster and a bright one without a drop shadow.
    static const QPointF arrow[] = { { 1.0, 1.0 },   { 1.0, 17.0 },  { 5.2, 13.0 },
                                     { 8.2, 19.4 }, { 10.8, 18.2 }, { 7.8, 12.2 },
                                     { 12.6, 12.2 } };
    QPainterPath body;
    body.addPolygon(QPolygonF(QList<QPointF>(std::begin(arrow), std::end(arrow))));
    body.closeSubpath();
    p.setPen(QPen(Qt::white, 1.6));
    p.setBrush(Qt::black);
    p.drawPath(body);

    // The badge sits clear of the tip, so it never hides what is being pointed
    // at. Fixed colours, not the theme's: a cursor is drawn over the map, not
    // over the interface, and has to read on whatever the data happens to be.
    const QRectF badge(15.5, 15.5, 15.0, 15.0);
    p.setPen(QPen(Qt::white, 1.5));
    p.setBrush(QColor(0x1b, 0x1f, 0x26));
    p.drawEllipse(badge);
    QFont f = p.font();
    f.setPixelSize(11);
    f.setBold(true);
    p.setFont(f);
    p.setPen(Qt::white);
    p.drawText(badge, Qt::AlignCenter, QStringLiteral("i"));
    p.end();

    cached = QCursor(pm, 1, 1);   // hotspot on the tip
    return cached;
}

constexpr int kMaxDisplayDim = 4096;   // decimation cap for the display buffer
constexpr int kHistBins = 256;         // stats/CDF resolution
constexpr int kSliderBins = 128;       // histogram bars drawn in the slider

// Web-mercator world half-extent (EPSG:3857) and basemap tuning.
constexpr double kMercHalf = 20037508.342789244;
constexpr int kMaxTileZoom = 19;
constexpr int kMaxTilesPerUpdate = 160;
const char kTileUrl[] = "https://server.arcgisonline.com/ArcGIS/rest/services/"
                        "World_Imagery/MapServer/tile/%1/%2/%3";
const char kAttribution[] = "Basemap: Esri World Imagery — Source: Esri, Maxar, "
                            "Earthstar Geographics, and the GIS User Community";

// What the three stretches do. A raster holds numbers; a screen holds 256
// shades. The stretch is the rule that maps one onto the other, and changing it
// changes nothing in the data — only which differences you can see.
QString stretchHelpText()
{
    return QObject::tr(
        "<b>Stretch</b><br><br>"
        "The raster's values have to be mapped onto the colour ramp's range. "
        "The stretch is that mapping. It changes the <b>picture only</b>: no "
        "value is altered, nothing is written, and switching between these is "
        "free.<br><br>"
        "<b>Min–Max</b><br>"
        "The lowest value in the layer takes the first colour, the highest "
        "takes the last, and everything in between is spread evenly. Honest "
        "and complete — you are looking at the whole range. Its weakness is "
        "that a single extreme cell decides the scale for all the others: one "
        "FETE cell crossed by 30,000 paths, or one spike of nodata read as a "
        "number, and the rest of the map collapses into the first two or three "
        "shades.<br><br>"
        "<b>Percentile 2–98</b> (the default)<br>"
        "The same mapping, but the ends are cut off first: the value below "
        "which 2% of the cells fall becomes the first colour, the value below "
        "which 98% fall becomes the last. The 2% at each end are not hidden — "
        "they are drawn in the end colours, flattened together. What you gain "
        "is the middle 96%, which now uses the whole ramp instead of a corner "
        "of it. This is the usual choice for a FETE density or a cost surface, "
        "where the interesting structure is well away from the extremes.<br><br>"
        "<b>Logarithmic</b><br>"
        "Maps the logarithm of the value rather than the value. Each step of "
        "colour then means a <i>multiplication</i>, not an addition: 1→10 gets "
        "as much of the ramp as 10→100. Use it when the quantity spans orders "
        "of magnitude — accumulated path counts typically do — and the low end "
        "is where the detail lives. Values at or below zero have no logarithm "
        "and are drawn at the bottom of the ramp.<br><br>"
        "The <b>Filter</b> slider underneath is a different thing entirely: it "
        "hides cells outside a range, while the stretch only recolours them.");
}

QString cvdSafeHelpText()
{
    return QObject::tr(
        "<b>Colour-blind safe ramps</b><br><br>"
        "About 8% of men and 0.5% of women cannot reliably tell red from green. "
        "A rainbow ramp — <i>Turbo</i> is the one here — turns into a muddle for "
        "them, and so does anything that puts green at one end and brown or red "
        "at the other, which is exactly what <i>Terrain</i> does.<br><br>"
        "Tick this and those ramps are greyed out rather than removed, so it is "
        "clear which ones are unavailable and why.<br><br>"
        "<b>What is left, and why each one works.</b><br>"
        "<i>Cividis</i> — designed for this specific purpose: blue to yellow, no "
        "red and no green anywhere, and lightness rising the whole way. Someone "
        "with deuteranomaly sees very nearly what everyone else does. The "
        "safest choice for a published figure.<br>"
        "<i>Viridis</i> and <i>Magma</i> — perceptually uniform ramps built to "
        "survive colour vision deficiency and greyscale printing.<br>"
        "<i>Grayscale</i> — carries no colour information at all, so there is "
        "nothing to confuse.<br>"
        "<i>Heat</i> — black through red and yellow to white: lightness rises "
        "monotonically and green never appears.<br><br>"
        "The test that matters for all of them is the same: if the figure still "
        "reads when printed in black and white, it will read for a colour-blind "
        "reader too.");
}

QString exportScalebarHelpText()
{
    return QObject::tr(
        "<b>Scalebar</b><br><br>"
        "Draws a bar in the bottom-left corner of the exported image, labelled "
        "with the distance it spans on the ground. On by default: an image of a "
        "result without a scale is not evidence of anything, and a reader has "
        "no way to recover it.<br><br>"
        "The length is chosen automatically — the largest round number "
        "(1, 2 or 5 times a power of ten) that fits in about a quarter of the "
        "image — and it is sized for the image, so a 300 dpi export gets a bar "
        "drawn for 300 dpi rather than one drawn for the screen.<br><br>"
        "It needs a raster with a projected CRS. Over a layer in plain degrees "
        "the bar is labelled in degrees, which is honest but rarely useful, "
        "since a degree of longitude is not a fixed distance.");
}

QString exportNorthHelpText()
{
    return QObject::tr(
        "<b>North arrow</b><br><br>"
        "Draws an arrow and an <i>N</i> in the top-right corner of the exported "
        "image. On by default.<br><br>"
        "It always points straight up, and that is correct rather than lazy: "
        "Trajecta never rotates the map, so the top of the view is the top of "
        "the raster. If your raster is itself stored rotated — unusual, but "
        "possible — the arrow follows the raster, not true north.");
}

QString exportResolutionHelpText()
{
    return QObject::tr(
        "<b>Export resolution</b><br><br>"
        "Two ways to say how big the image should be.<br><br>"
        "<b>By print density (dpi).</b> The familiar choice for anything going "
        "into a document: 300 dpi is the usual requirement for a printed "
        "figure, 600 for fine line work, 96 for a screenshot. Trajecta treats "
        "96 dpi as the view at its own resolution, so 300 dpi renders it 3.125 "
        "times larger and stamps the density into the file — the image then "
        "arrives in a page layout at the right physical size instead of "
        "whatever the application guesses. The dialog shows the pixel size you "
        "will actually get.<br><br>"
        "<b>By pixels.</b> When a specific pixel size is what matters — a "
        "journal that asks for a width in pixels, a slide, a web page. No "
        "density is written into the file.<br><br>"
        "<b>What is being enlarged.</b> The scene is re-rendered at the size "
        "you ask for, so vectors, paths and text come out genuinely sharper. "
        "The raster underneath cannot gain detail it does not have: past its "
        "own cell size it only becomes smoother, not more informative.<br><br>"
        "Very large exports need memory in proportion — a 20000 × 20000 image "
        "is 1.6 GB while it is being built. Trajecta refuses anything beyond "
        "400 megapixels rather than failing halfway through.");
}

// Behind the "?" next to the basemap switch. The credit is not decoration: the
// tile service's terms require it wherever the imagery is shown, which is why
// it is also painted into the canvas and into every export.
QString basemapHelpText()
{
    return QObject::tr(
        "<b>Satellite basemap</b><br><br>"
        "Draws satellite imagery behind the raster, so a result can be read "
        "against the real landscape instead of against an empty background. "
        "Lower the raster opacity to see through to it.<br><br>"
        "<b>Off by default</b>, and deliberately: this is the only feature in "
        "Trajecta that contacts the internet. Nothing is sent about your data — "
        "the requests carry only the tile coordinates of the area on screen — "
        "but a program that reaches out to the network without being asked is a "
        "program that surprises people, so it waits for you to switch it "
        "on.<br><br>"
        "<b>Requirements.</b> An internet connection, and a raster that carries "
        "a CRS: without one there is no way to know where on Earth to put the "
        "tiles. Downloaded tiles are cached on disk and reused in later "
        "sessions, so panning over the same area a second time costs "
        "nothing.<br><br>"
        "<b>Imagery credit.</b> Esri World Imagery — Source: Esri, Maxar, "
        "Earthstar Geographics, and the GIS User Community. The service's terms "
        "require that credit wherever the imagery appears, which is why "
        "Trajecta paints it in the corner of the map and keeps it in every "
        "image you export. It is not a watermark you should remove.");
}

// Painted by hand, so they follow the palette through ThemeManager rather
// than through the stylesheet. Functions, not constants: the theme can change
// while the page is alive.
// The card colour, so the canvas matches the panels on every other page.
// Opaque even where the cards are translucent: a map needs a solid backing.
inline QColor canvasBg()   { return ThemeManager::mapped("#1b1f26"); }

// Whether the map is paper over a picture rather than a solid slab. Asked of
// the theme rather than assumed, because it decides two things that would
// otherwise fight each other: the scene's background brush, and the corner
// wedges — both of which paint the canvas colour, and both of which would put
// that colour straight back over what the stylesheet just made translucent.
inline bool translucentCanvas()
{
    return ThemeManager::theme(ThemeManager::current()).translucentCanvas;
}
inline QColor overlayPen() { return ThemeManager::theme(ThemeManager::current()).overlayPen; }
inline QColor scalebarFg() { return ThemeManager::mapped("#e4e7ec"); }
inline QColor scalebarBg() { return ThemeManager::mapped("#14171c"); }
inline QColor hintFg()     { return ThemeManager::mapped("#99a1ac"); }

// A "nice" scalebar length: the largest 1/2/5 x 10^k that fits in `maxUnits`.
// Shared by the on-screen bar and the exported one so the two cannot disagree.
double niceBarLength(double maxUnits)
{
    if (!(maxUnits > 0.0) || !std::isfinite(maxUnits))
        return 0.0;
    const double mag = std::pow(10.0, std::floor(std::log10(maxUnits)));
    for (const double m : {5.0, 2.0, 1.0}) {
        if (mag * m <= maxUnits)
            return mag * m;
    }
    return mag;
}

QString scalebarLabel(double len, bool geographic)
{
    if (geographic)
        return QStringLiteral("%1°").arg(QString::number(len, 'g', 4));
    if (len >= 1000.0)
        return QStringLiteral("%1 km").arg(QString::number(len / 1000.0, 'g', 4));
    return QStringLiteral("%1 m").arg(QString::number(len, 'g', 4));
}

// What an exported image gets painted on top of it. Kept separate from
// MapView::drawForeground, which works in viewport coordinates and is tied to
// the window size: an export is a different canvas at a different scale, and
// the decorations have to be sized for it rather than for the screen.
struct ExportDecorations {
    bool scalebar = true;
    bool northArrow = true;
    QString attribution;      // never optional: the tile terms require it
    double unitsPerPx = 0.0;  // real-world units per pixel of the exported image
    bool geographic = false;
};

void paintExportDecorations(QPainter &p, const QSize &size, const ExportDecorations &d)
{
    // Everything scales with the image, so a 300 dpi export does not come out
    // with a scalebar drawn for a screen.
    const double k = std::max(1.0, std::min(size.width(), size.height()) / 700.0);
    const int pad = int(std::lround(16 * k));
    p.setRenderHint(QPainter::Antialiasing, true);

    if (d.scalebar && d.unitsPerPx > 0.0 && std::isfinite(d.unitsPerPx)) {
        const double len = niceBarLength(d.unitsPerPx * size.width() / 4.0);
        const int barPx = int(std::lround(len / d.unitsPerPx));
        if (len > 0.0 && barPx >= int(20 * k)) {
            const QString label = scalebarLabel(len, d.geographic);
            QFont f = p.font();
            f.setPixelSize(int(std::lround(11 * k)));
            p.setFont(f);
            const QRect textRect = p.fontMetrics().boundingRect(label);
            const int x = pad;
            const int y = size.height() - int(std::lround(20 * k));
            QColor bg = scalebarBg();
            bg.setAlpha(170);
            p.setPen(Qt::NoPen);
            p.setBrush(bg);
            p.drawRoundedRect(QRect(x - int(8 * k), y - textRect.height() - int(12 * k),
                                    std::max(barPx, textRect.width()) + int(16 * k),
                                    textRect.height() + int(22 * k)),
                              6 * k, 6 * k);
            p.setPen(QPen(scalebarFg(), std::max(1.0, 2.0 * k)));
            p.drawLine(x, y, x + barPx, y);
            p.drawLine(x, y - int(4 * k), x, y + int(4 * k));
            p.drawLine(x + barPx, y - int(4 * k), x + barPx, y + int(4 * k));
            p.drawText(x, y - int(8 * k), label);
        }
    }

    if (d.northArrow) {
        // A plain triangle with an N above it, top-right. North is up because
        // the view is north-up: Trajecta never rotates the map, so an arrow
        // that pointed anywhere else would be wrong.
        const double s = 26.0 * k;                 // arrow height
        const double cx = size.width() - pad - s * 0.5;
        const double cy = pad + s * 0.9;
        QPainterPath tri;
        tri.moveTo(cx, cy - s * 0.5);
        tri.lineTo(cx + s * 0.32, cy + s * 0.5);
        tri.lineTo(cx, cy + s * 0.26);
        tri.lineTo(cx - s * 0.32, cy + s * 0.5);
        tri.closeSubpath();

        QColor bg = scalebarBg();
        bg.setAlpha(170);
        p.setPen(Qt::NoPen);
        p.setBrush(bg);
        p.drawRoundedRect(QRectF(cx - s * 0.62, cy - s * 0.5 - 18 * k,
                                 s * 1.24, s * 1.0 + 24 * k),
                          6 * k, 6 * k);

        QFont f = p.font();
        f.setPixelSize(int(std::lround(12 * k)));
        f.setBold(true);
        p.setFont(f);
        p.setPen(scalebarFg());
        p.drawText(QRectF(cx - s * 0.62, cy - s * 0.5 - 17 * k, s * 1.24, 16 * k),
                   Qt::AlignHCenter | Qt::AlignVCenter, QStringLiteral("N"));

        p.setPen(Qt::NoPen);
        p.setBrush(scalebarFg());
        p.drawPath(tri);
    }

    if (!d.attribution.isEmpty()) {
        QFont f = p.font();
        f.setPixelSize(int(std::lround(10 * k)));
        f.setBold(false);
        p.setFont(f);
        const QRect textRect = p.fontMetrics().boundingRect(d.attribution);
        const int x = size.width() - textRect.width() - int(14 * k);
        const int y = size.height() - int(8 * k);
        QColor bg = scalebarBg();
        bg.setAlpha(170);
        p.setPen(Qt::NoPen);
        p.setBrush(bg);
        p.drawRoundedRect(QRect(x - int(6 * k), y - textRect.height() - int(4 * k),
                                textRect.width() + int(12 * k),
                                textRect.height() + int(8 * k)),
                          4 * k, 4 * k);
        p.setPen(hintFg());
        p.drawText(x, y - int(3 * k), d.attribution);
    }
}

inline QColor legendFrame(){ return ThemeManager::mapped("#333a44"); }

// ---------------------------------------------------------------------------
// LayerItemDelegate — draws a bin at the right edge of every row of the layer
// list and turns a click on it into a removal request.
//
// No Q_OBJECT: a std::function keeps this a plain class, so it can live in
// this file's anonymous namespace instead of needing a header and moc.
// ---------------------------------------------------------------------------
constexpr int kBinZoneWidth = 30;   // room reserved at the right of each row

void paintBin(QPainter *painter, const QRectF &box, const QColor &color)
{
    const qreal w = box.width();
    const qreal h = box.height();
    QPen pen(color, qMax(1.0, w * 0.09));
    pen.setCapStyle(Qt::RoundCap);
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);

    // Lid, with a handle above it.
    const qreal lidY = box.top() + h * 0.22;
    painter->drawLine(QPointF(box.left(), lidY), QPointF(box.right(), lidY));
    painter->drawLine(QPointF(box.left() + w * 0.36, box.top() + h * 0.08),
                      QPointF(box.left() + w * 0.64, box.top() + h * 0.08));
    // Body: tapered, with two slots.
    QPainterPath body;
    body.moveTo(box.left() + w * 0.14, lidY + h * 0.06);
    body.lineTo(box.left() + w * 0.22, box.bottom());
    body.lineTo(box.right() - w * 0.22, box.bottom());
    body.lineTo(box.right() - w * 0.14, lidY + h * 0.06);
    painter->drawPath(body);
    painter->drawLine(QPointF(box.left() + w * 0.38, lidY + h * 0.22),
                      QPointF(box.left() + w * 0.40, box.bottom() - h * 0.12));
    painter->drawLine(QPointF(box.right() - w * 0.38, lidY + h * 0.22),
                      QPointF(box.right() - w * 0.40, box.bottom() - h * 0.12));
}



// ---------------------------------------------------------------------------
// Color scales: anchor points interpolated to 256-entry LUTs.
// ---------------------------------------------------------------------------
struct ColorAnchor {
    double t;
    int r, g, b;
};

QVector<QRgb> makeLut(std::initializer_list<ColorAnchor> anchors)
{
    QVector<QRgb> lut(256);
    const std::vector<ColorAnchor> a(anchors);
    for (int i = 0; i < 256; ++i) {
        const double t = i / 255.0;
        size_t hi = 1;
        while (hi + 1 < a.size() && a[hi].t < t)
            ++hi;
        const ColorAnchor &c0 = a[hi - 1];
        const ColorAnchor &c1 = a[hi];
        const double f = (c1.t > c0.t)
            ? std::clamp((t - c0.t) / (c1.t - c0.t), 0.0, 1.0) : 0.0;
        lut[i] = qRgb(int(std::lround(c0.r + f * (c1.r - c0.r))),
                      int(std::lround(c0.g + f * (c1.g - c0.g))),
                      int(std::lround(c0.b + f * (c1.b - c0.b))));
    }
    return lut;
}

struct Colormap {
    const char *name;
    QVector<QRgb> lut;
    // Whether the ramp stays readable to someone with a colour vision
    // deficiency. What matters is that lightness rises monotonically along the
    // ramp and that no two ends rely on a red/green distinction, which is the
    // one about 8% of men cannot make.
    bool cvdSafe = true;
};

const QVector<Colormap> &colormaps()
{
    static const QVector<Colormap> maps = {
        {"Grayscale", makeLut({{0.0, 8, 10, 14}, {1.0, 245, 246, 248}})},
        {"Viridis", makeLut({{0.0, 68, 1, 84}, {0.125, 72, 40, 120},
                             {0.25, 62, 74, 137}, {0.375, 49, 104, 142},
                             {0.5, 38, 130, 142}, {0.625, 31, 158, 137},
                             {0.75, 53, 183, 121}, {0.875, 109, 205, 89},
                             {1.0, 253, 231, 37}})},
        {"Magma", makeLut({{0.0, 0, 0, 4}, {0.125, 28, 16, 68},
                           {0.25, 79, 18, 123}, {0.375, 129, 37, 129},
                           {0.5, 181, 54, 122}, {0.625, 229, 80, 100},
                           {0.75, 251, 135, 97}, {0.875, 254, 194, 135},
                           {1.0, 252, 253, 191}})},
        // Designed at the Berlin Institute of Technology specifically so that
        // someone with deuteranomaly sees very nearly what everyone else does.
        // Blue to yellow only: no red, no green, and lightness rises the whole
        // way. The safest choice for a figure that has to work in print, in
        // greyscale and for any reader.
        {"Cividis", makeLut({{0.0, 0, 32, 76}, {0.125, 0, 54, 111},
                             {0.25, 42, 74, 118}, {0.375, 71, 93, 122},
                             {0.5, 99, 112, 125}, {0.625, 129, 132, 123},
                             {0.75, 162, 154, 116}, {0.875, 197, 178, 102},
                             {1.0, 253, 231, 55}})},
        {"Turbo", makeLut({{0.0, 48, 18, 59}, {0.125, 70, 107, 227},
                           {0.25, 40, 187, 235}, {0.375, 32, 229, 181},
                           {0.5, 122, 252, 82}, {0.625, 218, 227, 25},
                           {0.75, 253, 154, 42}, {0.875, 224, 62, 7},
                           {1.0, 122, 4, 3}}), false},
        {"Terrain", makeLut({{0.0, 27, 120, 55}, {0.35, 166, 217, 106},
                             {0.55, 254, 224, 139}, {0.75, 166, 97, 26},
                             {1.0, 247, 247, 247}}), false},
        {"Heat", makeLut({{0.0, 0, 0, 0}, {0.4, 179, 0, 0},
                          {0.75, 255, 204, 0}, {1.0, 255, 255, 255}})},
    };
    return maps;
}

QString formatValue(double v)
{
    return QString::number(v, 'g', 5);
}

} // namespace

// ---------------------------------------------------------------------------
// Layer data
// ---------------------------------------------------------------------------
struct RasterLayer {
    QString label;
    QString path;
    bool loaded = false;
    bool failed = false;

    GDALDatasetH ds = nullptr;   // kept open for exact 1x1 hover reads
    int srcW = 0, srcH = 0;
    int dispW = 0, dispH = 0;
    double gt[6] = {0, 1, 0, 0, 0, -1};
    bool hasNoData = false;
    double noData = 0.0;

    QVector<float> data;         // decimated display buffer, dispW * dispH
    double minV = 0.0, maxV = 0.0;
    double p2 = 0.0, p98 = 0.0;
    QVector<float> sortedSample; // sorted sample of valid cells, for quantiles
    QVector<float> sliderHist;   // kSliderBins normalized bar heights

    QString crsName;
    QString wkt;                 // raw CRS WKT, for basemap reprojection
    bool geographic = false;

    ~RasterLayer()
    {
        if (ds && GdalApi::instance().isLoaded())
            GdalApi::instance().Close(ds);
    }
};

struct VectorOverlay {
    QString label;
    QString path;
    bool loaded = false;
    bool failed = false;
    // Geometry as it was read, in the layer's own CRS. Rings of a polygon are
    // kept as closed lines: the Viewer draws outlines, never fills, so a
    // boundary and a closed path are the same picture.
    QVector<QPolygonF> lines;
    QVector<QPointF> points;
    QString wkt;                 // the layer's CRS, empty when it declares none
    QString crsName;             // readable form, for the info strip
    bool geographic = false;     // degrees rather than metres: no scalebar
    QRectF extent;               // bounding box in the layer's own CRS
    bool hasExtent = false;      // a box around one point is legitimately empty

    // The same geometry moved into the CRS of whatever raster it is drawn over,
    // rebuilt when that raster changes. Equal to the source when the two match,
    // when either lacks a CRS, or when no transform could be built.
    QVector<QPolygonF> drawLines;
    QVector<QPointF> drawPoints;
    QRectF drawExtent;
    bool hasDrawExtent = false;
    QString drawWkt;             // CRS the fields above are in
    bool drawReady = false;

    // What the user chose for this layer, invalid until they choose. Kept per
    // overlay rather than per position, because the automatic colours come from
    // the position in the list and removing a layer shifts every one after it:
    // a choice tied to the position would silently move to another layer.
    QColor customColour;
    // 100 = the layer's ordinary marker and line weight; the slider under the
    // colour wheel moves this. Kept per overlay for the same reason as the
    // colour above.
    int sizePercent = 100;

    // --- attributes, for the info panel and for colouring ---
    //
    // One entry per geometry, not per feature: a multipoint contributes several
    // points and each of them has to be able to answer for itself when it is
    // clicked. The values are text because that is what they are shown as, and
    // because a column of mixed types is a real thing in a shapefile.
    QStringList fieldNames;
    QVector<QStringList> pointAttrs;
    QVector<QStringList> lineAttrs;

    // The same geometry in scene coordinates, kept so a click can be answered
    // without recomputing the projection for every point on every press.
    QVector<QPointF> scenePoints;
    QVector<QPolygonF> sceneLines;

    // Colouring by a numeric column. -1 means the layer is drawn in its own
    // single colour, which is what every layer did before and what most still
    // do: this switches on only when the column is one this application wrote.
    int colourField = -1;
    bool colourDiverging = false;   // centred on 50, for a score whose null is 50
    double colourLo = 0.0, colourHi = 100.0;
    QVector<double> pointValues;    // parsed colourField, NaN where unreadable
};

namespace {

// Min/max in both directions, kept as plain numbers. Everything here is a
// bounding box over points, and QRectF cannot do that job: a box around one
// point is "null" to Qt, and united() drops null boxes on the floor.
struct BoundsAccumulator {
    double minX = 0, minY = 0, maxX = 0, maxY = 0;
    bool valid = false;

    void add(const QPointF &p)
    {
        if (!valid) {
            minX = maxX = p.x();
            minY = maxY = p.y();
            valid = true;
            return;
        }
        minX = std::min(minX, p.x());
        maxX = std::max(maxX, p.x());
        minY = std::min(minY, p.y());
        maxY = std::max(maxY, p.y());
    }
    void add(const BoundsAccumulator &other)
    {
        if (!other.valid)
            return;
        add(QPointF(other.minX, other.minY));
        add(QPointF(other.maxX, other.maxY));
    }
    QRectF rect() const { return QRectF(minX, minY, maxX - minX, maxY - minY); }
};

// Open options that make GDAL's CSV driver produce geometry. Without them a
// table of coordinates opens as a layer of attributes and nothing at all is
// drawn — the commonest way a plain-text import silently "works" and shows an
// empty map. Passed only for the extensions the driver claims, since other
// drivers warn about open options they do not know.
QList<QByteArray> textOpenOptions(const QString &path)
{
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (suffix != QLatin1String("csv") && suffix != QLatin1String("txt")
        && suffix != QLatin1String("tsv")) {
        return {};
    }
    return {
        QByteArrayLiteral("X_POSSIBLE_NAMES=x,lon,lng,long,longitude,easting,east,xcoord,x_coord"),
        QByteArrayLiteral("Y_POSSIBLE_NAMES=y,lat,latitude,northing,north,ycoord,y_coord"),
        QByteArrayLiteral("GEOM_POSSIBLE_NAMES=geom,geometry,wkt,the_geom"),
        QByteArrayLiteral("KEEP_GEOM_COLUMNS=NO"),
        QByteArrayLiteral("AUTODETECT_TYPE=YES"),
    };
}

// Opens a vector dataset the way the Viewer wants it: content decides, not the
// extension, and a text table gets the hints it needs to yield coordinates.
GDALDatasetH openVectorDataset(const QString &path)
{
    GdalApi &api = GdalApi::instance();
    const QByteArray pathUtf8 = path.toUtf8();
    const QList<QByteArray> options = textOpenOptions(path);
    if (options.isEmpty())
        return api.OpenEx(pathUtf8.constData(), GdalApi::OF_Vector,
                          nullptr, nullptr, nullptr);

    QVector<const char *> argv;
    for (const QByteArray &opt : options)
        argv.append(opt.constData());
    argv.append(nullptr);
    GDALDatasetH ds = api.OpenEx(pathUtf8.constData(), GdalApi::OF_Vector,
                                 nullptr, argv.constData(), nullptr);
    // A CSV that already carries its own geometry column, or one the hints do
    // not fit, still deserves a plain attempt rather than an error.
    if (!ds)
        ds = api.OpenEx(pathUtf8.constData(), GdalApi::OF_Vector,
                        nullptr, nullptr, nullptr);
    return ds;
}

// Reads every geometry of every layer into `overlay`, in the file's own CRS.
// Returns false when the file cannot be opened or holds nothing drawable.
bool loadOverlayGeometry(VectorOverlay &overlay)
{
    GdalApi &api = GdalApi::instance();
    GDALDatasetH ds = openVectorDataset(overlay.path);
    if (!ds)
        return false;

    const auto collectLine = [&overlay, &api](OGRGeometryH geom) {
        const int n = api.G_GetPointCount(geom);
        if (n < 2)
            return;
        QPolygonF line;
        line.reserve(n);
        for (int i = 0; i < n; ++i) {
            double x = 0, y = 0, z = 0;
            api.G_GetPoint(geom, i, &x, &y, &z);
            line.append(QPointF(x, y));
        }
        overlay.lines.append(line);
    };
    const auto collectPoint = [&overlay, &api](OGRGeometryH geom) {
        if (api.G_GetPointCount(geom) < 1)
            return;
        double x = 0, y = 0, z = 0;
        api.G_GetPoint(geom, 0, &x, &y, &z);
        overlay.points.append(QPointF(x, y));
    };

    // Recursive: a multipolygon holds polygons, a polygon holds rings, and a
    // geometry collection holds anything at all, including further
    // collections. Depth-limited because the structure comes from a file.
    std::function<void(OGRGeometryH, int)> collect =
        [&](OGRGeometryH geom, int depth) {
            if (!geom || depth > 8)
                return;
            const int type = GdalApi::flattenGeomType(api.G_GetGeometryType(geom));
            switch (type) {
            case GdalApi::WkbPoint:
                collectPoint(geom);
                break;
            case GdalApi::WkbLineString:
            case GdalApi::WkbLinearRing:
                collectLine(geom);
                break;
            case GdalApi::WkbPolygon:
            case GdalApi::WkbMultiPoint:
            case GdalApi::WkbMultiLineString:
            case GdalApi::WkbMultiPolygon:
            case GdalApi::WkbGeometryCollection: {
                const int parts = api.G_GetGeometryCount(geom);
                for (int p = 0; p < parts; ++p)
                    collect(api.G_GetGeometryRef(geom, p), depth + 1);
                break;
            }
            default:
                // Curves and surfaces (CircularString, CompoundCurve…). Their
                // parts are still reachable, so recursing is better than
                // dropping the feature.
                if (api.G_GetGeometryCount(geom) > 0) {
                    const int parts = api.G_GetGeometryCount(geom);
                    for (int p = 0; p < parts; ++p)
                        collect(api.G_GetGeometryRef(geom, p), depth + 1);
                } else if (api.G_GetPointCount(geom) > 1) {
                    collectLine(geom);
                }
                break;
            }
        };

    const int layerCount = api.DatasetGetLayerCount(ds);
    for (int li = 0; li < layerCount; ++li) {
        OGRLayerH vl = api.DatasetGetLayer(ds, li);
        if (!vl)
            continue;

        // The CRS of the first layer that declares one. A file mixing CRSs
        // between its layers is pathological; the alternative is refusing it.
        if (overlay.wkt.isEmpty() && api.L_GetSpatialRef && api.OSRExportToWkt) {
            if (OGRSpatialReferenceH srs = api.L_GetSpatialRef(vl)) {
                char *wkt = nullptr;
                if (api.OSRExportToWkt(srs, &wkt) == 0 && wkt && *wkt) {
                    overlay.wkt = QString::fromUtf8(wkt);
                    const char *name = api.OSRGetName(srs);
                    const char *authName = api.OSRGetAuthorityName(srs, nullptr);
                    const char *authCode = api.OSRGetAuthorityCode(srs, nullptr);
                    overlay.crsName = name ? QString::fromUtf8(name) : QString();
                    if (authName && authCode) {
                        overlay.crsName += QStringLiteral(" (%1:%2)")
                                               .arg(QString::fromUtf8(authName),
                                                    QString::fromUtf8(authCode));
                    }
                    overlay.geographic = api.OSRIsGeographic(srs) != 0;
                }
                if (wkt && api.VSIFree)
                    api.VSIFree(wkt);
            }
        }

        // Column names, from the first layer that has any. Read once: they
        // belong to the layer, not to the feature, and asking per feature was
        // measurably slower on a layer of a hundred thousand points.
        const bool fields = api.canReadFields();

        api.L_ResetReading(vl);
        while (OGRFeatureH f = api.L_GetNextFeature(vl)) {
            const int pointsBefore = overlay.points.size();
            const int linesBefore = overlay.lines.size();
            collect(api.F_GetGeometryRef(f), 0);

            // The feature's attributes are attached to every geometry it
            // produced, so a multipoint's parts all answer with their parent's
            // values. Padding rather than appending once keeps the two vectors
            // the same length as the geometry vectors, which is what the click
            // handler relies on.
            QStringList values;
            if (fields) {
                const int n = api.F_GetFieldCount(f);
                for (int i = 0; i < n; ++i) {
                    if (overlay.fieldNames.size() < n) {
                        if (OGRFieldDefnH fd = api.F_GetFieldDefnRef(f, i))
                            overlay.fieldNames << QString::fromUtf8(api.Fld_GetNameRef(fd));
                    }
                    values << QString::fromUtf8(api.F_GetFieldAsString(f, i));
                }
            }
            while (overlay.pointAttrs.size() < overlay.points.size())
                overlay.pointAttrs << (overlay.pointAttrs.size() >= pointsBefore
                                           ? values : QStringList());
            while (overlay.lineAttrs.size() < overlay.lines.size())
                overlay.lineAttrs << (overlay.lineAttrs.size() >= linesBefore
                                          ? values : QStringList());
            api.F_Destroy(f);
        }
    }
    api.Close(ds);

    // Colouring, decided here because it is a property of what was read.
    //
    // Only the two columns this application writes itself are recognised. A
    // guess at "the first numeric column" would colour a layer of site numbers
    // by site number, which looks meaningful and is not.
    // "score_prox" and "score_int" are what the coherence tool called its two
    // scores before the measures were rebuilt. Still recognised, so a layer
    // saved by an earlier version keeps the colouring it was read with instead
    // of silently falling through to something else.
    bool followLayer = false;
    for (int i = 0; i < overlay.fieldNames.size(); ++i) {
        const QString name = overlay.fieldNames.at(i).toLower();
        if (name == QLatin1String("prox_idx")) {
            overlay.colourField = i;
            overlay.colourDiverging = false;
            followLayer = true;
            break;
        }
        if (name == QLatin1String("score_prox")) {          // legacy
            overlay.colourField = i;
            overlay.colourDiverging = false;
            followLayer = false;
            break;
        }
        if ((name == QLatin1String("inten_idx") || name == QLatin1String("score_int"))
            && overlay.colourField < 0) {
            overlay.colourField = i;
            // 50 is what the average location scores, so the ramp is built
            // around it: above and below the null read as two different
            // things, and one continuous ramp would hide the boundary.
            overlay.colourDiverging = true;
        }
    }
    if (overlay.colourField >= 0) {
        overlay.pointValues.reserve(overlay.pointAttrs.size());
        for (const QStringList &attrs : std::as_const(overlay.pointAttrs)) {
            bool ok = false;
            const double v = overlay.colourField < attrs.size()
                                 ? attrs.at(overlay.colourField).toDouble(&ok)
                                 : 0.0;
            overlay.pointValues << (ok ? v : std::numeric_limits<double>::quiet_NaN());
        }
        overlay.colourLo = 0.0;
        overlay.colourHi = 100.0;   // the percentile scores are defined on 0-100
        if (followLayer) {
            // The proximity index is not. It is the share of the neighbourhood
            // that is corridor, and a corridor is a thin thing: on a real
            // surface the whole layer often sits under five per cent. Held to a
            // fixed 0-100 every site would come out the same colour, so the top
            // of the ramp follows the layer instead of the definition.
            double top = 0.0;
            for (double v : std::as_const(overlay.pointValues)) {
                if (!qIsNaN(v))
                    top = std::max(top, v);
            }
            overlay.colourHi = top > 0.0 ? top : 100.0;
        }
    }

    // Extent in the source CRS, used to frame a vector shown on its own.
    // Accumulated as four numbers rather than with QRectF::united(): a
    // rectangle around a single point has no width and no height, which is
    // what QRectF calls null, and united() answers a null rectangle by
    // ignoring it — so a layer of points would never grow a box at all.
    BoundsAccumulator bounds;
    for (const QPolygonF &line : std::as_const(overlay.lines)) {
        for (const QPointF &p : line)
            bounds.add(p);
    }
    for (const QPointF &p : std::as_const(overlay.points))
        bounds.add(p);
    overlay.extent = bounds.rect();
    overlay.hasExtent = bounds.valid;

    return !overlay.lines.isEmpty() || !overlay.points.isEmpty();
}

// Moves an overlay's geometry into `targetWkt`, caching the result: switching
// between two rasters in different systems must not re-project on every
// repaint. Falls back to the source coordinates whenever a transform cannot be
// built — which is the right answer when the two CRSs already agree, and the
// only possible one when PROJ is unhappy.
void projectOverlay(VectorOverlay &overlay, const QString &targetWkt)
{
    if (overlay.drawReady && overlay.drawWkt == targetWkt)
        return;

    const auto useSource = [&overlay, &targetWkt] {
        overlay.drawLines = overlay.lines;
        overlay.drawPoints = overlay.points;
        overlay.drawExtent = overlay.extent;
        overlay.hasDrawExtent = overlay.hasExtent;
        overlay.drawWkt = targetWkt;
        overlay.drawReady = true;
    };

    GdalApi &api = GdalApi::instance();
    if (targetWkt.isEmpty() || overlay.wkt.isEmpty() || overlay.wkt == targetWkt
        || !api.isLoaded()) {
        useSource();
        return;
    }

    const QByteArray srcWkt = overlay.wkt.toUtf8();
    const QByteArray dstWkt = targetWkt.toUtf8();
    OGRSpatialReferenceH src = api.OSRNewSpatialReference(srcWkt.constData());
    OGRSpatialReferenceH dst = api.OSRNewSpatialReference(dstWkt.constData());
    OGRCoordinateTransformationH ct = nullptr;
    if (src && dst) {
        // Traditional x/y order, as everywhere else here: GDAL 3 would
        // otherwise hand back lat/lon for some authority definitions.
        api.OSRSetAxisMappingStrategy(src, 0);
        api.OSRSetAxisMappingStrategy(dst, 0);
        ct = api.OCTNewCoordinateTransformation(src, dst);
    }
    if (src)
        api.OSRDestroySpatialReference(src);
    if (dst)
        api.OSRDestroySpatialReference(dst);
    if (!ct) {
        useSource();
        return;
    }

    BoundsAccumulator bounds;
    // One OCTTransform call per polyline rather than per vertex: PROJ sets up
    // its pipeline once per call, and a route can carry tens of thousands of
    // vertices.
    //
    // The return value is deliberately ignored. OCTTransform answers "did every
    // point convert", and it says no as soon as one of them falls outside the
    // area the transformation is defined for — while still converting all the
    // others and marking only the failures with HUGE_VAL. Treating that as a
    // failed call would throw away a whole layer because one point of five
    // thousand sits off the edge of the projection. The check that matters is
    // per point, below.
    const auto transform = [&api, ct](QVector<double> &xs, QVector<double> &ys) {
        if (!xs.isEmpty())
            api.OCTTransform(ct, xs.size(), xs.data(), ys.data(), nullptr);
    };

    overlay.drawLines.clear();
    overlay.drawLines.reserve(overlay.lines.size());
    QVector<double> xs, ys;
    for (const QPolygonF &line : std::as_const(overlay.lines)) {
        xs.resize(line.size());
        ys.resize(line.size());
        for (int i = 0; i < line.size(); ++i) {
            xs[i] = line[i].x();
            ys[i] = line[i].y();
        }
        transform(xs, ys);
        QPolygonF out;
        out.reserve(line.size());
        for (int i = 0; i < line.size(); ++i) {
            // PROJ marks a point it could not convert with HUGE_VAL rather
            // than failing the whole call.
            if (!std::isfinite(xs[i]) || !std::isfinite(ys[i]))
                continue;
            out.append(QPointF(xs[i], ys[i]));
            bounds.add(out.last());
        }
        if (out.size() >= 2)
            overlay.drawLines.append(out);
    }

    overlay.drawPoints.clear();
    if (!overlay.points.isEmpty()) {
        xs.resize(overlay.points.size());
        ys.resize(overlay.points.size());
        for (int i = 0; i < overlay.points.size(); ++i) {
            xs[i] = overlay.points[i].x();
            ys[i] = overlay.points[i].y();
        }
        transform(xs, ys);
        overlay.drawPoints.reserve(overlay.points.size());
        for (int i = 0; i < overlay.points.size(); ++i) {
            if (!std::isfinite(xs[i]) || !std::isfinite(ys[i]))
                continue;
            overlay.drawPoints.append(QPointF(xs[i], ys[i]));
            bounds.add(overlay.drawPoints.last());
        }
    }

    api.OCTDestroyCoordinateTransformation(ct);
    overlay.drawExtent = bounds.rect();
    overlay.hasDrawExtent = bounds.valid;
    overlay.drawWkt = targetWkt;
    overlay.drawReady = true;
}

} // namespace

// Paints the rows and nothing else. The click on the bin is picked up by an
// event filter on the popup's viewport instead of by editorEvent(): the combo's
// own container filters mouse releases on that viewport first, closes the popup
// and selects the row, so a delegate never sees the event at all.
class LayerItemDelegate : public QStyledItemDelegate
{
public:
    using QStyledItemDelegate::QStyledItemDelegate;

    void setHoveredRow(int row) { m_hoverRow = row; }

    static QRect binRect(const QRect &itemRect)
    {
        const int side = qMin(16, itemRect.height() - 6);
        QRect r(0, 0, side, side);
        r.moveCenter(QPoint(itemRect.right() - kBinZoneWidth / 2,
                            itemRect.center().y()));
        return r;
    }

    QSize sizeHint(const QStyleOptionViewItem &option,
                   const QModelIndex &index) const override
    {
        QSize s = QStyledItemDelegate::sizeHint(option, index);
        s.setWidth(s.width() + kBinZoneWidth);
        s.setHeight(qMax(s.height(), 26));
        return s;
    }

    void paint(QPainter *painter, const QStyleOptionViewItem &option,
               const QModelIndex &index) const override
    {
        // Keep the label clear of the bin.
        QStyleOptionViewItem text(option);
        text.rect.adjust(0, 0, -kBinZoneWidth, 0);
        QStyledItemDelegate::paint(painter, text, index);

        painter->save();
        painter->setRenderHint(QPainter::Antialiasing, true);
        QColor color = hintFg();
        color.setAlpha(m_hoverRow == index.row() ? 255 : 150);
        paintBin(painter, QRectF(binRect(option.rect)), color);
        painter->restore();
    }

private:
    int m_hoverRow = -1;
};

// ---------------------------------------------------------------------------
// MapView
// ---------------------------------------------------------------------------
MapView::MapView(QWidget *parent)
    : QGraphicsView(parent)
{
    setObjectName(QStringLiteral("ViewerCanvas"));
    setFrameShape(QFrame::NoFrame);
    // NoDrag, and the panning is done by hand below: ScrollHandDrag takes the
    // left button only and refuses to move past the scene rect.
    setDragMode(QGraphicsView::NoDrag);
    viewport()->setCursor(Qt::OpenHandCursor);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    applyCanvasBackground();
    setRenderHint(QPainter::SmoothPixmapTransform, true);
    viewport()->setMouseTracking(true);

    // The canvas must not accept drops, and this is not a detail.
    //
    // QGraphicsView switches drops on in its constructor so that a scene can
    // hand them to its items, and QAbstractScrollArea passes that on to the
    // viewport. We have no droppable items: every drop in this page belongs to
    // ViewerPage, which reads the file and adds a layer.
    //
    // Qt offers a drag to the deepest widget under the cursor that accepts
    // drops, walking up the parents until it finds one — and it stops at the
    // first, whether or not that widget does anything with it. So the viewport
    // silently swallowed every drag that landed on the map. It went unnoticed
    // because it only bites once there IS a map: on an empty Viewer the canvas
    // shows the placeholder label instead, which accepts nothing, so the walk
    // continued up to ViewerPage and the drop worked. That is why exactly the
    // first drop after each start behaved, and every one after it did nothing.
    setAcceptDrops(false);
    viewport()->setAcceptDrops(false);
}

void MapView::setUnitsPerScenePixel(double units, bool geographicCrs)
{
    m_unitsPerScenePx = units;
    m_geographic = geographicCrs;
    viewport()->update();
}

void MapView::fitAll()
{
    if (scene() && !scene()->sceneRect().isEmpty()) {
        // Cleared first: a pan-widened rect left over from the previous layer
        // would make "fit" leave the data as a small square in the middle.
        setSceneRect(QRectF());
        fitInView(scene()->sceneRect(), Qt::KeepAspectRatio);
        setRenderHint(QPainter::SmoothPixmapTransform, transform().m11() < 1.0);
        updatePanBounds();
    }
}

QPointF MapView::centreInScene() const
{
    return mapToScene(viewport()->rect().center());
}

// The counterpart of fitAll(): the same housekeeping, a different answer to
// "where should the view be looking". The view's own scene rect is cleared
// first for the same reason it is there — a pan-widened rect left over from
// the previous layer would otherwise clamp the centring to the wrong place.
void MapView::showAt(const QPointF &centreScene, double viewScale)
{
    if (!scene() || scene()->sceneRect().isEmpty())
        return;
    // The same range the wheel is held to. Two rasters of very different
    // resolutions can ask for a magnification outside it, and an unclamped
    // transform is how a view ends up blank and impossible to recover from.
    const double s = qBound(1e-4, viewScale, 1e4);
    setSceneRect(QRectF());
    setTransform(QTransform::fromScale(s, s));
    centerOn(centreScene);
    setRenderHint(QPainter::SmoothPixmapTransform, s < 1.0);
    updatePanBounds();
}

void MapView::updatePanBounds()
{
    if (!scene())
        return;
    // The scene's own rect, not the view's: the view's is the widened one this
    // function produces, and feeding it back would grow without bound.
    const QRectF base = scene()->sceneRect();
    if (base.isEmpty()) {
        setSceneRect(QRectF());
        return;
    }
    setSceneRect(QRectF());
    const QRectF visible = mapToScene(viewport()->rect()).boundingRect();
    const qreal mx = visible.width() * 0.9;
    const qreal my = visible.height() * 0.9;
    setSceneRect(base.adjusted(-mx, -my, mx, my));
}

void MapView::setAttribution(const QString &text)
{
    m_attribution = text;
    viewport()->update();
}

void MapView::wheelEvent(QWheelEvent *event)
{
    const double factor = event->angleDelta().y() > 0 ? 1.25 : 0.8;
    const double next = transform().m11() * factor;
    if (next > 1e-4 && next < 1e4) {
        scale(factor, factor);
        // Crisp cells when zoomed past 1:1, smooth decimation otherwise.
        setRenderHint(QPainter::SmoothPixmapTransform, transform().m11() < 1.0);
        // The free-pan margin is measured in scene units, so it has to be
        // recomputed whenever the scale changes.
        updatePanBounds();
        emit viewChanged();
    }
    event->accept();
}

void MapView::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    updatePanBounds();
    emit viewChanged();
}

void MapView::scrollContentsBy(int dx, int dy)
{
    QGraphicsView::scrollContentsBy(dx, dy);
    emit viewChanged();
}

void MapView::mousePressEvent(QMouseEvent *event)
{
    // Either button drags the map: left is what everyone reaches for, middle is
    // what people used to desktop GIS reach for.
    if (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton) {
        m_panning = true;
        m_panAnchor = event->position().toPoint();
        m_pressAt = m_panAnchor;
        viewport()->setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}

void MapView::mouseMoveEvent(QMouseEvent *event)
{
    emit hoverScenePos(mapToScene(event->position().toPoint()));
    if (m_panning) {
        const QPoint here = event->position().toPoint();
        const QPoint delta = here - m_panAnchor;
        m_panAnchor = here;
        // Scrollbar values, not translate(): the transform also carries the
        // zoom, and moving it directly would drift the anchor under the cursor.
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
        verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
        event->accept();
        return;
    }
    QGraphicsView::mouseMoveEvent(event);
}

void MapView::mouseReleaseEvent(QMouseEvent *event)
{
    if (m_panning
        && (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton)) {
        m_panning = false;
        viewport()->setCursor(Qt::OpenHandCursor);
        // Four pixels of slack: a hand on a mouse moves a little between
        // pressing and letting go, and asking for none would make the panel
        // impossible to open on a trackpad.
        const QPoint moved = event->position().toPoint() - m_pressAt;
        if (event->button() == Qt::LeftButton && moved.manhattanLength() <= 4)
            emit clicked(mapToScene(event->position().toPoint()));
        event->accept();
        return;
    }
    QGraphicsView::mouseReleaseEvent(event);
}

void MapView::leaveEvent(QEvent *event)
{
    emit hoverLeft();
    QGraphicsView::leaveEvent(event);
}

// The scene's backing, and whether the viewport paints one at all. On a theme
// that wants the picture to read through the map, both are dropped: the scene
// gets no brush and the viewport stops filling itself, so what shows in the
// empty parts of the canvas is the holder's translucent paper and, through it,
// the artwork. Everywhere else this is the plain opaque card colour it always
// was, which is also the fast path — a viewport that paints its own background
// never sends a repaint up to the window.
void MapView::applyCanvasBackground()
{
    const bool seeThrough = translucentCanvas();
    setBackgroundBrush(seeThrough ? QBrush(Qt::NoBrush) : QBrush(canvasBg()));
    viewport()->setAutoFillBackground(!seeThrough);
    viewport()->setAttribute(Qt::WA_OpaquePaintEvent, !seeThrough);
    m_cornerSize = QSize();   // the wedges are rebuilt on the next paint
}

void MapView::drawForeground(QPainter *painter, const QRectF &)
{
    painter->save();
    painter->resetTransform();

    // Rounded corners for the map. Painted, not masked: masking the viewport
    // makes it non-opaque, and every repaint then has to go up to the window
    // background first — which on a themed background is a full-size image
    // rescale, and enough of them in a row locks the window up. Four little
    // wedges in the holder's own colour cost nothing and antialias properly.
    if (m_cornerSize != viewport()->size()) {
        m_cornerSize = viewport()->size();
        const QRectF r(viewport()->rect());
        constexpr qreal kRadius = 11.0;
        QPainterPath full;
        full.addRect(r);
        QPainterPath rounded;
        // The top two corners only. The panel does not end with the map any
        // more — the information row continues below it — and a radius in the
        // middle of a panel reads as a notch cut out of it. Pushing the rounded
        // rectangle past the bottom edge leaves those corners square.
        rounded.addRoundedRect(r.adjusted(0, 0, 0, kRadius), kRadius, kRadius);
        m_cornerPath = full.subtracted(rounded);
    }
    // Not on a theme whose canvas is meant to show what is behind it: the wedge
    // would paint the paper back into the very corner the stylesheet rounded.
    if (!translucentCanvas()) {
        painter->setRenderHint(QPainter::Antialiasing, true);
        painter->fillPath(m_cornerPath, canvasBg());
    }


    // --- Scalebar, bottom-left ---
    const double unitsPerViewPx =
        m_unitsPerScenePx > 0.0 ? m_unitsPerScenePx / transform().m11() : 0.0;
    if (unitsPerViewPx > 0.0 && std::isfinite(unitsPerViewPx)) {
        // Nice 1/2/5 x 10^k length no wider than ~1/6 of the viewport, and in
        // any case stopping short of the feature panel in the opposite corner.
        // A scalebar is read by its label, not by its length, so shortening it
        // costs nothing — while running into the panel cost the panel.
        double availPx = viewport()->width() / 6.0;
        const double clearOfPanel =
            viewport()->width() - double(kFeaturePanelWidth) - 90.0;
        if (clearOfPanel > 40.0)
            availPx = std::min(availPx, clearOfPanel);
        const double targetUnits = unitsPerViewPx * availPx;
        const double mag = std::pow(10.0, std::floor(std::log10(targetUnits)));
        double len = mag;
        for (const double m : {5.0, 2.0, 1.0}) {
            if (mag * m <= targetUnits) {
                len = mag * m;
                break;
            }
        }
        const int barPx = int(std::lround(len / unitsPerViewPx));
        if (barPx >= 20) {
            QString label;
            if (m_geographic)
                label = QStringLiteral("%1°").arg(QString::number(len, 'g', 4));
            else if (len >= 1000.0)
                label = QStringLiteral("%1 km")
                            .arg(QString::number(len / 1000.0, 'g', 4));
            else
                label = QStringLiteral("%1 m").arg(QString::number(len, 'g', 4));

            const int x = 16;
            const int y = viewport()->height() - 20;
            QFont f = painter->font();
            f.setPixelSize(11);
            painter->setFont(f);
            const QRect textRect = painter->fontMetrics().boundingRect(label);

            QColor bg = scalebarBg();
            bg.setAlpha(170);
            painter->setPen(Qt::NoPen);
            painter->setBrush(bg);
            painter->drawRoundedRect(
                QRect(x - 8, y - textRect.height() - 12,
                      std::max(barPx, textRect.width()) + 16,
                      textRect.height() + 22),
                6, 6);

            painter->setPen(QPen(scalebarFg(), 2));
            painter->drawLine(x, y, x + barPx, y);
            painter->drawLine(x, y - 4, x, y + 4);
            painter->drawLine(x + barPx, y - 4, x + barPx, y + 4);
            painter->drawText(x, y - 8, label);
        }
    }

    // The Esri credit line used to be painted here, bottom-right — required by
    // the tile terms, but drawn *somewhere* on screen is all they ask, and the
    // info strip under the map now carries it instead (see ViewerPage's
    // m_attributionLabel), centred rather than tucked into a corner it shared
    // with the feature panel.

    painter->restore();
}

// ---------------------------------------------------------------------------
// LegendBar
// ---------------------------------------------------------------------------
namespace {
// The legend's horizontal make-up, in one place because two things need it:
// setState() adds it up to decide how wide the panel must be, and paintEvent()
// lays the pieces out. Kept as constants so those two cannot drift apart —
// which is exactly how the labels came to be cut off.
constexpr int kLegendPadL = 10;     // panel edge to colour bar
constexpr int kLegendBarW = 16;     // the colour bar itself
constexpr int kLegendTextGap = 5;   // colour bar to labels
constexpr int kLegendPadR = 11;     // labels to panel edge
constexpr int kLegendFontPx = 10;
} // namespace

LegendBar::LegendBar(QWidget *parent)
    : QWidget(parent)
{
    setMinimumHeight(140);
    // A placeholder only: the real width comes from the labels, in setState().
    setFixedWidth(kLegendPadL + kLegendBarW + kLegendTextGap + 40 + kLegendPadR);
}

void LegendBar::setState(const QVector<QRgb> &lut, double lowValue, double highValue)
{
    m_lut = lut;
    m_low = lowValue;
    m_high = highValue;

    // As wide as the widest of the three labels needs, and no wider. A fixed
    // width was enough for "1200" and cut "6.7745e+05" in half — the numbers
    // come from whatever raster is loaded, so the width has to come from them
    // too. Measured with the same font size paintEvent() draws with.
    QFont f = font();
    f.setPixelSize(kLegendFontPx);
    const QFontMetrics fm(f);
    int textW = 0;
    for (const double v : {m_high, (m_low + m_high) / 2.0, m_low})
        textW = qMax(textW, fm.horizontalAdvance(formatValue(v)));
    setFixedWidth(kLegendPadL + kLegendBarW + kLegendTextGap + textW + kLegendPadR);

    update();
}

void LegendBar::paintEvent(QPaintEvent *)
{
    if (m_lut.size() != 256)
        return;
    QPainter p(this);
    // The legend floats over the map, so it draws its own panel. Not from the
    // stylesheet: a QWidget subclass with its own paintEvent never gets the
    // styled background for free.
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setPen(legendFrame());
    p.setBrush(ThemeManager::mapped("#1b1f26"));
    p.drawRoundedRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5), 9, 9);
    p.setRenderHint(QPainter::Antialiasing, false);

    const int margin = 12;
    const QRect bar(kLegendPadL, margin, kLegendBarW, height() - margin * 2);
    for (int y = bar.top(); y <= bar.bottom(); ++y) {
        const double t = 1.0 - (y - bar.top()) / double(bar.height());
        p.setPen(QColor::fromRgb(m_lut[int(std::lround(t * 255))]));
        p.drawLine(bar.left(), y, bar.right(), y);
    }
    p.setPen(legendFrame());
    p.setBrush(Qt::NoBrush);
    p.drawRect(bar.adjusted(0, 0, -1, -1));

    QFont f = p.font();
    f.setPixelSize(kLegendFontPx);
    p.setFont(f);
    p.setPen(hintFg());
    const int tx = bar.right() + kLegendTextGap;
    p.drawText(tx, bar.top() + 8, formatValue(m_high));
    p.drawText(tx, bar.center().y() + 4, formatValue((m_low + m_high) / 2.0));
    p.drawText(tx, bar.bottom(), formatValue(m_low));
}

// ---------------------------------------------------------------------------
// ViewerPage — construction
// ---------------------------------------------------------------------------
ViewerPage::ViewerPage(QWidget *parent)
    : QWidget(parent)
{
    // Layers can be dropped straight onto the page — see dropEvent().
    setAcceptDrops(true);

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(28, 24, 28, 24);
    layout->setSpacing(10);

    // --- Controls card ---
    auto *card = new QFrame(this);
    card->setObjectName(QStringLiteral("Card"));
    m_controlsCard = card;
    auto *cardLayout = new QVBoxLayout(card);
    cardLayout->setContentsMargins(18, 12, 18, 12);
    cardLayout->setSpacing(8);

    auto *row1 = new QHBoxLayout;
    row1->setSpacing(10);
    row1->addWidget(new QLabel(tr("Layer"), card));
    // The only list that can grow without bound: a long session registers a
    // layer per output per run. Cap it, let it scroll, and give every row a
    // bin so the list can be pruned without restarting.
    auto *layerCombo = new SmoothComboBox(card);
    layerCombo->setVisibleItemCap(15);
    // Deliberately NOT setView(): the view QComboBox builds for itself moves
    // the highlight with the mouse, which is what makes the hover colour the
    // same teal as every other list in the app. A plain QListView would draw a
    // hover state instead, in the style's own colour.
    m_layerDelegate = new LayerItemDelegate(layerCombo);
    layerCombo->view()->setItemDelegate(m_layerDelegate);
    layerCombo->view()->setMouseTracking(true);
    // Ours is installed after the combo's own container filter, so it is asked
    // first and can claim the click on the bin before the popup closes on it.
    layerCombo->view()->viewport()->installEventFilter(this);
    m_layerCombo = layerCombo;
    m_layerCombo->setMinimumWidth(260);
    row1->addWidget(m_layerCombo, 1);
    auto *openBtn = new QToolButton(card);
    openBtn->setText(QStringLiteral("..."));
    openBtn->setToolTip(tr("Open a raster or a vector layer"));
    openBtn->setCursor(Qt::PointingHandCursor);
    row1->addWidget(openBtn);
    m_openButton = openBtn;
    // Reset view is added further down, on the same row as the basemap toggle:
    // both act on how the map is shown rather than on which layer is loaded.
    auto *resetBtn = new QToolButton(card);
    resetBtn->setText(tr("Reset view"));
    resetBtn->setToolTip(tr("Fit the active layer back into the window."));
    resetBtn->setCursor(Qt::PointingHandCursor);
    auto *exportBtn = new QToolButton(card);
    exportBtn->setText(tr("Export"));
    exportBtn->setCursor(Qt::PointingHandCursor);
    row1->addWidget(exportBtn);
    m_exportButton = exportBtn;
    cardLayout->addLayout(row1);

    auto *row2 = new QHBoxLayout;
    row2->setSpacing(10);
    row2->addWidget(new QLabel(tr("Color scale"), card));
    m_colormapCombo = new SmoothComboBox(card);
    for (const Colormap &c : colormaps())
        m_colormapCombo->addItem(QString::fromLatin1(c.name));
    // Not a colour ramp but a rendering: shaded relief from the raster's own
    // gradients. On a DEM it reads as terrain; handled in rebuildImage().
    m_colormapCombo->addItem(tr("Hillshade"));
    row2->addWidget(m_colormapCombo);
    row2->addSpacing(8);

    // Greys out the ramps that a reader with a colour vision deficiency cannot
    // follow, rather than removing them: the indices stay put, and the reason a
    // ramp is unavailable is visible instead of the ramp simply vanishing.
    m_cvdSafeToggle = new QCheckBox(tr("Colour-blind safe"), card);
    m_cvdSafeToggle->setChecked(false);
    row2->addWidget(TrajectaUi::withHelpDot(m_cvdSafeToggle, cvdSafeHelpText()));
    row2->addSpacing(8);
    row2->addWidget(new QLabel(tr("Stretch"), card));
    m_stretchCombo = new SmoothComboBox(card);
    m_stretchCombo->addItem(tr("Min–Max"));
    m_stretchCombo->addItem(tr("Percentile 2–98"));
    m_stretchCombo->addItem(tr("Logarithmic"));
    row2->addWidget(TrajectaUi::withHelpDot(m_stretchCombo, stretchHelpText()));
    row2->addSpacing(8);
    row2->addWidget(new QLabel(tr("Opacity"), card));
    m_opacitySlider = new QSlider(Qt::Horizontal, card);
    m_opacitySlider->setRange(0, 100);
    m_opacitySlider->setValue(100);
    m_opacitySlider->setFixedWidth(110);
    m_opacitySlider->setFixedHeight(22);  // room for the handle's QSS overshoot
    row2->addWidget(m_opacitySlider);
    row2->addSpacing(8);
    // Off unless the user asks for it: it is the only thing in Trajecta that
    // reaches out to the network, and that should never be a surprise.
    m_basemapToggle = new QCheckBox(tr("Satellite basemap"), card);
    m_basemapToggle->setEnabled(false);
    m_basemapToggle->setChecked(false);
    row2->addWidget(TrajectaUi::withHelpDot(m_basemapToggle, basemapHelpText()));
    row2->addSpacing(8);
    row2->addWidget(resetBtn);
    row2->addStretch(1);
    cardLayout->addLayout(row2);

    auto *row3 = new QHBoxLayout;
    row3->setSpacing(10);
    auto *filterLabel = new QLabel(tr("Filter"), card);
    filterLabel->setToolTip(tr("Hide cells outside the selected value range. "
                               "Use it to isolate e.g. the top 5%% of a density "
                               "raster."));
    row3->addWidget(filterLabel);
    m_rangeSlider = new RangeSlider(card);
    row3->addWidget(m_rangeSlider, 1);
    m_filterLoSpin = new QDoubleSpinBox(card);
    TrajectaUi::guardWheel(m_filterLoSpin);
    m_filterLoSpin->setKeyboardTracking(false);
    m_filterLoSpin->setMinimumWidth(110);
    row3->addWidget(m_filterLoSpin);
    row3->addWidget(new QLabel(QStringLiteral("–"), card));
    m_filterHiSpin = new QDoubleSpinBox(card);
    TrajectaUi::guardWheel(m_filterHiSpin);
    m_filterHiSpin->setKeyboardTracking(false);
    m_filterHiSpin->setMinimumWidth(110);
    row3->addWidget(m_filterHiSpin);
    m_percentToggle = new QToolButton(card);
    m_percentToggle->setText(QStringLiteral("%"));
    m_percentToggle->setCheckable(true);
    m_percentToggle->setCursor(Qt::PointingHandCursor);
    m_percentToggle->setToolTip(tr("Switch the filter boxes between absolute "
                                   "values and percentiles."));
    row3->addWidget(m_percentToggle);
    cardLayout->addLayout(row3);

    layout->addWidget(card);

    // --- Canvas ---
    // QFrame, not QWidget: a plain QWidget honours only the background
    // properties from a stylesheet, so border and border-radius on one are
    // silently dropped.
    auto *canvasHolder = new QFrame(this);
    canvasHolder->setObjectName(QStringLiteral("CanvasHolder"));
    m_canvasHolder = canvasHolder;

    // The holder is now a column: the map, and under it the line that says what
    // is on screen. That line used to sit outside the panel, on the page
    // background, where it read as a caption belonging to nothing — and on the
    // themes with artwork behind them it was barely legible at all. It is a
    // property of the canvas, so it lives in the canvas.
    auto *holderLayout = new QVBoxLayout(canvasHolder);
    // Five pixels, not one. The panel's corners are rounded by 12 px, and the
    // deepest that arc cuts into a corner is about 3.5 px: at a one-pixel
    // margin the map's square corner sat on top of the arc and filled it in, so
    // the panel read as a rectangle. It only showed on the themes with artwork
    // behind them, where the corner should have shown the picture — everywhere
    // else the page behind is the same colour and nobody could see it.
    //
    // The map is inset rather than masked deliberately: masking the viewport
    // makes it non-opaque, and then every repaint has to go up to the window
    // background first — a full-size image rescale on those very themes.
    holderLayout->setContentsMargins(5, 5, 5, 0);
    holderLayout->setSpacing(0);

    auto *canvasArea = new QWidget(canvasHolder);
    // An ordinary QWidget paints the opaque generic background (see the
    // AddChunkRow/ChunkHost comments below) — here that hid the holder's own
    // card colour behind a flat window-coloured square, most visible on the
    // light themes where the two are far enough apart to read as a mistake.
    canvasArea->setObjectName(QStringLiteral("CanvasArea"));
    m_canvasStack = new QStackedLayout(canvasArea);
    m_canvasStack->setContentsMargins(0, 0, 0, 0);
    m_view = new MapView(canvasArea);
    m_scene = new QGraphicsScene(this);
    m_view->setScene(m_scene);
    m_canvasStack->addWidget(m_view);
    m_placeholder = new QLabel(canvasArea);
    m_placeholder->setAlignment(Qt::AlignCenter);
    m_placeholder->setWordWrap(true);
    m_placeholder->setObjectName(QStringLiteral("HintLabel"));
    m_placeholder->setText(kEmptyViewerHint());
    m_canvasStack->addWidget(m_placeholder);
    m_canvasStack->setCurrentWidget(m_placeholder);

    // Overlays panel: a tick per vector layer, floating over the top-left of
    // the map rather than sitting in the toolbar. Tick boxes because the
    // overlays are not alternatives — routes and sample points are worth
    // seeing together — and next to the map because that is what they are
    // about.
    //
    // Parented to the view, NOT to its viewport: panning calls
    // QWidget::scroll() on the viewport, which drags the viewport's child
    // widgets along with the scene. As a sibling of the viewport the panel is
    // a fixed mask over the canvas — it stays in the corner however the map
    // is moved or zoomed.
    m_overlayPanel = new QWidget(m_view);
    m_overlayPanel->setObjectName(QStringLiteral("OverlayPanel"));
    m_overlayPanelLayout = new QVBoxLayout(m_overlayPanel);
    m_overlayPanelLayout->setContentsMargins(12, 9, 14, 10);
    m_overlayPanelLayout->setSpacing(5);
    auto *overlayTitle = new QLabel(tr("Overlays"), m_overlayPanel);
    overlayTitle->setObjectName(QStringLiteral("OverlayPanelTitle"));
    m_overlayPanelLayout->addWidget(overlayTitle);
    m_overlayPanel->move(14, 14);
    m_overlayPanel->hide();   // nothing to show until a run produces vectors

    // The colour bar floats over the top-right of the map, for the same reason
    // the Overlays panel floats over the top-left: as a widget in the row it
    // took its width out of the canvas, so the canvas was narrower than the
    // card above it whenever a colour ramp was in use — and jumped wider when
    // Hillshade hid it. Parented to the view, not the viewport, so panning
    // does not drag it along.
    m_legend = new LegendBar(m_view);
    m_legend->hide();   // nothing to explain until a ramp is on screen
    buildFeaturePanel();
    m_view->installEventFilter(this);
    connect(m_view, &MapView::clicked, this, &ViewerPage::onCanvasClicked);

    holderLayout->addWidget(canvasArea, 1);

    // --- Info strip, inside the panel and under the map ---
    auto *infoRow = new QWidget(canvasHolder);
    infoRow->setObjectName(QStringLiteral("CanvasInfoRow"));
    m_infoRow = infoRow;
    auto *info = new QHBoxLayout(infoRow);
    // Indented to the same margin the scalebar keeps from the left edge, so the
    // two line up, and given enough height to sit clear of the rounded corner.
    info->setContentsMargins(14, 5, 14, 7);
    info->setSpacing(10);
    m_crsLabel = new QLabel(infoRow);
    m_crsLabel->setObjectName(QStringLiteral("HintLabel"));
    info->addWidget(m_crsLabel);
    m_resLabel = new QLabel(infoRow);
    m_resLabel->setObjectName(QStringLiteral("HintLabel"));
    info->addWidget(m_resLabel);
    info->addStretch(1);
    m_cursorLabel = new QLabel(infoRow);
    m_cursorLabel->setObjectName(QStringLiteral("HintLabel"));
    info->addWidget(m_cursorLabel);
    // Not in `info`: a layout can only balance a widget between its
    // neighbours, and the neighbours here are not the same width — the
    // cursor readout on the right is empty whenever nothing is under the
    // pointer. Free-floating and repositioned by hand, this stays on the
    // row's true centre regardless of what either side is showing.
    m_attributionLabel = new QLabel(infoRow);
    m_attributionLabel->setObjectName(QStringLiteral("HintLabel"));
    infoRow->installEventFilter(this);
    holderLayout->addWidget(infoRow, 0);

    layout->addWidget(canvasHolder, 1);

    // --- State & wiring ---
    QSettings settings;
    m_colormapIndex = std::clamp(
        settings.value(QStringLiteral("viewer/colormap"), 1).toInt(),
        0, int(colormaps().size()));   // one past the ramps = Hillshade
    m_stretchIndex = std::clamp(
        settings.value(QStringLiteral("viewer/stretch"), 1).toInt(), 0, 2);
    m_colormapCombo->setCurrentIndex(m_colormapIndex);
    m_stretchCombo->setCurrentIndex(m_stretchIndex);

    m_rebuildTimer = new QTimer(this);
    m_rebuildTimer->setSingleShot(true);
    m_rebuildTimer->setInterval(40);
    connect(m_rebuildTimer, &QTimer::timeout, this, &ViewerPage::rebuildImage);

    // Basemap networking: persistent disk cache keeps tiles across sessions.
    m_net = new QNetworkAccessManager(this);
    auto *diskCache = new QNetworkDiskCache(this);
    diskCache->setCacheDirectory(
        QStandardPaths::writableLocation(QStandardPaths::CacheLocation)
        + QStringLiteral("/tiles"));
    diskCache->setMaximumCacheSize(256 * 1024 * 1024);
    m_net->setCache(diskCache);

    m_basemapTimer = new QTimer(this);
    m_basemapTimer->setSingleShot(true);
    m_basemapTimer->setInterval(150);
    connect(m_basemapTimer, &QTimer::timeout, this, &ViewerPage::updateBasemap);
    connect(m_view, &MapView::viewChanged, this, [this] {
        if (m_basemapToggle->isChecked())
            m_basemapTimer->start();
        // Pin the mask to the corner on every pan, zoom and resize, so its
        // position can never end up depending on where the map has been
        // dragged.
        if (m_overlayPanel && m_overlayPanel->isVisible()) {
            m_overlayPanel->move(14, 14);
            m_overlayPanel->raise();
        }
    });
    connect(m_cvdSafeToggle, &QCheckBox::toggled, this, [this](bool on) {
        QSettings().setValue(QStringLiteral("viewer/cvdSafe"), on);
        applyCvdSafeFilter();
    });
    {
        const bool cvdOn =
            QSettings().value(QStringLiteral("viewer/cvdSafe"), false).toBool();
        const QSignalBlocker blocker(m_cvdSafeToggle);
        m_cvdSafeToggle->setChecked(cvdOn);
        applyCvdSafeFilter();
    }
    connect(m_basemapToggle, &QCheckBox::toggled, this, [this](bool on) {
        QSettings().setValue(QStringLiteral("viewer/basemap"), on);
        m_view->setAttribution(on ? tr(kAttribution) : QString());
        m_attributionLabel->setText(on ? tr(kAttribution) : QString());
        repositionAttribution();
        if (on)
            updateBasemap();
        else
            clearBasemap(false);
    });
    {
        const bool baseOn =
            QSettings().value(QStringLiteral("viewer/basemap"), false).toBool();
        const QSignalBlocker blocker(m_basemapToggle);
        m_basemapToggle->setChecked(baseOn);
        if (baseOn) {
            m_view->setAttribution(tr(kAttribution));
            m_attributionLabel->setText(tr(kAttribution));
        }
        repositionAttribution();
    }

    connect(m_layerCombo, &QComboBox::currentIndexChanged,
            this, &ViewerPage::selectLayer);
    connect(m_colormapCombo, &QComboBox::currentIndexChanged, this, [this](int i) {
        m_colormapIndex = i;
        QSettings().setValue(QStringLiteral("viewer/colormap"), i);
        scheduleRebuild();
    });
    connect(m_stretchCombo, &QComboBox::currentIndexChanged, this, [this](int i) {
        m_stretchIndex = i;
        QSettings().setValue(QStringLiteral("viewer/stretch"), i);
        scheduleRebuild();
    });
    connect(m_opacitySlider, &QSlider::valueChanged, this, [this](int v) {
        if (m_pixmapItem)
            m_pixmapItem->setOpacity(v / 100.0);
    });
    connect(m_rangeSlider, &RangeSlider::rangeChanged, this,
            [this](double lo, double hi) {
                RasterLayer *layer = currentLayer();
                if (!layer || m_updatingUi)
                    return;
                m_filterLo = layer->minV + lo * (layer->maxV - layer->minV);
                m_filterHi = layer->minV + hi * (layer->maxV - layer->minV);
                updateFilterUi();
                scheduleRebuild();
            });
    connect(m_filterLoSpin, &QDoubleSpinBox::valueChanged, this, [this] {
        if (!m_updatingUi)
            applyFilterFromSpins();
    });
    connect(m_filterHiSpin, &QDoubleSpinBox::valueChanged, this, [this] {
        if (!m_updatingUi)
            applyFilterFromSpins();
    });
    connect(m_percentToggle, &QToolButton::toggled, this, [this](bool on) {
        m_percentMode = on;
        updateFilterUi();
    });
    connect(openBtn, &QToolButton::clicked, this, [this] {
        const QString file = QFileDialog::getOpenFileName(
            this, tr("Open layer"), QString(),
            tr("All supported layers (*.tif *.tiff *.vrt *.img *.asc *.shp "
               "*.gpkg *.geojson *.json *.kml *.kmz *.gml *.gpx *.csv *.tab "
               "*.mif *.dxf *.sqlite *.fgb);;"
               "Rasters (*.tif *.tiff *.vrt *.img *.asc);;"
               "Vector layers (*.shp *.gpkg *.geojson *.json *.kml *.kmz "
               "*.gml *.gpx *.csv *.tab *.mif *.dxf *.sqlite *.fgb);;"
               "All files (*)"));
        if (file.isEmpty())
            return;
        QString error;
        if (!openAnyFile(file, &error))
            TrajectaUi::notify(this, tr("Cannot open this file"), error);
    });
    connect(resetBtn, &QToolButton::clicked, this, [this] { m_view->fitAll(); });
    connect(exportBtn, &QToolButton::clicked, this, &ViewerPage::exportImage);
    connect(m_view, &MapView::hoverScenePos, this, &ViewerPage::onHover);
    connect(m_view, &MapView::hoverLeft, this, [this] {
        m_cursorLabel->clear();
        // Left the map with the identify pointer showing: put the hand back, or
        // the cursor keeps promising a feature over the panel beside it.
        if (m_hoveringFeature) {
            m_hoveringFeature = false;
            m_view->viewport()->setCursor(Qt::OpenHandCursor);
        }
    });

    updateInfoStrip();
}

ViewerPage::~ViewerPage()
{
    // Tear down GDAL/PROJ state (coordinate transformations, open datasets)
    // while the libraries are still healthy: leaving them alive until DLL
    // unload crashes PROJ's cleanup on exit.
    ++m_basemapGen;
    const QList<QNetworkReply *> replies = m_net->findChildren<QNetworkReply *>();
    for (QNetworkReply *reply : replies)
        reply->abort();
    clearBasemap(true);
    m_layers.clear();
    m_overlays.clear();
    // After the layers, and by hand rather than by leaving it to the members:
    // m_tourDir is declared last, so it would be destroyed *first*, and on
    // Windows a folder holding an open dataset does not delete — the tour's
    // example files would then be left in the temporary directory.
    m_tourDir.reset();
}

// ---------------------------------------------------------------------------
// GDAL access
// ---------------------------------------------------------------------------
void ViewerPage::configureGdal(const QStringList &dllDirs,
                               const QString &projDataDir, const QString &gdalDataDir)
{
    m_gdalDirs = dllDirs;
    m_projData = projDataDir;
    m_gdalData = gdalDataDir;
    m_gdalFailed = false;   // allow a retry after "Locate GDAL folder"
}

bool ViewerPage::ensureGdal()
{
    GdalApi &api = GdalApi::instance();
    if (api.isLoaded())
        return true;
    if (m_gdalFailed)
        return false;
    if (!api.load(m_gdalDirs, m_projData, m_gdalData)) {
        m_gdalFailed = true;
        m_placeholder->setText(
            tr("GDAL libraries could not be loaded (%1).\n\nInstall GDAL "
               "through OSGeo4W or use \"Locate GDAL folder...\" in the "
               "status bar, then come back to this page.")
                .arg(api.loadError()));
        m_canvasStack->setCurrentWidget(m_placeholder);
        m_legend->hide();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Layer registration / loading
// ---------------------------------------------------------------------------
void ViewerPage::registerRaster(const QString &label, const QString &path,
                                bool select)
{
    int index = -1;
    for (int i = 0; i < int(m_layers.size()); ++i) {
        if (QString::compare(m_layers[i]->path, path, Qt::CaseInsensitive) == 0) {
            index = i;
            // The file may have been rewritten by a new run: drop cached data.
            RasterLayer &layer = *m_layers[i];
            if (layer.ds && GdalApi::instance().isLoaded())
                GdalApi::instance().Close(layer.ds);
            layer.ds = nullptr;
            layer.loaded = false;
            layer.failed = false;
            layer.data.clear();
            break;
        }
    }
    if (index < 0) {
        auto layer = std::make_unique<RasterLayer>();
        layer->label = label;
        layer->path = path;
        m_layers.push_back(std::move(layer));
        m_layerCombo->addItem(QStringLiteral("%1 — %2")
                                  .arg(label, QFileInfo(path).fileName()));
        index = m_layerCombo->count() - 1;
    }

    if (select || index == m_layerCombo->currentIndex()) {
        if (isVisible()) {
            if (index == m_layerCombo->currentIndex())
                selectLayer(index);
            else
                m_layerCombo->setCurrentIndex(index);
        } else {
            // Defer the (potentially slow) GDAL read to the first show.
            const QSignalBlocker blocker(m_layerCombo);
            m_layerCombo->setCurrentIndex(index);
            m_deferredSelect = index;
        }
    }
}

void ViewerPage::positionLegend()
{
    if (!m_legend || !m_view)
        return;
    constexpr int kMargin = 14;
    // 112/0.496/272 rather than 140/0.62/340: 20% shorter across the board, so
    // a short window still leaves the legend clear of the feature panel below
    // it instead of the two touching.
    const int h = qBound(112, int(m_view->height() * 0.496), 272);
    m_legend->resize(m_legend->width(), h);
    m_legend->move(m_view->width() - m_legend->width() - kMargin, kMargin);
}

// True centre of the info strip, not a share of whatever room the CRS/
// resolution pair and the cursor readout leave around it — see the comment
// on m_attributionLabel for why a layout stretch cannot do this. Clamped
// against both neighbours' actual edges rather than centred unconditionally:
// a long CRS name (a UTM zone with its EPSG code routinely runs past 200 px)
// can reach past the row's true midpoint on its own, and a label that did not
// yield would print straight through it instead of merely drifting off
// centre — worse than the drift this replaced.
void ViewerPage::repositionAttribution()
{
    if (!m_attributionLabel || !m_infoRow)
        return;
    // A label's setText() only schedules the layout that resolves its
    // neighbours' new geometry — the actual pass runs once control returns to
    // the event loop, which is later than every call site here. Without this,
    // the clamp below would still be reading last frame's (narrower) CRS/
    // resolution width at the moment CRS text just grew, and let the two
    // touch anyway on the very update meant to prevent it.
    if (QLayout *lay = m_infoRow->layout())
        lay->activate();
    m_attributionLabel->adjustSize();
    const int rowW = m_infoRow->width();
    int x = (rowW - m_attributionLabel->width()) / 2;
    const int y = (m_infoRow->height() - m_attributionLabel->height()) / 2;

    constexpr int kGap = 16;
    if (m_resLabel)
        x = qMax(x, m_resLabel->geometry().right() + kGap);
    if (m_cursorLabel)
        x = qMin(x, m_cursorLabel->geometry().left() - kGap - m_attributionLabel->width());

    m_attributionLabel->move(qMax(0, x), qMax(0, y));
}

bool ViewerPage::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == m_view && event->type() == QEvent::Resize) {
        positionLegend();
        positionFeaturePanel();
        return false;
    }
    if (watched == m_infoRow && event->type() == QEvent::Resize) {
        repositionAttribution();
        return false;
    }

    QAbstractItemView *list = m_layerCombo ? m_layerCombo->view() : nullptr;
    if (!list || watched != list->viewport())
        return QWidget::eventFilter(watched, event);

    const QEvent::Type type = event->type();
    if (type != QEvent::MouseMove && type != QEvent::MouseButtonPress
        && type != QEvent::MouseButtonRelease) {
        return QWidget::eventFilter(watched, event);
    }

    const auto *mouse = static_cast<QMouseEvent *>(event);
    const QPoint pos = mouse->position().toPoint();
    const QModelIndex index = list->indexAt(pos);
    const bool onBin = index.isValid()
                       && LayerItemDelegate::binRect(list->visualRect(index)).contains(pos);

    if (type == QEvent::MouseMove) {
        // Light the bin up under the cursor, but let the event through so the
        // row highlight keeps following the mouse.
        if (m_layerDelegate) {
            m_layerDelegate->setHoveredRow(onBin ? index.row() : -1);
            list->viewport()->update();
        }
        return false;
    }
    if (!onBin)
        return false;

    // Claim both halves of the click: letting the press through would start a
    // selection, and letting the release through would close the popup and
    // switch layer instead of removing one.
    if (type == QEvent::MouseButtonPress)
        return true;

    const int row = index.row();
    m_layerCombo->hidePopup();
    // Out of the popup before opening a modal dialog: the popup holds a mouse
    // grab that a dialog underneath it would have to fight.
    QTimer::singleShot(0, this, [this, row] { confirmRemoveLayer(row); });
    return true;
}

void ViewerPage::confirmRemoveLayer(int index)
{
    if (index < 0 || index >= int(m_layers.size()))
        return;
    // The dialog is fixed-size: elide so a long name wraps instead of clipping.
    const QString label = TrajectaUi::elideForConfirm(m_layers[index]->label);
    const QString file =
        TrajectaUi::elideForConfirm(QFileInfo(m_layers[index]->path).fileName());

    if (!TrajectaUi::confirm(this, tr("Remove layer"),
                             tr("Remove \"%1\" (%2) from the Viewer?\n\n"
                                "The file on disk is not deleted.")
                                 .arg(label, file))) {
        return;
    }

    const int wasCurrent = m_layerCombo->currentIndex();
    {
        // Dropping the entry re-emits currentIndexChanged; let the explicit
        // selection below decide what to show instead of reacting twice.
        const QSignalBlocker blocker(m_layerCombo);
        m_layerCombo->removeItem(index);
    }
    m_layers.erase(m_layers.begin() + index);   // closes the dataset

    if (m_layers.empty()) {
        clearBasemap(true);
        m_scene->clear();
        m_pixmapItem = nullptr;
        m_overlayItems.clear();
        // No raster on screen, so no framing to carry to the next one.
        m_shownFrame = ViewFrame();
        m_legend->hide();
        m_basemapToggle->setEnabled(false);
        // Vectors may still be loaded: rather than vanishing with the raster
        // they happened to be drawn over, they take the scene for themselves.
        m_vectorScale = 0.0;
        rebuildOverlay();
        if (m_overlayItems.isEmpty()) {
            m_placeholder->setText(kEmptyViewerHint());
            m_canvasStack->setCurrentWidget(m_placeholder);
        }
        updateInfoStrip();
        return;
    }

    // Keep looking at the same layer where possible; otherwise fall back to
    // its neighbour.
    int next = wasCurrent;
    if (index < wasCurrent)
        --next;
    next = qBound(0, next, int(m_layers.size()) - 1);
    const QSignalBlocker blocker(m_layerCombo);
    m_layerCombo->setCurrentIndex(next);
    selectLayer(next);
}

// ---------------------------------------------------------------------------
// The guided walkthrough
// ---------------------------------------------------------------------------
bool ViewerPage::loadTourSamples()
{
    if (!m_tourDir) {
        m_tourDir = std::make_unique<QTemporaryDir>();
        if (!m_tourDir->isValid()) {
            m_tourDir.reset();
            return false;
        }
        // Out of the executable and onto disk, because GDAL opens paths, not
        // Qt resources.
        struct Sample { const char *resource; const char *name; QString *out; };
        const Sample samples[] = {
            {":/assets/tour/sample_dem.tif", "sample_dem.tif", &m_tourRasterPath},
            {":/assets/tour/sample_points.geojson", "sample_points.geojson",
             &m_tourVectorPath},
        };
        for (const Sample &sample : samples) {
            const QString target = m_tourDir->filePath(QString::fromLatin1(sample.name));
            QFile::remove(target);
            if (!QFile::copy(QString::fromLatin1(sample.resource), target)) {
                unloadTourSamples();
                return false;
            }
            // A file copied out of a resource comes out read-only, which would
            // stop the temporary directory removing itself afterwards.
            QFile::setPermissions(target, QFile::ReadOwner | QFile::WriteOwner);
            *sample.out = target;
        }
    }

    if (!ensureGdal())
        return false;
    // Which layer the user was looking at, so the tour can give it back. Taken
    // on the first call only, and guarded by its own flag rather than by the
    // index being negative: an empty Viewer answers -1, and testing for that
    // would read the index again on the next screen — by which time the answer
    // is the tour's own sample.
    if (!m_tourPrevLayerKnown) {
        m_tourPrevLayer = m_layerCombo->currentIndex();
        m_tourPrevLayerKnown = true;
    }
    registerRaster(tr("Sample DEM"), m_tourRasterPath, true);
    registerVectorOverlay(tr("Sample points"), m_tourVectorPath);
    return true;
}

void ViewerPage::unloadTourSamples()
{
    // Order matters, and on Windows it is not negotiable: forget the layers,
    // which closes their GDAL datasets, and only then delete the files. A file
    // still open cannot be removed, and the failure is silent.
    for (int i = int(m_layers.size()) - 1; i >= 0; --i) {
        if (m_layers[i]->path == m_tourRasterPath) {
            const QSignalBlocker blocker(m_layerCombo);
            m_layerCombo->removeItem(i);
            m_layers.erase(m_layers.begin() + i);   // closes the dataset
        }
    }
    for (int i = int(m_overlays.size()) - 1; i >= 0; --i) {
        if (m_overlays[i]->path == m_tourVectorPath)
            dropOverlayAt(i);
    }
    rebuildOverlayPanel();
    fitOverlayPanel();

    clearBasemap(true);
    m_scene->clear();
    m_pixmapItem = nullptr;
    m_overlayItems.clear();
    m_vectorScale = 0.0;
    m_shownFrame = ViewFrame();

    if (m_layers.empty()) {
        m_placeholder->setText(kEmptyViewerHint());
        m_canvasStack->setCurrentWidget(m_placeholder);
        m_legend->hide();
        m_basemapToggle->setEnabled(false);
    } else {
        // Something of the user's own was already here: put back the one they
        // were looking at before the tour borrowed the canvas.
        const int back = qBound(0, m_tourPrevLayer, int(m_layers.size()) - 1);
        const QSignalBlocker blocker(m_layerCombo);
        m_layerCombo->setCurrentIndex(back);
        selectLayer(back);
    }
    m_tourPrevLayer = -1;
    m_tourPrevLayerKnown = false;
    rebuildOverlay();
    updateInfoStrip();

    m_tourRasterPath.clear();
    m_tourVectorPath.clear();
    m_tourDir.reset();   // removes the folder and everything in it
}

QVector<TourStep> ViewerPage::walkthroughSteps()
{
    QVector<TourStep> steps;

    {   // Choosing what is drawn, and how
        TourStep s;
        s.lightCard(m_controlsCard);
        s.title = tr("Which layer, and in what colours");
        s.text = tr(
            "A colour ramp is an argument about the data, not a decoration: it "
            "decides which differences a reader can see at all.<br><br>"
            "Layers do not have to be opened through the button: <b>files can be "
            "dragged onto this page</b> from Explorer and dropped anywhere on it — "
            "several at once, rasters and vector layers together. Each one is "
            "read and added where it belongs, the rasters to the list below and "
            "the vectors to the overlay panel.");
        s.annotations = {
            { m_layerCombo, tr("Every layer loaded. Each row has a bin to drop it.") },
            { m_openButton, tr("Opens a raster or a vector file — or drag the "
                               "files onto the page instead.") },
            { m_colormapCombo, tr("The colour ramp, plus Hillshade for relief.") },
            { m_cvdSafeToggle, tr("Greys out the ramps a colour-blind reader "
                                  "cannot follow.") },
            { m_stretchCombo, tr("Which range of values the ramp is spread over.") },
        };
        steps.append(s);
    }

    {   // Reading the values
        TourStep s;
        s.lightCard(m_controlsCard);
        s.title = tr("Looking into the numbers");
        s.text = tr(
            "The filter is the tool that turns a density raster into an "
            "argument: hiding the low values leaves the corridors that actually "
            "carried routes, and the histogram behind the slider shows how much "
            "of the map each threshold keeps.");
        s.annotations = {
            { m_opacitySlider, tr("Fades the raster, to see what is under it.") },
            { m_basemapToggle, tr("Satellite imagery underneath, warped to the "
                                  "layer's own system.") },
            { m_rangeSlider, tr("Keeps only values inside the range.") },
            { m_percentToggle, tr("Switches the two boxes between values and "
                                  "percentiles.") },
        };
        steps.append(s);
    }

    {   // The vector overlay, and the canvas it is drawn on
        TourStep s;
        // The whole map frame, not the little panel alone: the panel is a list
        // of names, and what it is a list *of* is only visible on the canvas —
        // the elevation model in its colour ramp with the sample points on top
        // of it. Lighting the panel by itself pointed at the index of a book
        // while the book stayed in the dark. The panel floats over the canvas,
        // so this covers both.
        s.lightCard(m_canvasHolder);
        // The card reaches to within a few hundred pixels of the top of the
        // window, and a step describing it can only stand in that band — the
        // reason the text below is kept this short: at the plain 920 px
        // ceiling it already folds short enough to clear the card, without
        // the callout growing wide enough to reach the Overlays annotation
        // in the corner beside it.
        s.title = tr("Vectors over rasters");
        s.text = tr(
            "This frame is the map itself. Here you can visualize "
            "the results of your analysis.<br><br>"
            "Any combination of vectors can be shown at once, each in its own "
            "colour, the way a computed route is judged against a real one. A "
            "layer in a different CRS is reprojected automatically, and a "
            "vector opened alone gets the canvas to itself.<br><br>"
            "Colour and size can both be changed by <b>right-clicking a layer in "
            "the Overlays panel</b>.");
        s.annotations = {
            { m_overlayPanel, tr("Every vector layer loaded, and its colour. "
                                 "Tick to draw, untick to hide — right-click to "
                                 "recolour and resize.") },
        };
        steps.append(s);
    }

    {   // Getting a picture out
        TourStep s;
        s.targets = { m_exportButton };
        s.title = tr("Taking a picture out");
        s.text = tr(
            "Export writes what you are looking at as a PNG — the colours, the "
            "filter, the overlays and the zoom, exactly as they are on screen. "
            "It asks first how large the image should be, in DPI or in pixels, "
            "and whether to draw a scalebar and a north arrow on it.<br><br>"
            "When the satellite basemap is on, the attribution is written into "
            "the image as well, because the tiles are somebody else's work.");
        steps.append(s);
    }

    return steps;
}

void ViewerPage::applyTheme()
{
    if (m_view) {
        m_view->applyCanvasBackground();
        m_view->viewport()->update();
    }
    if (m_legend)
        m_legend->update();
    // The filter track and its histogram are painted by hand from the palette,
    // so they need a repaint to pick up the new colours.
    if (m_rangeSlider)
        m_rangeSlider->update();
    // The overlay's pen colour is baked into the item when it is built.
    rebuildOverlay();
}

void ViewerPage::releaseFiles()
{
    if (!GdalApi::instance().isLoaded())
        return;
    for (auto &layer : m_layers) {
        if (layer->ds) {
            GdalApi::instance().Close(layer->ds);
            layer->ds = nullptr;   // reopened on demand by onHover()
        }
    }
}

void ViewerPage::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    if (m_deferredSelect >= 0) {
        const int index = m_deferredSelect;
        m_deferredSelect = -1;
        selectLayer(index);
    }
    if (m_framedWhileHidden) {
        m_framedWhileHidden = false;
        // After the event loop, not now: the page has only just been made
        // visible and the view has not been given its real size yet.
        QTimer::singleShot(0, this, [this] {
            if (m_view)
                m_view->fitAll();
            positionFeaturePanel();
        });
    }
}

void ViewerPage::registerVectorOverlay(const QString &label, const QString &path)
{
    // Same file registered again (a re-run rewrote it): drop what was cached
    // and keep its position in the dropdown.
    int index = -1;
    for (int i = 0; i < int(m_overlays.size()); ++i) {
        if (QString::compare(m_overlays[i]->path, path, Qt::CaseInsensitive) == 0) {
            index = i;
            VectorOverlay &o = *m_overlays[i];
            o.loaded = false;
            o.failed = false;
            o.lines.clear();
            o.points.clear();
            o.wkt.clear();
            o.crsName.clear();
            o.extent = QRectF();
            o.hasExtent = false;
            o.drawLines.clear();
            o.drawPoints.clear();
            o.drawExtent = QRectF();
            o.hasDrawExtent = false;
            o.drawWkt.clear();
            o.drawReady = false;
            break;
        }
    }
    if (index < 0) {
        auto overlay = std::make_unique<VectorOverlay>();
        overlay->label = label;
        overlay->path = path;
        m_overlays.push_back(std::move(overlay));
        rebuildOverlayPanel();
    }
    fitOverlayPanel();
    rebuildOverlay();
}

// Takes an overlay out of the list together with its row in the panel. The two
// are kept in step by position, so removing one without the other would leave
// every later overlay wearing its neighbour's tick.
void ViewerPage::dropOverlayAt(int index)
{
    if (index < 0 || index >= int(m_overlays.size()))
        return;
    m_overlays.erase(m_overlays.begin() + index);
    if (index >= m_overlayChecks.size())
        return;
    QCheckBox *check = m_overlayChecks.takeAt(index);
    QWidget *row = check->parentWidget();
    if (!row || row == m_overlayPanel)
        row = check;
    // Out of the layout now, destroyed later: this can run inside the
    // clicked() handler of the very button being removed.
    m_overlayPanelLayout->removeWidget(row);
    row->hide();
    row->deleteLater();
}

// The panel is built from scratch rather than appended to: the colour an
// overlay is drawn in comes from its position in the list, so removing one
// re-colours every overlay after it and the ticks have to follow.
void ViewerPage::rebuildOverlayPanel()
{
    QList<bool> wasChecked;
    wasChecked.reserve(m_overlayChecks.size());
    for (QCheckBox *check : std::as_const(m_overlayChecks))
        wasChecked.append(check->isChecked());

    for (QCheckBox *check : std::as_const(m_overlayChecks)) {
        // The row widget owns the tick and the bin; taking it out takes both.
        QWidget *row = check->parentWidget();
        if (!row || row == m_overlayPanel)
            row = check;
        // Out of the layout at once, destroyed later: this can be running
        // inside the clicked() handler of the very button being removed, so
        // the object has to outlive the call — but if it stayed in the layout
        // until then, the panel would size itself around rows that are on
        // their way out.
        m_overlayPanelLayout->removeWidget(row);
        row->hide();
        row->deleteLater();
    }
    m_overlayChecks.clear();

    for (int i = 0; i < int(m_overlays.size()); ++i) {
        const VectorOverlay &overlay = *m_overlays[i];

        auto *row = new QWidget(m_overlayPanel);
        auto *rowLayout = new QHBoxLayout(row);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        rowLayout->setSpacing(6);

        auto *check = new QCheckBox(overlay.label, row);
        check->setToolTip(QDir::toNativeSeparators(overlay.path));
        // The tick carries the colour the overlay is drawn in, so the panel
        // doubles as the legend for the vectors.
        check->setStyleSheet(QStringLiteral("color:%1;")
                                 .arg(overlayColor(i).name()));
        // Shown straight away: it is either an output of the run just finished
        // or a file the user has this moment asked for, and the tick makes
        // turning it off obvious.
        check->setChecked(i < wasChecked.size() ? wasChecked.at(i) : true);
        connect(check, &QCheckBox::toggled, this, &ViewerPage::rebuildOverlay);
        rowLayout->addWidget(check, 1);

        auto *remove = new QToolButton(row);
        remove->setObjectName(QStringLiteral("OverlayRemove"));
        remove->setText(QStringLiteral("×"));
        remove->setToolTip(tr("Remove this overlay from the Viewer.\n"
                              "The file on disk is not deleted."));
        remove->setCursor(Qt::PointingHandCursor);
        remove->setAutoRaise(true);
        connect(remove, &QToolButton::clicked, this, [this, i] { removeOverlay(i); });
        rowLayout->addWidget(remove);

        // Right-click anywhere on the row — the tick, the name or the space
        // around them — opens the colour wheel for that layer. On the row
        // rather than on the checkbox alone, because the name is the part of it
        // people aim at, and it is a label, not a control.
        row->setContextMenuPolicy(Qt::CustomContextMenu);
        check->setContextMenuPolicy(Qt::CustomContextMenu);
        connect(row, &QWidget::customContextMenuRequested, this,
                [this, row, i](const QPoint &pos) {
                    pickOverlayColour(i, row->mapToGlobal(pos));
                });
        connect(check, &QWidget::customContextMenuRequested, this,
                [this, check, i](const QPoint &pos) {
                    pickOverlayColour(i, check->mapToGlobal(pos));
                });

        m_overlayPanelLayout->addWidget(row);
        // Shown explicitly. A widget added to the layout of a panel that is
        // already on screen is not visible until the layout next runs, and a
        // layout skips items it considers empty — a hidden widget being one.
        // The panel was therefore measured as though it held nothing but its
        // title, and collapsed to that: the bug appeared only from the second
        // overlay on, because the first arrived while the panel was still
        // hidden and was shown along with it.
        row->show();
        m_overlayChecks.append(check);
    }
}

void ViewerPage::setOverlaySizeForTest(int overlayIndex, int percent)
{
    if (overlayIndex < 0 || overlayIndex >= int(m_overlays.size()))
        return;
    m_overlays[overlayIndex]->sizePercent = std::clamp(percent, 25, 400);
    rebuildOverlay();
}

void ViewerPage::pickColourForTest(int overlayIndex)
{
    if (overlayIndex < 0 || overlayIndex >= m_overlayChecks.size())
        return;
    QWidget *row = m_overlayChecks.at(overlayIndex);
    pickOverlayColour(overlayIndex,
                      row->mapToGlobal(QPoint(row->width() / 2, row->height())));
}

void ViewerPage::pickOverlayColour(int index, const QPoint &globalPos)
{
    if (index < 0 || index >= int(m_overlays.size()))
        return;
    // `index` is captured by value in both callbacks below: it names the one
    // row that was right-clicked, fixed at the moment the wheel opened, so
    // every change it reports — colour or size, for as long as the popup
    // stays open — lands on that layer alone and never on whichever one
    // happens to be selected or listed first when the callback runs.
    TrajectaUi::pickColour(
        this, globalPos, overlayColor(index), m_overlays[index]->sizePercent,
        [this, index](const QColor &c) {
            if (index < 0 || index >= int(m_overlays.size()))
                return;   // a layer removed while the wheel was open
            m_overlays[index]->customColour = c;
            // Both at once, and that is the point: the tick's name is the
            // legend for what is on the map, so the two can never disagree.
            if (index < m_overlayChecks.size()) {
                m_overlayChecks.at(index)->setStyleSheet(
                    QStringLiteral("color:%1;").arg(c.name()));
            }
            rebuildOverlay();
        },
        [this, index](int percent) {
            if (index < 0 || index >= int(m_overlays.size()))
                return;   // a layer removed while the wheel was open
            m_overlays[index]->sizePercent = percent;
            rebuildOverlay();
        });
}

void ViewerPage::fitOverlayPanel()
{
    if (!m_overlayPanel)
        return;
    m_overlayPanel->setVisible(!m_overlays.empty());
    // Told to catch up before being measured: adjustSize() asks the layout for
    // a size hint, and a layout that has not run since the rows changed answers
    // for the rows it had.
    if (m_overlayPanelLayout)
        m_overlayPanelLayout->activate();
    m_overlayPanel->adjustSize();
    m_overlayPanel->move(14, 14);
    m_overlayPanel->raise();
}

void ViewerPage::removeOverlay(int index)
{
    if (index < 0 || index >= int(m_overlays.size()))
        return;
    const QString label = TrajectaUi::elideForConfirm(m_overlays[index]->label);
    const QString file =
        TrajectaUi::elideForConfirm(QFileInfo(m_overlays[index]->path).fileName());
    if (!TrajectaUi::confirm(this, tr("Remove overlay"),
                             tr("Remove \"%1\" (%2) from the Viewer?\n\n"
                                "The file on disk is not deleted.")
                                 .arg(label, file))) {
        return;
    }

    dropOverlayAt(index);
    // Rebuilt rather than patched: an overlay's colour and the index its bin
    // was built with both come from its position in the list, and everything
    // after the one just removed has moved up by one.
    rebuildOverlayPanel();
    fitOverlayPanel();

    // The one that framed a vector-only scene may have just gone: let the next
    // rebuild work out the framing again from whatever is left.
    if (m_vectorScale > 0.0)
        m_vectorScale = 0.0;
    rebuildOverlay();
    if (m_overlays.empty() && m_layers.empty()) {
        m_scene->clear();
        m_pixmapItem = nullptr;
        m_overlayItems.clear();
        m_placeholder->setText(kEmptyViewerHint());
        m_canvasStack->setCurrentWidget(m_placeholder);
        m_legend->hide();
    }
    updateInfoStrip();
}

// Distinct colours so two overlays drawn at once stay tellable apart; the
// first is the theme's own accent for vector work.
// The ramp a scored point layer is drawn with. Fixed colours, like the raster
// scales and for the same reason: these stand for values, and a value that
// changes hue when the interface theme changes is not a value.
//
// Sequential for the proximity index, which runs from "no corridor in reach"
// up to whatever share the busiest neighbourhood in the layer reaches.
// Diverging for the intensity index, which is built around 50 — the score the
// average location on the surface gets — so above and below that are two
// different statements and the ramp has to break there rather than slide
// through it.
QColor ViewerPage::scoreColour(double t, bool diverging)
{
    t = std::clamp(t, 0.0, 1.0);
    const auto mix = [](const QColor &a, const QColor &b, double u) {
        return QColor::fromRgbF(a.redF() + (b.redF() - a.redF()) * u,
                                a.greenF() + (b.greenF() - a.greenF()) * u,
                                a.blueF() + (b.blueF() - a.blueF()) * u);
    };
    if (diverging) {
        static const QColor low(0xb2, 0x18, 0x2b);    // below chance
        static const QColor mid(0xf2, 0xf2, 0xf2);
        static const QColor high(0x21, 0x66, 0xac);   // above chance
        return t < 0.5 ? mix(low, mid, t * 2.0) : mix(mid, high, (t - 0.5) * 2.0);
    }
    static const QColor far(0xdc, 0xdc, 0xdc);        // no corridor in range
    static const QColor near(0x0f, 0x6b, 0x35);       // on one
    return mix(far, near, t);
}

QColor ViewerPage::overlayColor(int index) const
{
    // A colour the user picked wins over the automatic one, and keeps winning:
    // nothing later — a theme change, another layer arriving, this one moving
    // up the list — puts it back.
    if (index >= 0 && index < int(m_overlays.size())
        && m_overlays[index]->customColour.isValid()) {
        return m_overlays[index]->customColour;
    }
    return automaticOverlayColor(index);
}

QColor ViewerPage::automaticOverlayColor(int index)
{
    static const QColor kExtra[] = {QColor(0x6e, 0xc8, 0xf0), QColor(0xd9, 0x8b, 0xf9),
                                    QColor(0x8f, 0xd9, 0x8a), QColor(0xff, 0x8a, 0x8a)};
    if (index <= 0)
        return overlayPen();
    return kExtra[(index - 1) % 4];
}

void ViewerPage::openRasterFile(const QString &path)
{
    registerRaster(QFileInfo(path).completeBaseName(), path, true);
}

// ---------------------------------------------------------------- drag & drop

namespace {

// The local files in a drag, in the order they were dragged. Anything that is
// not a file — a URL from a browser, dragged text — is dropped here rather than
// reported: it was never a layer and saying so would be noise.
QStringList droppedFiles(const QMimeData *mime)
{
    QStringList files;
    if (!mime || !mime->hasUrls())
        return files;
    const QList<QUrl> urls = mime->urls();
    for (const QUrl &url : urls) {
        const QString local = url.toLocalFile();
        if (!local.isEmpty())
            files << local;
    }
    return files;
}

} // namespace

void ViewerPage::dragEnterEvent(QDragEnterEvent *event)
{
    if (droppedFiles(event->mimeData()).isEmpty())
        return;                       // not files: leave the drag to whatever else wants it
    event->acceptProposedAction();
    // The canvas lights up while the drag is over the page, so that the drop
    // is known to be possible before the button is released.
    if (m_canvasHolder) {
        m_canvasHolder->setProperty("dropTarget", true);
        m_canvasHolder->style()->unpolish(m_canvasHolder);
        m_canvasHolder->style()->polish(m_canvasHolder);
    }
}

void ViewerPage::dragLeaveEvent(QDragLeaveEvent *event)
{
    Q_UNUSED(event);
    if (m_canvasHolder) {
        m_canvasHolder->setProperty("dropTarget", false);
        m_canvasHolder->style()->unpolish(m_canvasHolder);
        m_canvasHolder->style()->polish(m_canvasHolder);
    }
}

void ViewerPage::dropEvent(QDropEvent *event)
{
    dragLeaveEvent(nullptr);          // the highlight goes with the drag

    const QStringList files = droppedFiles(event->mimeData());
    if (files.isEmpty())
        return;
    event->acceptProposedAction();

    // Everything that fails is collected and reported once. Three refusals in
    // a row, each in its own box, is how a user learns to dismiss boxes without
    // reading them.
    QStringList refused;
    int loaded = 0;
    for (const QString &path : files) {
        QString error;
        if (openAnyFile(path, &error)) {
            ++loaded;
        } else {
            refused << tr("• %1 — %2")
                           .arg(QFileInfo(path).fileName(),
                                error.simplified());
        }
    }

    if (!refused.isEmpty()) {
        TrajectaUi::notify(
            this,
            loaded > 0 ? tr("Some layers could not be opened")
                       : tr("Nothing could be opened"),
            (refused.size() == 1
                 ? tr("This file could not be read as a raster or as a vector "
                      "layer:\n\n")
                 : tr("%1 files could not be read as a raster or as a vector "
                      "layer:\n\n").arg(refused.size()))
                + refused.join(QLatin1Char('\n')),
            QString(), 60);
    }
}

bool ViewerPage::openAnyFile(const QString &path, QString *error)
{
    if (!QFileInfo::exists(path)) {
        if (error)
            *error = tr("The file does not exist:\n%1").arg(QDir::toNativeSeparators(path));
        return false;
    }
    if (!ensureGdal()) {
        if (error) {
            *error = tr("GDAL could not be loaded, so no file can be read.\n\n"
                        "Use \"Locate GDAL folder...\" in the status bar.");
        }
        return false;
    }

    // Content decides. Asking GDAL to open it as a raster and then as a vector
    // is the only test that is right for every file: a shapefile has no bands,
    // a GeoTIFF has no layers, and an extension is only a hint about either.
    GdalApi &api = GdalApi::instance();
    const QByteArray pathUtf8 = path.toUtf8();
    if (GDALDatasetH ds = api.OpenEx(pathUtf8.constData(), GdalApi::OF_Raster,
                                     nullptr, nullptr, nullptr)) {
        const bool hasBand = api.GetRasterBand(ds, 1) != nullptr;
        api.Close(ds);
        if (hasBand) {
            openRasterFile(path);
            return true;
        }
    }

    if (GDALDatasetH ds = openVectorDataset(path)) {
        const int layers = api.DatasetGetLayerCount(ds);
        api.Close(ds);
        if (layers > 0) {
            registerVectorOverlay(QFileInfo(path).completeBaseName(), path);
            // registerVectorOverlay reads the geometry through rebuildOverlay();
            // if nothing drawable came out, say so instead of adding a tick box
            // for an overlay that will never appear.
            const int index = int(m_overlays.size()) - 1;
            if (index >= 0 && m_overlays[index]->path == path
                && m_overlays[index]->loaded && m_overlays[index]->failed) {
                dropOverlayAt(index);
                rebuildOverlayPanel();
                fitOverlayPanel();
                if (error) {
                    *error = tr("\"%1\" was read, but it holds no points, lines "
                                "or polygons that can be drawn.")
                                 .arg(QFileInfo(path).fileName());
                }
                return false;
            }
            return true;
        }
    }

    if (error) {
        *error = tr("\"%1\" could not be read as a raster or as a vector "
                    "layer.\n\nGeoTIFF and the usual vector formats "
                    "(Shapefile, GeoPackage, GeoJSON, KML, GML, CSV with "
                    "coordinates, MapInfo, DXF) are all accepted.")
                     .arg(QFileInfo(path).fileName());
    }
    return false;
}

RasterLayer *ViewerPage::currentLayer() const
{
    const int i = m_layerCombo->currentIndex();
    if (i < 0 || i >= int(m_layers.size()))
        return nullptr;
    RasterLayer *layer = m_layers[i].get();
    return (layer->loaded && !layer->failed) ? layer : nullptr;
}

namespace {

// Loads metadata + a decimated display buffer + stats. Returns false and
// flags the layer on any GDAL failure.
bool loadLayer(RasterLayer &layer)
{
    GdalApi &api = GdalApi::instance();
    const QByteArray pathUtf8 = layer.path.toUtf8();
    layer.ds = api.OpenEx(pathUtf8.constData(), GdalApi::OF_Raster,
                          nullptr, nullptr, nullptr);
    if (!layer.ds)
        return false;

    layer.srcW = api.GetRasterXSize(layer.ds);
    layer.srcH = api.GetRasterYSize(layer.ds);
    if (layer.srcW <= 0 || layer.srcH <= 0)
        return false;
    if (api.GetGeoTransform(layer.ds, layer.gt) != 0) {
        const double identity[6] = {0, 1, 0, 0, 0, -1};
        std::copy(identity, identity + 6, layer.gt);
    }

    GDALRasterBandH band = api.GetRasterBand(layer.ds, 1);
    if (!band)
        return false;
    int hasNoData = 0;
    layer.noData = api.GetRasterNoDataValue(band, &hasNoData);
    layer.hasNoData = hasNoData != 0;

    const double scale =
        std::min(1.0, double(kMaxDisplayDim) / std::max(layer.srcW, layer.srcH));
    layer.dispW = std::max(1, int(std::lround(layer.srcW * scale)));
    layer.dispH = std::max(1, int(std::lround(layer.srcH * scale)));
    layer.data.resize(layer.dispW * layer.dispH);
    if (api.RasterIO(band, GdalApi::ReadFlag, 0, 0, layer.srcW, layer.srcH,
                     layer.data.data(), layer.dispW, layer.dispH,
                     GdalApi::Float32, 0, 0) != 0)
        return false;

    // Stats over valid cells.
    double minV = std::numeric_limits<double>::infinity();
    double maxV = -std::numeric_limits<double>::infinity();
    for (const float v : layer.data) {
        if (std::isnan(v) || (layer.hasNoData && double(v) == layer.noData))
            continue;
        minV = std::min(minV, double(v));
        maxV = std::max(maxV, double(v));
    }
    if (!std::isfinite(minV)) {
        minV = 0.0;
        maxV = 1.0;
    }
    if (maxV <= minV)
        maxV = minV + 1.0;
    layer.minV = minV;
    layer.maxV = maxV;

    QVector<qint64> hist(kHistBins, 0);
    const double invRange = kHistBins / (maxV - minV);
    for (const float v : layer.data) {
        if (std::isnan(v) || (layer.hasNoData && double(v) == layer.noData))
            continue;
        const int bin = std::clamp(int((double(v) - minV) * invRange), 0,
                                   kHistBins - 1);
        ++hist[bin];
    }

    // Exact quantiles from a sorted sample: linear-binned CDFs are useless on
    // heavily skewed data (a FETE density raster packs >98% of its cells into
    // the first bin, degenerating percentile stretch to min-max).
    QVector<float> sample;
    const qsizetype total = layer.data.size();
    const qsizetype stride = std::max<qsizetype>(1, total / 600000);
    sample.reserve(int(total / stride) + 1);
    qsizetype zeroCount = 0;
    for (qsizetype i = 0; i < total; i += stride) {
        const float v = layer.data[i];
        if (std::isnan(v) || (layer.hasNoData && double(v) == layer.noData))
            continue;
        sample.append(v);
        if (v == 0.0f)
            ++zeroCount;
    }
    // Sparse rasters (path density) are mostly zero background; quantiles of
    // the zeros say nothing, so compute them over the traversed cells only.
    if (zeroCount > sample.size() / 2 && zeroCount < sample.size()) {
        QVector<float> nonZero;
        nonZero.reserve(int(sample.size() - zeroCount));
        for (const float v : sample) {
            if (v != 0.0f)
                nonZero.append(v);
        }
        sample = nonZero;
    }
    std::sort(sample.begin(), sample.end());
    layer.sortedSample = sample;

    auto pctValue = [&sample, minV, maxV](double pct) {
        if (sample.isEmpty())
            return pct < 50.0 ? minV : maxV;
        const int idx = std::clamp(
            int(std::lround(pct / 100.0 * (sample.size() - 1))),
            0, int(sample.size()) - 1);
        return double(sample.at(idx));
    };
    layer.p2 = pctValue(2.0);
    layer.p98 = pctValue(98.0);
    if (layer.p98 <= layer.p2) {
        layer.p2 = minV;
        layer.p98 = maxV;
    }

    // Log-scaled slider histogram (sqrt would hide sparse-density layers).
    layer.sliderHist.resize(kSliderBins);
    double histMax = 0.0;
    QVector<double> merged(kSliderBins, 0.0);
    const int group = kHistBins / kSliderBins;
    for (int i = 0; i < kSliderBins; ++i) {
        double sum = 0.0;
        for (int j = 0; j < group; ++j)
            sum += double(hist[i * group + j]);
        merged[i] = std::log1p(sum);
        histMax = std::max(histMax, merged[i]);
    }
    for (int i = 0; i < kSliderBins; ++i)
        layer.sliderHist[i] = histMax > 0 ? float(merged[i] / histMax) : 0.0f;

    // CRS description.
    const char *wkt = api.GetProjectionRef(layer.ds);
    if (wkt && *wkt) {
        layer.wkt = QString::fromUtf8(wkt);
        OGRSpatialReferenceH srs = api.OSRNewSpatialReference(wkt);
        if (srs) {
            const char *name = api.OSRGetName(srs);
            const char *authName = api.OSRGetAuthorityName(srs, nullptr);
            const char *authCode = api.OSRGetAuthorityCode(srs, nullptr);
            layer.crsName = name ? QString::fromUtf8(name) : QString();
            if (authName && authCode) {
                layer.crsName += QStringLiteral(" (%1:%2)")
                                     .arg(QString::fromUtf8(authName),
                                          QString::fromUtf8(authCode));
            }
            layer.geographic = api.OSRIsGeographic(srs) != 0;
            api.OSRDestroySpatialReference(srs);
        }
    }
    if (layer.crsName.isEmpty())
        layer.crsName = QObject::tr("No CRS");

    return true;
}


// Scene point -> the layer's own map coordinates. Scene units are the
// *decimated* display grid, so the source pixel is recovered first and only
// then run through the geotransform.
QPointF frameSceneToMap(const ViewFrame &f, const QPointF &scenePt)
{
    const double srcPx = scenePt.x() * f.srcW / f.dispW;
    const double srcPy = scenePt.y() * f.srcH / f.dispH;
    return QPointF(f.gt[0] + srcPx * f.gt[1] + srcPy * f.gt[2],
                   f.gt[3] + srcPx * f.gt[4] + srcPy * f.gt[5]);
}

// The inverse. False when the geotransform cannot be inverted — no raster GDAL
// will read has a degenerate one, but the arithmetic has to say so rather than
// divide by zero.
bool frameMapToScene(const ViewFrame &f, const QPointF &mapPt, QPointF &scenePt)
{
    const double det = f.gt[1] * f.gt[5] - f.gt[2] * f.gt[4];
    if (det == 0.0 || f.srcW <= 0 || f.srcH <= 0)
        return false;
    const double dx = mapPt.x() - f.gt[0];
    const double dy = mapPt.y() - f.gt[3];
    const double px = (f.gt[5] * dx - f.gt[2] * dy) / det;
    const double py = (-f.gt[4] * dx + f.gt[1] * dy) / det;
    scenePt = QPointF(px * f.dispW / f.srcW, py * f.dispH / f.srcH);
    return true;
}

ViewFrame frameOf(const RasterLayer &l)
{
    ViewFrame f;
    f.valid = l.loaded && !l.failed
              && l.srcW > 0 && l.srcH > 0 && l.dispW > 0 && l.dispH > 0;
    for (int i = 0; i < 6; ++i)
        f.gt[i] = l.gt[i];
    f.srcW = l.srcW;
    f.srcH = l.srcH;
    f.dispW = l.dispW;
    f.dispH = l.dispH;
    f.wkt = l.wkt;
    return f;
}

} // namespace

void ViewerPage::selectLayer(int comboIndex)
{
    if (comboIndex < 0 || comboIndex >= int(m_layers.size()))
        return;
    RasterLayer *layer = m_layers[comboIndex].get();

    if (!layer->loaded) {
        if (!ensureGdal())
            return;
        QApplication::setOverrideCursor(Qt::WaitCursor);
        layer->failed = !loadLayer(*layer);
        layer->loaded = true;
        QApplication::restoreOverrideCursor();
    }
    if (layer->failed) {
        m_placeholder->setText(tr("Could not read:\n%1").arg(layer->path));
        m_canvasStack->setCurrentWidget(m_placeholder);
        m_legend->hide();
        updateInfoStrip();
        return;
    }

    // Where the view is looking at this moment, held as a place on the ground
    // rather than a place on the scene, so it can be put back on top of the
    // next layer. This is what makes the Viewer comparative: zoom in on a
    // detail, change raster, and the detail is still under the cursor instead
    // of the whole extent being framed again.
    //
    // Two things are held. The centre, in map coordinates — scene units are the
    // layer's own decimated pixels and mean different distances on different
    // layers, so they cannot be carried across as they are. And the ground
    // covered by one pixel of the window, which is the magnification as the eye
    // understands it; carrying the raw zoom factor instead would show a
    // different area whenever the two rasters were decimated differently.
    //
    // Same CRS only. Reprojecting the centre would be possible, but comparing
    // rasters in different systems is not what this is for, and falling back to
    // "fit" is honest where a silently wrong centre would not be.
    bool keepView = false;
    QPointF heldCentreMap;
    double heldUnitsPerWindowPx = 0.0;
    if (m_shownFrame.valid && m_shownFrame.wkt == layer->wkt) {
        const double viewScale = m_view->transform().m11();
        if (viewScale > 0.0 && m_view->unitsPerScenePx() > 0.0) {
            heldCentreMap = frameSceneToMap(m_shownFrame, m_view->centreInScene());
            heldUnitsPerWindowPx = m_view->unitsPerScenePx() / viewScale;
            keepView = true;
        }
    }

    // Fresh scene for the new layer. Tile items and reprojection transforms
    // belong to the previous layer: drop them before the scene wipes them.
    clearBasemap(true);
    m_scene->clear();
    m_pixmapItem = nullptr;
    m_overlayItems.clear();
    m_scene->setSceneRect(0, 0, layer->dispW, layer->dispH);

    // Full range filter on layer switch.
    m_filterLo = layer->minV;
    m_filterHi = layer->maxV;
    m_updatingUi = true;
    m_rangeSlider->setRange(0.0, 1.0);
    m_rangeSlider->setHistogram(layer->sliderHist);
    m_updatingUi = false;
    updateFilterUi();

    rebuildImage();
    rebuildOverlay();

    const double unitsPerSrcPx = std::hypot(layer->gt[1], layer->gt[4]);
    m_view->setUnitsPerScenePixel(
        unitsPerSrcPx * layer->srcW / layer->dispW, layer->geographic);
    m_canvasStack->setCurrentWidget(m_view);
    m_canvasStack->activate();   // real size before fitting; see rebuildOverlay()

    // Back to the same ground at the same magnification, expressed on the new
    // layer's pixel grid. The containment test is the other half of it: a
    // raster that does not cover where the eye was is better framed whole than
    // scrolled to an empty corner of itself.
    const ViewFrame frame = frameOf(*layer);
    QPointF centreScene;
    if (keepView && frameMapToScene(frame, heldCentreMap, centreScene)
        && QRectF(0, 0, frame.dispW, frame.dispH).contains(centreScene)) {
        m_view->showAt(centreScene, m_view->unitsPerScenePx() / heldUnitsPerWindowPx);
    } else {
        m_view->fitAll();
        // A run that finishes while the user is on another page fills the
        // Viewer behind their back, and fitting to a page that is not on screen
        // frames the layer for a widget still at its default size — which is
        // why the map then opened as a thumbnail in the middle of the canvas.
        m_framedWhileHidden = !isVisible();
    }
    m_shownFrame = frame;
    updateInfoStrip();

    m_basemapToggle->setEnabled(!layer->wkt.isEmpty());
    if (m_basemapToggle->isChecked() && m_basemapToggle->isEnabled())
        updateBasemap();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
void ViewerPage::scheduleRebuild()
{
    m_rebuildTimer->start();
}

void ViewerPage::rebuildImage()
{
    RasterLayer *layer = currentLayer();
    if (!layer)
        return;

    // Stretch window.
    double lo = layer->minV, hi = layer->maxV;
    if (m_stretchIndex == 1) {
        lo = layer->p2;
        hi = layer->p98;
    }
    const double range = hi > lo ? hi - lo : 1.0;
    constexpr double kLog = 200.0;
    const double logDen = std::log1p(kLog);

    const bool hillshade = m_colormapIndex >= int(colormaps().size());
    const QVector<QRgb> &lut =
        colormaps().at(hillshade ? 0 : m_colormapIndex).lut;
    QImage img(layer->dispW, layer->dispH, QImage::Format_ARGB32);

    // Hillshade: Horn's 3x3 gradient with the conventional 315 deg / 45 deg
    // light. Computed on the decimated display buffer, so the cell size has to
    // be the decimated one or the relief comes out far too strong.
    const double cellX = std::abs(layer->gt[1]) * double(layer->srcW) / layer->dispW;
    const double cellY = std::abs(layer->gt[5]) * double(layer->srcH) / layer->dispH;
    // The light is quoted the cartographic way: 315 deg is a compass bearing
    // (from the north-west), 45 deg is its height above the horizon. The
    // trigonometry below wants a mathematical angle, measured anticlockwise
    // from east, so the bearing has to be converted — feeding the compass
    // value straight in turns the light round by 180 deg and the relief comes
    // out inverted, ridges reading as valleys.
    constexpr double kAzimuthDeg = 315.0;
    constexpr double kAltitudeDeg = 45.0;
    const double azimuthMath = std::fmod(360.0 - kAzimuthDeg + 90.0, 360.0);
    const double kAzimuth = azimuthMath * M_PI / 180.0;
    const double kZenith = (90.0 - kAltitudeDeg) * M_PI / 180.0;

    const auto sample = [&](int x, int y) {
        x = std::clamp(x, 0, layer->dispW - 1);
        y = std::clamp(y, 0, layer->dispH - 1);
        return double(layer->data.at(qsizetype(y) * layer->dispW + x));
    };

    for (int y = 0; y < layer->dispH; ++y) {
        QRgb *out = reinterpret_cast<QRgb *>(img.scanLine(y));
        const float *in = layer->data.constData() + qsizetype(y) * layer->dispW;
        for (int x = 0; x < layer->dispW; ++x) {
            const double v = in[x];
            if (std::isnan(in[x])
                || (layer->hasNoData && v == layer->noData)
                || v < m_filterLo || v > m_filterHi) {
                out[x] = qRgba(0, 0, 0, 0);
                continue;
            }
            if (hillshade) {
                const double a = sample(x - 1, y - 1), b = sample(x, y - 1), c = sample(x + 1, y - 1);
                const double d = sample(x - 1, y),                            f = sample(x + 1, y);
                const double g = sample(x - 1, y + 1), h = sample(x, y + 1), i = sample(x + 1, y + 1);
                const double dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cellX);
                const double dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * cellY);
                const double slope = std::atan(std::hypot(dzdx, dzdy));
                const double aspect = std::atan2(dzdy, -dzdx);
                double shade = std::cos(kZenith) * std::cos(slope)
                               + std::sin(kZenith) * std::sin(slope)
                                     * std::cos(kAzimuth - aspect);
                shade = std::clamp(shade, 0.0, 1.0);
                const int grey = int(std::lround(shade * 255.0));
                out[x] = qRgb(grey, grey, grey);
                continue;
            }
            double t = std::clamp((v - lo) / range, 0.0, 1.0);
            if (m_stretchIndex == 2)
                t = std::log1p(kLog * t) / logDen;
            out[x] = lut.at(int(t * 255.0));
        }
    }

    if (!m_pixmapItem) {
        m_pixmapItem = m_scene->addPixmap(QPixmap::fromImage(img));
        m_pixmapItem->setZValue(0);
    } else {
        m_pixmapItem->setPixmap(QPixmap::fromImage(img));
    }
    m_pixmapItem->setOpacity(m_opacitySlider->value() / 100.0);

    // A value ramp next to a relief image would claim the greys mean
    // elevation, which they do not: they mean illumination.
    // setState() first: the legend now sizes itself to its labels, and
    // positionLegend() measures from its right edge. The other way round it
    // would sit a label's width out of place until the next resize.
    m_legend->setState(lut, lo, hi);
    positionLegend();
    m_legend->setVisible(!hillshade);
    m_legend->raise();
}

void ViewerPage::rebuildOverlay()
{
    qDeleteAll(m_overlayItems);
    m_overlayItems.clear();

    // Read anything ticked that has not been read yet. Done before the frame is
    // decided, because with no raster loaded it is the vectors themselves that
    // define it.
    for (int oi = 0; oi < int(m_overlays.size()); ++oi) {
        if (oi >= m_overlayChecks.size() || !m_overlayChecks.at(oi)->isChecked())
            continue;
        VectorOverlay *overlay = m_overlays[oi].get();
        if (overlay->loaded)
            continue;
        if (!ensureGdal())
            return;
        overlay->loaded = true;
        overlay->failed = !loadOverlayGeometry(*overlay);
    }

    const auto drawable = [this](int oi) {
        if (oi >= m_overlayChecks.size() || !m_overlayChecks.at(oi)->isChecked())
            return false;
        const VectorOverlay *o = m_overlays[oi].get();
        return o->loaded && !o->failed
               && (!o->lines.isEmpty() || !o->points.isEmpty());
    };

    // Map coordinates -> scene coordinates, and the CRS those map coordinates
    // are expected to be in.
    QString targetWkt;
    std::function<QPointF(const QPointF &)> toScene;

    if (RasterLayer *layer = currentLayer()) {
        // Map coords -> source pixel (inverse geotransform) -> scene (display px).
        const double *gt = layer->gt;
        const double det = gt[1] * gt[5] - gt[2] * gt[4];
        if (det == 0.0)
            return;
        const double sx = double(layer->dispW) / layer->srcW;
        const double sy = double(layer->dispH) / layer->srcH;
        const double gt0 = gt[0], gt2 = gt[2], gt3 = gt[3], gt4 = gt[4],
                     gt1 = gt[1], gt5 = gt[5];
        targetWkt = layer->wkt;
        toScene = [=](const QPointF &mapPt) {
            const double dx = mapPt.x() - gt0;
            const double dy = mapPt.y() - gt3;
            const double px = (gt5 * dx - gt2 * dy) / det;
            const double py = (-gt4 * dx + gt1 * dy) / det;
            return QPointF(px * sx, py * sy);
        };
        m_vectorScale = 0.0;   // a raster owns the scene again
    } else {
        // Vectors on their own: they get a scene of their own, framed on their
        // shared extent. Without this an imported vector would open onto the
        // "no raster loaded" placeholder and look like a failed import.
        int first = -1;
        for (int oi = 0; oi < int(m_overlays.size()); ++oi) {
            if (drawable(oi)) {
                first = oi;
                break;
            }
        }
        if (first < 0) {
            m_vectorScale = 0.0;
            return;
        }
        targetWkt = m_overlays[first]->wkt;

        BoundsAccumulator bounds;
        for (int oi = 0; oi < int(m_overlays.size()); ++oi) {
            if (!drawable(oi))
                continue;
            projectOverlay(*m_overlays[oi], targetWkt);
            if (!m_overlays[oi]->hasDrawExtent)
                continue;
            const QRectF &e = m_overlays[oi]->drawExtent;
            bounds.add(e.topLeft());
            bounds.add(e.bottomRight());
        }
        if (!bounds.valid) {
            // Everything ticked failed to project. Forget the frame rather
            // than leaving the previous one behind: the info strip and the
            // cursor readout both invert it, and a stale one lies to them.
            m_vectorScale = 0.0;
            return;
        }
        QRectF extent = bounds.rect();
        // A single point, or a perfectly straight line, has no area to frame:
        // give it one, so the fit does not divide by zero.
        if (extent.width() <= 0.0 || extent.height() <= 0.0) {
            const double pad = std::max({extent.width(), extent.height(), 1.0}) * 0.5;
            extent.adjust(-pad, -pad, pad, pad);
        }

        // Scene units are arbitrary; ~1600 across keeps a cosmetic pen and the
        // scalebar in the same range they are in over a raster.
        constexpr double kVectorSceneSpan = 1600.0;
        const double scale = kVectorSceneSpan / std::max(extent.width(), extent.height());
        const double left = extent.left();
        const double bottom = extent.bottom();
        toScene = [=](const QPointF &mapPt) {
            return QPointF((mapPt.x() - left) * scale, (bottom - mapPt.y()) * scale);
        };

        const bool frameChanged = m_vectorScale <= 0.0 || m_vectorExtent != extent
                                  || m_vectorWkt != targetWkt;
        m_vectorScale = scale;
        m_vectorExtent = extent;
        m_vectorWkt = targetWkt;
        if (frameChanged) {
            // Only when the framing actually moved: a tick box toggled while
            // the user is zoomed in must not throw the view back to "fit".
            m_scene->setSceneRect(0, 0, extent.width() * scale,
                                  extent.height() * scale);
            m_view->setUnitsPerScenePixel(1.0 / scale,
                                          m_overlays[first]->geographic);
            m_canvasStack->setCurrentWidget(m_view);
            m_legend->hide();
            // The view is the hidden page of a stack until this moment, so it
            // still carries its default size; fitting to that would frame the
            // layer for a window 100 pixels wide.
            m_canvasStack->activate();
            m_view->fitAll();
            m_framedWhileHidden = !isVisible();
            updateInfoStrip();
        }
    }

    for (int oi = 0; oi < int(m_overlays.size()); ++oi) {
        if (!drawable(oi))
            continue;
        VectorOverlay *overlay = m_overlays[oi].get();
        // Onto whatever the frame is in. Cached, so this is a no-op except on
        // the first draw and after a layer switch that changes the CRS.
        projectOverlay(*overlay, targetWkt);
        if (overlay->drawLines.isEmpty() && overlay->drawPoints.isEmpty())
            continue;

        // The scene coordinates are kept as well as drawn: a click has to be
        // answered against the same geometry that is on screen, and projecting
        // a hundred thousand points again on every press would be felt.
        overlay->sceneLines.clear();
        overlay->scenePoints.clear();
        overlay->sceneLines.reserve(overlay->drawLines.size());
        overlay->scenePoints.reserve(overlay->drawPoints.size());

        QPainterPath path;
        for (const QPolygonF &line : std::as_const(overlay->drawLines)) {
            QPolygonF sceneLine;
            sceneLine.reserve(line.size());
            for (int i = 0; i < line.size(); ++i) {
                const QPointF scenePt = toScene(line[i]);
                sceneLine.append(scenePt);
                if (i == 0)
                    path.moveTo(scenePt);
                else
                    path.lineTo(scenePt);
            }
            overlay->sceneLines.append(sceneLine);
        }
        const bool hasPoints = !overlay->drawPoints.isEmpty();
        for (const QPointF &pt : std::as_const(overlay->drawPoints))
            overlay->scenePoints.append(toScene(pt));

        // A site marker is a target as well as a mark: at four pixels it was a
        // speck on a dense raster and all but impossible to aim a click at, so
        // it is drawn at twice that. Lines keep their own weight — a route is
        // read by its course, not by its thickness — which is why the two no
        // longer share a pen. Both then scale with the layer's own size
        // setting, so a route or a site set can still be made thicker or
        // thinner without changing what everyone else is drawn at.
        constexpr double kLineWidth = 1.6;
        constexpr double kPointWidth = 8.0;
        const double scale = std::clamp(overlay->sizePercent, 25, 400) / 100.0;
        const double lineWidth = kLineWidth * scale;
        const double pointWidth = kPointWidth * scale;

        if (overlay->colourField >= 0 && hasPoints
            && overlay->pointValues.size() == overlay->scenePoints.size()) {
            // Coloured by score. The points are bucketed and one path is built
            // per bucket, because a QGraphicsPathItem carries a single pen:
            // seven items instead of one, rather than one item per point, which
            // is what makes a layer of a hundred thousand sites unusable.
            constexpr int kBuckets = 7;
            QVector<QPainterPath> buckets(kBuckets);
            QPainterPath unknown;
            for (int i = 0; i < overlay->scenePoints.size(); ++i) {
                const double v = overlay->pointValues.at(i);
                const QPointF scenePt = overlay->scenePoints.at(i);
                QPainterPath &target =
                    std::isnan(v) ? unknown
                                  : buckets[std::clamp(
                                        int((v - overlay->colourLo)
                                            / std::max(1e-9, overlay->colourHi
                                                                 - overlay->colourLo)
                                            * kBuckets),
                                        0, kBuckets - 1)];
                target.moveTo(scenePt);
                target.lineTo(scenePt + QPointF(1e-3, 0.0));
            }
            for (int b = 0; b < kBuckets; ++b) {
                if (buckets[b].isEmpty())
                    continue;
                const double t = (b + 0.5) / kBuckets;
                QPen pen(scoreColour(t, overlay->colourDiverging), pointWidth + 1.0);
                pen.setCosmetic(true);
                pen.setCapStyle(Qt::RoundCap);
                QGraphicsPathItem *item = m_scene->addPath(buckets[b], pen);
                item->setZValue(1);
                m_overlayItems.append(item);
            }
            if (!unknown.isEmpty()) {
                QPen pen(QColor(0x9e, 0x9e, 0x9e), pointWidth);
                pen.setCosmetic(true);
                pen.setCapStyle(Qt::RoundCap);
                QGraphicsPathItem *item = m_scene->addPath(unknown, pen);
                item->setZValue(1);
                m_overlayItems.append(item);
            }
            // Lines, if the same layer has any, keep the layer's own colour.
            if (!path.isEmpty()) {
                QPen pen(overlayColor(oi), lineWidth);
                pen.setCosmetic(true);
                QGraphicsPathItem *item = m_scene->addPath(path, pen);
                item->setZValue(1);
                m_overlayItems.append(item);
            }
            continue;
        }

        // Points are drawn as near-zero-length subpaths: with a round-capped
        // cosmetic pen each one renders as a dot of constant screen size, so
        // the markers neither vanish when zooming out nor swell when zooming
        // in. One path item per layer keeps even 100k+ points responsive.
        QPainterPath dots;
        for (const QPointF &scenePt : std::as_const(overlay->scenePoints)) {
            dots.moveTo(scenePt);
            dots.lineTo(scenePt + QPointF(1e-3, 0.0));
        }

        if (!path.isEmpty()) {
            QPen pen(overlayColor(oi), lineWidth);
            pen.setCosmetic(true);
            QGraphicsPathItem *item = m_scene->addPath(path, pen);
            item->setZValue(1);
            m_overlayItems.append(item);
        }
        if (!dots.isEmpty()) {
            QPen pen(overlayColor(oi), pointWidth);
            pen.setCosmetic(true);
            pen.setCapStyle(Qt::RoundCap);
            QGraphicsPathItem *item = m_scene->addPath(dots, pen);
            item->setZValue(1);
            m_overlayItems.append(item);
        }
    }
}

// ---------------------------------------------------------------------------
// Filter UI
// ---------------------------------------------------------------------------
double ViewerPage::percentileToValue(const RasterLayer &layer, double pct) const
{
    const QVector<float> &s = layer.sortedSample;
    if (s.isEmpty())
        return pct < 50.0 ? layer.minV : layer.maxV;
    const int idx = std::clamp(
        int(std::lround(std::clamp(pct, 0.0, 100.0) / 100.0 * (s.size() - 1))),
        0, int(s.size()) - 1);
    return double(s.at(idx));
}

double ViewerPage::valueToPercentile(const RasterLayer &layer, double v) const
{
    const QVector<float> &s = layer.sortedSample;
    if (s.isEmpty())
        return 0.0;
    const auto it = std::upper_bound(s.begin(), s.end(), float(v));
    return 100.0 * double(it - s.begin()) / double(s.size());
}

void ViewerPage::updateFilterUi()
{
    RasterLayer *layer = currentLayer();
    if (!layer)
        return;
    m_updatingUi = true;

    const double range = layer->maxV - layer->minV;
    m_rangeSlider->setRange((m_filterLo - layer->minV) / range,
                            (m_filterHi - layer->minV) / range);

    for (QDoubleSpinBox *spin : {m_filterLoSpin, m_filterHiSpin}) {
        if (m_percentMode) {
            spin->setRange(0.0, 100.0);
            spin->setDecimals(1);
            spin->setSingleStep(1.0);
            spin->setSuffix(QStringLiteral(" %"));
        } else {
            spin->setRange(layer->minV, layer->maxV);
            spin->setDecimals(range >= 1000 ? 1 : (range >= 10 ? 2 : 4));
            spin->setSingleStep(range / 100.0);
            spin->setSuffix(QString());
        }
    }
    if (m_percentMode) {
        m_filterLoSpin->setValue(valueToPercentile(*layer, m_filterLo));
        m_filterHiSpin->setValue(valueToPercentile(*layer, m_filterHi));
    } else {
        m_filterLoSpin->setValue(m_filterLo);
        m_filterHiSpin->setValue(m_filterHi);
    }
    m_updatingUi = false;
}

void ViewerPage::applyFilterFromSpins()
{
    RasterLayer *layer = currentLayer();
    if (!layer)
        return;
    double lo = m_filterLoSpin->value();
    double hi = m_filterHiSpin->value();
    if (m_percentMode) {
        lo = percentileToValue(*layer, lo);
        hi = percentileToValue(*layer, hi);
    }
    m_filterLo = std::min(lo, hi);
    m_filterHi = std::max(lo, hi);
    updateFilterUi();
    scheduleRebuild();
}

// ---------------------------------------------------------------------------
// Info strip / hover
// ---------------------------------------------------------------------------
void ViewerPage::updateInfoStrip()
{
    RasterLayer *layer = currentLayer();
    if (!layer) {
        // Vectors on their own still have a system and an extent to report;
        // "Resolution" is a raster idea, so it says what the file is instead.
        if (m_vectorScale > 0.0) {
            QString crs;
            int shown = 0;
            for (int oi = 0; oi < int(m_overlays.size()); ++oi) {
                if (oi >= m_overlayChecks.size() || !m_overlayChecks.at(oi)->isChecked())
                    continue;
                const VectorOverlay &o = *m_overlays[oi];
                if (!o.loaded || o.failed)
                    continue;
                ++shown;
                if (crs.isEmpty())
                    crs = o.crsName;
            }
            m_crsLabel->setText(tr("CRS: %1").arg(crs.isEmpty() ? tr("No CRS") : crs));
            m_resLabel->setText(shown == 1 ? tr("Vector layer")
                                           : tr("%1 vector layers").arg(shown));
            m_cursorLabel->clear();
            repositionAttribution();
            return;
        }
        m_crsLabel->setText(tr("CRS: —"));
        m_resLabel->setText(tr("Resolution: —"));
        m_cursorLabel->clear();
        repositionAttribution();
        return;
    }
    m_crsLabel->setText(tr("CRS: %1").arg(layer->crsName));
    const double res = std::hypot(layer->gt[1], layer->gt[4]);
    m_resLabel->setText(tr("Resolution: %1%2/cell")
                            .arg(QString::number(res, 'g', 5),
                                 layer->geographic ? QStringLiteral("°")
                                                   : QStringLiteral(" m")));
    // Both labels just changed width, and repositionAttribution()'s clamp
    // against m_resLabel's edge is only as good as that edge is current.
    repositionAttribution();
}

// ---------------------------------------------------------------------------
// Feature information
// ---------------------------------------------------------------------------
//
// A click on a point or a line opens a small panel in the bottom-right corner
// of the map with that feature's own attributes. It is the question a map
// invites — "what is this one?" — and until now the Viewer could only answer it
// for raster cells, under the cursor, in the strip below.
//
void ViewerPage::buildFeaturePanel()
{
    m_featurePanel = new QFrame(m_view);
    // Its own name, styled to match the Overlays panel on the opposite corner:
    // the two are the same kind of thing — a small pane floating over the map —
    // and the outline is what separates this one from the raster behind it.
    m_featurePanel->setObjectName(QStringLiteral("FeaturePanel"));
    m_featurePanel->hide();
    auto *layout = new QVBoxLayout(m_featurePanel);
    layout->setContentsMargins(12, 8, 8, 10);
    layout->setSpacing(4);

    auto *head = new QHBoxLayout;
    head->setSpacing(6);
    m_featureTitle = new QLabel(m_featurePanel);
    m_featureTitle->setObjectName(QStringLiteral("CardTitle"));
    head->addWidget(m_featureTitle, 1);
    auto *close = new QToolButton(m_featurePanel);
    close->setObjectName(QStringLiteral("TourClose"));
    close->setText(QStringLiteral("✕"));
    close->setCursor(Qt::PointingHandCursor);
    close->setAutoRaise(true);
    connect(close, &QToolButton::clicked, this, [this] { m_featurePanel->hide(); });
    head->addWidget(close, 0, Qt::AlignTop);
    layout->addLayout(head);

    // Inside a scroll area, and bounded. A site layer can carry thirty columns
    // and one of them can hold a paragraph of references: unbounded, the panel
    // grew until it covered the map it was describing, which is the one thing
    // it must never do.
    m_featureBody = new QLabel(m_featurePanel);
    m_featureBody->setObjectName(QStringLiteral("HintLabel"));
    m_featureBody->setTextFormat(Qt::RichText);
    m_featureBody->setWordWrap(true);
    m_featureBody->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_featureBody->setAlignment(Qt::AlignTop | Qt::AlignLeft);

    auto *scroll = new QScrollArea(m_featurePanel);
    scroll->setWidget(m_featureBody);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setMinimumHeight(kFeaturePanelMaxHeight);
    scroll->setMaximumHeight(kFeaturePanelMaxHeight);
    scroll->viewport()->setAutoFillBackground(false);
    layout->addWidget(scroll);

    m_featurePanel->setMaximumWidth(kFeaturePanelWidth);
    m_featurePanel->setMinimumWidth(kFeaturePanelWidth);
}

void ViewerPage::positionFeaturePanel()
{
    if (!m_featurePanel || !m_featurePanel->isVisible() || !m_view)
        return;
    m_featurePanel->adjustSize();
    const int margin = 12;
    // Bottom right, clear of the bottom edge: flush against it, the panel read
    // as glued on rather than floating over the map. The credit line that used
    // to share this corner now lives in the info strip instead (see
    // ViewerPage's m_attributionLabel), so there is no second clearance to add
    // for it any more.
    // Narrowed by kFeaturePanelExtraBelow — never past the plain margin — so
    // part of the panel's own height sits below where it used to, rather than
    // all of it reaching upward. See kFeaturePanelMaxHeight above.
    const int bottomGap = std::max(margin, margin + 14 - kFeaturePanelExtraBelow);
    m_featurePanel->move(
        std::max(margin, m_view->width() - m_featurePanel->width() - margin),
        std::max(margin, m_view->height() - m_featurePanel->height() - bottomGap));
    m_featurePanel->raise();
}

void ViewerPage::clickFeatureForTest(int pointIndex)
{
    for (int oi = 0; oi < int(m_overlays.size()); ++oi) {
        const VectorOverlay &o = *m_overlays[oi];
        if (o.scenePoints.isEmpty())
            continue;
        const int idx = std::clamp(pointIndex, 0, int(o.scenePoints.size()) - 1);
        const QPoint viewPt = m_view->mapFromScene(o.scenePoints.at(idx));
        const QPointF local(viewPt);
        const QPointF global = m_view->viewport()->mapToGlobal(local);
        QMouseEvent press(QEvent::MouseButtonPress, local, global, Qt::LeftButton,
                          Qt::LeftButton, Qt::NoModifier);
        QMouseEvent release(QEvent::MouseButtonRelease, local, global, Qt::LeftButton,
                            Qt::NoButton, Qt::NoModifier);
        QApplication::sendEvent(m_view->viewport(), &press);
        QApplication::sendEvent(m_view->viewport(), &release);
        return;
    }
}

void ViewerPage::onCanvasClicked(QPointF scenePos)
{
    // The release has just put the open hand back. Whether the identify pointer
    // returns is decided by the next move, so what it remembers must not be
    // what it saw before the press.
    m_hoveringFeature = false;

    int overlayIndex = -1, geometryIndex = -1;
    bool isPoint = true;
    if (pickFeatureAt(scenePos, overlayIndex, isPoint, geometryIndex)) {
        showFeatureInfo(overlayIndex, isPoint, geometryIndex);
    } else if (m_featurePanel) {
        // Clicking empty ground closes the panel. It reads as "nothing here",
        // which is exactly what it means: the panel describes what was clicked,
        // and leaving the last feature on screen makes it describe something the
        // pointer has moved away from. Dragging the map is not a click, so the
        // panel survives being panned around.
        m_featurePanel->hide();
    }
}

bool ViewerPage::pickFeatureAt(const QPointF &scenePos, int &overlayIndex,
                               bool &isPoint, int &geometryIndex) const
{
    overlayIndex = -1;
    geometryIndex = -1;
    isPoint = true;
    if (m_overlays.empty() || !m_view)
        return false;
    // The tolerance is in screen pixels, converted through the current zoom, so
    // it stays the same size to the hand whatever the magnification.
    const double viewScale = m_view->transform().m11();
    if (viewScale <= 0.0)
        return false;
    const double tol = 12.0 / viewScale;

    int bestOverlay = -1, bestIndex = -1;
    bool bestIsPoint = true;
    double bestDist = tol;

    // The same test rebuildOverlay() uses, restated because it is a lambda in
    // there: a layer whose tick box is off is not on screen and must not answer
    // a click that landed on what is drawn over it.
    const auto pickable = [this](int oi) {
        if (oi >= m_overlayChecks.size() || !m_overlayChecks.at(oi)->isChecked())
            return false;
        const VectorOverlay *o = m_overlays[oi].get();
        return o->loaded && !o->failed;
    };

    for (int oi = 0; oi < int(m_overlays.size()); ++oi) {
        if (!pickable(oi))
            continue;
        const VectorOverlay &o = *m_overlays[oi];
        for (int i = 0; i < o.scenePoints.size(); ++i) {
            const QPointF d = o.scenePoints.at(i) - scenePos;
            const double dist = std::hypot(d.x(), d.y());
            if (dist < bestDist) {
                bestDist = dist;
                bestOverlay = oi;
                bestIndex = i;
                bestIsPoint = true;
            }
        }
    }
    // Points win over lines when both are in reach: a point is a smaller
    // target, so someone who hit one meant to.
    if (bestOverlay < 0) {
        for (int oi = 0; oi < int(m_overlays.size()); ++oi) {
            if (!pickable(oi))
                continue;
            const VectorOverlay &o = *m_overlays[oi];
            for (int li = 0; li < o.sceneLines.size(); ++li) {
                const QPolygonF &line = o.sceneLines.at(li);
                for (int i = 0; i + 1 < line.size(); ++i) {
                    const QPointF a = line.at(i), b = line.at(i + 1);
                    const QPointF ab = b - a;
                    const double len2 = ab.x() * ab.x() + ab.y() * ab.y();
                    double t = 0.0;
                    if (len2 > 0.0)
                        t = std::clamp(((scenePos.x() - a.x()) * ab.x()
                                        + (scenePos.y() - a.y()) * ab.y()) / len2,
                                       0.0, 1.0);
                    const QPointF foot = a + ab * t;
                    const double dist = std::hypot(scenePos.x() - foot.x(),
                                                   scenePos.y() - foot.y());
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestOverlay = oi;
                        bestIndex = li;
                        bestIsPoint = false;
                    }
                }
            }
        }
    }

    overlayIndex = bestOverlay;
    geometryIndex = bestIndex;
    isPoint = bestIsPoint;
    return bestOverlay >= 0;
}

void ViewerPage::showFeatureInfo(int overlayIndex, bool isPoint, int geometryIndex)
{
    if (overlayIndex < 0 || overlayIndex >= int(m_overlays.size()))
        return;
    const VectorOverlay &o = *m_overlays[overlayIndex];
    const QVector<QStringList> &table = isPoint ? o.pointAttrs : o.lineAttrs;
    const QStringList values =
        geometryIndex >= 0 && geometryIndex < table.size() ? table.at(geometryIndex)
                                                           : QStringList();

    m_featureTitle->setText(o.label);

    QString html = QStringLiteral("<table cellspacing='0' cellpadding='2'>");
    if (values.isEmpty() || o.fieldNames.isEmpty()) {
        // A layer with no attribute table is not an error — an exported path
        // often has none — and saying which feature was hit is still useful.
        html += QStringLiteral("<tr><td colspan='2'>%1</td></tr>")
                    .arg(isPoint ? tr("Point %1 — this layer carries no attributes")
                                       .arg(geometryIndex + 1)
                                 : tr("Line %1 — this layer carries no attributes")
                                       .arg(geometryIndex + 1));
    } else {
        for (int i = 0; i < o.fieldNames.size() && i < values.size(); ++i) {
            // A bibliography in one cell is legitimate data and unreadable in a
            // panel this size; it is cut, and the whole of it stays in the
            // table on disk.
            QString value = values.at(i).trimmed();
            if (value.size() > 90)
                value = value.left(89) + QChar(0x2026);
            html += QStringLiteral(
                        "<tr><td style='padding-right:10px'><b>%1</b></td><td>%2</td></tr>")
                        .arg(o.fieldNames.at(i).toHtmlEscaped(),
                             (value.isEmpty() ? QStringLiteral("&mdash;")
                                              : value.toHtmlEscaped()));
        }
    }
    html += QStringLiteral("</table>");
    m_featureBody->setText(html);

    m_featurePanel->show();
    positionFeaturePanel();
}

void ViewerPage::onHover(const QPointF &scenePos)
{
    // The pointer answers before the click does: over a feature it becomes the
    // identify arrow, everywhere else the open hand that means "drag me". Not
    // while a drag is actually under way — then the cursor belongs to the drag.
    if (m_view && !m_view->isPanning()) {
        int oi = -1, gi = -1;
        bool isPoint = true;
        const bool onFeature = pickFeatureAt(scenePos, oi, isPoint, gi);
        if (onFeature != m_hoveringFeature) {
            m_hoveringFeature = onFeature;
            m_view->viewport()->setCursor(onFeature ? identifyCursor()
                                                    : QCursor(Qt::OpenHandCursor));
        }
    } else {
        // A pan puts the closed hand up and keeps it there. Forgetting what was
        // under the pointer is what makes the cursor right again the moment the
        // drag ends, wherever it ended.
        m_hoveringFeature = false;
    }

    RasterLayer *layer = currentLayer();
    if (!layer || !layer->loaded) {
        // Vector-only: no cell to read, but the position under the pointer is
        // still worth having, and the frame is invertible.
        if (m_vectorScale > 0.0) {
            const double mx = m_vectorExtent.left() + scenePos.x() / m_vectorScale;
            const double my = m_vectorExtent.bottom() - scenePos.y() / m_vectorScale;
            m_cursorLabel->setText(QStringLiteral("%1, %2")
                                       .arg(QString::number(mx, 'f', 2),
                                            QString::number(my, 'f', 2)));
            return;
        }
        m_cursorLabel->clear();
        return;
    }
    if (!layer->ds) {
        // Released by releaseFiles() so a run could rewrite the file; the
        // exact-value readout needs it back.
        const QByteArray pathUtf8 = layer->path.toUtf8();
        layer->ds = GdalApi::instance().OpenEx(pathUtf8.constData(),
                                               GdalApi::OF_Raster,
                                               nullptr, nullptr, nullptr);
        if (!layer->ds) {
            m_cursorLabel->clear();
            return;
        }
    }
    const int dx = int(std::floor(scenePos.x()));
    const int dy = int(std::floor(scenePos.y()));
    if (dx < 0 || dy < 0 || dx >= layer->dispW || dy >= layer->dispH) {
        m_cursorLabel->clear();
        return;
    }
    // Display pixel -> source pixel; read the exact source value (the display
    // buffer may be decimated).
    const int sxPix = std::min(layer->srcW - 1,
                               int(double(dx) * layer->srcW / layer->dispW));
    const int syPix = std::min(layer->srcH - 1,
                               int(double(dy) * layer->srcH / layer->dispH));
    double value = 0.0;
    GdalApi &api = GdalApi::instance();
    GDALRasterBandH band = api.GetRasterBand(layer->ds, 1);
    if (!band || api.RasterIO(band, GdalApi::ReadFlag, sxPix, syPix, 1, 1,
                              &value, 1, 1, GdalApi::Float64, 0, 0) != 0) {
        m_cursorLabel->clear();
        return;
    }

    const double mapX = layer->gt[0] + (sxPix + 0.5) * layer->gt[1]
                        + (syPix + 0.5) * layer->gt[2];
    const double mapY = layer->gt[3] + (sxPix + 0.5) * layer->gt[4]
                        + (syPix + 0.5) * layer->gt[5];
    const int coordDecimals = layer->geographic ? 5 : 1;
    QString text = tr("E %1   N %2   ")
                       .arg(QString::number(mapX, 'f', coordDecimals),
                            QString::number(mapY, 'f', coordDecimals));
    if (std::isnan(value) || (layer->hasNoData && value == layer->noData))
        text += tr("no data");
    else
        text += tr("value: %1").arg(formatValue(value));
    m_cursorLabel->setText(text);
}

// ---------------------------------------------------------------------------
// Satellite basemap (Esri World Imagery)
// ---------------------------------------------------------------------------
namespace {

// Scene (display px) -> layer map coords -> web mercator. False on failure.
bool sceneToMerc(const RasterLayer &layer, void *ctLayerToMerc,
                 const QPointF &scenePt, double &mx, double &my)
{
    const double srcPx = scenePt.x() * layer.srcW / layer.dispW;
    const double srcPy = scenePt.y() * layer.srcH / layer.dispH;
    mx = layer.gt[0] + srcPx * layer.gt[1] + srcPy * layer.gt[2];
    my = layer.gt[3] + srcPx * layer.gt[4] + srcPy * layer.gt[5];
    return GdalApi::instance().OCTTransform(ctLayerToMerc, 1, &mx, &my,
                                            nullptr) != 0;
}

// Web mercator -> layer map coords -> scene (display px). False on failure.
bool mercToScene(const RasterLayer &layer, void *ctMercToLayer,
                 double mx, double my, QPointF &scenePt)
{
    if (!GdalApi::instance().OCTTransform(ctMercToLayer, 1, &mx, &my, nullptr))
        return false;
    const double *gt = layer.gt;
    const double det = gt[1] * gt[5] - gt[2] * gt[4];
    if (det == 0.0)
        return false;
    const double dx = mx - gt[0];
    const double dy = my - gt[3];
    const double px = (gt[5] * dx - gt[2] * dy) / det;
    const double py = (-gt[4] * dx + gt[1] * dy) / det;
    scenePt = QPointF(px * layer.dispW / layer.srcW,
                      py * layer.dispH / layer.srcH);
    return true;
}

} // namespace

bool ViewerPage::ensureBasemapTransforms(const RasterLayer &layer)
{
    if (layer.wkt.isEmpty())
        return false;
    if (m_ctLayerToMerc && m_ctMercToLayer && m_ctWkt == layer.wkt)
        return true;
    clearBasemap(true);

    GdalApi &api = GdalApi::instance();
    const QByteArray wktUtf8 = layer.wkt.toUtf8();
    OGRSpatialReferenceH src = api.OSRNewSpatialReference(wktUtf8.constData());
    OGRSpatialReferenceH merc = api.OSRNewSpatialReference(nullptr);
    if (!src || !merc || api.OSRImportFromEPSG(merc, 3857) != 0) {
        if (src)
            api.OSRDestroySpatialReference(src);
        if (merc)
            api.OSRDestroySpatialReference(merc);
        return false;
    }
    // Traditional x/y axis order regardless of what the authority says
    // (GDAL 3 would otherwise flip lat/lon for some CRS definitions).
    api.OSRSetAxisMappingStrategy(src, 0);
    api.OSRSetAxisMappingStrategy(merc, 0);
    m_ctLayerToMerc = api.OCTNewCoordinateTransformation(src, merc);
    m_ctMercToLayer = api.OCTNewCoordinateTransformation(merc, src);
    api.OSRDestroySpatialReference(src);
    api.OSRDestroySpatialReference(merc);
    if (!m_ctLayerToMerc || !m_ctMercToLayer) {
        clearBasemap(true);
        return false;
    }
    m_ctWkt = layer.wkt;
    return true;
}

void ViewerPage::updateBasemap()
{
    RasterLayer *layer = currentLayer();
    if (!layer || !m_basemapToggle->isChecked())
        return;
    if (!ensureGdal() || !ensureBasemapTransforms(*layer))
        return;

    // Visible scene area with a 20% margin, in mercator coordinates.
    QRectF vis = m_view->mapToScene(m_view->viewport()->rect()).boundingRect();
    vis.adjust(-vis.width() * 0.2, -vis.height() * 0.2,
               vis.width() * 0.2, vis.height() * 0.2);
    double minX = 0, minY = 0, maxX = 0, maxY = 0;
    bool first = true;
    for (const QPointF &corner :
         {vis.topLeft(), vis.topRight(), vis.bottomRight(), vis.bottomLeft()}) {
        double mx = 0, my = 0;
        if (!sceneToMerc(*layer, m_ctLayerToMerc, corner, mx, my))
            return;
        minX = first ? mx : std::min(minX, mx);
        maxX = first ? mx : std::max(maxX, mx);
        minY = first ? my : std::min(minY, my);
        maxY = first ? my : std::max(maxY, my);
        first = false;
    }
    minX = std::clamp(minX, -kMercHalf, kMercHalf);
    maxX = std::clamp(maxX, -kMercHalf, kMercHalf);
    minY = std::clamp(minY, -kMercHalf, kMercHalf);
    maxY = std::clamp(maxY, -kMercHalf, kMercHalf);

    // Zoom level whose tile resolution best matches the screen resolution.
    const QPoint vc = m_view->viewport()->rect().center();
    double cx0 = 0, cy0 = 0, cx1 = 0, cy1 = 0;
    if (!sceneToMerc(*layer, m_ctLayerToMerc, m_view->mapToScene(vc), cx0, cy0)
        || !sceneToMerc(*layer, m_ctLayerToMerc,
                        m_view->mapToScene(vc + QPoint(1, 0)), cx1, cy1))
        return;
    const double mercPerViewPx = std::hypot(cx1 - cx0, cy1 - cy0);
    if (mercPerViewPx <= 0.0 || !std::isfinite(mercPerViewPx))
        return;
    int z = std::clamp(
        int(std::lround(std::log2(2.0 * kMercHalf / (256.0 * mercPerViewPx)))),
        0, kMaxTileZoom);

    // Tile index range at z, shrinking z until the tile count is sane.
    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    for (;; --z) {
        const int n = 1 << z;
        const double span = 2.0 * kMercHalf / n;
        x0 = std::clamp(int(std::floor((minX + kMercHalf) / span)), 0, n - 1);
        x1 = std::clamp(int(std::floor((maxX + kMercHalf) / span)), 0, n - 1);
        y0 = std::clamp(int(std::floor((kMercHalf - maxY) / span)), 0, n - 1);
        y1 = std::clamp(int(std::floor((kMercHalf - minY) / span)), 0, n - 1);
        if (z == 0
            || qint64(x1 - x0 + 1) * qint64(y1 - y0 + 1) <= kMaxTilesPerUpdate)
            break;
    }

    QSet<QString> needed;
    for (int ty = y0; ty <= y1; ++ty) {
        for (int tx = x0; tx <= x1; ++tx) {
            const QString key = QStringLiteral("%1/%2/%3").arg(z).arg(tx).arg(ty);
            needed.insert(key);
            if (!m_tiles.contains(key) && !m_pendingTiles.contains(key))
                fetchTile(z, tx, ty);
        }
    }
    for (auto it = m_tiles.begin(); it != m_tiles.end();) {
        if (!needed.contains(it.key())) {
            delete it.value();
            it = m_tiles.erase(it);
        } else {
            ++it;
        }
    }
}

void ViewerPage::fetchTile(int z, int x, int y)
{
    const QString key = QStringLiteral("%1/%2/%3").arg(z).arg(x).arg(y);
    m_pendingTiles.insert(key);

    QNetworkRequest request(
        QUrl(QString::fromLatin1(kTileUrl).arg(z).arg(y).arg(x)));
    request.setAttribute(QNetworkRequest::CacheLoadControlAttribute,
                         QNetworkRequest::PreferCache);
    request.setHeader(QNetworkRequest::UserAgentHeader,
                      QStringLiteral("TrajectaStudio/%1")
                          .arg(QApplication::applicationVersion()));
    QNetworkReply *reply = m_net->get(request);
    const int gen = m_basemapGen;
    connect(reply, &QNetworkReply::finished, this,
            [this, reply, key, gen, z, x, y] {
                reply->deleteLater();
                m_pendingTiles.remove(key);
                if (gen != m_basemapGen || !m_basemapToggle->isChecked())
                    return;
                RasterLayer *layer = currentLayer();
                if (!layer || reply->error() != QNetworkReply::NoError)
                    return;
                const QImage img = QImage::fromData(reply->readAll());
                if (img.isNull())
                    return;

                // Warp the tile into the scene: its mercator corners become a
                // quad in display pixels (projective per-tile approximation).
                const int n = 1 << z;
                const double span = 2.0 * kMercHalf / n;
                const double mx0 = -kMercHalf + x * span;
                const double myTop = kMercHalf - y * span;
                QPointF tl, tr, br, bl;
                if (!mercToScene(*layer, m_ctMercToLayer, mx0, myTop, tl)
                    || !mercToScene(*layer, m_ctMercToLayer, mx0 + span, myTop, tr)
                    || !mercToScene(*layer, m_ctMercToLayer, mx0 + span,
                                    myTop - span, br)
                    || !mercToScene(*layer, m_ctMercToLayer, mx0, myTop - span, bl))
                    return;
                const QPolygonF srcQuad({QPointF(0, 0), QPointF(img.width(), 0),
                                         QPointF(img.width(), img.height()),
                                         QPointF(0, img.height())});
                const QPolygonF dstQuad({tl, tr, br, bl});
                QTransform t;
                if (!QTransform::quadToQuad(srcQuad, dstQuad, t))
                    return;

                auto *item = m_scene->addPixmap(QPixmap::fromImage(img));
                item->setZValue(-1);   // under the raster (0) and overlay (1)
                item->setTransformationMode(Qt::SmoothTransformation);
                item->setTransform(t);
                delete m_tiles.value(key, nullptr);
                m_tiles.insert(key, item);
            });
}

void ViewerPage::clearBasemap(bool dropTransforms)
{
    ++m_basemapGen;   // in-flight replies become stale
    for (QGraphicsPixmapItem *item : std::as_const(m_tiles))
        delete item;
    m_tiles.clear();
    m_pendingTiles.clear();
    if (dropTransforms && GdalApi::instance().isLoaded()) {
        if (m_ctLayerToMerc)
            GdalApi::instance().OCTDestroyCoordinateTransformation(m_ctLayerToMerc);
        if (m_ctMercToLayer)
            GdalApi::instance().OCTDestroyCoordinateTransformation(m_ctMercToLayer);
        m_ctLayerToMerc = nullptr;
        m_ctMercToLayer = nullptr;
        m_ctWkt.clear();
    }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
void ViewerPage::exportImage()
{
    if (!currentLayer())
        return;

    // What to draw and how big, asked before the file name so the dialog does
    // not appear after the user has already committed to a path.
    ExportSettings ex;
    if (!askExportSettings(&ex))
        return;

    // The format is picked in the save dialog itself: choosing "PNG image"
    // there, or simply typing a .png name, is what selects it.
    const QString jpegFilter = tr("JPEG image (*.jpg *.jpeg)");
    const QString pngFilter = tr("PNG image (*.png)");
    QString chosenFilter = jpegFilter;
    QString file = QFileDialog::getSaveFileName(
        this, tr("Export view"), QStringLiteral("trajecta_view.jpg"),
        jpegFilter + QStringLiteral(";;") + pngFilter, &chosenFilter);
    if (file.isEmpty())
        return;

    const QString suffix = QFileInfo(file).suffix().toLower();
    const char *format = "JPG";
    if (suffix == QLatin1String("png")) {
        format = "PNG";
    } else if (suffix.isEmpty()) {
        // No extension typed: follow the filter the user selected.
        const bool png = chosenFilter == pngFilter;
        format = png ? "PNG" : "JPG";
        file += png ? QStringLiteral(".png") : QStringLiteral(".jpg");
    }
    // Render the whole scene instead of grabbing the viewport: the export does
    // not depend on the window size or the current zoom, and includes basemap
    // and overlay as drawn.
    const QRectF rect = m_scene->sceneRect();
    const int nativeW = qMax(1, qRound(rect.width()));
    const int nativeH = qMax(1, qRound(rect.height()));

    int w = nativeW, h = nativeH;
    if (ex.byDpi) {
        // 96 dpi is the nominal screen density Qt assumes, so a 96 dpi export
        // is the scene at its own size and 300 dpi is 3.125 times that.
        const double scale = double(ex.dpi) / 96.0;
        w = qMax(1, qRound(nativeW * scale));
        h = qMax(1, qRound(nativeH * scale));
    } else {
        w = qMax(1, ex.width);
        h = qMax(1, ex.height);
    }
    // A cap, so a mistyped 30000 does not try to allocate several gigabytes.
    const qint64 pixels = qint64(w) * qint64(h);
    if (pixels > 400000000LL) {
        m_cursorLabel->setText(tr("Export cancelled: %1 x %2 is too large").arg(w).arg(h));
        return;
    }

    QImage img(w, h, QImage::Format_RGB32);
    if (img.isNull()) {
        m_cursorLabel->setText(tr("Export failed: not enough memory for %1 x %2").arg(w).arg(h));
        return;
    }
    img.fill(canvasBg());
    QPainter p(&img);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    m_scene->render(&p, QRectF(0, 0, w, h), rect);

    // Scalebar, north arrow and the basemap credit go on afterwards, in image
    // coordinates: MapView::drawForeground draws them for the viewport and does
    // not run here, which is why an exported image used to carry neither the
    // scale nor the attribution the tile terms require.
    ExportDecorations dec;
    dec.scalebar = ex.scalebar;
    dec.northArrow = ex.northArrow;
    dec.geographic = m_view->isGeographic();
    const double unitsPerScenePx = m_view->unitsPerScenePx();
    if (unitsPerScenePx > 0.0)
        dec.unitsPerPx = unitsPerScenePx * double(nativeW) / double(w);
    if (m_basemapToggle && m_basemapToggle->isChecked())
        dec.attribution = QString::fromLatin1(kAttribution);
    paintExportDecorations(p, img.size(), dec);
    p.end();

    // Stamp the density so the file prints at the size the dpi implies rather
    // than at whatever the viewer application assumes.
    if (ex.byDpi) {
        const int dpm = int(std::lround(ex.dpi / 0.0254));
        img.setDotsPerMeterX(dpm);
        img.setDotsPerMeterY(dpm);
    }

    const int quality = qstrcmp(format, "PNG") == 0 ? -1 : 92;
    if (!img.save(file, format, quality)) {
        m_cursorLabel->setText(tr("Export failed: %1").arg(file));
    } else {
        m_cursorLabel->setText(tr("Exported %1 x %2 px").arg(w).arg(h));
        // Only after a successful write, and only if it was asked for: opening
        // a file that failed to save would show whatever was there before.
        if (ex.openWhenDone)
            QDesktopServices::openUrl(QUrl::fromLocalFile(file));
    }
}

// Greys out the ramps a colour-blind reader cannot follow, and moves off one if
// it is the ramp currently in use. Disabling rather than removing keeps every
// combo index exactly where it was — Hillshade included, which lives one past
// the end of the ramp list.
void ViewerPage::applyCvdSafeFilter()
{
    if (!m_colormapCombo || !m_cvdSafeToggle)
        return;
    const bool restrict = m_cvdSafeToggle->isChecked();
    const auto &maps = colormaps();
    int firstSafe = -1;
    for (int i = 0; i < maps.size(); ++i) {
        const bool ok = !restrict || maps[i].cvdSafe;
        if (ok && firstSafe < 0)
            firstSafe = i;
        auto *model = qobject_cast<QStandardItemModel *>(m_colormapCombo->model());
        if (model && model->item(i))
            model->item(i)->setEnabled(ok);
        m_colormapCombo->setItemData(
            i,
            ok ? QString() : tr("Not readable with a colour vision deficiency"),
            Qt::ToolTipRole);
    }
    // Currently on a ramp that just became unavailable: move to the first that
    // is, rather than leaving the view showing something the switch forbids.
    const int cur = m_colormapCombo->currentIndex();
    if (restrict && cur >= 0 && cur < maps.size() && !maps[cur].cvdSafe && firstSafe >= 0)
        m_colormapCombo->setCurrentIndex(firstSafe);
}

// The export options, asked once and remembered. Deliberately a small dialog
// rather than a page of settings: the choices that matter are how big and
// what to draw on top.
bool ViewerPage::askExportSettings(ExportSettings *out)
{
    QSettings s;
    QDialog dlg(this);
    dlg.setWindowTitle(tr("Export view"));
    auto *form = new QVBoxLayout(&dlg);
    form->setSpacing(10);

    auto *scalebar = new QCheckBox(tr("Draw a scalebar"), &dlg);
    scalebar->setChecked(s.value(QStringLiteral("viewer/exportScalebar"), true).toBool());
    form->addWidget(TrajectaUi::withHelpDot(scalebar, exportScalebarHelpText()));

    auto *north = new QCheckBox(tr("Draw a north arrow"), &dlg);
    north->setChecked(s.value(QStringLiteral("viewer/exportNorth"), true).toBool());
    form->addWidget(TrajectaUi::withHelpDot(north, exportNorthHelpText()));

    auto *byDpi = new QRadioButton(tr("Resolution by print density"), &dlg);
    auto *bySize = new QRadioButton(tr("Resolution in pixels"), &dlg);
    const bool wasDpi = s.value(QStringLiteral("viewer/exportByDpi"), true).toBool();
    byDpi->setChecked(wasDpi);
    bySize->setChecked(!wasDpi);
    // Both could be ticked at once, and this is why: a radio button is
    // auto-exclusive only against its siblings, and withHelpDot() reparents the
    // widget it decorates into a wrapper of its own. That left one radio button
    // inside the wrapper and the other in the dialog — two only children, each
    // exclusive with nothing. A button group is exclusive by membership rather
    // than by parentage, so it survives being rearranged.
    auto *resolutionGroup = new QButtonGroup(&dlg);
    resolutionGroup->setExclusive(true);
    resolutionGroup->addButton(byDpi);
    resolutionGroup->addButton(bySize);
    form->addWidget(TrajectaUi::withHelpDot(byDpi, exportResolutionHelpText()));

    auto *dpiRow = new QWidget(&dlg);
    auto *dpiLayout = new QHBoxLayout(dpiRow);
    dpiLayout->setContentsMargins(22, 0, 0, 0);
    auto *dpiSpin = new QSpinBox(dpiRow);
    dpiSpin->setRange(48, 1200);
    dpiSpin->setSingleStep(50);
    dpiSpin->setValue(s.value(QStringLiteral("viewer/exportDpi"), 300).toInt());
    dpiSpin->setSuffix(tr(" dpi"));
    TrajectaUi::guardWheel(dpiSpin);
    dpiLayout->addWidget(dpiSpin);
    dpiLayout->addStretch(1);
    form->addWidget(dpiRow);

    form->addWidget(bySize);
    auto *sizeRow = new QWidget(&dlg);
    auto *sizeLayout = new QHBoxLayout(sizeRow);
    sizeLayout->setContentsMargins(22, 0, 0, 0);
    const QRectF sceneRect = m_scene->sceneRect();
    const int nativeW = qMax(1, qRound(sceneRect.width()));
    const int nativeH = qMax(1, qRound(sceneRect.height()));
    auto *wSpin = new QSpinBox(sizeRow);
    wSpin->setRange(64, 40000);
    wSpin->setValue(s.value(QStringLiteral("viewer/exportW"), nativeW).toInt());
    wSpin->setSuffix(tr(" px wide"));
    TrajectaUi::guardWheel(wSpin);
    auto *hSpin = new QSpinBox(sizeRow);
    hSpin->setRange(64, 40000);
    hSpin->setValue(s.value(QStringLiteral("viewer/exportH"), nativeH).toInt());
    hSpin->setSuffix(tr(" px high"));
    TrajectaUi::guardWheel(hSpin);
    sizeLayout->addWidget(wSpin);
    sizeLayout->addWidget(hSpin);
    sizeLayout->addStretch(1);
    form->addWidget(sizeRow);

    auto *nativeNote = new QLabel(
        tr("The view is %1 × %2 pixels at its own resolution.").arg(nativeW).arg(nativeH), &dlg);
    nativeNote->setObjectName(QStringLiteral("FieldHint"));
    form->addWidget(nativeNote);

    auto *openAfter = new QCheckBox(tr("Open the image when it has been written"), &dlg);
    openAfter->setChecked(s.value(QStringLiteral("viewer/exportOpen"), false).toBool());
    form->addWidget(TrajectaUi::withHelpDot(
        openAfter,
        tr("Hands the finished file to whatever your system opens images with, "
           "so you can check it without going to look for it. The file is "
           "written either way.")));

    // The size the user typed for themselves, kept aside: while the density
    // mode is on, the two boxes are used to *show* what that density works out
    // to, and switching back has to give the typed figures back rather than the
    // derived ones.
    int typedW = wSpin->value();
    int typedH = hSpin->value();

    const auto refresh = [&] {
        const bool dpiMode = byDpi->isChecked();
        dpiRow->setEnabled(dpiMode);
        // Not disabled: the boxes below are where the answer is now shown, and
        // grey digits are hard to read. Read-only says the same thing and keeps
        // the numbers legible.
        wSpin->setReadOnly(dpiMode);
        hSpin->setReadOnly(dpiMode);
        wSpin->setButtonSymbols(dpiMode ? QAbstractSpinBox::NoButtons
                                        : QAbstractSpinBox::UpDownArrows);
        hSpin->setButtonSymbols(dpiMode ? QAbstractSpinBox::NoButtons
                                        : QAbstractSpinBox::UpDownArrows);
        if (!dpiMode)
            return;
        // The result of the chosen density, written where the pixel size is
        // read. It used to be a separate "→ 1234 × 567 px" label beside the dpi
        // box, which said the same thing in a second place.
        const double scale = double(dpiSpin->value()) / 96.0;
        QSignalBlocker blockW(wSpin);
        QSignalBlocker blockH(hSpin);
        wSpin->setValue(qMax(1, qRound(nativeW * scale)));
        hSpin->setValue(qMax(1, qRound(nativeH * scale)));
    };
    connect(byDpi, &QRadioButton::toggled, &dlg, refresh);
    connect(bySize, &QRadioButton::toggled, &dlg, [&](bool on) {
        if (on) {
            QSignalBlocker blockW(wSpin);
            QSignalBlocker blockH(hSpin);
            wSpin->setValue(typedW);
            hSpin->setValue(typedH);
        }
        refresh();
    });
    connect(dpiSpin, QOverload<int>::of(&QSpinBox::valueChanged), &dlg, refresh);
    // Only what the user types counts as typed; the derived values are written
    // with the signals blocked and never land here.
    connect(wSpin, QOverload<int>::of(&QSpinBox::valueChanged), &dlg,
            [&](int v) { typedW = v; });
    connect(hSpin, QOverload<int>::of(&QSpinBox::valueChanged), &dlg,
            [&](int v) { typedH = v; });
    refresh();

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
    // The filled answer, as on the confirmations: this dialog is opened by
    // someone who has already decided to export.
    if (QPushButton *ok = buttons->button(QDialogButtonBox::Ok))
        ok->setProperty("fill", QStringLiteral("accent"));
    connect(buttons, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    form->addWidget(buttons);

    if (dlg.exec() != QDialog::Accepted)
        return false;

    out->scalebar = scalebar->isChecked();
    out->northArrow = north->isChecked();
    out->byDpi = byDpi->isChecked();
    out->dpi = dpiSpin->value();
    out->width = wSpin->value();
    out->height = hSpin->value();
    out->openWhenDone = openAfter->isChecked();

    s.setValue(QStringLiteral("viewer/exportScalebar"), out->scalebar);
    s.setValue(QStringLiteral("viewer/exportNorth"), out->northArrow);
    s.setValue(QStringLiteral("viewer/exportByDpi"), out->byDpi);
    s.setValue(QStringLiteral("viewer/exportDpi"), out->dpi);
    // The typed size, not what the density worked out to: the next export must
    // find the figures the user chose, not the ones the dialog showed them.
    s.setValue(QStringLiteral("viewer/exportW"), typedW);
    s.setValue(QStringLiteral("viewer/exportH"), typedH);
    s.setValue(QStringLiteral("viewer/exportOpen"), out->openWhenDone);
    return true;
}
