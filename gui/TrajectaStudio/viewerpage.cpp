#include "viewerpage.h"

#include "confirmdialog.h"
#include "gdalapi.h"
#include "rangeslider.h"
#include "smoothcombobox.h"
#include "thememanager.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGraphicsPathItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QNetworkAccessManager>
#include <QNetworkDiskCache>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPainter>
#include <QPainterPath>
#include <QRegion>
#include <QSettings>
#include <QSlider>
#include <QListView>
#include <QMessageBox>
#include <QStackedLayout>
#include <QStandardPaths>
#include <QStyledItemDelegate>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

namespace {

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

// Painted by hand, so they follow the palette through ThemeManager rather
// than through the stylesheet. Functions, not constants: the theme can change
// while the page is alive.
// The card colour, so the canvas matches the panels on every other page.
// Opaque even where the cards are translucent: a map needs a solid backing.
inline QColor canvasBg()   { return ThemeManager::mapped("#1b1f26"); }
inline QColor overlayPen() { return ThemeManager::theme(ThemeManager::current()).overlayPen; }
inline QColor scalebarFg() { return ThemeManager::mapped("#e4e7ec"); }
inline QColor scalebarBg() { return ThemeManager::mapped("#14171c"); }
inline QColor hintFg()     { return ThemeManager::mapped("#99a1ac"); }
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
        {"Turbo", makeLut({{0.0, 48, 18, 59}, {0.125, 70, 107, 227},
                           {0.25, 40, 187, 235}, {0.375, 32, 229, 181},
                           {0.5, 122, 252, 82}, {0.625, 218, 227, 25},
                           {0.75, 253, 154, 42}, {0.875, 224, 62, 7},
                           {1.0, 122, 4, 3}})},
        {"Terrain", makeLut({{0.0, 27, 120, 55}, {0.35, 166, 217, 106},
                             {0.55, 254, 224, 139}, {0.75, 166, 97, 26},
                             {1.0, 247, 247, 247}})},
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
    QVector<QPolygonF> lines;    // map coordinates
    QVector<QPointF> points;     // map coordinates
};

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
    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setBackgroundBrush(canvasBg());
    setRenderHint(QPainter::SmoothPixmapTransform, true);
    viewport()->setMouseTracking(true);
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
        fitInView(scene()->sceneRect(), Qt::KeepAspectRatio);
        setRenderHint(QPainter::SmoothPixmapTransform, transform().m11() < 1.0);
    }
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
        emit viewChanged();
    }
    event->accept();
}

void MapView::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);
    emit viewChanged();
}

void MapView::scrollContentsBy(int dx, int dy)
{
    QGraphicsView::scrollContentsBy(dx, dy);
    emit viewChanged();
}

void MapView::mouseMoveEvent(QMouseEvent *event)
{
    emit hoverScenePos(mapToScene(event->position().toPoint()));
    QGraphicsView::mouseMoveEvent(event);
}

void MapView::leaveEvent(QEvent *event)
{
    emit hoverLeft();
    QGraphicsView::leaveEvent(event);
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
        rounded.addRoundedRect(r, kRadius, kRadius);
        m_cornerPath = full.subtracted(rounded);
    }
    painter->setRenderHint(QPainter::Antialiasing, true);
    painter->fillPath(m_cornerPath, canvasBg());


    // --- Scalebar, bottom-left ---
    const double unitsPerViewPx =
        m_unitsPerScenePx > 0.0 ? m_unitsPerScenePx / transform().m11() : 0.0;
    if (unitsPerViewPx > 0.0 && std::isfinite(unitsPerViewPx)) {
        // Nice 1/2/5 x 10^k length no wider than ~1/4 of the viewport.
        const double targetUnits = unitsPerViewPx * viewport()->width() / 4.0;
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

    // --- Basemap attribution, bottom-right (required by the tile terms) ---
    if (!m_attribution.isEmpty()) {
        QFont f = painter->font();
        f.setPixelSize(10);
        painter->setFont(f);
        const QRect textRect =
            painter->fontMetrics().boundingRect(m_attribution);
        const int x = viewport()->width() - textRect.width() - 14;
        const int y = viewport()->height() - 8;
        QColor bg = scalebarBg();
        bg.setAlpha(170);
        painter->setPen(Qt::NoPen);
        painter->setBrush(bg);
        painter->drawRoundedRect(QRect(x - 6, y - textRect.height() - 4,
                                       textRect.width() + 12,
                                       textRect.height() + 8),
                                 4, 4);
        painter->setPen(hintFg());
        painter->drawText(x, y - 3, m_attribution);
    }

    painter->restore();
}

// ---------------------------------------------------------------------------
// LegendBar
// ---------------------------------------------------------------------------
LegendBar::LegendBar(QWidget *parent)
    : QWidget(parent)
{
    setFixedWidth(78);
    setMinimumHeight(140);
}

void LegendBar::setState(const QVector<QRgb> &lut, double lowValue, double highValue)
{
    m_lut = lut;
    m_low = lowValue;
    m_high = highValue;
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
    const QRect bar(10, margin, 16, height() - margin * 2);
    for (int y = bar.top(); y <= bar.bottom(); ++y) {
        const double t = 1.0 - (y - bar.top()) / double(bar.height());
        p.setPen(QColor::fromRgb(m_lut[int(std::lround(t * 255))]));
        p.drawLine(bar.left(), y, bar.right(), y);
    }
    p.setPen(legendFrame());
    p.setBrush(Qt::NoBrush);
    p.drawRect(bar.adjusted(0, 0, -1, -1));

    QFont f = p.font();
    f.setPixelSize(10);
    p.setFont(f);
    p.setPen(hintFg());
    const int tx = bar.right() + 5;
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
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(28, 24, 28, 24);
    layout->setSpacing(10);

    // --- Controls card ---
    auto *card = new QFrame(this);
    card->setObjectName(QStringLiteral("Card"));
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
    openBtn->setToolTip(tr("Open a raster file"));
    openBtn->setCursor(Qt::PointingHandCursor);
    row1->addWidget(openBtn);
    auto *resetBtn = new QToolButton(card);
    resetBtn->setText(tr("Reset view"));
    resetBtn->setCursor(Qt::PointingHandCursor);
    row1->addWidget(resetBtn);
    auto *exportBtn = new QToolButton(card);
    exportBtn->setText(tr("Export"));
    exportBtn->setCursor(Qt::PointingHandCursor);
    row1->addWidget(exportBtn);
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
    row2->addWidget(new QLabel(tr("Stretch"), card));
    m_stretchCombo = new SmoothComboBox(card);
    m_stretchCombo->addItem(tr("Min–Max"));
    m_stretchCombo->addItem(tr("Percentile 2–98"));
    m_stretchCombo->addItem(tr("Logarithmic"));
    row2->addWidget(m_stretchCombo);
    row2->addSpacing(8);
    row2->addWidget(new QLabel(tr("Opacity"), card));
    m_opacitySlider = new QSlider(Qt::Horizontal, card);
    m_opacitySlider->setRange(0, 100);
    m_opacitySlider->setValue(100);
    m_opacitySlider->setFixedWidth(110);
    m_opacitySlider->setFixedHeight(22);  // room for the handle's QSS overshoot
    row2->addWidget(m_opacitySlider);
    row2->addSpacing(8);
    m_basemapToggle = new QCheckBox(tr("Satellite basemap"), card);
    m_basemapToggle->setEnabled(false);
    m_basemapToggle->setToolTip(
        tr("Show Esri World Imagery satellite tiles behind the raster. "
           "Requires an internet connection and a raster with a CRS; lower "
           "the raster opacity to see through it."));
    row2->addWidget(m_basemapToggle);
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
    m_filterLoSpin->setKeyboardTracking(false);
    m_filterLoSpin->setMinimumWidth(110);
    row3->addWidget(m_filterLoSpin);
    row3->addWidget(new QLabel(QStringLiteral("–"), card));
    m_filterHiSpin = new QDoubleSpinBox(card);
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
    m_canvasStack = new QStackedLayout(canvasHolder);
    m_canvasStack->setContentsMargins(1, 1, 1, 1);   // clear of the border
    m_view = new MapView(canvasHolder);
    m_scene = new QGraphicsScene(this);
    m_view->setScene(m_scene);
    m_canvasStack->addWidget(m_view);
    m_placeholder = new QLabel(canvasHolder);
    m_placeholder->setAlignment(Qt::AlignCenter);
    m_placeholder->setWordWrap(true);
    m_placeholder->setObjectName(QStringLiteral("HintLabel"));
    m_placeholder->setText(tr("No raster loaded.\n\nOutputs appear here "
                              "automatically after a run, or use \"Open "
                              "raster...\" above."));
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
    m_view->installEventFilter(this);

    layout->addWidget(canvasHolder, 1);

    // --- Info strip ---
    auto *info = new QHBoxLayout;
    info->setSpacing(10);
    m_crsLabel = new QLabel(this);
    m_crsLabel->setObjectName(QStringLiteral("HintLabel"));
    info->addWidget(m_crsLabel);
    m_resLabel = new QLabel(this);
    m_resLabel->setObjectName(QStringLiteral("HintLabel"));
    info->addWidget(m_resLabel);
    info->addStretch(1);
    m_cursorLabel = new QLabel(this);
    m_cursorLabel->setObjectName(QStringLiteral("HintLabel"));
    info->addWidget(m_cursorLabel);
    layout->addLayout(info);

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
    connect(m_basemapToggle, &QCheckBox::toggled, this, [this](bool on) {
        QSettings().setValue(QStringLiteral("viewer/basemap"), on);
        m_view->setAttribution(on ? tr(kAttribution) : QString());
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
        if (baseOn)
            m_view->setAttribution(tr(kAttribution));
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
            this, tr("Open raster"), QString(),
            tr("GeoTIFF (*.tif *.tiff);;All files (*)"));
        if (!file.isEmpty())
            openRasterFile(file);
    });
    connect(resetBtn, &QToolButton::clicked, this, [this] { m_view->fitAll(); });
    connect(exportBtn, &QToolButton::clicked, this, &ViewerPage::exportImage);
    connect(m_view, &MapView::hoverScenePos, this, &ViewerPage::onHover);
    connect(m_view, &MapView::hoverLeft, this, [this] {
        m_cursorLabel->clear();
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
    const int h = qBound(140, int(m_view->height() * 0.62), 340);
    m_legend->resize(m_legend->width(), h);
    m_legend->move(m_view->width() - m_legend->width() - kMargin, kMargin);
}

bool ViewerPage::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == m_view && event->type() == QEvent::Resize) {
        positionLegend();
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
        m_placeholder->setText(tr("No raster loaded.\n\nOutputs appear here "
                                  "automatically after a run, or use the \"...\" "
                                  "button above."));
        m_canvasStack->setCurrentWidget(m_placeholder);
        m_legend->hide();
        m_basemapToggle->setEnabled(false);
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

void ViewerPage::applyTheme()
{
    if (m_view) {
        m_view->setBackgroundBrush(canvasBg());
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
}

void ViewerPage::registerVectorOverlay(const QString &label, const QString &path)
{
    // Same file registered again (a re-run rewrote it): drop what was cached
    // and keep its position in the dropdown.
    int index = -1;
    for (int i = 0; i < int(m_overlays.size()); ++i) {
        if (QString::compare(m_overlays[i]->path, path, Qt::CaseInsensitive) == 0) {
            index = i;
            m_overlays[i]->loaded = false;
            m_overlays[i]->failed = false;
            m_overlays[i]->lines.clear();
            m_overlays[i]->points.clear();
            break;
        }
    }
    if (index < 0) {
        auto overlay = std::make_unique<VectorOverlay>();
        overlay->label = label;
        overlay->path = path;
        m_overlays.push_back(std::move(overlay));
        index = int(m_overlays.size()) - 1;

        auto *check = new QCheckBox(label, m_overlayPanel);
        check->setToolTip(QDir::toNativeSeparators(path));
        // The tick carries the colour the overlay is drawn in, so the panel
        // doubles as the legend for the vectors.
        check->setStyleSheet(QStringLiteral("color:%1;")
                                 .arg(overlayColor(index).name()));
        // Shown straight away: it is an output of the run just finished, and
        // the tick makes turning it off obvious.
        check->setChecked(true);
        connect(check, &QCheckBox::toggled, this, &ViewerPage::rebuildOverlay);
        m_overlayPanelLayout->addWidget(check);
        m_overlayChecks.append(check);
    }
    m_overlayPanel->setVisible(!m_overlays.empty());
    m_overlayPanel->adjustSize();
    m_overlayPanel->raise();
    rebuildOverlay();
}

// Distinct colours so two overlays drawn at once stay tellable apart; the
// first is the theme's own accent for vector work.
QColor ViewerPage::overlayColor(int index)
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
    m_view->fitAll();
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
    positionLegend();
    m_legend->setVisible(!hillshade);
    m_legend->raise();
    m_legend->setState(lut, lo, hi);
}

void ViewerPage::rebuildOverlay()
{
    qDeleteAll(m_overlayItems);
    m_overlayItems.clear();

    RasterLayer *layer = currentLayer();
    if (!layer)
        return;

    // Map coords -> source pixel (inverse geotransform) -> scene (display px).
    const double *gt = layer->gt;
    const double det = gt[1] * gt[5] - gt[2] * gt[4];
    if (det == 0.0)
        return;
    const double sx = double(layer->dispW) / layer->srcW;
    const double sy = double(layer->dispH) / layer->srcH;
    const auto toScene = [&](const QPointF &mapPt) {
        const double dx = mapPt.x() - gt[0];
        const double dy = mapPt.y() - gt[3];
        const double px = (gt[5] * dx - gt[2] * dy) / det;
        const double py = (-gt[4] * dx + gt[1] * dy) / det;
        return QPointF(px * sx, py * sy);
    };

    for (int oi = 0; oi < int(m_overlays.size()); ++oi) {
        if (oi >= m_overlayChecks.size() || !m_overlayChecks.at(oi)->isChecked())
            continue;
        VectorOverlay *overlay = m_overlays[oi].get();

        if (!overlay->loaded) {
            if (!ensureGdal())
                return;
            overlay->loaded = true;
            GdalApi &api = GdalApi::instance();
            const QByteArray pathUtf8 = overlay->path.toUtf8();
            GDALDatasetH ds = api.OpenEx(pathUtf8.constData(), GdalApi::OF_Vector,
                                         nullptr, nullptr, nullptr);
            if (!ds) {
                overlay->failed = true;
                continue;
            }
            auto collectLine = [overlay, &api](OGRGeometryH geom) {
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
                overlay->lines.append(line);
            };
            auto collectPoint = [overlay, &api](OGRGeometryH geom) {
                if (api.G_GetPointCount(geom) < 1)
                    return;
                double x = 0, y = 0, z = 0;
                api.G_GetPoint(geom, 0, &x, &y, &z);
                overlay->points.append(QPointF(x, y));
            };
            const int layerCount = api.DatasetGetLayerCount(ds);
            for (int li = 0; li < layerCount; ++li) {
                OGRLayerH vl = api.DatasetGetLayer(ds, li);
                if (!vl)
                    continue;
                api.L_ResetReading(vl);
                while (OGRFeatureH f = api.L_GetNextFeature(vl)) {
                    OGRGeometryH g = api.F_GetGeometryRef(f);
                    if (g) {
                        const int type = GdalApi::flattenGeomType(api.G_GetGeometryType(g));
                        if (type == GdalApi::WkbLineString) {
                            collectLine(g);
                        } else if (type == GdalApi::WkbMultiLineString) {
                            const int parts = api.G_GetGeometryCount(g);
                            for (int p = 0; p < parts; ++p)
                                collectLine(api.G_GetGeometryRef(g, p));
                        } else if (type == GdalApi::WkbPoint) {
                            collectPoint(g);
                        } else if (type == GdalApi::WkbMultiPoint) {
                            const int parts = api.G_GetGeometryCount(g);
                            for (int p = 0; p < parts; ++p)
                                collectPoint(api.G_GetGeometryRef(g, p));
                        }
                    }
                    api.F_Destroy(f);
                }
            }
            api.Close(ds);
        }
        if (overlay->failed || (overlay->lines.isEmpty() && overlay->points.isEmpty()))
            continue;

        QPainterPath path;
        for (const QPolygonF &line : std::as_const(overlay->lines)) {
            for (int i = 0; i < line.size(); ++i) {
                const QPointF scenePt = toScene(line[i]);
                if (i == 0)
                    path.moveTo(scenePt);
                else
                    path.lineTo(scenePt);
            }
        }
        // Points are drawn as near-zero-length subpaths: with a round-capped
        // cosmetic pen each one renders as a dot of constant screen size, so
        // the markers neither vanish when zooming out nor swell when zooming
        // in. One path item per layer keeps even 100k+ points responsive.
        const bool hasPoints = !overlay->points.isEmpty();
        for (const QPointF &pt : std::as_const(overlay->points)) {
            const QPointF scenePt = toScene(pt);
            path.moveTo(scenePt);
            path.lineTo(scenePt + QPointF(1e-3, 0.0));
        }

        QPen pen(overlayColor(oi), hasPoints && overlay->lines.isEmpty() ? 4.0 : 1.6);
        pen.setCosmetic(true);
        if (hasPoints)
            pen.setCapStyle(Qt::RoundCap);
        QGraphicsPathItem *item = m_scene->addPath(path, pen);
        item->setZValue(1);
        m_overlayItems.append(item);
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
        m_crsLabel->setText(tr("CRS: —"));
        m_resLabel->setText(tr("Resolution: —"));
        m_cursorLabel->clear();
        return;
    }
    m_crsLabel->setText(tr("CRS: %1").arg(layer->crsName));
    const double res = std::hypot(layer->gt[1], layer->gt[4]);
    m_resLabel->setText(tr("Resolution: %1%2/cell")
                            .arg(QString::number(res, 'g', 5),
                                 layer->geographic ? QStringLiteral("°")
                                                   : QStringLiteral(" m")));
}

void ViewerPage::onHover(const QPointF &scenePos)
{
    RasterLayer *layer = currentLayer();
    if (!layer || !layer->loaded) {
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
    // Render the whole scene at the display buffer's resolution instead of
    // grabbing the viewport: the export no longer depends on the window size
    // or the current zoom, and includes basemap and overlay as drawn.
    const QRectF rect = m_scene->sceneRect();
    const int w = qMax(1, qRound(rect.width()));
    const int h = qMax(1, qRound(rect.height()));
    QImage img(w, h, QImage::Format_RGB32);
    img.fill(canvasBg());
    QPainter p(&img);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    m_scene->render(&p, QRectF(0, 0, w, h), rect);
    p.end();
    const int quality = qstrcmp(format, "PNG") == 0 ? -1 : 92;
    if (!img.save(file, format, quality)) {
        m_cursorLabel->setText(tr("Export failed: %1").arg(file));
    }
}
