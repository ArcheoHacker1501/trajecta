#pragma once

#include <QGraphicsView>
#include <QHash>
#include <QPainterPath>
#include <QSet>
#include <QStringList>
#include <QVector>
#include <QWidget>

#include <memory>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGraphicsPathItem;
class QGraphicsPixmapItem;
class QGraphicsScene;
class QLabel;
class QVBoxLayout;
class QNetworkAccessManager;
class QSlider;
class QStackedLayout;
class QTimer;
class QToolButton;
class RangeSlider;

struct RasterLayer;
struct VectorOverlay;
class LayerItemDelegate;

// ---------------------------------------------------------------------------
// MapView — canvas with wheel zoom, drag pan and a scalebar overlay.
// ---------------------------------------------------------------------------
class MapView : public QGraphicsView
{
    Q_OBJECT

public:
    explicit MapView(QWidget *parent = nullptr);

    // Real-world units represented by one scene pixel (for the scalebar).
    void setUnitsPerScenePixel(double units, bool geographicCrs);
    void fitAll();

    // Bottom-right credit line, shown while the satellite basemap is active.
    void setAttribution(const QString &text);

signals:
    void hoverScenePos(QPointF scenePos);
    void hoverLeft();
    void viewChanged();   // zoom / pan / resize — the basemap listens to this

protected:
    void wheelEvent(QWheelEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void leaveEvent(QEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void scrollContentsBy(int dx, int dy) override;
    void drawForeground(QPainter *painter, const QRectF &rect) override;

private:
    double m_unitsPerScenePx = 0.0;
    bool m_geographic = false;
    QString m_attribution;
    // Four corner wedges covering the map, so it ends in the same radius as
    // the panel behind it. Cached: the boolean op that builds them is far too
    // slow to repeat on every repaint, and drawForeground runs on all of them.
    QPainterPath m_cornerPath;
    QSize m_cornerSize;
};

// ---------------------------------------------------------------------------
// LegendBar — vertical colorbar reflecting the active color scale + stretch.
// ---------------------------------------------------------------------------
class LegendBar : public QWidget
{
    Q_OBJECT

public:
    explicit LegendBar(QWidget *parent = nullptr);
    void setState(const QVector<QRgb> &lut, double lowValue, double highValue);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QVector<QRgb> m_lut;
    double m_low = 0.0;
    double m_high = 1.0;
};

// ---------------------------------------------------------------------------
// ViewerPage — the "Viewer" tab: layer picker, color scale, stretch, value
// filter, opacity, LCPA path overlay, scalebar/CRS/cursor readout, PNG export.
// GDAL is loaded dynamically from the detected OSGeo4W installation.
// ---------------------------------------------------------------------------
class ViewerPage : public QWidget
{
    Q_OBJECT

public:
    explicit ViewerPage(QWidget *parent = nullptr);
    ~ViewerPage() override;

    // Where to look for gdal*.dll and its PROJ/GDAL data (from MainWindow's
    // environment discovery). May be called again after "Locate GDAL folder".
    void configureGdal(const QStringList &dllDirs,
                       const QString &projDataDir, const QString &gdalDataDir);

    // Layer registration. Rasters appear in the layer dropdown (deduplicated
    // by path); vector files appear in the separate "Overlay" dropdown and are
    // drawn on top of whichever raster is showing. Lines and points are both
    // accepted (LCPA routes, generated sample points).
    void registerRaster(const QString &label, const QString &path,
                        bool select = false);
    void registerVectorOverlay(const QString &label, const QString &path);

    // "Open raster..." / hidden --viewer-load hook.
    void openRasterFile(const QString &path);

    // Close every open GDAL dataset without forgetting the layers. Called
    // before a run: the engine deletes and recreates its output rasters, and
    // Windows refuses to delete a file this page still holds open.
    void releaseFiles();

    // Re-reads the colour theme. The map canvas, the scalebar and the vector
    // overlay are painted by hand, so the stylesheet cannot reach them.
    void applyTheme();

protected:
    // Deferred layer loading: selections made while the page is hidden (e.g.
    // auto-registration at run end) only hit GDAL on first show.
    void showEvent(QShowEvent *event) override;
    // Catches clicks on the bin in the layer list before the combo's own
    // container turns them into a selection.
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    bool ensureGdal();
    RasterLayer *currentLayer() const;
    void selectLayer(int comboIndex);
    // Asks, then drops the layer from the list and releases its dataset. The
    // file on disk is not touched.
    void confirmRemoveLayer(int index);
    void scheduleRebuild();
    void rebuildImage();
    void rebuildOverlay();
    void updateLegend();
    // Parks the floating colour bar in the top-right of the map. Called on
    // every view resize, since it is positioned by hand rather than laid out.
    void positionLegend();
    void updateInfoStrip();
    void updateFilterUi();
    void applyFilterFromSpins();
    void onHover(const QPointF &scenePos);
    void exportImage();

    double percentileToValue(const RasterLayer &layer, double pct) const;
    double valueToPercentile(const RasterLayer &layer, double v) const;

    // Satellite basemap (Esri World Imagery). Web-mercator tiles are warped
    // into the raster's scene with per-tile quad transforms.
    bool ensureBasemapTransforms(const RasterLayer &layer);
    void updateBasemap();
    void clearBasemap(bool dropTransforms);
    void fetchTile(int z, int x, int y);

    // GDAL configuration
    QStringList m_gdalDirs;
    QString m_projData;
    QString m_gdalData;
    bool m_gdalFailed = false;

    // Layers
    std::vector<std::unique_ptr<RasterLayer>> m_layers;
    // Overlays are independent of each other: each has a tick box in the panel
    // floating over the canvas, and any combination can be drawn at once.
    std::vector<std::unique_ptr<VectorOverlay>> m_overlays;
    QList<QCheckBox *> m_overlayChecks;
    LayerItemDelegate *m_layerDelegate = nullptr;
    QList<QGraphicsPathItem *> m_overlayItems;
    static QColor overlayColor(int index);

    // Display state
    int m_colormapIndex = 0;
    int m_stretchIndex = 1;      // default: percentile 2-98
    double m_filterLo = 0.0;     // absolute values of the current layer
    double m_filterHi = 0.0;
    bool m_percentMode = false;
    bool m_updatingUi = false;
    int m_deferredSelect = -1;

    // Basemap state
    QNetworkAccessManager *m_net = nullptr;
    QHash<QString, QGraphicsPixmapItem *> m_tiles;   // "z/x/y" -> scene item
    QSet<QString> m_pendingTiles;
    int m_basemapGen = 0;        // invalidates in-flight replies on layer switch
    void *m_ctLayerToMerc = nullptr;   // OGRCoordinateTransformationH
    void *m_ctMercToLayer = nullptr;
    QString m_ctWkt;             // CRS the transforms above were built for
    QTimer *m_basemapTimer = nullptr;

    // Widgets
    QComboBox *m_layerCombo = nullptr;
    QComboBox *m_colormapCombo = nullptr;
    QComboBox *m_stretchCombo = nullptr;
    QSlider *m_opacitySlider = nullptr;
    QWidget *m_overlayPanel = nullptr;   // floats over the top-left of the canvas
    QVBoxLayout *m_overlayPanelLayout = nullptr;
    QCheckBox *m_basemapToggle = nullptr;
    RangeSlider *m_rangeSlider = nullptr;
    QDoubleSpinBox *m_filterLoSpin = nullptr;
    QDoubleSpinBox *m_filterHiSpin = nullptr;
    QToolButton *m_percentToggle = nullptr;
    MapView *m_view = nullptr;
    QGraphicsScene *m_scene = nullptr;
    QGraphicsPixmapItem *m_pixmapItem = nullptr;
    QStackedLayout *m_canvasStack = nullptr;
    QLabel *m_placeholder = nullptr;
    LegendBar *m_legend = nullptr;
    QLabel *m_crsLabel = nullptr;
    QLabel *m_resLabel = nullptr;
    QLabel *m_cursorLabel = nullptr;
    QTimer *m_rebuildTimer = nullptr;
};
