#pragma once

#include "walkthrough.h"

#include <QGraphicsView>
#include <QHash>
#include <QPainterPath>
#include <QSet>
#include <QStringList>
#include <QVector>
#include <QWidget>

#include <memory>

class QTemporaryDir;

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFrame;
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

// Just enough of a raster to turn a point on the scene into a point on the
// ground: the geotransform, and the two grids the scene sits between (the
// file's own pixels and the decimated ones the display buffer holds).
//
// Kept as a copy of the layer that is on screen rather than as an index into
// the layer list, because that list can lose entries — an index would then
// quietly point at a different raster.
struct ViewFrame {
    bool valid = false;
    double gt[6] = {0, 1, 0, 0, 0, -1};
    int srcW = 0, srcH = 0;
    int dispW = 0, dispH = 0;
    QString wkt;   // the CRS those map coordinates are in
};

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
    // The same two values, for the export: it draws its own scalebar at its own
    // size and needs the scale the view is working with.
    double unitsPerScenePx() const { return m_unitsPerScenePx; }
    bool isGeographic() const { return m_geographic; }
    void fitAll();

    // Where the view is looking, in scene units, and its counterpart: put the
    // view *there*, at that magnification. Together they let a framing survive
    // a change of layer, where the scene is rebuilt on a different pixel grid
    // — see ViewerPage::selectLayer().
    QPointF centreInScene() const;
    void showAt(const QPointF &centreScene, double viewScale);

    // Re-reads from the theme whether the canvas paints its own background or
    // lets what is behind it show through, and applies the answer.
    void applyCanvasBackground();

    // Bottom-right credit line, shown while the satellite basemap is active.
    void setAttribution(const QString &text);
    bool hasAttribution() const { return !m_attribution.isEmpty(); }
    // The feature panel follows the pointer's meaning: while the map is being
    // dragged the cursor belongs to the drag, and nothing may take it back.
    bool isPanning() const { return m_panning; }

signals:
    void hoverScenePos(QPointF scenePos);
    void hoverLeft();
    void viewChanged();   // zoom / pan / resize — the basemap listens to this
    // A press and a release in the same place. Not mousePressEvent: every
    // press here begins a pan, and a map you cannot drag because the drag is
    // read as a click would be a poor trade for an information panel.
    void clicked(QPointF scenePos);

protected:
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void leaveEvent(QEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void scrollContentsBy(int dx, int dy) override;
    void drawForeground(QPainter *painter, const QRectF &rect) override;

private:
    // Panning is done here rather than through ScrollHandDrag: that mode only
    // accepts the left button and stops dead at the edge of the scene, so the
    // map could never be dragged clear of the window the way a web map can.
    bool m_panning = false;
    QPoint m_panAnchor;
    QPoint m_pressAt;        // where the press landed, to tell a click from a drag
    // Widens the view's own scene rect by about a screen on each side, which is
    // what lets the drag continue past the edge of the data.
    void updatePanBounds();

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
    // The "..." button and the --viewer-load switch: opens whatever the file
    // turns out to be. GDAL decides from the content, not from the extension,
    // so a vector saved as .txt still arrives in the overlay list and a GeoTIFF
    // named .dat still becomes a raster. Returns false, with `error` filled,
    // when the file is neither.
    bool openAnyFile(const QString &path, QString *error = nullptr);

    // Close every open GDAL dataset without forgetting the layers. Called
    // before a run: the engine deletes and recreates its output rasters, and
    // Windows refuses to delete a file this page still holds open.
    void releaseFiles();

    // Re-reads the colour theme. The map canvas, the scalebar and the vector
    // overlay are painted by hand, so the stylesheet cannot reach them.
    void applyTheme();

    // The Viewer block of the guided walkthrough. Built here because the
    // controls it points at are private to this page.
    QVector<TourStep> walkthroughSteps();

    // The two example layers the tour shows: a small DEM and a set of sample
    // points, both carried inside the executable. They are unpacked into a
    // temporary folder, registered like any other layer, and taken away again
    // when the tour ends — so the Viewer is never explained against an empty
    // canvas, and nothing is left behind on disk.
    //
    // Returns false when GDAL is unavailable, in which case the tour carries on
    // and explains the page in words.
    bool loadTourSamples();
    void unloadTourSamples();

    // Hidden --pick-demo switch (testing): sends a real press and release to
    // the map at the position of one of the points on screen, so the whole
    // path — the view's click detection, the hit test, the panel — is
    // exercised rather than the panel being filled in directly.
    void clickFeatureForTest(int pointIndex = 0);
    // Opens the colour wheel for one overlay, so the popup can be photographed
    // without the right-click a screenshot run cannot make.
    void pickColourForTest(int overlayIndex = 0);
    // Sets one overlay's size directly, the way the wheel's slider would,
    // without opening it — for confirming that a size change lands on the one
    // layer it was meant for and nowhere else.
    void setOverlaySizeForTest(int overlayIndex, int percent);

protected:
    // Files dragged onto the page from Explorer, Finder or a file manager.
    //
    // Accepted anywhere on the Viewer, not only over the canvas: the layer
    // list, the controls and the map are all one target, because a person
    // dragging a DEM onto this page has already said what they want and should
    // not have to aim. Several files at once is the normal case — two rasters
    // to compare, a route and the DEM under it — so the whole drop is taken and
    // whatever cannot be read is reported once, at the end.
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dragLeaveEvent(QDragLeaveEvent *event) override;
    void dropEvent(QDropEvent *event) override;

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
    // How big the exported image should be and what goes on top of it. Asked
    // before the file name, and remembered between runs.
    struct ExportSettings {
        bool scalebar = true;
        bool northArrow = true;
        bool byDpi = true;
        int dpi = 300;
        int width = 0;
        int height = 0;
        // Hand the finished file to the system's image viewer.
        bool openWhenDone = false;
    };
    bool askExportSettings(ExportSettings *out);
    // Greys out the colour ramps that a colour-blind reader cannot follow.
    void applyCvdSafeFilter();

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
    // The colour this overlay is drawn in: the one the user picked for it, or
    // the automatic one for its position in the list.
    QColor overlayColor(int index) const;
    static QColor automaticOverlayColor(int index);
    // Right-click on a row: opens the wheel, applies every change live.
    void pickOverlayColour(int index, const QPoint &globalPos);
    // The ramp for a point layer coloured by one of the coherence scores.
    static QColor scoreColour(double t, bool diverging);

    // Removes an imported vector from the list, its tick box with it. Rasters
    // have the bin in their dropdown; without this a vector opened by mistake
    // could only be hidden, never taken back out.
    void removeOverlay(int index);
    // Drops one overlay and its row together; the panel is then rebuilt so the
    // colours and the bins line up with the new positions.
    void dropOverlayAt(int index);
    void rebuildOverlayPanel();
    // Shows or hides the Overlays panel and sizes it to whatever it now holds.
    // One place, because measuring it correctly turned out to need more than a
    // call to adjustSize() and the four callers had no business each knowing
    // what — see the definition.
    void fitOverlayPanel();

    // Vector-only display. With no raster to hang them on, the overlays define
    // the scene themselves: this is the frame that does it, and m_vectorScale
    // is zero whenever a raster owns the scene instead.
    double m_vectorScale = 0.0;   // scene pixels per map unit
    QRectF m_vectorExtent;        // map extent the scene was framed on
    QString m_vectorWkt;          // CRS of that extent

    // The raster currently on screen, as far as geometry goes. Written at the
    // end of every successful selectLayer() and read by the next one, which is
    // what lets the view stay where it was when the layer underneath changes.
    ViewFrame m_shownFrame;

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
    // The two panels the page is made of, kept because the walkthrough lights
    // whole cards: the strip of controls, and the frame the map is drawn in.
    QWidget *m_controlsCard = nullptr;
    QWidget *m_canvasHolder = nullptr;
    QComboBox *m_layerCombo = nullptr;
    QComboBox *m_colormapCombo = nullptr;
    QComboBox *m_stretchCombo = nullptr;
    QSlider *m_opacitySlider = nullptr;
    QWidget *m_overlayPanel = nullptr;   // floats over the top-left of the canvas
    QVBoxLayout *m_overlayPanelLayout = nullptr;
    QCheckBox *m_basemapToggle = nullptr;
    // Restricts the ramp list to the ones a colour-blind reader can follow.
    QCheckBox *m_cvdSafeToggle = nullptr;
    RangeSlider *m_rangeSlider = nullptr;
    QDoubleSpinBox *m_filterLoSpin = nullptr;
    QDoubleSpinBox *m_filterHiSpin = nullptr;
    QToolButton *m_percentToggle = nullptr;
    QToolButton *m_openButton = nullptr;
    QToolButton *m_exportButton = nullptr;
    MapView *m_view = nullptr;
    QGraphicsScene *m_scene = nullptr;
    QGraphicsPixmapItem *m_pixmapItem = nullptr;
    QStackedLayout *m_canvasStack = nullptr;
    QLabel *m_placeholder = nullptr;
    LegendBar *m_legend = nullptr;
    // The scene was framed while this page was not on screen, so the view it
    // was fitted to was the stacked page's default size and not the real one.
    // Set at every fit, acted on the next time the page is shown.
    bool m_framedWhileHidden = false;
    // Whether the identify pointer is currently showing. Cached rather than
    // recomputed: the hit test walks every point of every overlay, and a mouse
    // move that changes nothing must not pay for it twice.
    bool m_hoveringFeature = false;

    // --- feature information ---
    // What a click on a point or a line opens: a small panel in the bottom
    // right of the map, listing that feature's own attributes. One at a time —
    // clicking another feature replaces it, the cross closes it.
    QFrame *m_featurePanel = nullptr;
    QLabel *m_featureTitle = nullptr;
    QLabel *m_featureBody = nullptr;
    void buildFeaturePanel();
    void positionFeaturePanel();
    void onCanvasClicked(QPointF scenePos);
    // What is within reach of `scenePos`, if anything. Shared by the click that
    // opens the panel and the hover that changes the cursor, so the pointer can
    // never promise a feature the click then fails to find.
    bool pickFeatureAt(const QPointF &scenePos, int &overlayIndex, bool &isPoint,
                       int &geometryIndex) const;
    void showFeatureInfo(int overlayIndex, bool isPoint, int geometryIndex);
    QLabel *m_crsLabel = nullptr;
    QLabel *m_resLabel = nullptr;
    // The Esri credit line, moved off the canvas itself. Not managed by the
    // info strip's QHBoxLayout: a stretch on each side only balances it
    // between whatever the CRS/resolution pair on the left and the cursor
    // readout on the right each happen to need, which is not the same as the
    // row's true centre when the two differ — the cursor readout is empty
    // whenever nothing is under the pointer, and the label visibly drifted
    // right. It is instead a free-floating child of m_infoRow, kept centred
    // on the row's actual midpoint by repositionAttribution(), independent of
    // whatever else the row holds. See setAttribution() call sites in the
    // constructor.
    QLabel *m_attributionLabel = nullptr;
    // The info strip m_attributionLabel floats over. Kept as a member only
    // for repositionAttribution() and the resize filter that calls it.
    QWidget *m_infoRow = nullptr;
    void repositionAttribution();
    QLabel *m_cursorLabel = nullptr;
    QTimer *m_rebuildTimer = nullptr;

    // Where the tour's example layers were unpacked, and what they were called
    // once registered. The directory removes itself with the object, but the
    // tour removes it earlier than that.
    std::unique_ptr<QTemporaryDir> m_tourDir;
    QString m_tourRasterPath;
    QString m_tourVectorPath;
    // The layer that was selected before the tour put its own on screen; -1
    // when the tour is not running.
    int m_tourPrevLayer = -1;
    bool m_tourPrevLayerKnown = false;
};
