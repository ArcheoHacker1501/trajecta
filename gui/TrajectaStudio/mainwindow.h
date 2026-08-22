#pragma once

#include <QElapsedTimer>
#include <QMainWindow>

#include "checkpointstore.h"
#include "largepages.h"
#include "trajectarunner.h"
// For RunTicker, which the status bar owns and the run signals feed.
#include "uiwidgets.h"

class QCheckBox;
class QDialog;
class QMenu;
class QComboBox;
class QDoubleSpinBox;
class QGridLayout;
class QGroupBox;
class QHBoxLayout;
class QLabel;
class QLineEdit;
class QListWidget;
class QProgressBar;
class QPushButton;
class QScrollArea;
class QSpinBox;
class QStackedWidget;
class QTextBrowser;
class QTimer;
class QToolButton;
class QUrl;
class QVariantAnimation;
class QAction;

class BatchPage;
class ConsoleView;
class PathPicker;
class PostBatchPage;
class SmoothComboBox;
class TourOverlay;
class ViewerPage;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

    // Used by the hidden --autorun switch (testing): fills the form from the
    // saved settings and immediately starts the analysis.
    void triggerRun();

    // Used by the hidden --autorun-interp switch (testing): starts the NNI
    // post-processing run from the saved settings.
    void triggerInterpRun();

    // Used by the hidden --autorun-points switch (testing): generates the
    // sample points. With thenAnalysis, chains straight into the analysis once
    // they are written, which is how the "the run consumes the layer you
    // inspected" path gets exercised end to end.
    void triggerPointsRun(bool thenAnalysis = false);

    // Used by the hidden --page switch (testing):
    // 0 setup, 1 post, 2 viewer, 3 guide. ("run" is an alias for the setup
    // page, whose tail the live run panel now is; "about" for the guide,
    // landing on its own About section, last in the sidebar.)
    void showPage(int index);

    // Used by the hidden --guide-page switch (testing): opens one page of the
    // Guide, which is otherwise only reachable by clicking the list on the
    // left. 0 is Overview.
    void showGuideSectionForTest(int index) { showGuideSection(index); }

    // Used by the hidden --page about switch (testing): the About section is
    // always last, but how many sections precede it depends on how many
    // <!--nav:--> markers the guide text carries, so main.cpp cannot just
    // hardcode an index.
    int guidePageCount() const;

    // Used by the hidden --mode switch (testing): "fete", "lcpa" or "batch".
    // Batch mode is otherwise only reachable by clicking its card.
    void selectMode(const QString &name);

    // Used by the hidden --autorun-compare switch (testing): fills the two
    // pickers and runs the known-route comparison.
    void triggerRouteComparison(const QString &computed, const QString &known,
                                double tolerance);

    // Used by the hidden --open-log switch (testing): unfolds every log canvas,
    // which is otherwise a click the user makes and a screenshot cannot.
    void openAllLogs();

    // The guided walkthrough. Reached from the "tutorial" link in the Guide and
    // from the offer made on a first run; --tour / --tour-step drive it for
    // screenshots. `index` is 0-based.
    void startWalkthrough(int index = 0);
    // Maximises the window first. What the Guide's link and the first-run offer
    // both call, because both of them say so in the question they ask.
    void startWalkthroughMaximised();
    // Used by the hidden --tour-autoclose switch (testing).
    void closeWalkthrough();

    // Used by the hidden --post-mode switch (testing): "nni" or "compare".
    // Same reason as --mode: the choice is a card, and a screenshot run cannot
    // click one.
    void selectPostMode(const QString &name);

    // Used by the hidden --batch-load / --batch-run switches (testing): fills
    // the batch page from a .trjbatch file and starts it.
    void loadBatchFile(const QString &path);
    void triggerBatchRun();

    // Used by the hidden --post-batch-load / --post-batch-run switches
    // (testing): the same two hooks, for the post-processing batch page.
    void loadPostBatchFile(const QString &path);
    void triggerPostBatchRun();

    // Used by the hidden --viewer-load switch (testing): opens a raster or a
    // vector layer in the Viewer page. May be passed more than once.
    void viewerLoadFile(const QString &path);

    // Used by the hidden --pick-demo switch (testing): clicks a point on the
    // map so the feature information panel can be photographed.
    void pickFeatureForTest(int pointIndex);
    // Used by the hidden --wheel-demo switch (testing): opens the colour wheel
    // for one overlay so the popup can be photographed.
    void pickColourForTest(int overlayIndex);
    // Used by the hidden --wheel-size-demo switch (testing): sets one
    // overlay's size directly, to confirm the change stays on that layer.
    void setOverlaySizeForTest(int overlayIndex, int percent);

    // Used by the hidden --scroll-end switch (testing): scrolls the long page
    // currently showing (setup form or guide) to `fraction` of its range.
    void scrollSetupToEnd(double fraction = 1.0);

    // Used by the hidden --open-combo switch (testing): drops open the n-th
    // combo box of the page on screen, so the popup can be screenshotted.
    void openComboForTest(int index);

    // Used by the hidden --progress switch (testing): puts the run panel's bar
    // at a given percentage without an engine behind it, so the sweep along a
    // half-filled bar can be photographed.
    // `paused` also dresses the status-bar ticker in its paused colours, which
    // is the other half of what --progress exists to photograph.
    void setProgressForTest(int percent, bool paused = false);
    // Hidden --theme switch: dresses this launch in another palette so a
    // screen can be checked in all of them. main() puts the saved choice back.
    void applyThemeForTest(int index) { applyTheme(index); }
    // Hidden --drop-demo switch: sends the Viewer a real drag-and-drop of the
    // given files, so the handlers a file manager would exercise are exercised
    // by a test run too.
    void dropOnViewerForTest(const QStringList &paths);
    // Hidden --ticker-drawer switch, to photograph the drawer open.
    void openTickerDrawerForTest()
    {
        if (m_ticker)
            m_ticker->openDrawerForTest();
    }
    // Hidden --advanced-settings switch, to photograph the dialog open: it is
    // modal, so nothing else on the CLI can reach the button that opens it.
    void showAdvancedSettingsForTest() { showAdvancedSettings(); }

protected:
    void closeEvent(QCloseEvent *event) override;
    void changeEvent(QEvent *event) override;
    // The window accepts dropped files as a fallback, so that a file dropped
    // anywhere on it opens in the Viewer. The Viewer and the path fields sit
    // deeper in the widget tree and are found first, so they keep their own
    // behaviour; this only catches what would otherwise fall through.
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;
    bool eventFilter(QObject *watched, QEvent *event) override;
#ifdef Q_OS_WIN
    // Frameless windows lose the frame the OS resizes by. Answering
    // WM_NCHITTEST ourselves hands that job back to Windows, so edge resizing,
    // Aero Snap and the maximise-respects-the-taskbar behaviour all survive.
    bool nativeEvent(const QByteArray &eventType, void *message,
                     qintptr *result) override;
#endif

private:
    // Page construction
    QWidget *buildTopBar();
    QWidget *buildStatusBar();
    QWidget *buildSetupPage();
    QWidget *buildPostPage();
    QWidget *buildGuidePage();
    // The Guide's home: what Trajecta is and what its core tools are.
    // Widgets rather than a document, because the intro paragraph needs to
    // link back into handleGuideLink like every other page's prose.
    QWidget *buildGuideOverviewPage();
    // The Guide's last section: the logo and the project links. Also, when
    // kShowAboutTab is flipped on in mainwindow.cpp, a standalone top-level
    // tab — the same widget serves both, laid out the same way in each.
    QWidget *buildAboutPage();
    void showGuideSection(int index);
    void handleGuideLink(const QUrl &url);
    // titleHelp, when not empty, puts a "?" badge right after the title
    // instead of the usual subtitle line below it — for a card whose
    // description belongs behind a click, not permanently on screen.
    QWidget *makeCard(const QString &title, const QString &subtitle, QWidget *content,
                      const QString &titleHelp = QString());

    // Behaviour
    void switchPage(int index);
    void updateModeUi();
    void updatePointsSourceUi();
    void updateGeneratedPointsPreview();
    void updateEnvironmentStatus();
    void configureViewerGdal();
    QString validationError() const;
    TrajectaRunner::Parameters collectParameters() const;
    // Fingerprint of everything that determines the generated point layer.
    // Lets the analysis reuse a layer produced by "Generate points" instead of
    // writing it again, and tells the user when it went stale.
    QString generationKey() const;
    void updateGeneratedPointsStatus();
    void startRun();
    void startPointsRun();
    void startInterpRun();
    void beginRun(const TrajectaRunner::Parameters &params);

    // True while anything owns the engine — a single run, a points generation,
    // an interpolation or a batch. One question with one answer: the batch has
    // a runner of its own, so "is the single-run runner busy" was never the
    // whole story, and the start buttons that did not ask it could put two
    // engines on the machine at once.
    bool engineBusy() const;
    // Says so, and refuses. Returns true when the caller must give up.
    bool refuseIfEngineBusy();
    // Rebuilds the status bar's ticker from whatever is running. Cheap, and
    // called from everywhere a run changes state — including once a second
    // while one is in progress, which is what keeps the estimate honest.
    void refreshRunTicker();
    // The walkthrough's demonstration of that ticker: nothing is running during
    // a tour, and a screen explaining an invisible widget explains nothing.
    void setTourTicker(bool on);
    void onRunFinished(TrajectaRunner::Outcome outcome, const QString &report);
    void onPauseStateChanged(bool paused);
    void openOutputFolder();
    void locateEngine();
    void locateGdal();
    // Repaints everything the stylesheet cannot reach (gear icon, console,
    // map canvas, guide) after a palette change.
    void applyTheme(int index);
    // Font choice is stored separately from the palette and survives a theme
    // change, so the two menus are genuinely independent.
    void applyUiFont(int index);
    // Opens the Advanced settings dialog behind the gear menu (large memory
    // pages today; a sidebar list of whatever else joins it later).
    void showAdvancedSettings();
    // Window controls living in the top bar, in place of the native title bar.
    void toggleMaximised();
    void refreshWindowButtons();

    // Environment discovery
    struct GdalEnvironment {
        bool found = false;  // GDAL DLLs reachable somewhere
        QString binDir;      // to prepend to PATH (empty = already reachable)
        QString projData;    // folder containing proj.db (empty = unknown)
        QString gdalData;    // GDAL_DATA folder (empty = unknown)
    };
    QString engineExePath() const;
    GdalEnvironment detectGdalEnvironment() const;
    // Every folder that may hold gdal*.dll, most specific first.
    QStringList gdalDllDirs() const;
    // GDAL is loaded on demand — the interface has to start without it — so
    // anything that reads a raster or a vector has to ask for it first. Returns
    // false when the library genuinely is not there.
    bool ensureGdalLoaded();
    static bool dirHasGdal(const QString &dir);

    // Persistence
    void loadSettings();
    void saveSettings() const;

    // --- Navigation ---
    QStackedWidget *m_pages = nullptr;
    QList<QPushButton *> m_navButtons;
    QScrollArea *m_setupScroll = nullptr;

    // --- Setup page widgets ---
    // Single vs batch, asked before which tool: picking Batch hides the tool
    // card and the single-run form and shows the batch page in their place,
    // on the same Analysis Setup page rather than on a separate tab.
    QPushButton *m_modeSingle = nullptr;
    QPushButton *m_modeBatch = nullptr;
    QWidget *m_cardAnalysisType = nullptr;
    QPushButton *m_modeFete = nullptr;
    QPushButton *m_modeLcpa = nullptr;
    BatchPage *m_batchPage = nullptr;
    // Every card of the single-run form, collected once so batch mode can hide
    // them all without each one needing its own member.
    QList<QWidget *> m_singleRunCards;

    QWidget *m_pointsSourceLabel = nullptr;
    QComboBox *m_pointsSourceCombo = nullptr;
    QWidget *m_pointsLabel = nullptr;
    QWidget *m_originLabel = nullptr;
    QWidget *m_destinationsLabel = nullptr;
    PathPicker *m_demPicker = nullptr;
    PathPicker *m_pointsPicker = nullptr;
    PathPicker *m_originPicker = nullptr;
    PathPicker *m_destinationsPicker = nullptr;
    PathPicker *m_outputDirPicker = nullptr;

    // --- Sample point generation (FETE, "Generate from the DEM" source) ---
    // The whole group is hidden and disabled while the points come from a
    // file, and nothing in it is read when collecting the run parameters.
    QGroupBox *m_generateGroup = nullptr;
    QComboBox *m_genDensityCombo = nullptr;
    QWidget *m_genSpacingLabel = nullptr;
    QSpinBox *m_genSpacingSpin = nullptr;
    QWidget *m_genTargetLabel = nullptr;
    QSpinBox *m_genTargetSpin = nullptr;
    QComboBox *m_genArrangementCombo = nullptr;
    QWidget *m_genSeedLabel = nullptr;
    QSpinBox *m_genSeedSpin = nullptr;
    QSpinBox *m_genEdgeSpin = nullptr;
    QLineEdit *m_genNameEdit = nullptr;
    QLabel *m_genPreviewLabel = nullptr;
    QPushButton *m_genPointsButton = nullptr;
    QLabel *m_genStatusLabel = nullptr;
    // Layer written by the last successful "Generate points". While the form
    // still matches the fingerprint it was made with, the analysis consumes
    // this exact file instead of regenerating one.
    QString m_previewedPointsPath;
    QString m_previewedPointsKey;
    // Estimating the point count needs the share of DEM cells a point can sit
    // on. It comes from one decimated band read, cached per DEM path so
    // dragging a spin box does not re-read the raster.
    QString m_genCachedDem;
    double m_genValidFraction = -1.0;   // < 0: unknown (no GDAL, unreadable DEM)
    int m_genDemWidth = 0;
    int m_genDemHeight = 0;

    QGroupBox *m_modifiersGroup = nullptr;
    PathPicker *m_costVectorPicker = nullptr;
    QSpinBox *m_polylineBufferSpin = nullptr;
    PathPicker *m_costRasterPicker = nullptr;
    QCheckBox *m_barrierCheck = nullptr;
    QDoubleSpinBox *m_barrierSpin = nullptr;

    QComboBox *m_neighboursCombo = nullptr;
    // Shown only while the combo sits on "Custom…"; carries the real number.
    QSpinBox *m_neighboursCustom = nullptr;
    void refreshNeighboursCustom();
    // The number the engine will be given: the preset, or the custom box.
    int selectedNeighbours() const;
    // Says which unit the chosen cost function produces. Herzog is energy, the
    // rest are time, and the outputs must not be read as the same thing.
    QLabel *m_costUnitsNote = nullptr;
    void refreshCostUnitsNote();
    // Slope cut-off: moves steeper than this are refused outright. Off by
    // default, so nothing changes for anyone who does not ask for it.
    // Cost corridor: a second, optional LCPA output. The width stays
    // visible while the option is off, so its cost can be read first.
    QWidget *m_corridorRow = nullptr;
    QCheckBox *m_corridorCheck = nullptr;
    QDoubleSpinBox *m_corridorWidthSpin = nullptr;
    QWidget *m_corridorNameLabel = nullptr;
    QLineEdit *m_corridorNameEdit = nullptr;
    QCheckBox *m_slopeCapCheck = nullptr;
    QSpinBox *m_slopeCapUp = nullptr;
    QSpinBox *m_slopeCapDown = nullptr;
    QComboBox *m_costFunctionCombo = nullptr;
    QSpinBox *m_smoothingSpin = nullptr;

    QSpinBox *m_threadsSpin = nullptr;
    QSpinBox *m_ramSpin = nullptr;
    QCheckBox *m_verboseCheck = nullptr;
    QCheckBox *m_manifestCheck = nullptr;

    QLineEdit *m_slopeNameEdit = nullptr;
    QLineEdit *m_costNameEdit = nullptr;
    QWidget *m_additionalNameLabel = nullptr;
    QLineEdit *m_additionalNameEdit = nullptr;
    QWidget *m_totalNameLabel = nullptr;
    QLineEdit *m_totalNameEdit = nullptr;
    QWidget *m_densityNameLabel = nullptr;
    QLineEdit *m_densityNameEdit = nullptr;
    QWidget *m_pathRasterNameLabel = nullptr;
    QLineEdit *m_pathRasterNameEdit = nullptr;
    QWidget *m_pathLinesNameLabel = nullptr;
    QLineEdit *m_pathLinesNameEdit = nullptr;

    // Guided walkthrough: the overlay, and the list of screens it iterates.
    // Everything Trajecta-specific about the tour lives in buildWalkthrough(),
    // never in the overlay itself.
    TourOverlay *m_tour = nullptr;
    void buildWalkthrough();
    // R2: the tour has to navigate in order to show anything, and navigating
    // here changes state that is written to the settings when the window
    // closes. Where the user was is photographed before the tour starts and
    // put back when it ends, however it ends.
    void restoreAfterWalkthrough();
    // The ✕ on the callout. Asks before stopping — a tour abandoned by a
    // mis-click is not easy to notice, and there is no undo — and says where to
    // find it again afterwards, because "it is in the Guide" is exactly the
    // thing a user who has just left the tour has not been told yet.
    void confirmCloseWalkthrough();
    // True once the tour has put its example layers in the Viewer, so they are
    // taken away again exactly once, whichever screen the tour is closed on.
    bool m_viewerSamplesLoaded = false;
    int m_tourReturnPage = 0;
    QString m_tourReturnMode;
    QString m_tourReturnPostMode;
    // Whether each log canvas was open when the tour began, in the order
    // { run, post-processing, comparison }. The tour folds them all away and
    // puts them back at the end; empty while no tour is running.
    QVector<bool> m_tourReturnLogsOpen;
    // And whether each batch chunk was folded away, in page order. The tour
    // does the opposite here — it unfolds them — for the same reason: a folded
    // chunk is a header with nothing under it, and the screens about the rows
    // have nothing to point at. Empty while no tour is running.
    QVector<bool> m_tourReturnChunksFolded;
    // Same, for the post-processing batch page's own chunks.
    QVector<bool> m_tourReturnPostChunksFolded;

    QPushButton *m_runButton = nullptr;

    // The cards of the setup page, in the order they appear on it. Kept for one
    // reason: the walkthrough lights a whole card — its heading, its note and
    // its fields — rather than the controls inside it, because a section of a
    // form is what the reader is being shown and the heading is half of what
    // says which section it is.
    // Also carries the hardware-resources fields now — see buildSetupPage().
    QWidget *m_cardMode = nullptr;
    QWidget *m_cardInput = nullptr;
    QWidget *m_cardModifiers = nullptr;
    QWidget *m_cardAlgorithm = nullptr;
    QWidget *m_cardOutputs = nullptr;
    // The same, on the post-processing page.
    QWidget *m_cardPostTool = nullptr;

    // --- Post-processing page widgets (NNI) ---
    PathPicker *m_interpInputPicker = nullptr;
    PathPicker *m_interpOutputDirPicker = nullptr;
    QDoubleSpinBox *m_interpThresholdSpin = nullptr;
    QSpinBox *m_interpSpacingSpin = nullptr;
    // Keeps each block's real maximum as well as the grid cell; enabled
    // only while the spacing is actually discarding cells.
    QCheckBox *m_interpPeaksCheck = nullptr;
    // Known-route comparison: pure geometry over two vector layers, run
    // in the interface rather than by the engine.
    PathPicker *m_cmpComputedPicker = nullptr;
    PathPicker *m_cmpKnownPicker = nullptr;
    QDoubleSpinBox *m_cmpToleranceSpin = nullptr;
    QPushButton *m_cmpButton = nullptr;
    // The comparison gets the same furniture as an engine run — a state chip, a
    // foldable log, a summary card — because from the outside it is the same
    // thing: press a button, wait, read what came out. That it happens in the
    // interface rather than in the engine is an implementation detail.
    QWidget *m_cmpRunRow = nullptr;
    QWidget *m_cmpPanel = nullptr;
    QLabel *m_cmpChip = nullptr;
    QLabel *m_cmpPhase = nullptr;
    ConsoleView *m_cmpConsole = nullptr;
    QToolButton *m_cmpLogHandle = nullptr;
    QWidget *m_cmpSummaryCard = nullptr;
    QLabel *m_cmpSummaryTitle = nullptr;
    QLabel *m_cmpResult = nullptr;
    void runRouteComparison();
    void setCmpState(const QString &text, const QString &state);
    // Builds m_cmpPanel: the same card, chip, foldable log and summary as
    // buildRunPanel, without the parts that only make sense for a subprocess
    // (progress bar, Pause, Cancel, Open output folder).
    QWidget *buildComparePanel(QWidget *parent);
    // --- Site-corridor coherence ---
    // The third post-processing tool. Like the comparison it runs inside the
    // interface rather than in the engine: it is arithmetic over a raster and a
    // point layer, and finishes in seconds.
    PathPicker *m_cohRasterPicker = nullptr;
    PathPicker *m_cohPointsPicker = nullptr;
    PathPicker *m_cohOutPicker = nullptr;
    QDoubleSpinBox *m_cohRadiusSpin = nullptr;
    SmoothComboBox *m_cohThresholdCombo = nullptr;
    QDoubleSpinBox *m_cohThresholdSpin = nullptr;
    QLabel *m_cohCellNote = nullptr;
    QCheckBox *m_cohNullCheck = nullptr;
    SmoothComboBox *m_cohNullModeCombo = nullptr;
    QSpinBox *m_cohRepsSpin = nullptr;
    QCheckBox *m_cohSensCheck = nullptr;
    QLineEdit *m_cohSensEdit = nullptr;
    QLineEdit *m_cohEcdfEdit = nullptr;
    QCheckBox *m_cohEdgeCheck = nullptr;
    QCheckBox *m_cohRScriptCheck = nullptr;
    SmoothComboBox *m_cohVectorCombo = nullptr;
    QCheckBox *m_cohRasterCheck = nullptr;
    QLineEdit *m_cohPrefixEdit = nullptr;
    QWidget *m_cohRunRow = nullptr;
    QPushButton *m_cohButton = nullptr;
    QWidget *m_cohPanel = nullptr;
    QLabel *m_cohChip = nullptr;
    QLabel *m_cohPhase = nullptr;
    ConsoleView *m_cohConsole = nullptr;
    QToolButton *m_cohLogHandle = nullptr;
    QWidget *m_cohSummaryCard = nullptr;
    QLabel *m_cohSummaryTitle = nullptr;
    QLabel *m_cohResult = nullptr;
    bool m_cohRunning = false;
    QWidget *buildCoherencePanel(QWidget *parent);
    void runCoherence();
public:
    // Testing hook: fills the pickers and presses the button, so the whole
    // interface path can be exercised from the command line.
    void triggerCoherence(const QString &raster, const QString &points,
                          double radiusMetres = 0.0);
private:
    void setCohState(const QString &text, const QString &state);
    void refreshCoherenceUi();

    // Single vs batch, asked before which tool — the Post-processing page's
    // own "Analysis type" card, the same relationship the Processing page's
    // m_modeSingle/m_modeBatch have with its tool card.
    QPushButton *m_postModeSingle = nullptr;
    QPushButton *m_postModeBatch = nullptr;
    QWidget *m_postCardAnalysisType = nullptr;
    // The Post-processing page holds three unrelated tools; these pick which
    // one is on screen while Single analysis is selected.
    QPushButton *m_postModeNni = nullptr;
    QPushButton *m_postModeCompare = nullptr;
    QPushButton *m_postModeCoherence = nullptr;
    QWidget *m_postNniCard = nullptr;
    QWidget *m_postCompareCard = nullptr;
    QWidget *m_postCoherenceCard = nullptr;
    QWidget *m_postRunRow = nullptr;
    QWidget *m_postRunPanel = nullptr;
    // Every card of the single-tool forms on the Post-processing page,
    // collected the same way m_singleRunCards is on the Processing page, so
    // the BP card can hide them all without each one needing its own member.
    QList<QWidget *> m_postSingleRunCards;
    PostBatchPage *m_postBatchPage = nullptr;
    void updatePostModeUi();
    QSpinBox *m_interpRadiusSpin = nullptr;
    QLineEdit *m_interpNameEdit = nullptr;
    QPushButton *m_runInterpButton = nullptr;
    QScrollArea *m_postScroll = nullptr;

    // NNI's own hardware-resources card — shown only while NNI is the
    // selected post-processing tool, since Compare and Coherence run in the
    // interface and never touch this. Deliberately without a verbose
    // checkbox: NNI's single pass is not worth a debug transcript, and a
    // single-run NNI does not get a verbose option at all rather than
    // silently inheriting whatever the Processing page's checkbox happens to
    // hold. Large pages is not among these fields any more — see
    // advancedsettingsdialog.h — so both hardware boxes ask the same two
    // questions now: threads and RAM. Folded into m_cardPostTool rather than a
    // card of its own — see buildPostPage() — so this is a plain box, shown
    // only while NNI is selected.
    QWidget *m_postHardwareBox = nullptr;
    QSpinBox *m_postThreadsSpin = nullptr;
    QSpinBox *m_postRamSpin = nullptr;
    QCheckBox *m_postManifestCheck = nullptr;

    // --- Run panels ---
    // The same set of live-run widgets exists twice: on the "Run & results"
    // page (FETE/LCPA) and on the "Post-processing" page (NNI). m_activeUi
    // points at the panel of the run in progress.
    struct RunUi {
        QLabel *chip = nullptr;
        QLabel *phase = nullptr;
        QLabel *elapsed = nullptr;
        QProgressBar *progress = nullptr;
        ConsoleView *console = nullptr;
        // Folds the console away. Collapsed by default: the progress bar and
        // the summary are enough for a run that goes well.
        QToolButton *logHandle = nullptr;
        QPushButton *pauseButton = nullptr;
        QPushButton *cancelButton = nullptr;
        QPushButton *openFolderButton = nullptr;
        QLabel *summaryTitle = nullptr;
        QLabel *summaryBody = nullptr;
        QWidget *summaryCard = nullptr;
    };
    // `withCheckpointLinks` adds the save/resume pair to the log row: true for
    // the analysis panel, false for post-processing, which has no state worth
    // keeping.
    QWidget *buildRunPanel(RunUi &ui, QWidget *parent, const QString &idlePhrase,
                           QWidget *leadingButton = nullptr,
                           bool withCheckpointLinks = false);
    RunUi m_runUi;
    RunUi m_postUi;
    RunUi *m_activeUi = &m_runUi;
    // The FETE/LCPA run panel, now the last card of the setup page rather than
    // a page of its own. Kept so starting a run can scroll straight to it.
    QWidget *m_runPanel = nullptr;
    void revealRunPanel();

    // The GitHub button on the About page. Kept because its mark is dyed with
    // the palette's text colour, which changes with the theme.
    QPushButton *m_githubButton = nullptr;

    // --- Status bar environment indicators ---
    // The strip itself, which is what the walkthrough lights.
    QWidget *m_statusBar = nullptr;
    // The run ticker in the middle of it. Everything else that reports on a run
    // belongs to the page that started it; this is the one that is on screen
    // wherever the user has gone.
    RunTicker *m_ticker = nullptr;
    // What the single-run engine is doing, kept because the ticker has to name
    // it long after start() and the runner does not hold on to its parameters.
    QString m_runKind;
    QString m_runHardware;
    double m_runPercent = -1.0;
    QLabel *m_engineStatus = nullptr;
    // Kept so the walkthrough can point at them.
    QPushButton *m_locateEngineButton = nullptr;
    QPushButton *m_locateGdalButton = nullptr;
    QLabel *m_gdalStatus = nullptr;

    // --- Appearance ---
    QToolButton *m_gearButton = nullptr;
    // Lazily built on first use; outlives repeated opens so its own state
    // (which sidebar entry was last selected) survives closing and reopening it.
    QDialog *m_advancedSettings = nullptr;
    QWidget *m_topBar = nullptr;
    QToolButton *m_minButton = nullptr;
    QToolButton *m_maxButton = nullptr;
    QToolButton *m_closeButton = nullptr;
    QList<QAction *> m_themeActions;
    QList<QAction *> m_fontActions;
    // Automatic saving of a running analysis, in the gear menu next to the
    // appearance settings.
    QAction *m_autosaveAction = nullptr;
    QMenu *m_autosaveIntervalMenu = nullptr;
    QList<QAction *> m_autosaveIntervalActions;
    QAction *m_autosaveFolderAction = nullptr;
    // Kept so the walkthrough can find this one entry's own geometry inside
    // the rendered menu picture, for a caption of its own — see
    // renderMenuPicture() and step "2 — the gear" in buildWalkthrough().
    QAction *m_advancedSettingsAction = nullptr;
    // Mode cards: minimum width taken from their own captions, redone after
    // every theme or font change because both can change the type.
    void refreshModeCardWidths();
    // Every card of the setup page keeps its field labels in one grid column.
    // Left alone, each column is only as wide as its own longest label, so the
    // fields — and the "?" badges pinned to the right of the labels — start at
    // a different place in every card. These give them all the widest column,
    // and are redone whenever the type changes.
    QList<QPair<QGridLayout *, int>> m_labelColumns;
    void alignLabelColumns();
    void refreshAutosaveMenu();
    void chooseAutosaveFolder();
    // Called once, shortly after the window appears: offers to resume a run
    // that never got to finish.
    void offerCrashRecovery();
    // The one-time offer at the first start. Suppressed under the hidden
    // testing switches, and skipped without a trace when there is an
    // interrupted analysis to deal with first.
    void offerWalkthroughOnFirstRun();
    void resumeFromCheckpoint(const Checkpoint::Session &session);
    // The two checkpoint buttons on every run panel: keep a copy of the state
    // of a run in progress, and pick a saved one back up. The automatic offer
    // at start-up stays exactly as it was; these make the same state reachable
    // on purpose rather than only by accident of a crash.
    void exportCheckpointCopy();
    void importCheckpointAndResume();
    // Appends the pair to an existing button row, rather than returning a
    // widget of its own: they belong on the run panel's row of buttons, at its
    // left-hand end, and a wrapper would only get in the layout's way.
    void buildCheckpointButtons(QHBoxLayout *row, QWidget *parent);
    // The analysis panel's pair, kept so the walkthrough can light them.
    QPushButton *m_ckptSaveButton = nullptr;
    QPushButton *m_ckptLoadButton = nullptr;
    // Only one unfinished analysis can be kept at a time: the checkpoint folder
    // holds one pair of slots. Returns false when the user would rather keep
    // what is already saved than start something that overwrites it.
    bool confirmOverwritingSavedProcess(const QString &dir);
    // Records that the run being abandoned is being abandoned on purpose.
    // Called from closeEvent, where the runner's finished() signal will never
    // be delivered: without it, quitting during an analysis makes the next
    // start-up announce a crash that never happened.
    void markSessionDeliberate();
    // Engine path and GDAL/PROJ locations only; the rest of the run is filled
    // in by whoever is starting it.
    TrajectaRunner::Parameters currentEnvironment() const;
    // The folder the run in progress is saving into, empty when it is not
    // saving. Kept so the end of a run only ever clears its own state.
    QString m_lastRunCheckpointDir;
    // The guide is a small site now: a list of pages on the left, the
    // documents in the middle, the current page's sections on the right.
    QListWidget *m_guideNav = nullptr;
    QStackedWidget *m_guidePages = nullptr;
    // The "On this page" caption and the list below it, hidden and shown
    // together — see showGuideSection().
    QWidget *m_guideTocPanel = nullptr;
    QListWidget *m_guideToc = nullptr;
    // One entry per page, in the same order, holding that page's h3
    // headings. Index 0 is Overview, which has none.
    QList<QStringList> m_guideSections;

    // --- Viewer ---
    ViewerPage *m_viewer = nullptr;
    // Output files expected from the run in progress; registered with the
    // Viewer page (when they exist) once the run succeeds.
    struct PendingOutput {
        QString label;
        QString path;
    };
    QList<PendingOutput> m_pendingOutputs;
    QString m_pendingVector;   // LCPA paths shapefile
    QString m_pendingPoints;   // generated sample points shapefile (FETE)

    // --- Runtime ---
    TrajectaRunner *m_runner = nullptr;
    QElapsedTimer m_elapsed;
    QTimer *m_elapsedTimer = nullptr;
    // Pause bookkeeping: the elapsed label shows working time only, so time
    // spent paused is accumulated here and subtracted from m_elapsed.
    QElapsedTimer m_pauseClock;
    qint64 m_pausedMs = 0;
    QString m_lastOutputDir;
    TrajectaRunner::Mode m_lastRunMode = TrajectaRunner::Mode::Fete;
    QString m_lastDensityPath;  // FETE output, used to prefill the NNI input
};
