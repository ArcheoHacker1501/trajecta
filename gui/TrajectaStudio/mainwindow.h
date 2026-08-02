#pragma once

#include <QElapsedTimer>
#include <QMainWindow>

#include "trajectarunner.h"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QScrollArea;
class QSpinBox;
class QStackedWidget;
class QTextBrowser;
class QTimer;
class QToolButton;
class QVariantAnimation;
class QAction;

class ConsoleView;
class PathPicker;
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
    // 0 setup, 1 run, 2 post, 3 viewer, 4 guide, 5 about.
    void showPage(int index);

    // Used by the hidden --viewer-load switch (testing): opens a raster in the
    // Viewer page, or registers a vector file as an overlay. May be passed
    // more than once.
    void viewerLoadFile(const QString &path);

    // Used by the hidden --scroll-end switch (testing): scrolls the long page
    // currently showing (setup form or guide) to `fraction` of its range.
    void scrollSetupToEnd(double fraction = 1.0);

    // Used by the hidden --open-combo switch (testing): drops open the n-th
    // combo box of the page on screen, so the popup can be screenshotted.
    void openComboForTest(int index);

protected:
    void closeEvent(QCloseEvent *event) override;
    void changeEvent(QEvent *event) override;
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
    QWidget *buildRunPage();
    QWidget *buildPostPage();
    QWidget *buildGuidePage();
    QWidget *buildAboutPage();
    QWidget *makeCard(const QString &title, const QString &subtitle, QWidget *content);

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
    // Large memory pages: reflect which of the three gates the user is at, and
    // grant the Windows privilege through an elevated helper.
    void refreshLargePagesStatus();
    void setUpLargePages();
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
    static bool dirHasGdal(const QString &dir);

    // Persistence
    void loadSettings();
    void saveSettings() const;

    // --- Navigation ---
    QStackedWidget *m_pages = nullptr;
    QList<QPushButton *> m_navButtons;
    QScrollArea *m_setupScroll = nullptr;

    // --- Setup page widgets ---
    QPushButton *m_modeFete = nullptr;
    QPushButton *m_modeLcpa = nullptr;

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
    QComboBox *m_costFunctionCombo = nullptr;
    QSpinBox *m_smoothingSpin = nullptr;

    QSpinBox *m_threadsSpin = nullptr;
    QSpinBox *m_ramSpin = nullptr;
    QCheckBox *m_verboseCheck = nullptr;

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

    QPushButton *m_runButton = nullptr;

    // --- Post-processing page widgets (NNI) ---
    PathPicker *m_interpInputPicker = nullptr;
    PathPicker *m_interpOutputDirPicker = nullptr;
    QDoubleSpinBox *m_interpThresholdSpin = nullptr;
    QSpinBox *m_interpSpacingSpin = nullptr;
    QSpinBox *m_interpRadiusSpin = nullptr;
    QLineEdit *m_interpNameEdit = nullptr;
    QPushButton *m_runInterpButton = nullptr;
    QScrollArea *m_postScroll = nullptr;

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
        QPushButton *pauseButton = nullptr;
        QPushButton *cancelButton = nullptr;
        QPushButton *openFolderButton = nullptr;
        QLabel *summaryTitle = nullptr;
        QLabel *summaryBody = nullptr;
        QWidget *summaryCard = nullptr;
    };
    QWidget *buildRunPanel(RunUi &ui, QWidget *parent, const QString &idlePhrase,
                           QWidget *leadingButton = nullptr);
    RunUi m_runUi;
    RunUi m_postUi;
    RunUi *m_activeUi = &m_runUi;

    // --- Status bar environment indicators ---
    QLabel *m_engineStatus = nullptr;
    QLabel *m_gdalStatus = nullptr;

    // --- Appearance ---
    QCheckBox *m_largePagesCheck = nullptr;
    QLabel *m_largePagesStatus = nullptr;
    QPushButton *m_largePagesSetup = nullptr;
    QToolButton *m_gearButton = nullptr;
    QWidget *m_topBar = nullptr;
    QToolButton *m_minButton = nullptr;
    QToolButton *m_maxButton = nullptr;
    QToolButton *m_closeButton = nullptr;
    QList<QAction *> m_themeActions;
    QList<QAction *> m_fontActions;
    QTextBrowser *m_guideBrowser = nullptr;

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
