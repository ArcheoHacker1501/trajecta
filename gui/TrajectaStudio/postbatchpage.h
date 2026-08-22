#pragma once

#include <QWidget>

#include "postbatchcontroller.h"
#include "postbatchmodel.h"
// For RunTicker::State, which this page assembles for the status bar.
#include "uiwidgets.h"
// For TourStep, which walkthroughSteps() returns.
#include "walkthrough.h"

#include <QElapsedTimer>

#include <functional>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFrame;
class QGridLayout;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QPropertyAnimation;
class QToolButton;
class QVBoxLayout;

class ConsoleView;
class PathPicker;
class SmoothComboBox;

// One chunk: everything one NNI interpolation, one route comparison or one
// coherence score needs, plus the buttons that move, duplicate or delete it.
// Simpler than BatchChunkWidget: there is no table of rows underneath it,
// because here the chunk already is the row (see postbatchmodel.h).
class PostBatchChunkWidget : public QWidget
{
    Q_OBJECT

public:
    PostBatchChunkWidget(PostBatch::Mode mode, QWidget *parent = nullptr);

    void setMode(PostBatch::Mode mode);
    void setIndex(int index);   // shown in the title, 1-based

    PostBatch::Chunk chunk() const;
    void setChunk(const PostBatch::Chunk &chunk);

    void setEditingEnabled(bool enabled);

    bool isCollapsed() const { return m_collapsed; }
    void setCollapsed(bool collapsed, bool animate = true);

    // What the guided tour points at inside a chunk. Only the widgets that
    // exist regardless of mode are exposed this way; the tour picks its own
    // screen per mode and reads the matching accessor.
    QWidget *nniAnchor() const { return m_nniBox; }
    QWidget *compareAnchor() const { return m_cmpBox; }
    QWidget *coherenceAnchor() const { return m_cohBox; }

signals:
    void removeRequested();
    void duplicateRequested();
    void moveUpRequested();
    void moveDownRequested();
    void changed();

private:
    void buildNniFields(QWidget *body, QVBoxLayout *bodyLayout);
    void buildCompareFields(QWidget *body, QVBoxLayout *bodyLayout);
    void buildCoherenceFields(QWidget *body, QVBoxLayout *bodyLayout);
    void refreshCoherenceEnablement();

    PostBatch::Mode m_mode;
    QLabel *m_title = nullptr;
    QWidget *m_body = nullptr;
    QToolButton *m_collapseButton = nullptr;
    QPropertyAnimation *m_collapseAnim = nullptr;
    bool m_collapsed = false;

    // --- NNI ---
    QWidget *m_nniBox = nullptr;
    PathPicker *m_nniInputPicker = nullptr;
    PathPicker *m_nniOutputDirPicker = nullptr;
    QDoubleSpinBox *m_nniThresholdSpin = nullptr;
    QSpinBox *m_nniSpacingSpin = nullptr;
    QCheckBox *m_nniPeaksCheck = nullptr;
    QSpinBox *m_nniRadiusSpin = nullptr;
    QLineEdit *m_nniNameEdit = nullptr;

    // --- Compare with a known route ---
    QWidget *m_cmpBox = nullptr;
    PathPicker *m_cmpComputedPicker = nullptr;
    PathPicker *m_cmpKnownPicker = nullptr;
    QDoubleSpinBox *m_cmpToleranceSpin = nullptr;

    // --- Site-corridor coherence ---
    QWidget *m_cohBox = nullptr;
    PathPicker *m_cohRasterPicker = nullptr;
    PathPicker *m_cohPointsPicker = nullptr;
    QDoubleSpinBox *m_cohRadiusSpin = nullptr;
    SmoothComboBox *m_cohThresholdCombo = nullptr;
    QDoubleSpinBox *m_cohThresholdSpin = nullptr;
    QCheckBox *m_cohNullCheck = nullptr;
    SmoothComboBox *m_cohNullModeCombo = nullptr;
    QSpinBox *m_cohRepsSpin = nullptr;
    QCheckBox *m_cohSensCheck = nullptr;
    QLineEdit *m_cohSensEdit = nullptr;
    QLineEdit *m_cohEcdfEdit = nullptr;
    QCheckBox *m_cohEdgeCheck = nullptr;
    QCheckBox *m_cohRScriptCheck = nullptr;
    PathPicker *m_cohOutPicker = nullptr;
    QLineEdit *m_cohPrefixEdit = nullptr;
    SmoothComboBox *m_cohVectorCombo = nullptr;
    QCheckBox *m_cohRasterCheck = nullptr;

    // Present in all three modes, hidden for Compare — see postbatchpage.cpp
    // for why a comparison has no spatial result to register with the Viewer.
    QCheckBox *m_loadInViewer = nullptr;
};

// The whole post-processing batch: tool, hardware (NNI only), the chunks and
// the run controls. Lives inside the Post-processing page and replaces the
// single-tool forms when the "BP" card is selected — the same relationship
// BatchPage has with the Processing page's single-run form.
class PostBatchPage : public QWidget
{
    Q_OBJECT

public:
    explicit PostBatchPage(QWidget *parent = nullptr);

    void setEnvironment(const TrajectaRunner::Parameters &env);
    bool isRunning() const;
    void cancelForShutdown();
    void applyTheme();
    void openLogs();

    // Ensures GDAL is loaded in this process before a Compare or Coherence
    // batch starts — the same idempotent call the single-run buttons make.
    // Wired by MainWindow, which already owns engine/GDAL discovery; the page
    // does not know how that is done, only that it has to happen first.
    void setGdalLoader(std::function<bool()> loader) { m_gdalLoader = std::move(loader); }

    QVector<bool> unfoldChunks();
    void restoreChunkFolds(const QVector<bool> &folded);

    RunTicker::State tickerState() const;
    QVector<TourStep> walkthroughSteps();

    void setStartAllowed(bool allowed);

    bool loadBatchFile(const QString &path, QString *error = nullptr);
    void startBatchNow() { startBatch(); }

    QString saveState() const;
    void restoreState(const QString &json);

    void resumeJob(const QJsonObject &job, int chunkIndex);

signals:
    void runningChanged(bool running);
    void exportCheckpointRequested();
    void importCheckpointRequested();
    void viewerLayersReady(const QStringList &rasters, const QStringList &vectors);
    void tickerChanged();

private:
    PostBatch::Job buildJob() const;
    void applyJob(const PostBatch::Job &job);
    void addChunk(const PostBatch::Chunk &chunk);
    void removeChunk(PostBatchChunkWidget *w);
    void renumberChunks();
    void setEditingEnabled(bool enabled);
    void updateRunButtons();
    void updateHardwareVisibility();

    void startBatch();
    void saveToFile();
    void loadFromFile();
    void setChipState(const QString &state, const QString &text);
    void publishChunkLayers(int chunkIndex);
    void writeSessionForChunk(int chunkIndex);

    int m_resumeChunkIndex = -1;

    PostBatch::Mode mode() const;

    PostBatchController *m_controller = nullptr;
    TrajectaRunner::Parameters m_env;
    std::function<bool()> m_gdalLoader;

    QFrame *m_settingsCard = nullptr;
    QFrame *m_runCard = nullptr;

    QPushButton *m_modeNni = nullptr;
    QPushButton *m_modeCompare = nullptr;
    QPushButton *m_modeCoherence = nullptr;
    QWidget *m_hardwareRow = nullptr;
    QSpinBox *m_threads = nullptr;
    QSpinBox *m_ram = nullptr;
    QCheckBox *m_manifest = nullptr;

    QWidget *m_chunkHost = nullptr;
    QVBoxLayout *m_chunkLayout = nullptr;
    QList<PostBatchChunkWidget *> m_chunks;

    QPushButton *m_runButton = nullptr;
    QPushButton *m_pauseButton = nullptr;
    QPushButton *m_skipButton = nullptr;
    QPushButton *m_stopButton = nullptr;
    QProgressBar *m_progress = nullptr;
    QLabel *m_chip = nullptr;
    QLabel *m_status = nullptr;
    QLabel *m_summary = nullptr;
    ConsoleView *m_console = nullptr;
    QToolButton *m_logHandle = nullptr;
    bool m_startAllowed = true;

    QElapsedTimer m_batchClock;
    int m_rowsDone = 0;
    double m_rowPercent = 0.0;
};
