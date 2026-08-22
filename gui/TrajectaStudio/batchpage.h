#pragma once

#include <QWidget>

#include "batchcontroller.h"
#include "batchmodel.h"
#include "batchtable.h"
// For RunTicker::State, which this page assembles for the status bar.
#include "uiwidgets.h"
#include "walkthrough.h"

#include <QElapsedTimer>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFrame;
class QGroupBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QPropertyAnimation;
class QToolButton;
class QVBoxLayout;

class ConsoleView;

// One chunk: the algorithm and cost modifiers that apply to all its rows, plus
// the table of rows themselves.
class BatchChunkWidget : public QWidget
{
    Q_OBJECT

public:
    BatchChunkWidget(TrajectaRunner::Mode mode, QWidget *parent = nullptr);

    void setMode(TrajectaRunner::Mode mode);
    void setIndex(int index);  // shown in the title, 1-based

    Batch::Chunk chunk() const;
    void setChunk(const Batch::Chunk &chunk);

    BatchTableModel *model() const { return m_model; }
    void setEditingEnabled(bool enabled);

    // What the guided tour points at inside a chunk. Accessors rather than a
    // list of screens: the widgets belong to the chunk, the words that explain
    // them belong to the tour, and neither should have to know the other.
    // Defined in the .cpp: up here QGroupBox and BatchTableView are only
    // forward declarations, and a pointer to an incomplete type cannot be
    // converted to its base.
    QWidget *algorithmAnchor() const;
    QWidget *modifiersAnchor() const;
    QWidget *tableAnchor() const;
    QList<QWidget *> rowButtons() const { return m_rowButtons; }

    // Folding a chunk away is the only way a batch of six chunks stays
    // navigable; the state is part of the document, not of the session.
    bool isCollapsed() const { return m_collapsed; }
    void setCollapsed(bool collapsed, bool animate = true);

signals:
    void removeRequested();
    void duplicateRequested();
    void moveUpRequested();
    void moveDownRequested();
    void changed();

private:
    void addRowsFromFiles();
    // Shows or hides the rows that only mean something in one of the two
    // modes. Called from the constructor as well as setMode().
    void applyModeVisibility();

    TrajectaRunner::Mode m_mode;
    QLabel *m_title = nullptr;
    QComboBox *m_neighbours = nullptr;
    // Shown only while the combo sits on "Custom…"; carries the real number.
    QSpinBox *m_neighboursCustom = nullptr;
    // Slope cut-off for every row of this chunk.
    QCheckBox *m_slopeCap = nullptr;
    QSpinBox *m_slopeCapUp = nullptr;
    QSpinBox *m_slopeCapDown = nullptr;
    // Cost corridor, shown only for LCPA chunks.
    QWidget *m_corridorRow = nullptr;
    QCheckBox *m_corridorCheck = nullptr;
    QDoubleSpinBox *m_corridorWidth = nullptr;
    QComboBox *m_costFunction = nullptr;
    QSpinBox *m_smoothing = nullptr;
    QGroupBox *m_modifiers = nullptr;
    QLineEdit *m_costVector = nullptr;
    QSpinBox *m_polylineBuffer = nullptr;
    QLineEdit *m_costRaster = nullptr;
    QCheckBox *m_barrier = nullptr;
    QDoubleSpinBox *m_barrierValue = nullptr;
    QCheckBox *m_keepExtra = nullptr;
    QCheckBox *m_loadRasters = nullptr;
    // Result vectors, shown only for LCPA chunks: the row and its help dot
    // together, so hiding it does not leave the dot behind on its own.
    QWidget *m_loadVectorsRow = nullptr;
    QCheckBox *m_loadVectors = nullptr;
    BatchTableModel *m_model = nullptr;
    BatchTableView *m_table = nullptr;
    QList<QWidget *> m_rowButtons;

    QWidget *m_body = nullptr;
    QToolButton *m_collapseButton = nullptr;
    QPropertyAnimation *m_collapseAnim = nullptr;
    bool m_collapsed = false;
};

// The whole batch: mode, hardware, the chunks, and the run controls.
//
// It lives inside the Analysis Setup page and replaces the single-run form
// when the Batch card is selected.
class BatchPage : public QWidget
{
    Q_OBJECT

public:
    explicit BatchPage(QWidget *parent = nullptr);

    // Engine path and GDAL/PROJ locations, refreshed by MainWindow whenever it
    // re-detects the environment.
    void setEnvironment(const TrajectaRunner::Parameters &env);
    bool isRunning() const;
    void cancelForShutdown();

    void applyTheme();
    // Used by the hidden --open-log switch (testing): unfolds the batch log.
    void openLogs();

    // Unfolds every chunk and hands back the fold states it found, so the
    // caller can put them back afterwards. For the guided tour: a folded chunk
    // hides its own rows, and the screens that describe them — the table, the
    // row buttons — then point at widgets that are not on screen.
    QVector<bool> unfoldChunks();
    void restoreChunkFolds(const QVector<bool> &folded);

    // What the status bar's ticker should say while this batch runs. Assembled
    // here rather than by MainWindow: which chunk a queue position belongs to,
    // and how far the running row has got, are this page's business.
    // Inactive (State::active false) whenever no batch is running.
    RunTicker::State tickerState() const;

    // The batch block of the guided walkthrough. Built here because every
    // widget it points at is private to this page; MainWindow supplies the
    // navigation that has to happen before them and splices the result into
    // the rest of the tour.
    QVector<TourStep> walkthroughSteps();

    // False while the single-run panel owns the engine: blocks "Run batch" and
    // nothing else, so the page stays editable throughout.
    void setStartAllowed(bool allowed);

    // Hidden testing hooks (--batch-load / --batch-run), so a batch can be set
    // up and executed without clicking through the table.
    bool loadBatchFile(const QString &path, QString *error = nullptr);
    void startBatchNow() { startBatch(); }

    // The batch as it stands, for QSettings: a hand-typed table must survive
    // closing the window, not only an explicit Save.
    QString saveState() const;
    void restoreState(const QString &json);

    // Crash recovery: reload an interrupted batch and pick it up at the row it
    // stopped on, resuming that row from `checkpointPath` and running the rest
    // from the start as usual.
    void resumeJob(const QJsonObject &job, int queueIndex, const QString &checkpointPath);

signals:
    void runningChanged(bool running);
    // The two links on the log row. The page has no idea where checkpoints are
    // kept — MainWindow owns that folder and the resume path — so it asks
    // rather than acts.
    void exportCheckpointRequested();
    void importCheckpointRequested();
    // Outputs of a row that finished, for the chunks that asked for them. The
    // page has no reference to the Viewer; MainWindow owns both and connects
    // this to it.
    void viewerLayersReady(const QStringList &rasters, const QStringList &vectors);
    // Something the ticker shows has moved. Carries nothing: the receiver asks
    // tickerState(), which keeps one description of the batch instead of one
    // per signal.
    void tickerChanged();

private:
    Batch::Job buildJob() const;
    void applyJob(const Batch::Job &job);
    void addChunk(const Batch::Chunk &chunk);
    void removeChunk(BatchChunkWidget *w);
    void renumberChunks();
    void setEditingEnabled(bool enabled);
    void updateRunButtons();

    void startBatch();
    void saveToFile();
    void loadFromFile();
    // "idle", "running", "paused", "success", "failed" — the same vocabulary the
    // setup page's chip uses, so theme.qss styles both from one rule.
    void setChipState(const QString &state, const QString &text);
    void publishRowLayers(const Batch::Index &at);
    void writeSessionForRow(int queueIndex);

    // Set by resumeJob() and consumed by the next startBatch().
    int m_resumeQueueIndex = -1;
    QString m_resumeCheckpoint;

    TrajectaRunner::Mode mode() const;

    BatchController *m_controller = nullptr;
    TrajectaRunner::Parameters m_env;

    // The two cards the page is framed by, kept because the walkthrough lights
    // whole cards rather than the controls inside them.
    QFrame *m_settingsCard = nullptr;   // mode, hardware, per-batch options
    QFrame *m_runCard = nullptr;        // the buttons, the bar and the log

    QPushButton *m_modeFete = nullptr;
    QPushButton *m_modeLcpa = nullptr;
    QSpinBox *m_threads = nullptr;
    QSpinBox *m_ram = nullptr;
    QCheckBox *m_folderPerRow = nullptr;
    QCheckBox *m_verbose = nullptr;
    QCheckBox *m_manifest = nullptr;

    QWidget *m_chunkHost = nullptr;
    QVBoxLayout *m_chunkLayout = nullptr;
    QList<BatchChunkWidget *> m_chunks;

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

    // What the ticker needs and the controller does not keep: how long the
    // batch has been going, how many rows are behind it and how far the one in
    // front has got. A batch's estimate is per row rather than per percent —
    // rows are the unit that actually repeats.
    QElapsedTimer m_batchClock;
    int m_rowsDone = 0;
    double m_rowPercent = 0.0;
};
