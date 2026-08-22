#pragma once

#include <QElapsedTimer>
#include <QList>
#include <QObject>
#include <QString>

#include "postbatchmodel.h"
#include "trajectarunner.h"

// Runs a post-processing batch: one chunk at a time, from first to last,
// never stopping for a failure. Mirrors BatchController, with one structural
// difference forced by what the three tools actually are:
//
// An NNI chunk is a trajecta.exe subprocess, exactly like a Processing row —
// so it reuses the same TrajectaRunner, and Pause really does freeze it mid
// computation. A Compare or Coherence chunk runs synchronously, in this
// process, on the same thread that owns the event loop — the same thing the
// single-run "Compare"/"Score the sites" buttons already do (see
// MainWindow::runRouteComparison() / runCoherence()), for the same reason:
// neither is worth a subprocess. That call excludes user input from the
// events it pumps to keep the log and the window repainting, so for these two
// tools Pause, Skip row and Stop batch can only take effect between chunks —
// there is no click to receive while one is running. A job runs one tool for
// its whole length (the mode is chosen once, like Batch::Job::mode), so a
// single controller instance is never asked to mix the two execution paths.
class PostBatchController : public QObject
{
    Q_OBJECT

public:
    enum class RowState {
        Pending,
        Running,
        Done,
        Failed,
        Invalid,
        Cancelled
    };

    struct RowResult {
        int chunk = -1;
        RowState state = RowState::Pending;
        QString message;
        double percent = 0.0;
        qint64 elapsedMs = 0;
    };

    // Picking a batch up where a crash left it. Coarser than BatchController's:
    // none of the three tools here can resume mid-computation (see the class
    // comment), so the chunk that was running restarts from its beginning —
    // the chunks before it are taken as already done and never run again.
    struct Resume {
        int chunkIndex = -1;
        bool isValid() const { return chunkIndex >= 0; }
    };

    explicit PostBatchController(QObject *parent = nullptr);

    // `env` supplies the engine path and the GDAL/PROJ locations for NNI
    // chunks; Compare and Coherence chunks ignore it. Returns false, with
    // nothing started, if the pre-flight check found a job-level problem.
    bool start(const PostBatch::Job &job, const TrajectaRunner::Parameters &env,
              QString *error, const Resume &resume);
    bool start(const PostBatch::Job &job, const TrajectaRunner::Parameters &env,
              QString *error = nullptr);

    void skipCurrentRow();
    void stopBatch();

    // For an NNI job this freezes the running subprocess immediately, exactly
    // like the single-run panel. For a Compare or Coherence job there is no
    // process to freeze; it holds the queue instead, so the chunk in progress
    // finishes normally and the next one does not start until resumed.
    void pause();
    void resume();
    bool isPaused() const;
    bool isRunning() const;
    bool wasStopped() const { return m_stopRequested; }

    const QList<RowResult> &results() const { return m_results; }
    const PostBatch::Job &job() const { return m_job; }
    int currentChunkIndex() const { return m_current; }
    int total() const { return m_job.chunks.size(); }

signals:
    void rowStarted(int chunkIndex);
    void rowProgress(int chunkIndex, double percent);
    void rowFinished(int chunkIndex, PostBatchController::RowState state,
                     const QString &message);
    void batchProgress(int finished, int total);
    void consoleOutput(const QString &rawText);
    void consoleErrorLine(const QString &line);
    void statusChanged(const QString &status);
    void pauseStateChanged(bool paused);
    void batchFinished(const QString &report);
    // Emitted once a chunk finishes successfully, so the page can register its
    // output with the Viewer without the controller knowing the Viewer exists.
    void chunkLayersReady(int chunkIndex);

private:
    void startNext();
    void finishRow(RowState state, const QString &message);
    void runNniChunk(const PostBatch::Chunk &chunk);
    void runInProcessChunk(const PostBatch::Chunk &chunk);
    QString buildReport() const;

    TrajectaRunner *m_runner = nullptr;   // NNI chunks only
    PostBatch::Job m_job;
    TrajectaRunner::Parameters m_env;
    QList<RowResult> m_results;
    int m_current = -1;
    bool m_running = false;
    bool m_stopRequested = false;
    bool m_skipRequested = false;   // in-process modes: checked once the call returns
    bool m_paused = false;          // in-process modes: checked before the next chunk
    QElapsedTimer m_rowClock;
};
