#pragma once

#include <QElapsedTimer>
#include <QList>
#include <QObject>
#include <QString>

#include "batchmodel.h"
#include "trajectarunner.h"

// Runs a batch: one row at a time, from first chunk to last, never stopping
// for a failure.
//
// It owns a single TrajectaRunner and reuses it for every row — the runner
// tears its QProcess down and rebuilds it on each start(), so one instance is
// enough and the engine gets a clean process (and clean RAM) per row.
//
// A row that fails is recorded and skipped; the batch carries on. That is the
// whole point: a batch is left running unattended, and one bad path in row 12
// must not throw away the other thirty.
class BatchController : public QObject
{
    Q_OBJECT

public:
    enum class RowState {
        Pending,   // not reached yet
        Running,
        Done,      // the engine reported success
        Failed,    // the engine refused an input, or crashed
        Invalid,   // rejected by the pre-flight check, never launched
        Cancelled  // the user abandoned this row or the whole batch
    };

    struct RowResult {
        Batch::Index where;
        RowState state = RowState::Pending;
        QString message;   // why it failed, when it did
        double percent = 0.0;
        qint64 elapsedMs = 0;
    };

    explicit BatchController(QObject *parent = nullptr);

    // Picking a batch up where a crash left it: the rows before `queueIndex`
    // are taken as already done, and `queueIndex` itself restarts from the
    // engine checkpoint rather than from its first source point.
    struct Resume {
        int queueIndex = -1;
        // May be empty: a batch stopped between two rows has no engine state to
        // pick up, but the rows before `queueIndex` are still done and only the
        // rest need running.
        QString checkpointPath;
        bool isValid() const { return queueIndex >= 0; }
    };

    // `env` carries what the batch cannot know: the engine path and the GDAL /
    // PROJ locations the GUI detected. Returns false, with nothing started, if
    // the pre-flight check found a problem that dooms the whole batch.
    // Two overloads rather than a defaulted `Resume()`: a default argument that
    // constructs the enclosing class's own nested type is not allowed inside
    // the class definition.
    bool start(const Batch::Job &job, const TrajectaRunner::Parameters &env,
               QString *error, const Resume &resume);
    bool start(const Batch::Job &job, const TrajectaRunner::Parameters &env,
               QString *error = nullptr);

    // Abandon the row in progress and move to the next one.
    void skipCurrentRow();
    // Abandon the row in progress and stop; the rows left stay Pending.
    void stopBatch();

    void pause();
    void resume();
    bool isPaused() const;
    bool isRunning() const;
    // True when the batch ended because stopBatch() was called rather than
    // because it ran out of rows. The rows left are still Pending, so the state
    // on disk is worth keeping.
    bool wasStopped() const { return m_stopRequested; }

    const QList<RowResult> &results() const { return m_results; }
    const Batch::Job &job() const { return m_job; }
    int currentQueueIndex() const { return m_current; }
    int total() const { return m_queue.size(); }

signals:
    void rowStarted(int queueIndex);
    void rowProgress(int queueIndex, double percent);
    void rowFinished(int queueIndex, BatchController::RowState state,
                     const QString &message);
    void batchProgress(int finished, int total);
    // Forwarded from the running row, so the console shows a live transcript.
    void consoleOutput(const QString &rawText);
    void consoleErrorLine(const QString &line);
    void statusChanged(const QString &status);
    void pauseStateChanged(bool paused);
    void batchFinished(const QString &report);

private:
    void startNext();
    void finishRow(RowState state, const QString &message);
    QString buildReport() const;

    TrajectaRunner *m_runner = nullptr;
    Batch::Job m_job;
    TrajectaRunner::Parameters m_env;
    QList<Batch::Index> m_queue;
    QList<RowResult> m_results;
    int m_current = -1;
    bool m_running = false;
    // Set while a stop is in flight, so the runner's Cancelled outcome can be
    // told apart from a row the user only wanted to skip.
    bool m_stopRequested = false;
    // Consumed by the first row that runs after a resume, and empty from then
    // on: only that one row has state to pick up.
    QString m_pendingResumeCheckpoint;
    QElapsedTimer m_rowClock;
};
