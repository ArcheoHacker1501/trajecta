#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>

#include "trajectarunner.h"

// The interface's half of checkpointing.
//
// The engine writes the propagation state (see src/checkpoint.h): how many
// sources are done and the density so far. That is enough to carry on
// computing, but not enough to *start* the engine again — the DEM, the output
// names, the algorithm and, in a batch, the rows still to run all live here.
//
// So a run that has checkpointing on drops a session file next to the
// checkpoint, and removes it when it ends in any orderly way: finished,
// failed, cancelled, application closed. A session file still present at the
// next start therefore means one thing only — the last run did not get to say
// goodbye — and that is what the recovery prompt is built on.
namespace Checkpoint {

struct Settings {
    // On by default, every half hour. It was off to begin with, on the
    // argument that a short analysis is quicker to repeat than to save — which
    // is true, and beside the point: the runs that matter are the long ones,
    // and a user who has just lost four days of computation is not consoled by
    // having saved a few seconds of disk writes. Half an hour is the interval
    // at which the cost is invisible on any run worth checkpointing.
    bool enabled = true;
    int minutes = 30;
    QString dir;   // empty means defaultDir()
};

Settings settings();
void setSettings(const Settings &s);

// %LOCALAPPDATA%\Trajecta\checkpoints — the same place the engine picks, and
// never the install directory, which a normal account cannot write to.
QString defaultDir();
// settings().dir, falling back to defaultDir().
QString activeDir();

// What the engine left behind.
struct Info {
    bool found = false;
    QString path;
    int nextSource = 0;
    int sources = 0;
    qint64 sizeBytes = 0;
    QString modified;   // human readable, for the recovery dialog
};
Info latest(const QString &dir);

// The run that was in progress, as the interface knew it.
struct Session {
    bool valid = false;
    bool batch = false;
    // batch only: which of the two batch pages `job` belongs to — Processing's
    // Batch::toJson() or the post-processing page's PostBatch::toJson(). Both
    // land in the same `job` field and the same checkpoint folder (there is
    // only ever one unfinished analysis kept at a time, whichever page it
    // came from), so resumeFromCheckpoint() needs this to know which page and
    // which fromJson() to hand it to. False — the Processing page — for every
    // session written before this field existed, which is what that page's
    // batches already were.
    bool isPostBatch = false;
    QJsonObject params;      // toJson(TrajectaRunner::Parameters)
    QJsonObject job;         // batch only: Batch::toJson(job) or PostBatch::toJson(job)
    int queueIndex = -1;     // batch only: the row/chunk that was running
    QString startedAt;
    QString label;           // "FETE — DEM30m" etc., shown in the dialog
    // True when the user stopped the run themselves (Cancel, or closing the
    // window). The state is just as resumable either way; only the wording of
    // the prompt changes, and calling a deliberate stop a crash would be a lie.
    bool deliberate = false;
};

QString sessionPath(const QString &dir);
bool writeSession(const QString &dir, const Session &s);
Session readSession(const QString &dir);
void clearSession(const QString &dir);

// Removes the checkpoints and the session file: the user chose not to resume.
void discard(const QString &dir);

// Copies the checkpoint and its session into a folder of the user's choosing,
// leaving the originals where they are. This is what the "Save a copy" button
// on the run panels does, and it is safe to press in the middle of a run: the
// engine only ever renames a finished file into place, so what is copied is
// always a whole checkpoint, at worst one interval old.
bool copyTo(const QString &dir, const QString &targetDir, QString *error,
            QStringList *copiedNames = nullptr);

// The same, and then removes the originals: used when the user declines to
// resume and asks to keep the state instead of losing it.
bool exportTo(const QString &dir, const QString &targetDir, QString *error);
// The counterpart: point the working directory at a previously exported set.
bool importFrom(const QString &sourceDir, const QString &dir, QString *error);

// TrajectaRunner::Parameters <-> JSON. Only the fields that describe the run;
// the engine and GDAL locations are re-detected at load time, because the
// application may well have been reinstalled since.
QJsonObject toJson(const TrajectaRunner::Parameters &p);
TrajectaRunner::Parameters fromJson(const QJsonObject &o);

} // namespace Checkpoint
