#pragma once

#include <QString>
#include <QStringList>

class QDialog;
class QWidget;

// The Yes/No confirmation used across Trajecta Studio.
//
// QMessageBox was the obvious choice and the wrong one: it sizes itself around
// its text, so two confirmations never came out the same size, and its internal
// label inherits the opaque background that theme.qss gives every QWidget,
// which does not match the background Fusion paints for the dialog itself —
// hence the visible box behind the text. This is a plain QDialog with a fixed
// size, a transparent centred label and centred buttons, styled entirely from
// theme.qss so it follows the active palette like everything else.
namespace TrajectaUi {

// Returns true when the user confirms. Modal on `parent`. The button labels
// default to Yes/No; a question that is not a yes/no question reads better with
// its own verbs ("Resume" / "No").
//
// `extraHeight` grows the box for the few messages that are a short paragraph
// rather than a question — the welcome offer is the one that asked for it. The
// width is left alone: it is what makes every confirmation in the application
// the same object.
//
// Which of the two buttons, if either, wears the accent fill rather than the
// outline. Most confirmations leave both plain: they are about something the
// user has just asked for, and two buttons of equal weight ask the question
// without answering it. Where one answer is plainly the better one, the
// interface may as well say so — "start the walkthrough?" expects yes,
// "stop the walkthrough?" expects no.
// AcceptDanger is the two-colour form: the accepted answer wears the accent and
// the refusal wears the destructive red. For the one question in the
// application that is genuinely two-sided — granting a Windows privilege, which
// raises a UAC prompt — where a pair of identical outlined buttons made a
// consequential choice look like a passive notice.
enum class Fill { None, Accept, Reject, AcceptDanger };

bool confirm(QWidget *parent, const QString &title, const QString &message,
             const QString &acceptText = QString(),
             const QString &rejectText = QString(),
             int extraHeight = 0, Fill fill = Fill::None);

// A statement rather than a question: one button, and nothing to decide.
// Same shape and same styling as the confirmation, so a message that reports
// a refused file does not arrive as a differently-dressed system box.
// `extraHeight` grows the box, as on confirm(): a couple of these messages are
// a short paragraph and a half, and a fixed box would clip the last line.
void notify(QWidget *parent, const QString &title, const QString &message,
            const QString &buttonText = QString(), int extraHeight = 0);

// Three-way version, for the one place where "no" is not the whole answer.
// Returns the index of the chosen button. The first option must be the safe
// one: it is what Escape and the window's close button also produce.
int choose(QWidget *parent, const QString &title, const QString &message,
           const QStringList &options);

// Longest layer/file name rendered inside a confirmation before it is elided.
// The dialog is fixed-size, so unbounded text would be clipped instead.
QString elideForConfirm(const QString &text, int maxChars = 42);

// Puts a dialog in the middle of the screen the application is on.
//
// Exported because the dialogs in this file are not the only ones that need it:
// Qt's own placement works from the parent's *frame* geometry, and Trajecta's
// window is frameless on Windows, so anything that leaves the position to Qt
// lands off to one side. Call it after the size is settled — it reads size().
void centreOnScreen(QDialog &dialog, QWidget *parent);

} // namespace TrajectaUi
