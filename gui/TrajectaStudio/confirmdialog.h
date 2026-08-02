#pragma once

#include <QString>

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

// Returns true when the user confirms. Modal on `parent`.
bool confirm(QWidget *parent, const QString &title, const QString &message);

// Longest layer/file name rendered inside a confirmation before it is elided.
// The dialog is fixed-size, so unbounded text would be clipped instead.
QString elideForConfirm(const QString &text, int maxChars = 42);

} // namespace TrajectaUi
