#pragma once

// The whole header, not a forward declaration: the default argument below
// constructs a QString, which needs the complete type.
#include <QString>

class QWidget;

// The report form behind the "report form" link on the Guide page.
//
// What the user writes here reaches the project's mailbox without that address
// ever appearing in this program: the message is handed to a form-to-email
// relay which holds the destination in its own configuration. See the constants
// at the top of reportform.cpp for what that means in practice — and for the
// one value that has to be filled in before a release.
//
// Modal, centred on the screen and dressed by the active theme, like every
// other dialog in the application.
namespace TrajectaUi {

// `selfTestMessage` belongs to the hidden --report-selftest switch: when it is
// not empty the form fills itself in with that text, attaches one generated
// image and presses Send by itself.
//
// It exists because the send path cannot be exercised any other way — it takes
// a person typing and clicking — and the join it crosses is the one where a
// mistake would hide: the multipart written by Qt has to be understood by the
// relay that parses it. Left empty, which is every real use, the form behaves
// as if the parameter were not there.
void showReportForm(QWidget *parent, const QString &selfTestMessage = QString());

} // namespace TrajectaUi
