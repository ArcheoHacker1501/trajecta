#include "confirmdialog.h"

#include <QDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

namespace TrajectaUi {

namespace {
// Both confirmations get exactly these dimensions, whatever they say. Sized so
// the longest message in the app ("Remove ... from the Viewer?" plus the
// reassurance line) sits on three lines with room to spare.
constexpr int kDialogWidth = 440;
constexpr int kDialogHeight = 200;
constexpr int kButtonWidth = 92;
} // namespace

QString elideForConfirm(const QString &text, int maxChars)
{
    if (text.size() <= maxChars)
        return text;
    return text.left(maxChars - 1) + QChar(0x2026);  // horizontal ellipsis
}

bool confirm(QWidget *parent, const QString &title, const QString &message)
{
    QDialog dialog(parent);
    dialog.setObjectName(QStringLiteral("ConfirmDialog"));
    dialog.setWindowTitle(title);
    // No context-help button in the title bar: there is nothing behind it.
    dialog.setWindowFlags(dialog.windowFlags() & ~Qt::WindowContextHelpButtonHint);
    dialog.setFixedSize(kDialogWidth, kDialogHeight);

    auto *layout = new QVBoxLayout(&dialog);
    layout->setContentsMargins(28, 26, 28, 24);
    layout->setSpacing(20);

    auto *text = new QLabel(message, &dialog);
    text->setObjectName(QStringLiteral("ConfirmText"));
    text->setWordWrap(true);
    text->setAlignment(Qt::AlignCenter);
    text->setTextInteractionFlags(Qt::NoTextInteraction);
    layout->addWidget(text, 1);

    auto *buttons = new QHBoxLayout;
    buttons->setSpacing(12);
    buttons->addStretch(1);

    auto *yes = new QPushButton(QObject::tr("Yes"), &dialog);
    auto *no = new QPushButton(QObject::tr("No"), &dialog);
    for (QPushButton *b : { yes, no }) {
        b->setObjectName(QStringLiteral("ConfirmButton"));
        b->setFixedWidth(kButtonWidth);
        b->setCursor(Qt::PointingHandCursor);
        b->setAutoDefault(false);
        buttons->addWidget(b);
    }
    buttons->addStretch(1);
    layout->addLayout(buttons);

    // No is the safe answer, so it takes Enter and Escape.
    no->setDefault(true);
    QObject::connect(yes, &QPushButton::clicked, &dialog, &QDialog::accept);
    QObject::connect(no, &QPushButton::clicked, &dialog, &QDialog::reject);

    return dialog.exec() == QDialog::Accepted;
}

} // namespace TrajectaUi
