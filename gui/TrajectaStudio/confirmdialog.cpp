#include "confirmdialog.h"

#include "framelessdialog.h"

#include <QDialog>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScreen>
#include <QVBoxLayout>

namespace TrajectaUi {

namespace {
// Both confirmations get exactly these dimensions, whatever they say. Sized so
// the longest message in the app ("Remove ... from the Viewer?" plus the
// reassurance line) sits on three lines with room to spare. Height is content
// only — FramelessDialog::kTitleBarHeight is added on top of it at every use
// below, for the title row a native title bar used to provide free.
constexpr int kDialogWidth = 440;
constexpr int kDialogHeight = 200;
constexpr int kButtonWidth = 92;

// Every dialog here is parented to the window, never to the widget that raised
// it.
//
// Not tidiness — correctness. theme.qss clears the background of everything
// inside a card ("QFrame#Card QWidget"), and that selector has a higher
// specificity than "QDialog#ConfirmDialog": it reaches any dialog whose parent
// happens to sit in a card, because the stylesheet walks the parent chain and
// does not care that a dialog is a separate window. The background then
// resolves to transparent, and a transparent top-level window has nothing
// behind it to show — so the box comes up black, with only its buttons
// painted. The large-pages confirmation, raised from an option inside a card,
// was where it showed.
//
// Making the window the parent takes the dialog out of every descendant rule
// in one move, and is what a modal box should be parented to in any case.
QWidget *dialogHost(QWidget *parent)
{
    return parent ? parent->window() : nullptr;
}

} // namespace

// Centres the dialog on the screen the application is on.
//
// Qt does this itself — QDialog::adjustPosition() runs when a dialog is shown
// without having been moved — but it works from the parent's *frame* geometry,
// and Trajecta's window is frameless on Windows: what Qt is given is not the
// rectangle the user sees, and the box lands off to one side. Moving it here
// also sets WA_Moved, which is what tells Qt to leave it where we put it.
//
// Called after the size is fixed, so size() is final.
void centreOnScreen(QDialog &dialog, QWidget *parent)
{
    QWidget *host = parent ? parent->window() : nullptr;
    const QScreen *screen = host ? host->screen() : nullptr;
    if (!screen)
        return;   // nothing to centre on: Qt's own placement is as good as any
    // Centred on the screen rather than on the parent window. Two reasons, and
    // the second is the one that was actually wrong: a window dragged half off
    // the screen used to drag the dialog with it; and the recovery prompt at
    // start-up is raised before the main window has settled into its final
    // geometry, so centring on the window put it wherever the window happened
    // to be at that instant — visibly off-centre. The application is normally
    // maximised, where the two rules agree anyway.
    const QRect avail = screen->availableGeometry();
    const QSize sz = dialog.size();
    QPoint p(avail.center().x() - sz.width() / 2, avail.center().y() - sz.height() / 2);
    p.setX(qBound(avail.left(), p.x(), qMax(avail.left(), avail.right() - sz.width())));
    p.setY(qBound(avail.top(), p.y(), qMax(avail.top(), avail.bottom() - sz.height())));
    dialog.move(p);
}

QString elideForConfirm(const QString &text, int maxChars)
{
    if (text.size() <= maxChars)
        return text;
    return text.left(maxChars - 1) + QChar(0x2026);  // horizontal ellipsis
}

bool confirm(QWidget *parent, const QString &title, const QString &message,
             const QString &acceptText, const QString &rejectText,
             int extraHeight, Fill fill)
{
    FramelessDialog dialog(dialogHost(parent), title);
    dialog.setObjectName(QStringLiteral("ConfirmDialog"));
    dialog.setFixedSize(kDialogWidth,
                        FramelessDialog::kTitleBarHeight + kDialogHeight + qMax(0, extraHeight));

    auto *layout = dialog.contentLayout();
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

    auto *yes = new QPushButton(
        acceptText.isEmpty() ? QObject::tr("Yes") : acceptText, &dialog);
    auto *no = new QPushButton(
        rejectText.isEmpty() ? QObject::tr("No") : rejectText, &dialog);
    for (QPushButton *b : { yes, no }) {
        b->setObjectName(QStringLiteral("ConfirmButton"));
        // A minimum, so a longer verb than "Yes" is not clipped, and the two
        // buttons still match each other.
        b->setMinimumWidth(kButtonWidth);
        b->setCursor(Qt::PointingHandCursor);
        b->setAutoDefault(false);
        buttons->addWidget(b);
    }
    // A property rather than a second object name: the button keeps every rule
    // written for #ConfirmButton — size, radius, the per-theme shape overrides
    // — and only the fill is switched. The same mechanism the mode cards use
    // for their colours.
    if (fill == Fill::Accept || fill == Fill::Reject) {
        QPushButton *filled = (fill == Fill::Accept) ? yes : no;
        filled->setProperty("fill", QStringLiteral("accent"));
    } else if (fill == Fill::AcceptDanger) {
        yes->setProperty("fill", QStringLiteral("accent"));
        no->setProperty("fill", QStringLiteral("danger"));
    }
    buttons->addStretch(1);
    layout->addLayout(buttons);

    // No is the safe answer, so it takes Enter and Escape.
    no->setDefault(true);
    QObject::connect(yes, &QPushButton::clicked, &dialog, &QDialog::accept);
    QObject::connect(no, &QPushButton::clicked, &dialog, &QDialog::reject);

    centreOnScreen(dialog, parent);
    return dialog.exec() == QDialog::Accepted;
}

void notify(QWidget *parent, const QString &title, const QString &message,
            const QString &buttonText, int extraHeight)
{
    FramelessDialog dialog(dialogHost(parent), title);
    dialog.setObjectName(QStringLiteral("ConfirmDialog"));
    // Taller than the question form: what a message like this has to say is
    // usually a sentence longer, and there is only one button under it.
    dialog.setFixedSize(kDialogWidth,
                        FramelessDialog::kTitleBarHeight + kDialogHeight + 40
                            + qMax(0, extraHeight));

    auto *layout = dialog.contentLayout();
    layout->setContentsMargins(28, 26, 28, 24);
    layout->setSpacing(20);

    auto *text = new QLabel(message, &dialog);
    text->setObjectName(QStringLiteral("ConfirmText"));
    text->setWordWrap(true);
    text->setAlignment(Qt::AlignCenter);
    text->setTextInteractionFlags(Qt::NoTextInteraction);
    layout->addWidget(text, 1);

    auto *buttons = new QHBoxLayout;
    buttons->addStretch(1);
    auto *ok = new QPushButton(
        buttonText.isEmpty() ? QObject::tr("OK") : buttonText, &dialog);
    ok->setObjectName(QStringLiteral("ConfirmButton"));
    ok->setMinimumWidth(kButtonWidth);
    ok->setCursor(Qt::PointingHandCursor);
    ok->setAutoDefault(false);
    ok->setDefault(true);
    buttons->addWidget(ok);
    buttons->addStretch(1);
    layout->addLayout(buttons);

    QObject::connect(ok, &QPushButton::clicked, &dialog, &QDialog::accept);
    centreOnScreen(dialog, parent);
    dialog.exec();
}

int choose(QWidget *parent, const QString &title, const QString &message,
           const QStringList &options)
{
    if (options.isEmpty())
        return -1;

    FramelessDialog dialog(dialogHost(parent), title);
    dialog.setObjectName(QStringLiteral("ConfirmDialog"));
    // Wider than the two-button form: three verbs plus their spacing do not fit
    // in 440, and the fixed height still holds the same three lines of text.
    dialog.setFixedSize(kDialogWidth + 120, FramelessDialog::kTitleBarHeight + kDialogHeight);

    auto *layout = dialog.contentLayout();
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

    // Qt's done() codes: 0 is Rejected, so the choice index is offset by one
    // and index 0 — the safe option — is what closing the dialog also yields.
    for (int i = 0; i < options.size(); ++i) {
        auto *b = new QPushButton(options.at(i), &dialog);
        b->setObjectName(QStringLiteral("ConfirmButton"));
        b->setMinimumWidth(kButtonWidth);
        b->setCursor(Qt::PointingHandCursor);
        b->setAutoDefault(false);
        if (i == 0)
            b->setDefault(true);
        QObject::connect(b, &QPushButton::clicked, &dialog,
                         [&dialog, i] { dialog.done(i + 1); });
        buttons->addWidget(b);
    }
    buttons->addStretch(1);
    layout->addLayout(buttons);

    centreOnScreen(dialog, parent);
    const int code = dialog.exec();
    return code > 0 ? code - 1 : 0;
}

} // namespace TrajectaUi
