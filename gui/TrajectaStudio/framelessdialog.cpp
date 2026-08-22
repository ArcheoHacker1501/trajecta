#include "framelessdialog.h"

#include "thememanager.h"
#include "uiwidgets.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWindow>

#ifdef Q_OS_WIN
#include <dwmapi.h>
#include <windows.h>
#endif

namespace TrajectaUi {

namespace {

// The only interactive part of the title row besides the close button: press
// anywhere else on it and drag, the same gesture the native title bar it
// replaces would have offered. startSystemMove() hands the drag to the
// window manager itself rather than tracking the mouse by hand, which is
// what keeps it feeling native — Aero Snap included, on Windows.
class DialogTitleBar : public QWidget
{
public:
    using QWidget::QWidget;

protected:
    void mousePressEvent(QMouseEvent *event) override
    {
        if (event->button() == Qt::LeftButton) {
            if (QWindow *win = window()->windowHandle()) {
                win->startSystemMove();
                return;
            }
        }
        QWidget::mousePressEvent(event);
    }
};

} // namespace

FramelessDialog::FramelessDialog(QWidget *parent, const QString &title)
    : QDialog(parent)
{
    // Qt::Dialog keeps every ordinary dialog behaviour — modality, staying
    // above its parent, Escape closing it; FramelessWindowHint is the one
    // flag that takes the window manager's title bar away, minimise and
    // maximise/restore with it. There was never a WindowMinMaxButtonsHint to
    // remove: a plain QDialog does not carry one. What put those two buttons
    // on every popup in the application was the native frame itself, which
    // this line removes wholesale.
    setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint);
    // Makes the window itself a per-pixel-alpha surface, so paintEvent()
    // below can leave the four corners genuinely transparent instead of
    // filling a square and hoping the window manager clips it to match —
    // see the class comment in framelessdialog.h.
    setAttribute(Qt::WA_TranslucentBackground);

    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(0);

    auto *titleBar = new DialogTitleBar(this);
    titleBar->setObjectName(QStringLiteral("FramelessTitleBar"));
    titleBar->setFixedHeight(kTitleBarHeight);
    auto *titleLayout = new QHBoxLayout(titleBar);
    titleLayout->setContentsMargins(18, 0, 10, 0);
    titleLayout->setSpacing(8);

    m_titleLabel = new QLabel(title, titleBar);
    m_titleLabel->setObjectName(QStringLiteral("FramelessTitleLabel"));
    titleLayout->addWidget(m_titleLabel, 1);

    // The same #WindowCloseButton the main window's own close button is —
    // same drawn "X", same hover-red, same pressed shade — so a dialog's
    // close control and the application's are one design, not two that
    // happen to agree today.
    auto *close = new QToolButton(titleBar);
    close->setObjectName(QStringLiteral("WindowCloseButton"));
    close->setIcon(TrajectaUi::makeWindowIcon(
        TrajectaUi::WindowGlyph::Close, ThemeManager::mapped("#99a1ac"), 20));
    close->setCursor(Qt::PointingHandCursor);
    // Not in the tab order: Escape already closes the dialog, and a button
    // that only ever does what Escape does is not worth a stop on the way
    // through the form.
    close->setFocusPolicy(Qt::NoFocus);
    connect(close, &QToolButton::clicked, this, &QDialog::reject);
    titleLayout->addWidget(close);

    outer->addWidget(titleBar);

    // A layout, not a wrapper QWidget: an unnamed QWidget here would be
    // caught by the "QMainWindow, QWidget" rule at the top of theme.qss and
    // paint its own opaque rectangle over the dialog's own background — the
    // same failure the Viewer's canvas hit before it was given an object
    // name of its own. A bare layout has no background to get wrong.
    m_content = new QVBoxLayout;
    outer->addLayout(m_content, 1);
}

// The stylesheet's own "#ConfirmDialog { background-color: ... }" rule and
// its siblings on the other dialogs built on this class do not reliably
// reach a frameless top-level widget — the same failure theme.qss's own
// palette-rule note documents for a Qt::Popup, and Qt::FramelessWindowHint
// turns out to be enough on its own to trigger it too, even though this is
// an ordinary Qt::Dialog otherwise. Painted directly instead, in the one
// colour every dialog built on this class already asks the stylesheet for,
// so the two can never show a different shade — and painted first, so
// whatever the base class's own paintEvent() does afterwards layers over an
// already-correct background rather than under one.
//
// Filled and stroked as one rounded path, not a plain fillRect(): with
// Qt::WA_TranslucentBackground set in the constructor, whatever this leaves
// outside the path stays transparent rather than reading as a stray white
// or black square behind the curve, and the same #ModeCard border — colour
// and width both — that outlines every mode card in the app now outlines
// this window too. Inset by half the pen width first, or Qt centres the
// stroke on the path and the outer half gets clipped at the window edge.
void FramelessDialog::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const qreal borderWidth = 1.0;
    const QRectF area = QRectF(rect()).adjusted(
        borderWidth / 2.0, borderWidth / 2.0, -borderWidth / 2.0, -borderWidth / 2.0);
    QPainterPath path;
    path.addRoundedRect(area, kCornerRadius, kCornerRadius);

    painter.fillPath(path, ThemeManager::mapped("#1b1f26"));
    painter.setPen(QPen(ThemeManager::mapped("#2a2f38"), borderWidth));
    painter.drawPath(path);

    QDialog::paintEvent(event);
}

#ifdef Q_OS_WIN
void FramelessDialog::showEvent(QShowEvent *event)
{
    QDialog::showEvent(event);
    // Same call the main window makes for the same reason (see its own
    // nativeEvent()): with the native frame gone, this is what still gives
    // the window a drop shadow. The rounded corners themselves are this
    // class's own doing now (see paintEvent()), not left to DWM's default —
    // this call is purely for the shadow. showEvent() can fire more than
    // once for the same dialog; re-issuing the call each time costs
    // nothing, so there is no need to guard it.
    const HWND hwnd = reinterpret_cast<HWND>(winId());
    const MARGINS shadow{ 0, 0, 0, 1 };
    DwmExtendFrameIntoClientArea(hwnd, &shadow);
}
#endif

} // namespace TrajectaUi
