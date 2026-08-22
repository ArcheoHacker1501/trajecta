#include "smoothcombobox.h"

#include <QAbstractItemView>
#include <QFrame>
#include <QGuiApplication>
#include <QPainter>
#include <QPainterPath>
#include <QPropertyAnimation>
#include <QScreen>
#include <QStyleOptionComboBox>
#include <QWheelEvent>

namespace {

// Width reserved for the arrow on the right edge; must match the
// QComboBox::drop-down width in theme.qss so the text never runs under it.
constexpr int kArrowZoneWidth = 26;
constexpr qreal kArrowHalfWidth = 4.5;
constexpr qreal kArrowHeight = 3.0;
constexpr int kArrowDurationMs = 170;
constexpr int kPopupGap = 4;      // breathing room between field and list
// Must track the border-radius on QComboBox QAbstractItemView in theme.qss.
constexpr int kPopupRadius = 8;

const QScreen &screenFor(const QRect &rect)
{
    if (const QScreen *s = QGuiApplication::screenAt(rect.center()))
        return *s;
    return *QGuiApplication::primaryScreen();
}

} // namespace

SmoothComboBox::SmoothComboBox(QWidget *parent)
    : QComboBox(parent)
{
    m_arrowAnim = new QPropertyAnimation(this, "arrowAngle", this);
    m_arrowAnim->setDuration(kArrowDurationMs);
    m_arrowAnim->setEasingCurve(QEasingCurve::OutCubic);
}

void SmoothComboBox::setArrowAngle(qreal angle)
{
    if (qFuzzyCompare(m_arrowAngle, angle))
        return;
    m_arrowAngle = angle;
    update();
}

void SmoothComboBox::animateArrowTo(qreal angle)
{
    m_arrowAnim->stop();
    m_arrowAnim->setStartValue(m_arrowAngle);
    m_arrowAnim->setEndValue(angle);
    m_arrowAnim->start();
}

// Qt wraps the list in a QComboBoxPrivateContainer: a top-level QFrame that
// paints its own square panel around the view. That panel is what shows through
// at the corners as a square menu sitting under the rounded one, so it is taken
// off here. The rounding itself is done by masking the container in showPopup()
// — a shaped window, and deliberately not a translucent one.
//
// WA_TranslucentBackground was tried, to be rid of the container's silhouette
// on the platforms where the mask is not honoured. It cost the popup its
// background entirely: the list came up black on every theme, which only
// showed on the light ones, where black on black had been passing for correct.
// A popup that cannot be read is a far worse fault than a square corner, so
// the container stays an ordinary opaque window and the mask does the shaping.
void SmoothComboBox::preparePopupChrome()
{
    if (m_popupChromeReady)
        return;
    QAbstractItemView *list = view();
    QWidget *container = list ? list->window() : nullptr;
    if (!container || container == window())
        return;   // container not built yet; try again on the next open

    container->setAutoFillBackground(false);
    if (auto *frame = qobject_cast<QFrame *>(container)) {
        frame->setFrameShape(QFrame::NoFrame);
        frame->setContentsMargins(0, 0, 0, 0);
    }
    m_popupChromeReady = true;
}

void SmoothComboBox::showPopup()
{
    // Qt caps the list at maxVisibleItems (10 by default) and scrolls past
    // that; the cap belongs to the widget, not to an arbitrary default.
    setMaxVisibleItems(m_visibleCap > 0 ? m_visibleCap : qMax(1, count()));

    // Before the container is on screen: changing window flags on a visible
    // window would hide it.
    preparePopupChrome();

    QComboBox::showPopup();
    animateArrowTo(180.0);

    QAbstractItemView *list = view();
    QWidget *popup = list ? list->window() : nullptr;
    if (!popup)
        return;

    QRect target = popup->geometry();
    if (target.height() <= 2)
        return;

    // Qt sizes the popup from the item heights alone. The padding the
    // stylesheet puts on the list is not part of that sum, so the last row is
    // clipped and a scrollbar appears over a list that would otherwise fit.
    // Measure what the rows actually need and grow the container by the
    // shortfall, rather than trying to guess the stylesheet's metrics.
    if (count() > 0) {
        const int rowHeight = list->sizeHintForRow(0);
        const int rows = m_visibleCap > 0 ? qMin(count(), m_visibleCap) : count();
        const int shortfall = rowHeight * rows - list->viewport()->height();
        if (rowHeight > 0 && shortfall > 0) {
            target.setHeight(target.height() + shortfall);
            popup->setGeometry(target);
        }
    }

    // Qt is willing to lay the list over the field, aligning the current item
    // with it. Keep it strictly under the row instead, so the field one is
    // choosing for stays readable while the list is open.
    {
        const QRect avail = screenFor(target).availableGeometry();
        const int belowY = mapToGlobal(rect().bottomLeft()).y() + kPopupGap;
        target.moveTop(belowY);
        if (target.bottom() > avail.bottom()) {
            // No room underneath: put it entirely above the field rather than
            // let it creep back over it.
            const int aboveY = mapToGlobal(rect().topLeft()).y() - kPopupGap - target.height();
            target.moveTop(aboveY >= avail.top() ? aboveY : avail.top());
        }
        popup->setGeometry(target);
    }

    // Qt paints a widget's children without clipping them to its border-radius,
    // so the highlighted row squares off the corners the stylesheet rounded.
    // Masking the container to the same radius clips everything inside it —
    // rows, scrollbar, frame — to one rounded silhouette. Has to be redone
    // here because the mask is in widget coordinates and the size just changed.
    {
        QPainterPath rounded;
        rounded.addRoundedRect(QRectF(QPointF(0, 0), QSizeF(popup->size())),
                               kPopupRadius, kPopupRadius);
        popup->setMask(QRegion(rounded.toFillPolygon().toPolygon()));
    }

    // The list appears at its full size, without a fade. A window opacity is
    // what makes a window *layered* on Windows, and a layered window ignores
    // the mask above — which is how the square corners this whole function
    // exists to remove came back. Two frames of fade are not worth them.
    //
    // Animating the height was the other candidate and is worse: at one pixel
    // tall Qt decides the container needs its scroller arrows, and those arrows
    // go on to steal the height that would have made them unnecessary, leaving
    // the last row clipped for good.
}

void SmoothComboBox::hidePopup()
{
    QComboBox::hidePopup();
    animateArrowTo(0.0);
}

void SmoothComboBox::wheelEvent(QWheelEvent *event)
{
    event->ignore();
}

void SmoothComboBox::paintEvent(QPaintEvent *event)
{
    QComboBox::paintEvent(event);

    // The frame, the text and the drop-down zone are drawn by the stylesheet;
    // only the arrow inside that zone is ours.
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QColor color = palette().color(isEnabled() ? QPalette::Active : QPalette::Disabled,
                                   QPalette::WindowText);
    color.setAlpha(isEnabled() ? 210 : 120);

    const QPointF centre(width() - kArrowZoneWidth / 2.0 - 4.0, height() / 2.0);
    painter.translate(centre);
    painter.rotate(m_arrowAngle);

    QPainterPath triangle;
    triangle.moveTo(-kArrowHalfWidth, -kArrowHeight);
    triangle.lineTo(kArrowHalfWidth, -kArrowHeight);
    triangle.lineTo(0.0, kArrowHeight);
    triangle.closeSubpath();

    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    painter.drawPath(triangle);
}
