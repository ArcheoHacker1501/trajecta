#pragma once

#include <QComboBox>

class QPropertyAnimation;

// SmoothComboBox — a QComboBox that says it is one.
//
// Two things the stylesheet alone cannot do:
//
//  * the arrow. QSS can only swap one static image for another on the popup's
//    :on state, which snaps. Here the triangle is painted by hand and its angle
//    is an animated property, so it turns over as the list opens and turns back
//    as it closes.
//  * the popup. Qt shows it at full height immediately. The container's
//    geometry is animated instead, unrolling from the edge the list is anchored
//    to — downwards normally, upwards when the list had to flip because it was
//    near the bottom of the screen.
//
// Everything else, including sizing and the item view, is left to QComboBox and
// to the stylesheet (which hides the style's own arrow: see QComboBox::down-arrow
// in theme.qss).
class SmoothComboBox : public QComboBox
{
    Q_OBJECT
    Q_PROPERTY(qreal arrowAngle READ arrowAngle WRITE setArrowAngle)

public:
    explicit SmoothComboBox(QWidget *parent = nullptr);

    qreal arrowAngle() const { return m_arrowAngle; }
    void setArrowAngle(qreal angle);

    // How many rows the open list may show before it has to scroll.
    // 0 (the default) means "as many as there are": a list the user has to
    // scroll to see three options is worse than a tall list.
    void setVisibleItemCap(int rows) { m_visibleCap = rows; }

    void showPopup() override;
    void hidePopup() override;

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    void animateArrowTo(qreal angle);
    // Strips the popup container's own square frame and background, once.
    void preparePopupChrome();

    qreal m_arrowAngle = 0.0;   // degrees; 0 = pointing down, 180 = flipped
    int m_visibleCap = 0;       // 0 = show every item
    bool m_popupChromeReady = false;
    QPropertyAnimation *m_arrowAnim = nullptr;
};
