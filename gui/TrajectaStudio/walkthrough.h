#pragma once

#include <QColor>
#include <QPixmap>
#include <QPointer>
#include <QRect>
#include <QString>
#include <QVector>
#include <QWidget>

#include <functional>

class QFrame;
class QLabel;
class QPushButton;
class QToolButton;
class QPropertyAnimation;
class QVariantAnimation;

// The guided tour: a dimming layer over the whole window with one section lit
// at a time, a description beside it, and no way through to the application
// underneath.
//
// The engine knows nothing about Trajecta. It is handed a list of steps, each
// naming widgets to light and strings to show, and it iterates them. Everything
// Trajecta-specific — which widget, in what order, with what words — lives in
// the code that builds the list, never in here. That is what keeps this file
// reviewable on its own and reusable if the interface is rearranged.

// A leader line from a caption to one widget inside the lit area. Used when a
// step has to point at several controls at once ("this is the DEM, that is the
// output folder") rather than describe the section as a whole.
struct TourAnnotation {
    // Either a live widget…
    QPointer<QWidget> anchor;
    QString text;
    // …or, when there is no widget, a rectangle inside the step's inset
    // picture, in the picture's own coordinates: a way to point at part of
    // something that is shown rather than present. Last, so that the ordinary
    // { widget, text } form stays the short one.
    QRect insetRect;
    // Put this caption in the row *above* the lit area instead of the one
    // below it. Splitting a wide set in two is what keeps the leaders from
    // crossing when the controls are stacked but the captions are not: the
    // ones belonging to the top controls go over the top edge, the rest stay
    // underneath. Ignored when there is no lit area to be above.
    bool above = false;
};

struct TourStep {
    // Widgets to light. Empty: no cutout at all, and the description is centred
    // on screen — which is what an introductory or closing screen wants. More
    // than one: the cutout is their bounding box, for the cases where the thing
    // being described is a row of controls rather than a single widget.
    QVector<QPointer<QWidget>> targets;
    QString title;
    QString text;
    // Empty for a "what this section is for" screen; filled for the screen that
    // follows it and points at the individual parameters.
    QVector<TourAnnotation> annotations;
    // Navigation to run before the step is measured: switch page, pick a mode.
    // Anything that makes the targets exist and be visible.
    std::function<void()> onEnter;
    // Breathing room around the cutout. Most fields are wrapped in a help-dot
    // container, and a cut flush to the edge clips the focus ring.
    int padding = 8;
    // Corner radius of the cutout. -1 asks for the default, which suits a lit
    // control; a step that lights a whole card passes the card's own radius so
    // the lit shape is the card, exactly, and not a rounded rectangle that
    // happens to be over it. See TourStep::forCard().
    int radius = -1;
    // Extra wait before measuring, for steps that land on something animated
    // (a batch chunk unfolding). Zero for everything else.
    int settleMs = 0;
    // Normally the callout is made wider — and so shorter — until it clears the
    // lit area, because a panel drawn across the very card being described
    // hides what the reader was told to look at. Set false where that trade is
    // not worth making: a step whose card is taller than the window leaves no
    // gap to fit into at any width, and chasing one only produces a callout as
    // wide as the screen.
    bool avoidLitArea = true;
    // Last rung of the width ladder fitCalloutWidth() climbs (see there for why
    // it stops at 920 by default). Raised only for a step whose text is long
    // enough that no width up to 920 leaves the callout short enough to clear a
    // tall lit area without covering it — a step describing a card that fills
    // most of the window's height, which is otherwise a narrow band to stand
    // a multi-paragraph callout in above it.
    int calloutWidthCap = 920;
    // Try the band above the lit area before the one below it. Only ever
    // worth setting on a step whose captions already claim most of the room
    // below — a card lighting six controls at once, say — where the normal
    // below-first order would otherwise rest the panel over one of them.
    // Silently falls back to the usual below/above/sideways cascade when
    // above does not fit either, so this is never a way to force an overlap,
    // only to try the other order first.
    bool preferAbove = false;
    // A still picture drawn under the lit widget, where the thing it shows
    // would really appear. The gear menu is the case this exists for: a QMenu
    // is a top-level popup that takes a mouse and keyboard grab, so opening the
    // real one during the tour would paint over the overlay, stay clickable,
    // and swallow the click meant for Continue. A picture of it cannot.
    QPixmap inset;

    // Light one card as itself: the whole panel, heading and all, cut out in
    // the card's own shape. Most screens describe a section of the form rather
    // than a control in it, and lighting the fields alone left the heading that
    // names them in the dark — the reader had to be told which part of the page
    // they were looking at by the callout, when the page says so itself.
    void lightCard(QWidget *card);
};

// QPointer and not a raw pointer: batch chunks are created and destroyed while
// the application runs, and the step list is built once at start-up.

class TourOverlay : public QWidget
{
    Q_OBJECT

public:
    // `host` is the window to cover — the overlay makes itself its child.
    explicit TourOverlay(QWidget *host);

    void setSteps(const QVector<TourStep> &steps);
    bool isActive() const { return m_active; }

    // Index of the screen shown when the tour was last closed, so a caller can
    // offer to resume. -1 when the tour has never run or ran to the end.
    int lastIndex() const { return m_lastIndex; }

public slots:
    void start();
    void startAt(int index);   // testing hook (--tour-step)
    void next();
    void prev();
    void closeTour();

signals:
    void tourFinished();
    void stepChanged(int index, int total);
    // The ✕ was pressed. Deliberately not closeTour() itself: leaving a tour
    // half way is worth one question, and what to ask — and what to say
    // afterwards about where the tour can be found again — is the
    // application's business, not this widget's. A receiver that decides to
    // close calls closeTour(); one that does not, does nothing.
    void closeRequested();

protected:
    void paintEvent(QPaintEvent *event) override;
    // Every one of these swallows the event. The tour is static: nothing under
    // the overlay may be clicked, dragged, scrolled or typed into.
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void contextMenuEvent(QContextMenuEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    void buildChrome();
    // render() is the cross-fade in front of renderNow(), which is the step
    // change itself. Anything that needs the new step on screen at once —
    // the first one, and a resize — calls renderNow directly.
    void render(int index);
    void renderNow(int index);
    double calloutOpacity() const;
    void measureAndShow();
    // Frames the step's targets in their scroll area. True when it actually
    // moved the page, which means the geometry is not settled yet.
    bool ensureTargetVisible(const TourStep &step);
    QRect targetRect(const TourStep &step) const;
    void animateSpotlightTo(const QRect &target);
    // Picks the narrowest width at which the step's text is comfortable to read
    // and, where the step asks for it, clears `occupied` — the lit area with
    // its captions. A null rectangle means "use the whole band".
    void fitCalloutWidth(const QRect &occupied);
    void placeCallout(const QRect &spot);
    // Final correction, from the geometry actually on screen: moves the callout
    // clear of the lit area and its captions if it has landed on them.
    void keepCalloutClear(const QRect &occupied);
    // `avoid` is the lit area. The bar normally sits at the bottom of the
    // window; when the lit area reaches down that far it moves to the top
    // instead, rather than being drawn across the very thing the step is about.
    void placeNavBar(const QRect &avoid);
    // The band the callout, the captions and the inset may occupy: whichever
    // end of the window the navigation bar is not at.
    int topLimit() const;
    int bottomLimit() const;
    void refreshNav();
    // Back and Continue/Finish sized as one pair, on every screen.
    void matchFootButtons();
    // The ‹ › centred on the bar's axis rather than on their line box.
    void centreNavGlyphs();
    void fadeCalloutIn();
    void repaintFor(const QRect &a, const QRect &b);
    qreal spotRadius() const;
    QColor accent() const;

    struct PlacedAnnotation {
        QRect anchor;      // in overlay coordinates
        QRect chip;        // caption box
        QString text;
    };
    void layoutAnnotations(const QRect &spot);
    QRect placeInset(const QRect &spot) const;

    QWidget *m_host = nullptr;
    QVector<TourStep> m_steps;
    int m_index = -1;
    int m_lastIndex = -1;
    bool m_active = false;
    bool m_completed = false;   // reached the last screen rather than abandoned

    QRect m_spotlight;                  // current cutout, animated
    QVariantAnimation *m_spotAnim = nullptr;
    QVariantAnimation *m_scrimAnim = nullptr;
    QVariantAnimation *m_annotAnim = nullptr;
    // The callout's own fade, kept rather than created per use: the fade-out
    // has to chain into the step it was started for, and a chain needs
    // something to hang the connection on.
    QPropertyAnimation *m_calloutFade = nullptr;
    int m_pendingIndex = -1;
    // How many times this step has been measured. Scrolling the page invalidates
    // the measurement, so the first pass usually asks for another; the count
    // stops that turning into a loop when something is still animating.
    int m_measurePasses = 0;
    double m_scrim = 0.0;               // 0..1, faded in at start
    double m_annotProgress = 0.0;       // 0..1, drives the staggered reveal
    QVector<PlacedAnnotation> m_annotations;
    // Captions under the lit area rather than beside it — chosen per step from
    // how much room the target leaves at its sides.
    bool m_annotBelow = false;
    // The navigation bar has been moved to the top of the window for this step,
    // because the lit area reaches the bottom of it.
    bool m_navAtTop = false;
    // The padding rule currently on the ‹ ›, kept so it is only re-applied when
    // the font behind it changes: setting a stylesheet re-polishes the widget.
    QString m_navGlyphCss;
    // Where this step's inset picture is drawn, empty when it has none.
    QRect m_insetRect;

    QFrame *m_callout = nullptr;
    QLabel *m_title = nullptr;
    QLabel *m_body = nullptr;
    QPushButton *m_back = nullptr;
    QPushButton *m_continue = nullptr;
    QToolButton *m_close = nullptr;
    // "7 / 44", between Back and Continue in the panel's own foot.
    QLabel *m_footCounter = nullptr;

    // The navigation bar. Built, kept, and not shown: see kShowNavBar.
    QFrame *m_nav = nullptr;
    QToolButton *m_prev = nullptr;
    QToolButton *m_next = nullptr;
    QLabel *m_counter = nullptr;
};
