#include "walkthrough.h"

#include "thememanager.h"

#include <QApplication>
#include <QContextMenuEvent>
#include <QCoreApplication>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFontMetrics>
#include <QFrame>
#include <QGraphicsOpacityEffect>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPropertyAnimation>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QTimer>
#include <QToolButton>
#include <QVarLengthArray>
#include <QVBoxLayout>
#include <QVariantAnimation>
#include <QWheelEvent>

#include <algorithm>
#include <climits>

namespace {

// Darkness of the dimmed area at full fade. The lit section still has to be
// the thing the eye goes to, but the rest of the window is not meant to
// disappear: half of what a step says is *where* on the page the section is,
// and at 205 the page around it was gone. 155 keeps the layout readable as a
// shape while leaving the cutout plainly brighter than everything else.
constexpr int kScrimAlpha    = 155;
constexpr int kSpotRadius    = 10;    // a lit control; a lit card uses its own
constexpr int kSpotMoveMs    = 300;
constexpr int kScrimFadeMs   = 200;
constexpr int kCalloutFadeMs = 170;   // in
constexpr int kCalloutOutMs  = 110;   // out, deliberately quicker than in
constexpr int kCalloutRisePx = 10;    // how far it travels while it appears
constexpr int kAnnotRevealMs = 260;
constexpr int kMargin        = 16;
constexpr int kNavBottomGap  = 24;
constexpr int kCalloutWidth  = 560;
constexpr int kChipMaxWidth  = 300;
// One inset for both kinds of panel: the callout with its Continue button and
// the little captions on their leader lines. They are the same family of
// object, so the text sits the same distance from the edge in both — and the
// vertical figure is deliberately the smaller of the two, because on the
// callout every pixel of height is taken from the page underneath.
constexpr int kPanelPadH     = 16;
constexpr int kPanelPadV     = 12;
// Width of the callout's ✕, and of the empty column that balances it on the
// other side of the heading.
constexpr int kCloseWidth    = 26;
// Between the ✕, the heading and the column that balances the ✕.
constexpr int kHeadSpacing   = 8;
// Air above the heading and below it, and the same figure again under the
// paragraph. The heading is pinned to the exact height of its own text (see
// fitCalloutWidth), so this is the *only* thing that decides how the top of a
// panel breathes — and it is one number, so every panel breathes alike.
constexpr int kTitleGap      = 16;
// The ‹ › of the navigation bar, square so the glyph centres in them.
constexpr int kNavButtonSize = 34;
// The floor for Back and Continue at the foot of the panel. Not their size:
// matchFootButtons() gives the pair whichever of the two needs more, so that
// they are one pair of buttons rather than two of different builds.
constexpr int kFootButtonW   = 130;
constexpr int kFootButtonH   = 34;
constexpr int kLeaderGap     = 14;
constexpr int kGlowRings     = 4;

// The height a callout is allowed to reach before it is made wider instead.
// Nothing to do with what fits on screen: a paragraph poured into a narrow
// column is a worse thing to read long before it runs off the bottom, and the
// widths below are there to be used.
constexpr int kCalloutEasyHeight = 330;
// Gap between the lit area and the callout beside it.
constexpr int kCalloutGap = 18;
// And when the panel has no choice but to overlap the lit area — a card that
// fills the window, the chunk on the batch page — how much of the lit edge it
// must leave in sight. Drawn flush against it the panel covered the frame, and
// the light stopped reading as an outline around a section at all.
constexpr int kFrameReveal = 20;
// Below this, no width makes the box fit: a heading, one line and a button do
// not fold any smaller.
constexpr int kCalloutFloor = 150;

// The scrim has to clear the navigation bar, or the callout can sit on top of
// it when a step is near the bottom of the window.
constexpr int kNavReserve = 76;

// The travelling ‹ 7 / 44 › bar at the foot of the window: switched off, and
// kept whole rather than deleted, so it can be brought back by flipping this
// one line. The count now sits inside the panel, between Back and Continue,
// where the eye already is — and with it goes the reason the bar existed, since
// the two arrows only ever repeated what those two buttons and the ← → keys
// already do. Everything the bar needs still works: it places itself, moves out
// of the lit area's way, and keeps its own copy of the counter.
constexpr bool kShowNavBar = false;

// Where a leader line meets the control it points at.
//
// Aiming at the centre of the anchor is right for a control the size of a spin
// box and quite wrong for a field that spans a card: three stacked fields have
// the *same* centre, so three captions spread across the page would all point
// at one column of pixels and their leaders would cross on the way. The meeting
// point therefore slides along the anchor's own edge, as near as it will go to
// the caption below it.
// How far a string's ink sits below the middle of the line box, in pixels.
//
// A widget centres text by centring the *line* the font would set it on —
// ascent above the baseline, descent below — and that box is not centred on the
// ink. A chevron or a row of digits has nothing under the baseline, so the
// descent is empty space beneath it and the mark ends up riding low. Negative
// when the ink rides high instead.
int inkDrop(const QFontMetrics &fm, const QString &text)
{
    const QRect ink = fm.tightBoundingRect(text);
    if (ink.isEmpty())
        return 0;
    const double inkCentre = (ink.top() + ink.bottom()) / 2.0;
    const double boxCentre = (double(fm.descent()) - fm.ascent()) / 2.0;
    // Truncated towards zero, not rounded, and deliberately. The ink the
    // rasteriser actually lays down sits a fraction above what
    // tightBoundingRect() reports — the faint antialiased edge rows are ink to
    // the metric and not to the eye — so rounding the last half pixel up
    // over-corrects, and the mark ends up as far high as it started low.
    // Measured on screen at 150%: rounding left the chevrons a pixel high,
    // truncating leaves them a third of one, which is the best either can do.
    return int(inkCentre - boxCentre);
}

// The rule that takes that drop out again: text centred in a box rises by half
// of whatever padding is taken off the bottom, so the padding is twice the drop.
// The selector carries an id because the theme sets padding through one, and a
// bare type selector here would lose to it on specificity.
QString centringCss(const QString &selector, int drop)
{
    if (drop > 0)
        return QStringLiteral("%1 { padding-bottom: %2px; }").arg(selector).arg(2 * drop);
    if (drop < 0)
        return QStringLiteral("%1 { padding-top: %2px; }").arg(selector).arg(-2 * drop);
    return QString();
}

int leaderAttachX(const QRect &anchor, const QRect &chip)
{
    const int inset = qMin(10, anchor.width() / 4);
    return qBound(anchor.left() + inset, chip.center().x(),
                  qMax(anchor.left() + inset, anchor.right() - inset));
}

} // namespace

// Defined here rather than in the header: the card radius is a theme's
// business, and this keeps ThemeManager out of every file that builds steps.
void TourStep::lightCard(QWidget *card)
{
    targets = { card };
    padding = 0;
    radius = ThemeManager::cardRadius();
}

TourOverlay::TourOverlay(QWidget *host)
    : QWidget(host)
    , m_host(host)
{
    setObjectName(QStringLiteral("TourOverlay"));
    setVisible(false);
    setFocusPolicy(Qt::StrongFocus);
    // Without this a file dragged from Explorer would still land in the path
    // fields underneath: Qt delivers drag events to the topmost widget that
    // *accepts drops*, and silently skips one that does not.
    setAcceptDrops(true);
    // Unhandled mouse events must die here rather than climb to the window.
    setAttribute(Qt::WA_NoMousePropagation, true);

    buildChrome();

    m_spotAnim = new QVariantAnimation(this);
    m_spotAnim->setDuration(kSpotMoveMs);
    m_spotAnim->setEasingCurve(QEasingCurve::InOutCubic);
    connect(m_spotAnim, &QVariantAnimation::valueChanged, this,
            [this](const QVariant &v) {
        const QRect before = m_spotlight;
        m_spotlight = v.toRect();
        repaintFor(before, m_spotlight);
    });
    // The captions arrive once the light has settled, never while it travels.
    connect(m_spotAnim, &QVariantAnimation::finished, this, [this] {
        if (m_annotations.isEmpty())
            return;
        m_annotAnim->stop();
        m_annotAnim->start();
    });

    m_scrimAnim = new QVariantAnimation(this);
    m_scrimAnim->setDuration(kScrimFadeMs);
    m_scrimAnim->setEasingCurve(QEasingCurve::OutCubic);
    connect(m_scrimAnim, &QVariantAnimation::valueChanged, this,
            [this](const QVariant &v) {
        m_scrim = v.toDouble();
        update();
    });

    m_annotAnim = new QVariantAnimation(this);
    m_annotAnim->setDuration(kAnnotRevealMs);
    m_annotAnim->setEasingCurve(QEasingCurve::OutCubic);
    m_annotAnim->setStartValue(0.0);
    m_annotAnim->setEndValue(1.0);
    connect(m_annotAnim, &QVariantAnimation::valueChanged, this,
            [this](const QVariant &v) {
        m_annotProgress = v.toDouble();
        QRect dirty;
        for (const PlacedAnnotation &a : m_annotations)
            dirty = dirty.isNull() ? a.chip.united(a.anchor) : dirty.united(a.chip).united(a.anchor);
        update(dirty.adjusted(-8, -8, 8, 8));
    });

    m_host->installEventFilter(this);
}

// The two floating pieces. Both carry object names so theme.qss dresses them
// like the rest of the application instead of having colours written in here.
void TourOverlay::buildChrome()
{
    m_callout = new QFrame(this);
    m_callout->setObjectName(QStringLiteral("TourCallout"));
    m_callout->setFixedWidth(kCalloutWidth);
    auto *box = new QVBoxLayout(m_callout);
    // The same inset as a caption chip, so the two read as one family. Tight
    // vertically on purpose: every pixel of height here is a pixel the panel
    // takes from the page it is describing, while width costs nothing — the box
    // is placed above or below the lit area, never beside it.
    //
    // The spacing between the rows is 0 and every gap is an explicit item. A
    // layout spacing would apply around the spacers as well as between the
    // rows, so "the gap under the heading" would be three numbers added
    // together and nobody could read the intended figure off the code.
    box->setContentsMargins(kPanelPadH, kTitleGap, kPanelPadH, kPanelPadV);
    box->setSpacing(0);

    auto *head = new QHBoxLayout;
    head->setContentsMargins(0, 0, 0, 0);
    head->setSpacing(kHeadSpacing);

    // The only way out of the tour. Deliberately here and not in the window's
    // top-right corner: that is where Trajecta's own close button lives, and
    // two crosses a few pixels apart — one closing the tour, one closing the
    // program — is a mistake waiting to happen.
    m_close = new QToolButton(m_callout);
    m_close->setObjectName(QStringLiteral("TourClose"));
    m_close->setText(QStringLiteral("✕"));
    m_close->setCursor(Qt::PointingHandCursor);
    m_close->setAutoRaise(true);
    m_close->setFocusPolicy(Qt::NoFocus);
    // Fixed in both directions. The width balances the heading (below); the
    // height matters because this button, not the heading, is what decides how
    // tall the top row of the panel is — left to its size hint it came out
    // several pixels taller than the title and opened a gap under it.
    m_close->setFixedSize(kCloseWidth, kCloseWidth - 4);
    m_close->setToolTip(tr("Close the walkthrough"));
    connect(m_close, &QToolButton::clicked, this, &TourOverlay::closeRequested);

    // An empty column the width of the ✕, so the heading is centred on the
    // panel and not on the space left over beside the button. Which is why the
    // button has a fixed width: the two have to agree, and a size hint that
    // changes with the font would not.
    //
    // Fixed in *both* directions, and that is not a detail. addSpacing() would
    // give a spacer that is Minimum vertically — free to grow — and a row with
    // one growable item in it is a row that swallows every spare pixel the
    // panel has. Which is what opened the gap around the heading: the panel is
    // sized from the layout's hint, the hint over-estimates how many lines the
    // paragraph needs at a wide setting, and the surplus went straight into
    // this row, where the vertical centring split it above and below the title.
    head->addItem(new QSpacerItem(kCloseWidth, 0, QSizePolicy::Fixed, QSizePolicy::Fixed));
    m_title = new QLabel(m_callout);
    m_title->setObjectName(QStringLiteral("TourTitle"));
    m_title->setWordWrap(true);
    m_title->setAlignment(Qt::AlignCenter);
    // Fixed vertically, and given an exact height at every step by
    // fitCalloutWidth(). Left to itself a wrapping label asks the layout for a
    // height it guesses from no width in particular, and the guess is wrong in
    // both directions.
    m_title->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    head->addWidget(m_title, 1, Qt::AlignVCenter);
    head->addWidget(m_close, 0, Qt::AlignTop);
    box->addLayout(head);
    box->addSpacing(kTitleGap);

    m_body = new QLabel(m_callout);
    m_body->setObjectName(QStringLiteral("TourBody"));
    m_body->setWordWrap(true);
    // Top-aligned rather than the QLabel default of vertically centred: if the
    // box ever ends up a few pixels taller than the text needs, the paragraph
    // should stay where it was put instead of drifting down half of them.
    // Justified, so the paragraph ends flush on the right as it does on the
    // left and the air left over is the same width on both sides. The last
    // line of a paragraph stays ranged left, as justified text always does.
    m_body->setAlignment(Qt::AlignJustify | Qt::AlignTop);
    m_body->setTextInteractionFlags(Qt::NoTextInteraction);
    box->addWidget(m_body);
    box->addSpacing(kTitleGap);

    auto *foot = new QHBoxLayout;
    foot->setContentsMargins(0, 0, 0, 0);

    // The way back, mirroring Continue at the other end of the row. Quiet — no
    // fill, no border — because it is the second thing anyone wants from this
    // panel and it should not compete with the first. The ‹ in the navigation
    // bar does the same job; this one is simply where the hand already is.
    m_back = new QPushButton(tr("Back"), m_callout);
    m_back->setObjectName(QStringLiteral("TourBackButton"));
    m_back->setCursor(Qt::PointingHandCursor);
    m_back->setMinimumSize(kFootButtonW, kFootButtonH);
    m_back->setFocusPolicy(Qt::NoFocus);
    connect(m_back, &QPushButton::clicked, this, &TourOverlay::prev);
    foot->addWidget(m_back);

    foot->addStretch(1);
    // The count, in the middle of the panel's own foot rather than in a bar
    // floating over the page. Two equal stretches put it at the centre, which
    // *is* the centre of the panel because the two buttons beside it are one
    // pair of the same width (matchFootButtons).
    m_footCounter = new QLabel(m_callout);
    m_footCounter->setObjectName(QStringLiteral("TourCounter"));
    m_footCounter->setAlignment(Qt::AlignCenter);
    foot->addWidget(m_footCounter, 0, Qt::AlignVCenter);
    foot->addStretch(1);

    m_continue = new QPushButton(tr("Continue"), m_callout);
    m_continue->setObjectName(QStringLiteral("RunButton"));
    m_continue->setCursor(Qt::PointingHandCursor);
    m_continue->setMinimumSize(kFootButtonW, kFootButtonH);
    // No focus on any of the tour's buttons: the overlay swallows the keyboard
    // and nothing here should be reachable with Tab or fire on Space.
    m_continue->setFocusPolicy(Qt::NoFocus);
    connect(m_continue, &QPushButton::clicked, this, &TourOverlay::next);
    foot->addWidget(m_continue);
    box->addLayout(foot);

    auto *effect = new QGraphicsOpacityEffect(m_callout);
    effect->setOpacity(1.0);
    m_callout->setGraphicsEffect(effect);

    m_calloutFade = new QPropertyAnimation(effect, "opacity", this);
    m_calloutFade->setEasingCurve(QEasingCurve::OutCubic);
    // The one place the fade-out turns into the next step. Guarded by the
    // pending index: the fade-in uses the same animation and must not be
    // mistaken for it, and stop() — which is what interrupts a fade-out when
    // the user presses Continue twice quickly — does not emit finished() at
    // all, so a discarded transition simply leaves the index behind.
    connect(m_calloutFade, &QPropertyAnimation::finished, this, [this] {
        if (m_pendingIndex < 0)
            return;
        const int index = m_pendingIndex;
        m_pendingIndex = -1;
        renderNow(index);
    });

    m_nav = new QFrame(this);
    m_nav->setObjectName(QStringLiteral("TourNavBar"));
    auto *navRow = new QHBoxLayout(m_nav);
    navRow->setContentsMargins(8, 4, 8, 4);
    navRow->setSpacing(10);

    // Square, fixed, and vertically centred in the row. A tool button sized by
    // its text hint comes out as tall as one line of type, which puts a 26 px
    // chevron off the middle of a bar whose height is set by the counter
    // beside it; a square the style can centre the glyph in does not.
    const QSize navButton(kNavButtonSize, kNavButtonSize);

    m_prev = new QToolButton(m_nav);
    m_prev->setObjectName(QStringLiteral("TourNavButton"));
    m_prev->setText(QStringLiteral("‹"));
    m_prev->setCursor(Qt::PointingHandCursor);
    m_prev->setAutoRaise(true);
    m_prev->setFocusPolicy(Qt::NoFocus);
    m_prev->setFixedSize(navButton);
    // The keyboard does the same job; the tooltip is where anyone who did not
    // read the first screen finds that out.
    m_prev->setToolTip(tr("Previous screen  (←)"));
    connect(m_prev, &QToolButton::clicked, this, &TourOverlay::prev);
    navRow->addWidget(m_prev, 0, Qt::AlignVCenter);

    m_counter = new QLabel(m_nav);
    m_counter->setObjectName(QStringLiteral("TourCounter"));
    m_counter->setAlignment(Qt::AlignCenter);
    m_counter->setMinimumWidth(64);
    navRow->addWidget(m_counter, 0, Qt::AlignVCenter);

    m_next = new QToolButton(m_nav);
    m_next->setObjectName(QStringLiteral("TourNavButton"));
    m_next->setText(QStringLiteral("›"));
    m_next->setCursor(Qt::PointingHandCursor);
    m_next->setAutoRaise(true);
    m_next->setFocusPolicy(Qt::NoFocus);
    m_next->setFixedSize(navButton);
    m_next->setToolTip(tr("Next screen  (→)"));
    connect(m_next, &QToolButton::clicked, this, &TourOverlay::next);
    navRow->addWidget(m_next, 0, Qt::AlignVCenter);
}

void TourOverlay::setSteps(const QVector<TourStep> &steps)
{
    m_steps = steps;
}

// The cutout's corner radius for the step on screen. A step that lights a card
// carries the card's own; everything else gets the default, which suits the
// controls it is cut around.
qreal TourOverlay::spotRadius() const
{
    if (m_index < 0 || m_index >= m_steps.size())
        return kSpotRadius;
    const int r = m_steps.at(m_index).radius;
    return r >= 0 ? qreal(r) : qreal(kSpotRadius);
}

QColor TourOverlay::accent() const
{
    // The accent of whatever palette is on, so the glow belongs to the theme
    // instead of being a colour bolted on top of it.
    return ThemeManager::mapped("#7ea8a0");
}

void TourOverlay::start()
{
    startAt(0);
}

void TourOverlay::startAt(int index)
{
    if (m_steps.isEmpty() || m_active)
        return;
    m_active = true;
    m_lastIndex = -1;
    m_completed = false;
    setGeometry(m_host->rect());
    show();
    raise();
    setFocus(Qt::OtherFocusReason);
    qApp->installEventFilter(this);

    // Out of sight until the first step has been measured and placed. Shown
    // straight away they paint once at wherever they happened to be left —
    // top-left, over a page the scrim has not darkened yet — which is a real
    // flash on a machine slow enough to get a frame in before the placement.
    m_callout->hide();
    m_nav->hide();

    m_scrim = 0.0;
    m_scrimAnim->stop();
    m_scrimAnim->setStartValue(0.0);
    m_scrimAnim->setEndValue(1.0);
    m_scrimAnim->start();

    m_spotlight = QRect();
    m_pendingIndex = -1;
    renderNow(qBound(0, index, m_steps.size() - 1));
}

void TourOverlay::next()
{
    if (!m_active)
        return;
    if (m_index >= m_steps.size() - 1) {
        m_completed = true;        // ran to the end: nothing to resume
        closeTour();
        return;
    }
    render(m_index + 1);
}

void TourOverlay::prev()
{
    if (!m_active || m_index <= 0)
        return;
    render(m_index - 1);
}

void TourOverlay::closeTour()
{
    if (!m_active)
        return;
    m_active = false;
    // Where it was abandoned, so a caller can offer to pick it up again; -1
    // when it ran to the end and there is nothing to resume. Reading the old
    // value to decide was a plain mistake: startAt() had just set it to -1, so
    // the test was never true and the index was never recorded.
    m_lastIndex = m_completed ? -1 : m_index;
    qApp->removeEventFilter(this);
    m_spotAnim->stop();
    m_annotAnim->stop();
    m_annotations.clear();

    m_scrimAnim->stop();
    m_scrimAnim->setStartValue(m_scrim);
    m_scrimAnim->setEndValue(0.0);
    // Disconnected after it fires, so a later fade-in does not hide the tour.
    auto *conn = new QMetaObject::Connection;
    *conn = connect(m_scrimAnim, &QVariantAnimation::finished, this, [this, conn] {
        disconnect(*conn);
        delete conn;
        hide();
        emit tourFinished();
    });
    m_scrimAnim->start();
}

void TourOverlay::render(int index)
{
    // A step change is a cross-fade, not a cut. Swapping the words in place and
    // jumping the box to its new position in the same frame is what makes a
    // guided tour feel like a slideshow of screenshots; taking the old panel
    // away first, changing the page while nothing is being read, and bringing
    // the new one up where it belongs is what makes it feel like one thing
    // moving. The page switch and the layout settling happen underneath the
    // fade, so none of it is watched.
    if (!m_callout->isVisible()) {
        renderNow(index);
        return;
    }
    m_pendingIndex = index;
    m_calloutFade->stop();   // stop() does not emit finished(), so no chaining
    m_calloutFade->setDuration(kCalloutOutMs);
    m_calloutFade->setStartValue(calloutOpacity());
    m_calloutFade->setEndValue(0.0);
    m_calloutFade->start();
}

void TourOverlay::renderNow(int index)
{
    m_index = index;
    m_measurePasses = 0;
    const TourStep &step = m_steps.at(index);

    if (step.onEnter)
        step.onEnter();

    // A single turn of the event loop is not enough here: the pages are a
    // QStackedWidget, choosing a mode rewrites the form, and some containers
    // animate. Force the layouts to settle first; the scrolling and the
    // measuring both happen in measureAndShow(), once they have.
    if (m_host->layout())
        m_host->layout()->activate();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::LayoutRequest);

    if (step.settleMs > 0)
        QTimer::singleShot(step.settleMs, this, &TourOverlay::measureAndShow);
    else
        QTimer::singleShot(0, this, &TourOverlay::measureAndShow);
}

// Scrolls the page so the whole of what is about to be lit is on screen.
// Answers whether it moved anything, which is the caller's cue to measure on
// the next turn of the event loop rather than this one.
//
// QScrollArea::ensureWidgetVisible() is the obvious call and the wrong one: it
// scrolls the least it can get away with, so a card taller than the viewport
// is left with its top edge resting on the bottom of the window — technically
// visible, and useless to look at. Since a step now lights a whole card, and
// the cards on this form are routinely half a screen tall, the scrolling has to
// be done explicitly: centre what fits, and top-align what does not.
bool TourOverlay::ensureTargetVisible(const TourStep &step)
{
    // Breathing room above a card that has been brought to the top.
    constexpr int kScrollMargin = 24;

    QScrollArea *area = nullptr;
    QRect wanted;   // union of the targets, in the scrolled content's own space
    for (const QPointer<QWidget> &w : step.targets) {
        if (!w || !w->isVisible())
            continue;
        QScrollArea *own = nullptr;
        for (QWidget *p = w->parentWidget(); p; p = p->parentWidget()) {
            if ((own = qobject_cast<QScrollArea *>(p)))
                break;
            if (p == m_host)
                break;
        }
        if (!own || !own->widget())
            continue;
        // Targets spread over two scroll areas cannot both be framed; the
        // first one wins, which is the one the step is really about.
        if (!area)
            area = own;
        else if (own != area)
            continue;
        const QRect r(w->mapTo(area->widget(), QPoint(0, 0)), w->size());
        wanted = wanted.isNull() ? r : wanted.united(r);
    }
    if (!area || wanted.isNull())
        return false;

    QScrollBar *bar = area->verticalScrollBar();
    const int viewport = area->viewport()->height();
    const int value = (wanted.height() + 2 * kScrollMargin <= viewport)
                          ? wanted.top() - (viewport - wanted.height()) / 2
                          : wanted.top() - kScrollMargin;
    const int before = bar->value();
    bar->setValue(qBound(bar->minimum(), value, bar->maximum()));
    // The value it settled on, not the one asked for: it is clamped to the
    // range, and the range depends on a layout that may still be catching up.
    return bar->value() != before;
}

// In overlay coordinates, clipped to every scroll viewport on the way up.
//
// The clipping is the part that matters: almost every target lives inside a
// QScrollArea, and a widget scrolled out of sight still reports isVisible() ==
// true. Without the intersection the light would be drawn over a stretch of
// window where the widget is not.
QRect TourOverlay::targetRect(const TourStep &step) const
{
    QRect united;
    for (const QPointer<QWidget> &w : step.targets) {
        if (!w || !w->isVisible())
            continue;
        // mapFromGlobal(mapToGlobal(...)) rather than mapTo(): it does not
        // require the target to be a descendant of the host.
        QRect r(m_host->mapFromGlobal(w->mapToGlobal(QPoint(0, 0))), w->size());
        for (QWidget *p = w->parentWidget(); p; p = p->parentWidget()) {
            if (auto *area = qobject_cast<QScrollArea *>(p)) {
                const QRect vp(m_host->mapFromGlobal(area->viewport()->mapToGlobal(QPoint(0, 0))),
                               area->viewport()->size());
                r = r.intersected(vp);
            }
            if (p == m_host)
                break;
        }
        if (r.isValid() && !r.isEmpty())
            united = united.isNull() ? r : united.united(r);
    }
    if (!united.isNull())
        united.adjust(-step.padding, -step.padding, step.padding, step.padding);
    return united;
}

void TourOverlay::measureAndShow()
{
    if (!m_active || m_index < 0 || m_index >= m_steps.size())
        return;
    const TourStep &step = m_steps.at(m_index);

    // Scrolled here rather than in renderNow(): a scroll bar's range comes from
    // the laid-out size of the page, and at the moment the step is entered the
    // page may not have one yet — on the very first screen it certainly does
    // not, and the value asked for is silently clamped to a maximum of nearly
    // zero. By now the layout has settled.
    //
    // And if it did scroll, everything below is measured on the *next* turn of
    // the event loop instead of this one. Widget positions after a scroll are
    // not reliably settled within the same turn, and measuring through that
    // window put the light where the card had been rather than where it now
    // is — a whole card out, with the captions pointing into the gap. The
    // second pass finds the value already right, reports no change, and
    // measures for real, so this recurses exactly once.
    //
    // Bounded, and not by faith: a chunk unfolding on the batch page changes
    // its own height for 180 ms, so "the scroll position moved again" can stay
    // true for a while. Three passes is more than any of them needs, and after
    // that the step is drawn wherever things have got to rather than spinning.
    if (ensureTargetVisible(step) && ++m_measurePasses < 3) {
        QTimer::singleShot(0, this, &TourOverlay::measureAndShow);
        return;
    }

    const QRect spot = targetRect(step);
    // Back to the bottom before anything is placed. Where the navigation bar
    // ends up cannot be decided until the lit area *and* the picture below it
    // are known, and the picture is placed against the free band — so the band
    // has to start from a known state rather than from whatever the previous
    // step happened to leave.
    m_navAtTop = false;
    // The picture hangs where the real thing would drop, and from here on it
    // counts as part of the lit area: the captions go beside *it*, and the
    // callout keeps clear of both.
    m_insetRect = placeInset(spot);
    QRect reference = spot;
    if (!m_insetRect.isNull())
        reference = reference.isNull() ? m_insetRect : reference.united(m_insetRect);

    m_title->setText(step.title);
    m_body->setText(step.text);
    m_continue->setText(m_index == m_steps.size() - 1 ? tr("Finish") : tr("Continue"));
    // After the labels, before anything is measured: the pair's size is part of
    // how tall the panel comes out.
    matchFootButtons();
    refreshNav();
    // Needs the counter's text, and is needed before the panel is measured: the
    // padding it sets is part of the foot row's height.
    centreNavGlyphs();

    // Everything painted for the step before this one goes now, in one whole
    // repaint. It is the only place that asks for one, and it has to: the
    // captions, their leader lines and the inset picture are painted by this
    // widget, and every other repaint in here is a *partial* one, computed from
    // where the light is travelling. A caption that belonged to the previous
    // step lies outside that region, so nothing ever invalidated it and it
    // stayed on screen over the new step — until something else forced a full
    // repaint, which is why clicking outside the window "fixed" it.
    //
    // One full repaint per step change costs nothing; the partial ones exist
    // for the sixty a second while the light is moving, and they keep that job.
    update();

    // The navigation bar first: it decides which end of the window everything
    // else has to keep clear of, and it is the only piece whose own position
    // depends on nothing but the lit area.
    placeNavBar(reference);

    // Captions next: where they land decides where the callout can go, since
    // the two must not overlap.
    m_annotations.clear();
    m_annotProgress = 0.0;
    layoutAnnotations(reference);

    QRect occupied = reference;
    for (const PlacedAnnotation &a : std::as_const(m_annotations))
        occupied = occupied.isNull() ? a.chip : occupied.united(a.chip);

    // Sized and placed against the *lit area alone*, not against the union with
    // the captions. The two constraints are not the same kind: standing clear
    // of the light is the hard one — a panel over it hides the very thing the
    // step is about — while a caption is a small box that keepCalloutClear()
    // below can nudge out of the way afterwards.
    //
    // Measuring against the union is what broke the Viewer step. One caption
    // sits above the canvas, so `occupied` began at that caption's top edge and
    // the free band above collapsed to a sliver a couple of hundred pixels
    // deep. The panel was then made as wide as the window trying to grow short
    // enough for a sliver it could never fit, and ended up drawn over the map
    // regardless — while the band actually above the canvas, some five hundred
    // pixels of toolbar, went unused.
    fitCalloutWidth(step.avoidLitArea ? reference : QRect());
    placeCallout(reference);
    // Clear of the lit area when the step asks for it, and clear of its own
    // captions always: a panel drawn over a caption hides half of what the
    // screen is saying, and unlike the lit area there is never a reason to
    // accept it. `occupied` is the union of the two, so the first call covers
    // both; the second is for the steps that do not ask to avoid the light but
    // still have captions to keep out from under.
    if (step.avoidLitArea)
        keepCalloutClear(occupied);
    else if (!m_annotations.isEmpty()) {
        QRect chips;
        for (const PlacedAnnotation &a : std::as_const(m_annotations))
            chips = chips.isNull() ? a.chip : chips.united(a.chip);
        keepCalloutClear(chips);
    }
    fadeCalloutIn();

    animateSpotlightTo(spot);
    if (m_annotations.isEmpty() && m_spotAnim->state() != QAbstractAnimation::Running)
        update();

    emit stepChanged(m_index, m_steps.size());
}

void TourOverlay::animateSpotlightTo(const QRect &target)
{
    m_spotAnim->stop();
    if (m_spotlight.isNull() || target.isNull()) {
        // Nothing to travel between: appear where it belongs.
        const QRect before = m_spotlight;
        m_spotlight = target;
        repaintFor(before, target);
        if (!m_annotations.isEmpty()) {
            m_annotAnim->stop();
            m_annotAnim->start();
        }
        return;
    }
    m_spotAnim->setStartValue(m_spotlight);
    m_spotAnim->setEndValue(target);
    m_spotAnim->start();
}

void TourOverlay::repaintFor(const QRect &a, const QRect &b)
{
    // Only the ground the light left and the ground it reached, widened for the
    // glow. Repainting the whole window sixty times a second while an analysis
    // has every core busy is exactly the cost this avoids.
    QRect dirty = a.isNull() ? b : (b.isNull() ? a : a.united(b));
    if (!m_insetRect.isNull())
        dirty = dirty.isNull() ? m_insetRect : dirty.united(m_insetRect);
    if (dirty.isNull()) {
        update();
        return;
    }
    update(dirty.adjusted(-3 * kGlowRings - 6, -3 * kGlowRings - 6,
                          3 * kGlowRings + 6, 3 * kGlowRings + 6));
}

// As narrow as it can be while still being comfortable to read *and* fitting in
// the space the lit area leaves.
//
// A single fixed width cannot manage either: the shortest screens are one
// sentence and the longest run to a dozen lines, and the interface font is a
// setting — at a large one, a paragraph that fitted on a 1440-tall screen runs
// off the bottom of a laptop's. So the box is measured at each candidate width
// and grows sideways until it is short enough.
//
// Short enough means two things. "Comfortable" is deliberately stricter than
// "it fits on screen": a paragraph poured into a narrow column becomes a tall
// ribbon of six-word lines long before it runs out of room. And it must clear
// the lit area — a callout that ends up drawn across the very card the screen
// is describing hides what the reader was told to look at. Widening it is
// exactly the remedy, because every extra pixel of width takes lines off the
// height.
//
// `occupied` is the lit area together with its captions; a null one means the
// screen lights nothing and the box may use the whole band.
void TourOverlay::fitCalloutWidth(const QRect &occupied)
{
    // Tried in order, and the first one that fits the target height wins, so a
    // wide panel is only ever reached when nothing narrower will do. The list
    // stops at 920 on purpose: past that the line is too long to read
    // comfortably, and a panel that still does not fit at 920 is telling you
    // the band it was measured against is the wrong one — unless the step
    // itself has said as much and raised calloutWidthCap, in which case a few
    // more rungs are added up to that width. Only ever wider than the default,
    // and only for the one or two steps that ask for it.
    QVarLengthArray<int, 8> widths{kCalloutWidth, 650, 740, 830, 920};
    const int cap = m_steps.at(m_index).calloutWidthCap;
    if (cap > 920) {
        for (int w = 1100; w < cap; w += 200)
            widths.append(w);
        widths.append(cap);
    }
    // Both labels are measured below, and a label that has never been polished
    // is still carrying the application font rather than the stylesheet's.
    m_title->ensurePolished();
    m_body->ensurePolished();
    // Never wider than the window itself has room for, whatever the text.
    const int widest = qMax(kCalloutWidth / 2, width() - 2 * kMargin);
    const int room = qMax(160, bottomLimit() - topLimit());

    int target = qMin(room, kCalloutEasyHeight);
    if (!occupied.isNull()) {
        // The taller of the two gaps the callout could stand in, measured the
        // way placeCallout() will measure them.
        const int below = bottomLimit() - (occupied.bottom() + kCalloutGap);
        const int above = (occupied.top() - kCalloutGap) - topLimit();
        const int free = qMax(below, above);
        // Below kCalloutFloor no width can help — a title, a line of text and a
        // button do not fold any smaller — so chasing it would buy a few pixels
        // at the price of a box the width of the window. Leave those to overlap.
        if (free >= kCalloutFloor)
            target = qMin(target, free);
    }

    for (const int candidate : widths) {
        const int w = qMin(candidate, widest);

        // The heading, pinned to the height its own text needs at this width.
        //
        // A word-wrapping QLabel with a Fixed vertical policy is handed
        // sizeHint().height(), and a wrapping label's size *hint* is a guess
        // made without knowing the width it will get: it comes out at two or
        // three lines for a title that will be drawn on one, and the surplus is
        // split above and below by the vertical centring. That is the gap.
        //
        // The width is worked out here rather than read back from the label,
        // and that is the whole repair. m_title->width() is only true once the
        // layout has run at the new callout width, which it has not: setting a
        // fixed width on a widget whose resize event has not been delivered
        // leaves the children where the *previous* step put them. So the pin
        // was computed at whatever width the label happened to still have —
        // three lines' worth on one screen, one line's worth on the next, which
        // is exactly the pair of symptoms: some panels gaping, some cramped.
        //
        // The row is: an empty column the width of the ✕, the heading, the ✕.
        // Justified reads well over several lines and badly over one: a single
        // line justified is either left-ranged (nothing to stretch) or, worse,
        // stretched across the whole column with holes between the words. A
        // paragraph that comes out one line long is therefore centred instead.
        // Decided here, at each candidate width, because whether the text is
        // one line *is* a function of the width being tried.
        const int bodyWidth = w - 2 * kPanelPadH;
        // Measured a little narrower than the label will actually be given. A
        // paragraph that fits on one line by a handful of pixels gets measured
        // as one line and drawn as two: the metrics asked here and the glyphs
        // the label finally lays out do not agree down to the last pixel, and
        // at a fractional display scaling they cannot. What that produced was
        // the worst of both — a centred full-width first line with one word
        // stranded and centred underneath it. An em of slack settles those
        // borderline cases as "it wraps", which is the safe way round: a line
        // that already reaches within an em of both margins looks the same
        // centred as it does ranged left, so deciding it wrongly costs nothing,
        // while deciding a two-line paragraph wrongly is plain to see.
        const int slack = m_body->fontMetrics().horizontalAdvance(QLatin1Char('m'));
        const int bodyHeight = m_body->heightForWidth(qMax(1, bodyWidth - slack));
        // One line, with room for the taller ascenders a bold run brings: two
        // lines could never come in under one and a half.
        const bool singleLine =
            bodyHeight > 0
            && bodyHeight < (m_body->fontMetrics().lineSpacing() * 3) / 2;
        m_body->setAlignment((singleLine ? Qt::AlignHCenter : Qt::AlignJustify)
                             | Qt::AlignTop);

        const int titleWidth = w - 2 * kPanelPadH - 2 * kCloseWidth - 2 * kHeadSpacing;
        const int titleHeight = m_title->heightForWidth(titleWidth);
        m_title->setFixedHeight(titleHeight > 0 ? titleHeight
                                                : m_title->fontMetrics().height());

        m_callout->setFixedWidth(w);
        m_callout->adjustSize();
        // And the height the layout actually needs *at this width*, which is
        // not what adjustSize() just used. adjustSize() takes the layout's size
        // hint, and a wrapping paragraph's hint is a height guessed at some
        // natural width of its own choosing — at 920 px it is routinely a line
        // or two too tall. Every one of those spare pixels then has to go
        // somewhere inside the panel, and no arrangement of them is right.
        if (QLayout *lay = m_callout->layout(); lay && lay->hasHeightForWidth())
            m_callout->resize(w, lay->totalHeightForWidth(w));

        if (m_callout->height() <= target || w >= widest)
            return;
    }
}

void TourOverlay::placeCallout(const QRect &spot)
{
    const QSize sz = m_callout->size();
    const int top = topLimit();
    const int bottom = bottomLimit();
    int x = (width() - sz.width()) / 2;
    int y = (height() - sz.height()) / 2;

    if (!spot.isNull()) {
        x = spot.center().x() - sz.width() / 2;
        // Above, tried first and only when the step asks for it and it
        // actually fits: everything else below is the untouched below-first
        // cascade, reached exactly as before whenever this does not apply.
        if (m_steps.at(m_index).preferAbove) {
            const int above = spot.top() - kCalloutGap - sz.height();
            if (above >= top) {
                m_callout->move(qBound(kMargin, x, qMax(kMargin, width() - sz.width() - kMargin)), above);
                m_callout->raise();
                return;
            }
        }
        y = spot.bottom() + kCalloutGap;
        if (y + sz.height() > bottom) {
            const int above = spot.top() - kCalloutGap - sz.height();
            if (above >= top) {
                y = above;
            } else if (spot.top() - top >= sz.height() - kFrameReveal) {
                // Short of the full gap, but only just: pressed against the top
                // margin the panel still stands above the lit area and laps its
                // edge by no more than the sliver the light needs to stay
                // visible. The alternative below is a panel across the middle of
                // the very thing being described — which is what a canvas that
                // fills the window used to get, and it hid the map.
                y = top;
            } else {
                // Neither under nor over: go beside it. Falling straight to
                // "centred" put the callout across the middle of the gear menu
                // and hid the very rows the captions were pointing at.
                const int leftOf = spot.left() - kCalloutGap - sz.width();
                const int rightOf = spot.right() + kCalloutGap;
                if (leftOf >= kMargin) {
                    x = leftOf;
                    y = qBound(top, spot.top(), qMax(top, bottom - sz.height()));
                } else if (rightOf + sz.width() <= width() - kMargin) {
                    x = rightOf;
                    y = qBound(top, spot.top(), qMax(top, bottom - sz.height()));
                } else {
                    // A card that fills the window leaves nowhere to stand:
                    // overlap it as little as possible by going flush to
                    // whichever end has more room, rather than sitting across
                    // its middle — which is where the fields being described
                    // are. Flush, but kFrameReveal short of it, so the lit
                    // edge on that side stays in sight under the panel.
                    y = (bottom - spot.bottom() >= spot.top() - top)
                            ? qMin(bottom - sz.height(),
                                   spot.bottom() - kFrameReveal - sz.height())
                            : qMax(top, spot.top() + kFrameReveal);
                }
            }
        }
    }
    x = qBound(kMargin, x, qMax(kMargin, width() - sz.width() - kMargin));
    y = qBound(top, y, qMax(top, bottom - sz.height()));
    m_callout->move(x, y);
    m_callout->raise();
}

// Bottom centre, unless the lit area is down there.
//
// A step that lights a whole card, on a form whose cards are half a screen
// tall, routinely reaches the bottom of the window — and the bar was then
// drawn across the last row of the very card being described. Moving it to the
// top is better than shrinking the light or covering the page in reserved
// bands: the bar is small, and there are only ever two places it can be, so it
// stays findable.
//
// If both ends are occupied the bottom wins: that is where it has been all
// along, and a bar that hops about for no visible gain is worse than one that
// overlaps.
// The last word on where the callout sits, and it belongs to the geometry
// rather than to the arithmetic above.
//
// placeCallout() works from the rectangles it was handed, and those are one
// round of measurement away from the truth: the panel is sized before the
// layout has quite finished with it, so a few pixels of disagreement are
// enough to leave it resting on the lit card or on a caption. On a caption the
// result is not even visible as an overlap — the captions are painted by this
// widget and the callout is a child widget on top of them, so the caption
// simply is not there.
//
// So: measure what is actually on screen, and if it touches what it is meant
// to be explaining, move it clear. Down first, since that is where the room
// usually is; up if not; and if neither end has space, leave it — at that point
// the window is too small for any arrangement and moving it only trades one
// overlap for another.
void TourOverlay::keepCalloutClear(const QRect &occupied)
{
    if (occupied.isNull())
        return;
    const QRect box(m_callout->pos(), m_callout->size());
    // A gap, not a touch: the glow is drawn outside the cutout, and a panel
    // resting against it reads as part of the card.
    const QRect avoid = occupied.adjusted(0, -kCalloutGap, 0, kCalloutGap);
    if (!box.intersects(avoid))
        return;

    const int below = occupied.bottom() + kCalloutGap;
    const int above = occupied.top() - kCalloutGap - box.height();
    if (below + box.height() <= bottomLimit()) {
        m_callout->move(box.left(), below);
        return;
    }
    if (above >= topLimit()) {
        m_callout->move(box.left(), above);
        return;
    }

    // Sideways, when neither end has room. Something tall and narrow — the gear
    // menu is the case, a list that drops almost the full height of the window —
    // leaves nothing above it and nothing below, while the whole left half of
    // the screen stands empty. Without this the panel simply stayed where it
    // was, on top of a caption.
    //
    // The side with more room is tried first, and the panel is only moved if it
    // genuinely fits: a move that merely swaps which caption is covered is not
    // worth making.
    const int leftOf = occupied.left() - kCalloutGap - box.width();
    const int rightOf = occupied.right() + kCalloutGap;
    const bool leftFirst =
        occupied.left() - kMargin >= (width() - kMargin) - occupied.right();
    const int first = leftFirst ? leftOf : rightOf;
    const int second = leftFirst ? rightOf : leftOf;
    for (const int x : {first, second}) {
        if (x >= kMargin && x + box.width() <= width() - kMargin) {
            m_callout->move(x, box.top());
            return;
        }
    }
}

// The ‹ 7 / 44 › of the navigation bar, all three on the bar's own axis.
//
// Measured rather than nudged by a constant, because the interface font is a
// setting: seven families, each with its own bearings, and the chevrons are set
// at twice the size of the counter beside them. Left alone the marks sat almost
// four pixels low and the digits one and a half, which is exactly the sort of
// thing that reads as "crooked" without being nameable.
void TourOverlay::centreNavGlyphs()
{
    const QString marks = centringCss(QStringLiteral("QToolButton#TourNavButton"),
                                      inkDrop(m_prev->fontMetrics(), m_prev->text()));
    const QString count = centringCss(QStringLiteral("QLabel#TourCounter"),
                                      inkDrop(m_counter->fontMetrics(), m_counter->text()));
    // Applied only when it changes: setting a stylesheet re-polishes the widget,
    // and this runs at every step.
    if (marks + count == m_navGlyphCss)
        return;
    m_navGlyphCss = marks + count;
    m_prev->setStyleSheet(marks);
    m_next->setStyleSheet(marks);
    m_counter->setStyleSheet(count);
    // The count in the panel's foot gets the same treatment, and needs it more:
    // there it sits between two buttons, and a row of digits a pixel low beside
    // two words that are not is exactly what "crooked" looks like.
    m_footCounter->setStyleSheet(count);
}

void TourOverlay::placeNavBar(const QRect &avoid)
{
    if (!kShowNavBar) {
        m_nav->hide();
        // Nothing is at the top, so nothing has to be kept clear of it: the
        // limits below read this flag through m_navAtTop.
        m_navAtTop = false;
        return;
    }
    m_nav->adjustSize();
    const QSize sz = m_nav->size();
    const int x = (width() - sz.width()) / 2;
    const QRect atBottom(x, height() - sz.height() - kNavBottomGap, sz.width(), sz.height());
    const QRect atTop(x, kNavBottomGap, sz.width(), sz.height());
    // Grazing the lit edge counts as overlapping: the glow is drawn outside the
    // cutout, and a bar resting on it reads as part of the card.
    const QRect danger = avoid.isNull() ? QRect() : avoid.adjusted(-10, -10, 10, 10);

    m_navAtTop = !danger.isNull() && danger.intersects(atBottom)
                 && !danger.intersects(atTop);
    m_nav->move(m_navAtTop ? atTop.topLeft() : atBottom.topLeft());
    m_nav->show();
    m_nav->raise();
}

// The band the panel, the captions and the picture may use. Both ends are the
// plain margin when there is no navigation bar to keep clear of — reserving 76
// px for a hidden widget would push every screen up for nothing.
int TourOverlay::topLimit() const
{
    return (kShowNavBar && m_navAtTop) ? kNavReserve : kMargin;
}

int TourOverlay::bottomLimit() const
{
    return (kShowNavBar && !m_navAtTop) ? height() - kNavReserve : height() - kMargin;
}

// Back and Continue are one pair of buttons, not two that happen to share a
// row. Left to their size hints they came out plainly different objects: the
// labels differ in length, "Continue" becomes "Finish" on the last screen, and
// the styles are not the same either — the filled one is set in a larger,
// heavier face than the outlined one, which makes it taller as well as wider.
// So the pair takes whichever of the two needs more, in both directions, with a
// floor under it; and it is done at every step, because the right-hand label
// changes.
void TourOverlay::matchFootButtons()
{
    // Asked before the size is pinned, and unaffected by a pin left from the
    // previous step: a button's size hint comes from its text and its style,
    // not from its minimum and maximum.
    const QSize back = m_back->sizeHint();
    const QSize next = m_continue->sizeHint();
    const QSize pair(qMax(kFootButtonW, qMax(back.width(), next.width())),
                     qMax(kFootButtonH, qMax(back.height(), next.height())));
    m_back->setFixedSize(pair);
    m_continue->setFixedSize(pair);
}

void TourOverlay::refreshNav()
{
    m_prev->setEnabled(m_index > 0);
    // Hidden rather than greyed on the first screen: there is nothing behind it
    // to go back to, and a dead button in a panel with only two is noise. The
    // stretch beside it keeps Continue where it was.
    m_back->setVisible(m_index > 0);
    // Both copies: the one in the panel is what anyone reads, the one in the
    // bar is there for the day the bar is switched back on.
    const QString count = QStringLiteral("%1 / %2").arg(m_index + 1).arg(m_steps.size());
    m_counter->setText(count);
    m_footCounter->setText(count);
}

double TourOverlay::calloutOpacity() const
{
    auto *effect = qobject_cast<QGraphicsOpacityEffect *>(m_callout->graphicsEffect());
    return effect ? effect->opacity() : 1.0;
}

void TourOverlay::fadeCalloutIn()
{
    auto *effect = qobject_cast<QGraphicsOpacityEffect *>(m_callout->graphicsEffect());
    if (!effect)
        return;
    m_calloutFade->stop();
    m_calloutFade->setDuration(kCalloutFadeMs);
    m_calloutFade->setStartValue(0.0);
    m_calloutFade->setEndValue(1.0);

    // A short rise as it appears, rather than a bare fade. The eye reads the
    // movement as the panel arriving; a box that simply materialises in place
    // reads as a redraw.
    const QPoint resting = m_callout->pos();
    auto *rise = new QPropertyAnimation(m_callout, "pos", m_callout);
    rise->setDuration(kCalloutFadeMs + 60);
    rise->setEasingCurve(QEasingCurve::OutCubic);
    rise->setStartValue(resting + QPoint(0, kCalloutRisePx));
    rise->setEndValue(resting);

    m_callout->show();
    m_calloutFade->start();
    rise->start(QAbstractAnimation::DeleteWhenStopped);
}

// The inset hangs under the lit widget and aligned to its right edge, which is
// where a menu dropping from a button in the top right would actually appear.
QRect TourOverlay::placeInset(const QRect &spot) const
{
    const QPixmap &pm = m_steps.at(m_index).inset;
    if (pm.isNull())
        return QRect();
    const QSize sz = pm.deviceIndependentSize().toSize();
    int x = spot.isNull() ? (width() - sz.width()) / 2 : spot.right() - sz.width();
    int y = spot.isNull() ? topLimit() : spot.bottom() + 6;
    x = qBound(kMargin, x, qMax(kMargin, width() - sz.width() - kMargin));
    y = qBound(topLimit(), y, qMax(topLimit(), bottomLimit() - sz.height()));
    return QRect(QPoint(x, y), sz);
}

// Where the captions go.
//
// Two arrangements, because this application has both shapes. A tall, narrow
// target (a column of fields) leaves room at its side, and the captions stack
// there, each level with the control it names. A wide one — a row of cards, a
// card spanning the page, which is most of Trajecta — leaves no side room at
// all, and the captions go underneath instead, each centred on its own control.
// Choosing by the target's centre, as a first attempt did, pushed them into a
// 140-pixel gutter where every caption wrapped to six lines.
void TourOverlay::layoutAnnotations(const QRect &spot)
{
    const QVector<TourAnnotation> &list = m_steps.at(m_index).annotations;
    if (list.isEmpty())
        return;

    // Resolve every anchor to a rectangle first: some name a live widget, some
    // name a part of the step's picture, and from here on they are all the same
    // thing — somewhere on the overlay to point at.
    struct Resolved { QRect rect; QString text; bool above; };
    QVector<Resolved> items;
    for (const TourAnnotation &a : list) {
        if (a.anchor && a.anchor->isVisible()) {
            items.append({ QRect(m_host->mapFromGlobal(a.anchor->mapToGlobal(QPoint(0, 0))),
                                 a.anchor->size()), a.text, a.above });
        } else if (!a.insetRect.isNull() && !m_insetRect.isNull()) {
            items.append({ a.insetRect.translated(m_insetRect.topLeft()), a.text, a.above });
        }
    }
    if (items.isEmpty())
        return;

    const QFontMetrics fm(font());
    const int roomLeft = spot.isNull() ? width() / 2 : spot.left();
    const int roomRight = spot.isNull() ? width() / 2 : width() - spot.right();
    const bool onLeft = roomLeft >= roomRight;
    const int side = qMax(roomLeft, roomRight) - kLeaderGap - 2 * kMargin;

    // The shape of the captions follows the shape of what they point at, which
    // is the only thing that keeps the leaders short. Controls spread across the
    // page (the three mode cards) get a row of captions underneath, one beneath
    // each. Controls stacked one above the other (a column of fields) get a
    // column of captions at the side, each level with its own. Deciding by the
    // free space instead — the first attempt — put three captions in a row under
    // a vertical stack, and the leaders crossed the whole window.
    int minCx = INT_MAX;
    int maxCx = INT_MIN;
    for (const Resolved &r : items) {
        minCx = qMin(minCx, r.rect.center().x());
        maxCx = qMax(maxCx, r.rect.center().x());
    }
    const bool spreadAcross = (maxCx > minCx) && (maxCx - minCx) > 120;
    // A sliver too narrow to read is worse than a long leader, so a very tight
    // side sends even a stacked set underneath.
    m_annotBelow = spreadAcross || side < 110;

    if (!m_annotBelow) {
        int cursor = topLimit();
        for (const Resolved &r : items) {
            const int textW = qMin(kChipMaxWidth, side);
            const QRect textRect = fm.boundingRect(QRect(0, 0, textW - 2 * kPanelPadH, 0),
                                                   Qt::TextWordWrap, r.text);
            const int chipW = textRect.width() + 2 * kPanelPadH;
            const int chipH = textRect.height() + 2 * kPanelPadV;

            const int chipX = onLeft ? spot.left() - kLeaderGap - chipW
                                     : spot.right() + kLeaderGap;
            int chipY = qMax(r.rect.center().y() - chipH / 2, cursor);
            chipY = qBound(topLimit(), chipY,
                           qMax(topLimit(), bottomLimit() - chipH));
            cursor = chipY + chipH + 8;
            m_annotations.append({ r.rect, QRect(chipX, chipY, chipW, chipH), r.text });
        }
        return;
    }

    // Underneath, side by side: each caption gets an equal share of the width
    // and sits under the control it points at.
    //
    // A step may ask for some of its captions to be put *above* the lit area
    // instead (TourAnnotation::above). The set is then laid out as two
    // independent rows, by the same rules and with a different top edge. It is
    // what a card of stacked controls needs: with every caption queueing along
    // the one edge underneath, the leaders of the upper controls have to cross
    // the whole card and each other to get there.
    const int gap = 10;

    auto placeRow = [&](const QVector<Resolved> &rowItems, bool wantAbove) {
        if (rowItems.isEmpty())
            return;

        const int share =
            (width() - 2 * kMargin - (rowItems.size() - 1) * gap) / rowItems.size();
        const int chipW = qBound(120, qMin(kChipMaxWidth, share), qMax(120, share));

        // One height for all of them: a row of boxes with ragged bottoms reads
        // as a mistake rather than as a set. Measured before the row is placed,
        // because whether it fits depends on how tall it is.
        int tallest = 0;
        for (const Resolved &r : rowItems) {
            const QRect textRect = fm.boundingRect(QRect(0, 0, chipW - 2 * kPanelPadH, 0),
                                                   Qt::TextWordWrap, r.text);
            tallest = qMax(tallest, textRect.height() + 2 * kPanelPadV);
        }

        const int overSpot = spot.top() - kLeaderGap - 8 - tallest;
        const int underSpot = spot.bottom() + kLeaderGap + 8;
        int top;
        if (spot.isNull()) {
            top = topLimit();
        } else if (wantAbove) {
            // Asked for above, but the top edge wins over the request: a row
            // drawn off the screen would be worse than a crossed leader.
            top = (overSpot >= topLimit())
                      ? overSpot
                      : qBound(topLimit(), underSpot,
                               qMax(topLimit(), bottomLimit() - tallest));
        } else {
            // A section at the foot of the page has no room beneath it — the
            // run panel is the case — and the row would be drawn off the bottom
            // edge. Put it above the lit area instead; the leaders simply point
            // upwards.
            top = underSpot;
            if (top + tallest > bottomLimit()) {
                top = (overSpot >= topLimit())
                          ? overSpot
                          : qBound(topLimit(), top,
                                   qMax(topLimit(), bottomLimit() - tallest));
            }
        }

        // Left to right in the order the controls actually appear, whatever
        // order the step happened to list them in: the row is placed with a
        // cursor that only ever moves right, so an unsorted set would leave the
        // captions crossing over each other.
        //
        // The tie-break by y is not a detail. A card of full-width fields gives
        // every anchor the same centre x, and without it the row order would be
        // whatever std::sort happened to produce — so the captions under a
        // column of fields read left to right in the order the fields are
        // stacked.
        QVector<Resolved> ordered = rowItems;
        std::sort(ordered.begin(), ordered.end(),
                  [](const Resolved &a, const Resolved &b) {
            if (a.rect.center().x() != b.rect.center().x())
                return a.rect.center().x() < b.rect.center().x();
            return a.rect.center().y() < b.rect.center().y();
        });

        const int firstIndex = m_annotations.size();
        int cursorX = kMargin;
        for (const Resolved &r : ordered) {
            // Centred under its own control, but never on top of the one
            // before: the clamp to the window edge has to come first, or it can
            // drag a chip back over its neighbour.
            int chipX = r.rect.center().x() - chipW / 2;
            chipX = qBound(kMargin, chipX, qMax(kMargin, width() - chipW - kMargin));
            chipX = qMax(chipX, cursorX);
            cursorX = chipX + chipW + gap;
            m_annotations.append({ r.rect, QRect(chipX, top, chipW, tallest), r.text });
        }

        // A second pass, right to left. The cursor above only ever pushes a
        // chip away from its anchor and never pulls one back, so a set whose
        // last controls sit near the right edge ends up hanging off it — while
        // a gap opens in the middle of the row where nothing needed the space.
        // Walking backwards, each chip is pulled in far enough to fit behind
        // the one after it, which closes that gap and brings the tail inside
        // the window. The row fits by construction (that is what `share`
        // computes), so the pass always ends with the first chip at or after
        // the margin.
        int limit = width() - kMargin - chipW;
        for (int i = m_annotations.size() - 1; i >= firstIndex; --i) {
            QRect &chip = m_annotations[i].chip;
            if (chip.left() > limit)
                chip.moveLeft(qMax(kMargin, limit));
            limit = chip.left() - gap - chipW;
        }

        // A third pass: untangle.
        //
        // The row is ordered by the anchors' centres, which is crossing-free as
        // long as every leader meets its anchor at that centre. It does not: a
        // leader slides along a wide anchor to reach its own caption
        // (leaderAttachX), so a wide control's meeting point can overtake a
        // narrow control's and the two lines cross. When that happens the two
        // captions have simply been dealt the wrong boxes — swapping what they
        // hold, rather than where the boxes are, uncrosses the pair and leaves
        // the evenly spaced row exactly as it was.
        for (int pass = firstIndex; pass < m_annotations.size(); ++pass) {
            bool swapped = false;
            for (int i = firstIndex; i + 1 < m_annotations.size(); ++i) {
                PlacedAnnotation &a = m_annotations[i];
                PlacedAnnotation &b = m_annotations[i + 1];
                if (leaderAttachX(a.anchor, a.chip) <= leaderAttachX(b.anchor, b.chip))
                    continue;
                std::swap(a.anchor, b.anchor);
                std::swap(a.text, b.text);
                swapped = true;
            }
            if (!swapped)
                break;
        }
    };

    // The two rows are placed top one first, so that the chips of each row sit
    // together at the end of m_annotations while it is being built: every pass
    // inside placeRow() works on the range from its own firstIndex onwards, and
    // that is only the current row if the previous one is already complete.
    QVector<Resolved> aboveItems;
    QVector<Resolved> belowItems;
    for (const Resolved &r : items) {
        // With nothing lit there is no "above" to be on, and the whole set goes
        // in the single row that layout produces.
        if (r.above && !spot.isNull())
            aboveItems.append(r);
        else
            belowItems.append(r);
    }
    placeRow(aboveItems, true);
    placeRow(belowItems, false);
}

void TourOverlay::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    QColor scrim(0, 0, 0);
    scrim.setAlpha(int(kScrimAlpha * m_scrim));

    if (m_spotlight.isNull()) {
        p.fillRect(rect(), scrim);
    } else {
        const qreal r = spotRadius();
        QPainterPath full;
        full.addRect(QRectF(rect()));
        QPainterPath hole;
        hole.addRoundedRect(QRectF(m_spotlight), r, r);
        p.fillPath(full.subtracted(hole), scrim);

        // A soft edge rather than a drawn line: concentric strokes of falling
        // alpha read as light coming off the cutout.
        const QColor base = accent();
        for (int i = 0; i < kGlowRings; ++i) {
            QColor c = base;
            c.setAlpha(int((160 - i * 36) * m_scrim));
            p.setPen(QPen(c, 2.0));
            p.setBrush(Qt::NoBrush);
            const qreal grow = i * 2.5;
            p.drawRoundedRect(QRectF(m_spotlight).adjusted(-grow, -grow, grow, grow),
                              r + grow, r + grow);
        }
    }

    // The still picture of something that would otherwise have to be opened for
    // real. Drawn at full strength — it is part of what the step is showing,
    // not part of the dimmed background.
    if (!m_insetRect.isNull() && !m_steps.at(m_index).inset.isNull()) {
        p.save();
        QPainterPath clip;
        clip.addRoundedRect(QRectF(m_insetRect), 8, 8);
        p.setClipPath(clip);
        p.drawPixmap(m_insetRect, m_steps.at(m_index).inset);
        p.restore();
        QColor edge = accent();
        edge.setAlphaF(0.55 * m_scrim);
        p.setPen(QPen(edge, 1.5));
        p.setBrush(Qt::NoBrush);
        p.drawRoundedRect(QRectF(m_insetRect).adjusted(0.5, 0.5, -0.5, -0.5), 8, 8);
    }

    if (m_annotations.isEmpty())
        return;

    const QColor chipBg = ThemeManager::mapped("#1a1e24");
    const QColor chipFg = ThemeManager::mapped("#e4e7ec");
    const QColor line = accent();
    const int n = m_annotations.size();

    for (int i = 0; i < n; ++i) {
        // Each caption starts a little after the one before it, so they arrive
        // in reading order instead of all at once.
        const double slot = 1.0 / double(n + 1);
        const double a = qBound(0.0, (m_annotProgress - i * slot) / (2.0 * slot), 1.0);
        if (a <= 0.0)
            continue;
        const PlacedAnnotation &an = m_annotations.at(i);

        QColor pen = line;
        pen.setAlphaF(0.85 * a);
        p.setPen(QPen(pen, 1.6));
        QPointF from;
        QPointF to;
        const auto meet = [&an](int y) {
            return QPointF(leaderAttachX(an.anchor, an.chip), y);
        };
        if (an.chip.top() >= an.anchor.bottom()) {
            // Caption underneath: a short vertical leader, which reads far
            // better than a diagonal when several sit side by side.
            from = QPointF(an.chip.center().x(), an.chip.top());
            to = meet(an.anchor.bottom());
        } else if (an.chip.bottom() <= an.anchor.top()) {
            from = QPointF(an.chip.center().x(), an.chip.bottom());
            to = meet(an.anchor.top());
        } else {
            const bool fromLeft = an.chip.center().x() < an.anchor.center().x();
            from = QPointF(fromLeft ? an.chip.right() : an.chip.left(), an.chip.center().y());
            to = QPointF(fromLeft ? an.anchor.left() : an.anchor.right(), an.anchor.center().y());
        }
        p.drawLine(from, to);
        p.setBrush(pen);
        p.drawEllipse(to, 3.0, 3.0);

        QColor bg = chipBg;
        bg.setAlphaF(0.96 * a);
        QColor border = line;
        border.setAlphaF(0.75 * a);
        p.setBrush(bg);
        p.setPen(QPen(border, 1.0));
        p.drawRoundedRect(QRectF(an.chip), 7, 7);

        QColor fg = chipFg;
        fg.setAlphaF(a);
        p.setPen(fg);
        // Centred, both ways. A row of captions is measured to one common
        // height, so a left-aligned two-line caption beside a three-line one
        // reads as a mistake in the row rather than as a shorter sentence.
        p.drawText(an.chip.adjusted(kPanelPadH, kPanelPadV, -kPanelPadH, -kPanelPadV),
                   Qt::TextWordWrap | Qt::AlignCenter, an.text);
    }
}

// --- Everything below exists to make the tour static -----------------------

void TourOverlay::mousePressEvent(QMouseEvent *event)        { event->accept(); }
void TourOverlay::mouseReleaseEvent(QMouseEvent *event)      { event->accept(); }
void TourOverlay::mouseDoubleClickEvent(QMouseEvent *event)  { event->accept(); }
void TourOverlay::mouseMoveEvent(QMouseEvent *event)         { event->accept(); }
void TourOverlay::wheelEvent(QWheelEvent *event)             { event->accept(); }
void TourOverlay::contextMenuEvent(QContextMenuEvent *event) { event->accept(); }
// Swallowed, never acted on: the tour is driven with the mouse alone.
void TourOverlay::keyPressEvent(QKeyEvent *event)            { event->accept(); }
void TourOverlay::keyReleaseEvent(QKeyEvent *event)          { event->accept(); }
void TourOverlay::dragEnterEvent(QDragEnterEvent *event)     { event->ignore(); }
void TourOverlay::dropEvent(QDropEvent *event)               { event->ignore(); }

bool TourOverlay::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == m_host && event->type() == QEvent::Resize) {
        setGeometry(m_host->rect());
        if (m_active && m_index >= 0 && m_index < m_steps.size()) {
            raise();
            // No animation on a resize: the light belongs where the widget now
            // is, immediately. Everything derived from the geometry has to be
            // recomputed in the same order as a normal step, the inset picture
            // included — leaving it where it was left it detached from the
            // widget it hangs under.
            m_spotAnim->stop();
            m_annotAnim->stop();
            m_spotlight = targetRect(m_steps.at(m_index));
            m_insetRect = placeInset(m_spotlight);
            QRect reference = m_spotlight;
            if (!m_insetRect.isNull())
                reference = reference.isNull() ? m_insetRect : reference.united(m_insetRect);

            placeNavBar(reference);
            m_annotations.clear();
            layoutAnnotations(reference);
            m_annotProgress = 1.0;

            QRect occupied = reference;
            for (const PlacedAnnotation &a : std::as_const(m_annotations))
                occupied = occupied.isNull() ? a.chip : occupied.united(a.chip);
            // The width is chosen from the free space, and a resize is exactly
            // when that changes.
            fitCalloutWidth(m_steps.at(m_index).avoidLitArea ? occupied : QRect());
            placeCallout(occupied);
            update();
        }
        return false;
    }

    // While the tour is up, no key reaches the application: not a shortcut, not
    // a mnemonic, not a character typed into a field that happens to have kept
    // the focus. They are consumed here and do nothing.
    //
    // Unless a modal dialog is up, and one is whenever the tour asks something
    // — "stop the walkthrough?" is the case. That dialog is not the application
    // underneath, it is part of the asking, and a dialog whose buttons answer
    // to neither Enter nor Escape is broken. The filter is on qApp, so without
    // this it would swallow the keys meant for it too.
    if (m_active && !QApplication::activeModalWidget()
        && (event->type() == QEvent::KeyPress
            || event->type() == QEvent::KeyRelease
            || event->type() == QEvent::ShortcutOverride)) {
        // Two exceptions, and they are the tour's own: ← and → turn the pages,
        // the same two moves as the ‹ › in the navigation bar. Caught here, in
        // the filter on qApp, because the overlay itself never takes the focus
        // — the keys are on their way to whatever field had it when the tour
        // started, and this is the only place they pass through.
        //
        // Only on the press, with no modifier, and never on auto-repeat: a held
        // key would fire thirty times a second into a panel that takes 110 ms to
        // cross-fade, and the tour would blur past a dozen screens.
        if (event->type() == QEvent::KeyPress) {
            auto *key = static_cast<QKeyEvent *>(event);
            // Every modifier except the keypad one, which is not a modifier in
            // any sense that matters here: Windows reports the arrow keys as
            // extended keys and Qt turns that into KeypadModifier, so an arrow
            // can arrive "modified" with nothing held down at all. Demanding a
            // bare NoModifier made the keys do nothing.
            const Qt::KeyboardModifiers mods = key->modifiers() & ~Qt::KeypadModifier;
            if (!key->isAutoRepeat() && mods == Qt::NoModifier) {
                if (key->key() == Qt::Key_Right)
                    next();
                else if (key->key() == Qt::Key_Left)
                    prev();
            }
        }
        // Everything else: the tour's own widgets have no focus policy, so
        // nothing of ours needs keys, and nothing underneath may have them.
        event->accept();
        return true;
    }

    return QWidget::eventFilter(watched, event);
}
