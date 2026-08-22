#include "uiwidgets.h"

// The admissible neighbourhood sizes are a property of the square grid, and the
// engine derives them from this header. The form asks the same header rather
// than repeating the list, so the two can never disagree.
#include "neighbourhood.h"

// For the pause mark, which has to be drawn in the active palette's ink.
#include "thememanager.h"

#include <QAbstractSpinBox>
#include <QApplication>
#include <QComboBox>
#include <QEvent>
#include <QFontMetrics>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QIcon>
#include <QKeyEvent>
#include <QLabel>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QObject>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>
#include <QPropertyAnimation>
#include <QPushButton>
#include <QResizeEvent>
#include <QGuiApplication>
#include <QScreen>
#include <QSlider>
#include <QToolButton>

#include <algorithm>
#include <cmath>
#include <QStyle>
#include <QStyleOptionGroupBox>
#include <QTextDocument>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

namespace {

// One pass of the highlight, and how often it is redrawn. 1.6 s is slow enough
// to read as movement rather than flicker; 33 ms is one frame at 30 Hz, which
// is smooth for a shape this simple and costs nothing next to the engine.
constexpr int kSweepMs = 1600;
constexpr int kFrameMs = 33;

// See TrajectaUi::guardWheel.
//
// Installed on the widget itself, deliberately, and not on the application: an
// application-wide filter is consulted once, before Qt starts walking up the
// parent chain looking for someone to scroll, so swallowing the event there
// would stop the page dead under the pointer. A filter on the widget runs
// inside that walk, and refusing the event there hands it to the scroll area
// exactly as if the control were not in the way.
class WheelGuard : public QObject
{
public:
    using QObject::QObject;

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        if (event->type() == QEvent::Wheel) {
            event->ignore();
            return true;
        }
        return QObject::eventFilter(watched, event);
    }
};

// ---------------------------------------------------------------- pause mark
//
// Two filled bars, painted rather than typed.
//
// The mark they answer to — the ▶ on *Run analysis*, *Run batch*, *Compare* and
// *Resume* — is the text glyph U+25B6 sitting inside the button's own label, and
// no interface font Trajecta offers actually contains it: Windows fetches it
// from Segoe UI Symbol, macOS from a face of its own choosing. Its size and
// weight are therefore decided by whatever that fallback happens to be, and a
// companion typed as ❚❚ would be a second, unrelated lottery — that character
// is missing from even more fonts, and where it exists it is often an emoji.
//
// So the pause is drawn, and drawn to the triangle's own measurements: the ink
// the play glyph really puts on screen at this button's font, asked for at the
// moment the mark is built. Change the interface font or the theme and the pair
// stays matched, because both come from the same measurement.
constexpr qreal kPauseBarWidth = 0.30;   // of the mark's height
constexpr qreal kPauseBarGap   = 0.26;   // ditto — the two make up the width
// A little air between the bars and the word, so the mark does not read as a
// letter of it. The play buttons get theirs from two spaces in the string.
constexpr int kPauseTextGap = 5;

QPixmap pauseBars(const QFont &font, const QColor &colour)
{
    const QFontMetrics fm(font);
    // tightBoundingRect, not boundingRect: the triangle sits inside a great
    // deal of side bearing, and bars built on the advance box would tower over
    // it. A font with no glyph at all reports nothing, hence the fallback.
    const QRect ink = fm.tightBoundingRect(QStringLiteral("▶"));
    const int h = ink.height() > 0 ? ink.height() : qRound(fm.ascent() * 0.72);

    const qreal barW = h * kPauseBarWidth;
    const qreal gap = h * kPauseBarGap;
    const int w = qRound(barW * 2 + gap) + kPauseTextGap;

    // Painted at twice the size and marked as such, the same way every other
    // drawn icon in the program is: a 1× pixmap is visibly soft at 150%.
    QPixmap pm(w * 2, h * 2);
    pm.setDevicePixelRatio(2.0);
    pm.fill(Qt::transparent);

    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setPen(Qt::NoPen);
    p.setBrush(colour);
    // Square corners, deliberately: the triangle's corners are sharp, and a
    // rounded bar next to it belongs to a different set of icons.
    p.drawRect(QRectF(0, 0, barW, h));
    p.drawRect(QRectF(barW + gap, 0, barW, h));
    return pm;
}

// Rebuilds the mark whenever the answer could have changed. A QIcon is a fixed
// bitmap: the ink colour comes from the theme and the size from the interface
// font, so both a new palette and a new font make the one on the button wrong.
class PauseMark : public QObject
{
public:
    explicit PauseMark(QPushButton *button)
        : QObject(button), m_button(button)
    {
        button->installEventFilter(this);
    }

    void setActive(bool on)
    {
        m_active = on;
        refresh();
    }

private:
    void refresh()
    {
        if (!m_active) {
            m_button->setIcon(QIcon());
            return;
        }
        // The ink of an ordinary button in theme.qss, mapped through the active
        // palette — the same route the gear and the window buttons take.
        const QPixmap pm = pauseBars(m_button->font(), ThemeManager::mapped("#e4e7ec"));
        m_button->setIcon(QIcon(pm));
        m_button->setIconSize(pm.deviceIndependentSize().toSize());
    }

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        if (m_active && (event->type() == QEvent::StyleChange     // a theme
                         || event->type() == QEvent::FontChange)) // a font
            refresh();
        return QObject::eventFilter(watched, event);
    }

private:
    QPushButton *m_button = nullptr;
    bool m_active = false;
};

// Keeps a group's note beside the group's own title instead of at the far end
// of the line.
//
// The note shares the title band — that is where it belongs, since it is about
// the switch and not about the first field under it — and it used to be pushed
// to the right-hand end of the card, which on a wide window put it half a metre
// from the thing it qualifies. It has to start where the title ends, and where
// the title ends is not something this file can know: the title is drawn by the
// style, from a stylesheet that indents it and pads it, in a font that is a user
// setting, and its text is translated. So the style is asked, and asked again
// whenever the answer could have changed.
class TitleFollower : public QObject
{
public:
    TitleFollower(QGroupBox *group, QWidget *note)
        : QObject(note), m_group(group), m_note(note)
    {
        group->installEventFilter(this);
        apply();
    }

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        switch (event->type()) {
        case QEvent::Show:          // first real geometry
        case QEvent::Resize:        // the note's own x may have moved
        case QEvent::FontChange:    // a different interface font
        case QEvent::StyleChange:   // a different theme
            apply();
            break;
        default:
            break;
        }
        return QObject::eventFilter(watched, event);
    }

private:
    void apply()
    {
        auto *layout = m_note->layout();
        if (!layout)
            return;

        QStyleOptionGroupBox opt;
        opt.initFrom(m_group);
        opt.text = m_group->title();
        opt.textAlignment = Qt::AlignLeft | Qt::AlignVCenter;
        opt.subControls = QStyle::SC_GroupBoxFrame | QStyle::SC_GroupBoxLabel;
        if (m_group->isCheckable())
            opt.subControls |= QStyle::SC_GroupBoxCheckBox;
        const QRect label = m_group->style()->subControlRect(
            QStyle::CC_GroupBox, &opt, QStyle::SC_GroupBoxLabel, m_group);

        // The margin is measured from where the note itself begins, which the
        // grid has already decided and is not the left edge of the box. Setting
        // it changes the note's width but never its x, so one pass settles it.
        constexpr int kGap = 14;
        const int left = qMax(0, label.right() + kGap - m_note->x());
        QMargins m = layout->contentsMargins();
        if (m.left() == left)
            return;
        m.setLeft(left);
        layout->setContentsMargins(m);
    }

    QGroupBox *m_group = nullptr;
    QWidget *m_note = nullptr;
};

} // namespace

namespace TrajectaUi {

void guardWheel(QWidget *w)
{
    if (!w)
        return;
    // StrongFocus as well: with the default policy these controls also take
    // focus as the wheel passes over them, which is the same accident one step
    // removed.
    w->setFocusPolicy(Qt::StrongFocus);
    w->installEventFilter(new WheelGuard(w));
}

} // namespace TrajectaUi

// One decimal place. The bars run 0–1000 rather than 0–100 precisely so a
// tenth of a percent is representable; printing only whole percent threw that
// resolution away and left the number motionless for minutes at a time.
QString ActivityBar::text() const
{
    if (format() != QLatin1String("%p%"))
        return QProgressBar::text();          // a bar with its own wording
    const int span = maximum() - minimum();
    if (span <= 0)
        return QString();                     // busy-indicator mode: no number
    const double percent = 100.0 * double(value() - minimum()) / double(span);
    return QStringLiteral("%1%").arg(QString::number(percent, 'f', 1));
}

ActivityBar::ActivityBar(QWidget *parent)
    : QProgressBar(parent)
{
    m_timer = new QTimer(this);
    m_timer->setInterval(kFrameMs);
    connect(m_timer, &QTimer::timeout, this, [this] {
        m_phase += double(kFrameMs) / double(kSweepMs);
        if (m_phase >= 1.0)
            m_phase -= 1.0;
        update();
    });
    // valueChanged is what turns the animation on and off: a bar at 0 or at
    // 100 percent is not working on anything.
    connect(this, &QProgressBar::valueChanged, this, &ActivityBar::updateAnimation);
    updateAnimation();
}

void ActivityBar::updateAnimation()
{
    // An indeterminate bar (minimum == maximum) is busy by definition; Qt
    // already animates its chunk, and the sweep rides along with it.
    const bool busy = minimum() == maximum();
    const bool working = busy || (value() > minimum() && value() < maximum());
    const bool wanted = working && isVisible() && isEnabled();
    if (wanted == m_timer->isActive())
        return;
    if (wanted) {
        m_timer->start();
    } else {
        m_timer->stop();
        m_phase = 0.0;
        update();
    }
}

void ActivityBar::showEvent(QShowEvent *event)
{
    QProgressBar::showEvent(event);
    updateAnimation();
}

void ActivityBar::hideEvent(QHideEvent *event)
{
    QProgressBar::hideEvent(event);
    updateAnimation();
}

void ActivityBar::changeEvent(QEvent *event)
{
    QProgressBar::changeEvent(event);
    if (event->type() == QEvent::EnabledChange)
        updateAnimation();
}

void ActivityBar::paintEvent(QPaintEvent *event)
{
    // The bar itself first, stylesheet and all: the sweep is an addition to it,
    // never a replacement, so nothing about the theme has to be duplicated here.
    QProgressBar::paintEvent(event);
    if (!m_timer->isActive())
        return;

    // How much of the groove is filled. The border the stylesheet draws is one
    // pixel, and the highlight has to stay inside it.
    const QRectF inner = QRectF(rect()).adjusted(1, 1, -1, -1);
    if (inner.width() <= 2 || inner.height() <= 0)
        return;
    const bool busy = minimum() == maximum();
    const double span = double(maximum()) - double(minimum());
    const double fraction =
        busy ? 1.0
             : qBound(0.0, (double(value()) - double(minimum())) / (span > 0 ? span : 1.0), 1.0);
    QRectF filled = inner;
    filled.setWidth(inner.width() * fraction);
    if (filled.width() <= 1)
        return;

    // Clipped to the fill, with the same corner the stylesheet gives the chunk,
    // so the highlight cannot spill onto the empty part or over the rounding.
    QPainterPath clip;
    clip.addRoundedRect(filled, qMin(8.0, filled.width() / 2.0),
                        qMin(8.0, filled.height() / 2.0));

    // A soft band of light, wider than it is tall, travelling from the left
    // edge to the right and starting again. It is white at a low alpha rather
    // than a colour of its own: on a dark theme the fill is a light colour and
    // on a paper theme a dark one, and white lifts both without tinting either.
    const double bandWidth = qMax(60.0, inner.height() * 4.0);
    const double travel = filled.width() + bandWidth;
    const double x = filled.left() - bandWidth + m_phase * travel;

    QLinearGradient gradient(x, 0, x + bandWidth, 0);
    gradient.setColorAt(0.0, QColor(255, 255, 255, 0));
    gradient.setColorAt(0.5, QColor(255, 255, 255, 46));
    gradient.setColorAt(1.0, QColor(255, 255, 255, 0));

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setClipPath(clip);
    painter.fillRect(filled, gradient);
}

// --------------------------------------------------------------- help popup
//
// Wide enough for a paragraph without becoming a page: at 460 the longest of
// these texts is five or six lines, which is read in one go.
namespace {
constexpr int kHelpPopupWidth = 460;

// The ticker's bar. Small enough to read as a miniature of the panel's bar
// rather than as a second bar competing with it, wide enough for the number
// inside it to have room at any of the interface fonts.
constexpr int kTickerBarWidth = 210;
constexpr int kTickerBarHeight = 18;
}

// What a "?" badge opens on a click.
//
// A child of the main window rather than a window of its own. That is the
// unglamorous half of this class and the important one: a frameless top-level
// widget is outside the layout, outside the palette inheritance a stylesheet
// travels down, and on Windows it is composited by the system rather than
// painted into the application's own surface — which is how the first version
// of this ended up as a black rectangle with unreadable text. As a child it is
// an ordinary panel, drawn exactly like the cards it explains, in whatever
// theme is on.
//
// It closes on the next click anywhere, or on Escape, which is what a reader
// expects of something opened with one click.
class HelpPopup : public QFrame
{
public:
    static void showFor(QWidget *anchor, const QString &html)
    {
        QWidget *host = anchor ? anchor->window() : nullptr;
        if (!host)
            return;
        // One at a time. Clicking a second badge replaces the first answer
        // rather than stacking a second panel on top of it. A QPointer rather
        // than a search of the widget tree: this class has no Q_OBJECT of its
        // own, and one pointer is a smaller thing to keep than a metaobject.
        if (s_open)
            s_open->close();

        auto *popup = new HelpPopup(host, html);
        s_open = popup;
        popup->place(anchor);
        popup->show();
        popup->raise();
    }

private:
    // The one that is open, if any.
    static QPointer<QWidget> s_open;

    HelpPopup(QWidget *host, const QString &html)
        : QFrame(host)
    {
        setObjectName(QStringLiteral("HelpPopup"));
        setAttribute(Qt::WA_DeleteOnClose, true);

        auto *layout = new QVBoxLayout(this);
        layout->setContentsMargins(16, 14, 16, 14);

        auto *label = new QLabel(html, this);
        label->setObjectName(QStringLiteral("HelpPopupText"));
        label->setWordWrap(true);
        // The same reading Qt gives a tooltip, so a text written with <b> and
        // one written with blank lines both come out as they were meant.
        label->setTextFormat(Qt::mightBeRichText(html) ? Qt::RichText : Qt::PlainText);
        label->setTextInteractionFlags(Qt::NoTextInteraction);
        label->setFixedWidth(kHelpPopupWidth - 32);
        layout->addWidget(label);

        setFixedWidth(kHelpPopupWidth);
        adjustSize();

        // Installed on the application, not on the host: the next press may
        // land on any widget at all, and every one of them dismisses this.
        qApp->installEventFilter(this);
    }

    // Under the badge, kept inside the window on all four sides. Badges sit at
    // the right-hand edge of label columns and near the bottom of long pages,
    // and both would otherwise push the panel out of sight.
    void place(QWidget *anchor)
    {
        QWidget *host = parentWidget();
        if (!host || !anchor)
            return;
        const QPoint dot = anchor->mapTo(host, QPoint(0, 0));
        int x = dot.x() - 12;
        int y = dot.y() + anchor->height() + 8;
        if (y + height() > host->height())
            y = dot.y() - height() - 8;          // above it instead
        x = qBound(8, x, qMax(8, host->width() - width() - 8));
        y = qBound(8, y, qMax(8, host->height() - height() - 8));
        move(x, y);
    }

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        if (event->type() == QEvent::MouseButtonPress) {
            // Anything outside closes it — including the badge itself, so a
            // second click on the same "?" puts the answer away again.
            auto *w = qobject_cast<QWidget *>(watched);
            if (!w || !isAncestorOf(w))
                close();
        } else if (event->type() == QEvent::KeyPress) {
            const auto *key = static_cast<QKeyEvent *>(event);
            if (key->key() == Qt::Key_Escape)
                close();
        }
        return QFrame::eventFilter(watched, event);
    }
};

QPointer<QWidget> HelpPopup::s_open;

// The badge itself. A label rather than a button because it is one character in
// a circle drawn by the stylesheet, and because every caller already places it
// as a label beside a field.
class HelpDot : public QLabel
{
public:
    HelpDot(const QString &help, QWidget *parent)
        : QLabel(QStringLiteral("?"), parent), m_help(help)
    {
    }

protected:
    void mousePressEvent(QMouseEvent *event) override
    {
        if (event->button() == Qt::LeftButton && !m_help.isEmpty())
            HelpPopup::showFor(this, m_help);
        else
            QLabel::mousePressEvent(event);
    }

private:
    QString m_help;
};

// -------------------------------------------------------------- run ticker

RunTicker::RunTicker(QWidget *parent)
    : QWidget(parent)
{
    setObjectName(QStringLiteral("RunTicker"));
    setCursor(Qt::PointingHandCursor);
    setToolTip(tr("Click for details of the run in progress"));

    auto *row = new QHBoxLayout(this);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(8);

    // The same object name as the chip in the run panel, so every state colour
    // it already has applies here without a second table of colours. Only the
    // measurements are its own, hence the property.
    m_chip = new QLabel(tr("RUNNING"), this);
    m_chip->setObjectName(QStringLiteral("StateChip"));
    m_chip->setProperty("mini", true);
    m_chip->setProperty("state", QStringLiteral("running"));
    m_chip->setAlignment(Qt::AlignCenter);
    row->addWidget(m_chip);

    // An ActivityBar, not a plain QProgressBar: a miniature of the panel's bar
    // means the travelling highlight too — it is what says "moving" on a run
    // that sits on the same percentage for minutes.
    m_bar = new ActivityBar(this);
    m_bar->setObjectName(QStringLiteral("MiniProgress"));
    m_bar->setFixedSize(kTickerBarWidth, kTickerBarHeight);
    m_bar->setRange(0, 0);
    row->addWidget(m_bar);

    m_kind = new QLabel(this);
    m_kind->setObjectName(QStringLiteral("TickerKind"));
    row->addWidget(m_kind);

    hide();
    if (parent)
        parent->installEventFilter(this);
}

void RunTicker::setState(const State &s)
{
    const bool wasActive = m_state.active;
    m_state = s;

    if (!s.active) {
        if (m_drawerOpen)
            setDrawerOpen(false);
        hide();
        return;
    }

    m_chip->setText(s.paused ? tr("PAUSED") : tr("RUNNING"));
    m_chip->setProperty("state", s.paused ? QStringLiteral("paused")
                                          : QStringLiteral("running"));
    // A property only reaches the stylesheet after the style is asked again.
    m_chip->style()->unpolish(m_chip);
    m_chip->style()->polish(m_chip);

    // Paused is drawn the way an unavailable control is drawn — the same
    // washed-out treatment the interface already uses for "there, but not
    // doing anything" — so the difference is visible without reading the chip.
    m_bar->setProperty("paused", s.paused);
    m_bar->style()->unpolish(m_bar);
    m_bar->style()->polish(m_bar);

    if (s.percent < 0.0) {
        m_bar->setRange(0, 0);          // sweeping: nothing reported yet
    } else {
        m_bar->setRange(0, 1000);       // tenths, so the text carries a decimal
        m_bar->setValue(int(s.percent * 10.0));
    }
    m_kind->setText(s.kind);

    if (m_drawerChunks) {
        m_drawerChunks->setText(s.chunks);
        m_drawerChunks->setVisible(!s.chunks.isEmpty());
        m_drawerHardware->setText(s.hardware);
        m_drawerHardware->setVisible(!s.hardware.isEmpty());
        m_drawerRemaining->setText(s.remaining);
        m_drawerRemaining->setVisible(!s.remaining.isEmpty());
    }

    adjustSize();
    show();
    if (!wasActive)
        raise();
    reposition();
}

// Placed by hand, not by the status bar's layout. The two groups of controls
// in that bar are not the same width — two indicators on the left, two link
// buttons and their badges on the right — so a stretch on each side would put
// this visibly off centre. It floats above the bar instead and is centred on
// every resize, which is also what keeps it centred in the 36 px height.
void RunTicker::reposition()
{
    QWidget *bar = parentWidget();
    if (!bar || !m_state.active)
        return;

    move((bar->width() - width()) / 2, (bar->height() - height()) / 2);

    // On a narrow window the ticker would land on top of the indicators. The
    // free space between them is measured rather than guessed, so this holds
    // whatever the theme's font does to the width of those labels.
    int leftEdge = 0;
    int rightEdge = bar->width();
    const int middle = bar->width() / 2;
    const auto siblings = bar->findChildren<QWidget *>(QString(), Qt::FindDirectChildrenOnly);
    for (QWidget *sib : siblings) {
        if (sib == this || sib->isHidden())
            continue;
        const QRect g = sib->geometry();
        if (g.right() < middle)
            leftEdge = qMax(leftEdge, g.right());
        else if (g.left() > middle)
            rightEdge = qMin(rightEdge, g.left());
    }
    setVisible(rightEdge - leftEdge >= width() + 32);
}

bool RunTicker::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == parentWidget()
        && (event->type() == QEvent::Resize || event->type() == QEvent::Show)) {
        reposition();
    }
    return QWidget::eventFilter(watched, event);
}

void RunTicker::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    reposition();
}

void RunTicker::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        setDrawerOpen(!m_drawerOpen);
        return;
    }
    QWidget::mousePressEvent(event);
}

// Three lines and no more. What a person wants from a glance at a running job
// is how far through it is, what it is using and when it will be done; anything
// else belongs to the panel that started it, which is one click away.
void RunTicker::buildDrawer()
{
    if (m_drawer)
        return;
    QWidget *host = window();
    if (!host)
        return;

    m_drawer = new QFrame(host);
    m_drawer->setObjectName(QStringLiteral("TickerDrawer"));
    m_drawer->hide();

    auto *layout = new QVBoxLayout(m_drawer);
    layout->setContentsMargins(16, 12, 16, 12);
    layout->setSpacing(4);
    for (QLabel **slot : {&m_drawerChunks, &m_drawerHardware, &m_drawerRemaining}) {
        *slot = new QLabel(m_drawer);
        (*slot)->setObjectName(QStringLiteral("TickerDrawerLine"));
        layout->addWidget(*slot);
    }

    m_drawerAnim = new QPropertyAnimation(m_drawer, "geometry", this);
    m_drawerAnim->setDuration(180);
    m_drawerAnim->setEasingCurve(QEasingCurve::InOutCubic);
    connect(m_drawerAnim, &QPropertyAnimation::finished, this, [this] {
        if (!m_drawerOpen)
            m_drawer->hide();
    });
}

void RunTicker::setDrawerOpen(bool open)
{
    buildDrawer();
    QWidget *host = window();
    if (!m_drawer || !host)
        return;

    if (open) {
        m_drawerChunks->setText(m_state.chunks);
        m_drawerChunks->setVisible(!m_state.chunks.isEmpty());
        m_drawerHardware->setText(m_state.hardware);
        m_drawerHardware->setVisible(!m_state.hardware.isEmpty());
        m_drawerRemaining->setText(m_state.remaining);
        m_drawerRemaining->setVisible(!m_state.remaining.isEmpty());
    }

    // Rolled up rather than faded in: the drawer grows out of the top edge of
    // the ticker, so where it came from is never in doubt.
    const QPoint anchor = mapTo(host, QPoint(0, 0));
    const QSize wanted = m_drawer->sizeHint();
    const int w = qMax(wanted.width(), width());
    const int h = wanted.height();
    const int x = qBound(8, anchor.x() + width() / 2 - w / 2,
                         qMax(8, host->width() - w - 8));
    const int shut = anchor.y() - 6;

    const QRect closed(x, shut, w, 0);
    const QRect opened(x, shut - h, w, h);

    m_drawerAnim->stop();
    m_drawerOpen = open;
    if (open) {
        m_drawer->setGeometry(closed);
        m_drawer->show();
        m_drawer->raise();
        m_drawerAnim->setStartValue(closed);
        m_drawerAnim->setEndValue(opened);
    } else {
        m_drawerAnim->setStartValue(m_drawer->geometry());
        m_drawerAnim->setEndValue(closed);
    }
    m_drawerAnim->start();
}

namespace TrajectaUi {

QLabel *makeHelpDot(const QString &help, QWidget *parent)
{
    auto *dot = new HelpDot(help, parent);
    dot->setObjectName(QStringLiteral("HelpDot"));
    dot->setAlignment(Qt::AlignCenter);
    dot->setFixedSize(16, 16);
    // No tooltip: the text belongs to the popup now, and a badge that answered
    // twice — once on hover, once on click — would be two different sizes of
    // the same paragraph.
    dot->setCursor(Qt::PointingHandCursor);
    return dot;
}

QWidget *makeFieldLabel(const QString &text, const QString &help, QWidget *parent)
{
    auto *box = new QWidget(parent);
    auto *layout = new QHBoxLayout(box);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    auto *label = new QLabel(text, box);
    label->setObjectName(QStringLiteral("FieldLabel"));
    layout->addWidget(label);
    // The gap goes between the text and the badge, not after it. In a grid the
    // label column is as wide as its longest entry, so this pins every badge to
    // the same x — one column of "?" against the fields they explain, instead
    // of a ragged edge that follows the length of each word. In a horizontal
    // row there is no spare width to take, and the badge stays next to its
    // label as before.
    layout->addStretch(1);
    layout->addWidget(makeHelpDot(help, box));
    return box;
}

QWidget *withHelpDot(QWidget *w, const QString &help)
{
    // The wrapper takes over the parent, so the caller can hand the result
    // straight to a layout without the widget losing its styling context.
    auto *box = new QWidget(w->parentWidget());
    auto *layout = new QHBoxLayout(box);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    w->setParent(box);
    layout->addWidget(w);
    layout->addWidget(makeHelpDot(help, box));
    layout->addStretch(1);
    return box;
}

QWidget *makeGroupNote(QGroupBox *group, const QString &note, const QString &help)
{
    auto *row = new QWidget(group);
    auto *layout = new QHBoxLayout(row);
    // A starting indent only: TitleFollower replaces it with the real one as
    // soon as the box has been laid out and knows where its title ends.
    layout->setContentsMargins(24, 0, 0, 0);
    layout->setSpacing(6);
    auto *label = new QLabel(note, row);
    label->setObjectName(QStringLiteral("HintLabel"));
    layout->addWidget(label);
    layout->addWidget(makeHelpDot(help, row));
    layout->addStretch(1);

    // QGroupBox disables its children as it is unticked and re-enables them as
    // it is ticked; both happen just before `toggled` reaches us, so putting
    // the row back here is enough to keep it legible in either state. It still
    // greys out with the rest of the page while an analysis is running, which
    // is what should happen.
    QObject::connect(group, &QGroupBox::toggled, row, [row] { row->setEnabled(true); });
    row->setEnabled(true);
    new TitleFollower(group, row);
    return row;
}

void setPauseMark(QPushButton *button, bool on)
{
    if (!button)
        return;
    // One keeper per button, created on first use and living as its child: the
    // mark has to be repainted on every theme and font change, and a caller
    // that had to remember to do that would forget one of the two buttons.
    //
    // Found by walking the children rather than with findChild(), which insists
    // on Q_OBJECT — and a class local to this file has no moc output to put it
    // in. dynamic_cast is enough: QObject is polymorphic.
    PauseMark *mark = nullptr;
    for (QObject *child : button->children()) {
        if (auto *found = dynamic_cast<PauseMark *>(child)) {
            mark = found;
            break;
        }
    }
    if (!mark)
        mark = new PauseMark(button);
    mark->setActive(on);
}

QString largePagesHelpText()
{
    return QObject::tr(
        "<b>Large memory pages</b><br><br>"
        "Every memory access has to be translated from a virtual to a physical "
        "address. The CPU caches those translations in the TLB, which holds "
        "roughly 2,000 entries. With the standard 4 KB page that covers only "
        "about 8 MB, while a large analysis works over more than a gigabyte — "
        "so nearly every access pays a <i>page walk</i> on top of its cache "
        "miss.<br><br>"
        "A 2 MB page is the same memory described by one entry instead of 512. "
        "It raises the TLB's reach to about 4 GB, which on a big DEM typically "
        "makes the propagation phase <b>15–30% faster</b>.<br><br>"
        "<b>It cannot change your results.</b> A page is a unit of address "
        "translation and nothing else: the same bytes stay at the same "
        "addresses, no arithmetic is affected, and the output rasters are "
        "identical bit for bit whether this is on or off. It is optional only "
        "because it can fail to switch on, never because it is risky.<br><br>"
#ifdef Q_OS_WIN
        "<b>How to enable it</b><br>"
        "1. Press <i>Set up…</i> and approve the Windows prompt. This grants "
        "the “Lock pages in memory” privilege to your account, once.<br>"
        "2. <b>Sign out of Windows and back in.</b> This step cannot be "
        "skipped: the privilege list is fixed when you log on, so a new "
        "session is needed before it exists. Restarting Trajecta alone is not "
        "enough.<br>"
        "3. Look at the line beside the checkbox. It says which of the steps "
        "below you are still standing at.<br><br>"
        "<b>If your account is an administrator, there is a third step</b><br>"
        "Windows gives an administrator <i>two</i> identities at logon: the "
        "full one, and a restricted one with the powerful privileges — this "
        "one included — taken out. Every program you start normally, Trajecta "
        "and the analysis engine with it, receives the restricted identity. "
        "The privilege is then genuinely absent, and signing out again will "
        "not change that, however many times you try.<br>"
        "On such an account the option only becomes available if Trajecta is "
        "started with <b>Run as administrator</b> (right-click its icon). The "
        "status line says <i>“granted, but withheld from ordinary programs”</i> "
        "when this is what is happening.<br>"
        "On a standard, non-administrator account the question does not arise: "
        "there is only one identity, the privilege is in it, and a plain "
        "sign-out is enough.<br><br>"
        "<b>If it still does not engage</b><br>"
        "A 2 MB page needs 2 MB of physically contiguous memory. On a computer "
        "that has been running for a long time, memory is fragmented and the "
        "request can be refused. Restarting the PC before a large run is the "
        "usual remedy. Trajecta always falls back to normal pages on its own, "
        "and reports what actually happened at the end of the run.");
#else
        // The privilege, the sign-out and the administrator token are all
        // Windows. Rather than keep a second copy of this whole text in the
        // macOS tree — which is how the two drifted apart before — the one
        // paragraph that differs is chosen here, and the file stays identical
        // on both sides.
        "<b>Availability</b><br>"
        "Large memory pages are a Windows facility: they need a privilege that "
        "Windows grants to an account, and there is no equivalent to ask for "
        "here. The option is shown for completeness and stays unavailable; "
        "nothing about a run changes because of it, and the analysis is not "
        "slower for its absence in any way that could be measured against a "
        "Windows machine of the same class.");
#endif
}

QString manifestHelpText()
{
    return QObject::tr(
        "<b>Run manifest</b><br><br>"
        "A plain text file, written in the output folder next to the results, "
        "recording <b>everything this run was made of</b>: the version of "
        "Trajecta, every input file with its size and a content hash, every "
        "setting (neighbours, cost function, buffers, barriers), the hardware "
        "actually used, the statistics of the result and every file "
        "produced.<br><br>"
        "<b>Why it is on by default.</b> A raster on disk cannot answer, six "
        "months later, which DEM it came from or whether the barriers were "
        "enabled. The manifest travels with the output and answers exactly "
        "that — which is what makes a map reproducible, and a figure in a "
        "publication defensible.<br><br>"
        "The content hash matters more than it sounds: two DEMs of the same "
        "area and the same extent look identical in a file listing, and only "
        "the hash tells them apart.<br><br>"
        "<b>What it costs.</b> A few seconds at the end of the run, spent "
        "reading the input files to hash them. Nothing during the "
        "computation, and nothing at all in memory.<br><br>"
        "The file is named after the main output — <i>dens_manifest.txt</i> "
        "for <i>dens.tif</i> — so re-running an analysis replaces its own "
        "record and never another's.");
}

QString timeLeftText(qint64 workedMs, double percent)
{
    // Under a twentieth of a percent, or under eight seconds of work, the
    // arithmetic produces numbers in the days for a job that will take a
    // minute. Better to say nothing than to say that.
    if (percent <= 0.5 || workedMs < 8000)
        return QObject::tr("Time left: still measuring");

    const double total = double(workedMs) / (percent / 100.0);
    qint64 left = qint64(total) - workedMs;
    if (left < 0)
        left = 0;

    // Written out rather than left to Qt's %n plural form: without a
    // translation file that form ships the source string as it stands, and
    // "about 5 minute(s)" is not something to put in front of a user.
    const qint64 mins = left / 60000;
    if (mins < 1)
        return QObject::tr("Time left: under a minute");
    if (mins == 1)
        return QObject::tr("Time left: about a minute");
    if (mins < 60)
        return QObject::tr("Time left: about %1 minutes").arg(mins);
    const qint64 hours = mins / 60;
    if (hours < 24)
        return QObject::tr("Time left: about %1 h %2 min").arg(hours).arg(mins % 60);
    const qint64 days = hours / 24;
    if (days == 1)
        return QObject::tr("Time left: about a day");
    return QObject::tr("Time left: about %1 days").arg(days);
}

QString ramHeadroomNote()
{
    return QObject::tr(
        "<br><br>Beyond roughly "
        "4000 × 4000 cells the working set grows with the raster — the DEM, the "
        "slope and cost surfaces and the accumulation grid are all held at once "
        "— and the ceiling has to be raised with it. An analysis that reaches "
        "the ceiling stops with an out-of-memory error rather than slowing "
        "down, so on a large DEM it is worth granting the headroom before "
        "starting a run that would otherwise fail hours in.");
}

QList<int> admissibleNeighbourCounts()
{
    QList<int> out;
    for (int n : neighbourhood::admissibleSizes())
        out.append(n);
    return out;
}

int snapNeighbourCount(int wanted)
{
    return neighbourhood::snap(wanted);
}

QString neighboursHelpText()
{
    QStringList sizes;
    for (int n : admissibleNeighbourCounts())
        sizes << QString::number(n);

    return QObject::tr(
        "<b>Neighbours</b><br><br>"
        "How many directions a path may leave a cell in. With 8 it can only "
        "move to the cells touching it, so a route running at 22° has to be "
        "built out of alternating 0° and 45° steps and comes out longer than "
        "it really is. Adding directions removes that zig-zag.<br><br>"
        "<b>Which totals are allowed.</b> A neighbourhood has to look the same "
        "in every compass direction, otherwise some headings are cheaper than "
        "others and every result leans that way. Directions therefore come in "
        "whole symmetric groups — eight at a time in general, four when they "
        "fall on the axes or the diagonals — so only certain totals exist:"
        "<br><br><i>%1</i><br><br>"
        "They are mostly multiples of 8, but not all of them are. Type any "
        "number you like: the engine uses the largest admissible total that "
        "does not exceed it, and reports what it used.<br><br>"
        "<b>What it costs.</b> Time per cell grows in proportion to the "
        "number: 32 neighbours means roughly twice the work of 16, and 64 "
        "roughly four times. Memory does not change — the offsets are a "
        "handful of bytes — so this is a pure time trade.<br><br>"
        "<b>What it buys.</b> Much less than the cost, past a point. Going "
        "from 8 to 16 removes most of the length error and is nearly always "
        "worth it. 16 to 32 refines the diagonals and can matter on gentle "
        "terrain where routes run at shallow angles. Beyond 48 the paths "
        "barely move, and the extra long moves have a drawback of their own: "
        "a jump of three or four cells prices the slope between its endpoints "
        "and ignores whatever lies in between, so a route can hop over a gully "
        "the terrain would really force it around.<br><br>"
        "16 is the default because it is where the curve of benefit against "
        "cost turns over. Larger neighbour grids can be used for testing or "
        "experimentation.")
        .arg(sizes.join(QStringLiteral(", ")));
}

QString costFunctionHelpText()
{
    return QObject::tr(
        "<b>Cost function</b><br><br>"
        "The model that turns the slope of a single move into a walking speed, and "
        "therefore into the time that move costs.<br><br>"
        "<b>What is computed for every move.</b> Between two cell centres the engine "
        "takes <code>dh</code>, the horizontal distance in metres, and <code>dz</code>, "
        "the signed elevation difference in metres. The slope is "
        "<code>S = dz / dh</code> — a tangent, not an angle, and it keeps its sign, "
        "which is what makes all three models anisotropic. The cost is then "
        "<code>(dh / 1000) / v</code>, in <b>hours</b>.<br><br>"
        "<b>The six formulas, exactly as implemented:</b><br><br>"
        "<b>1 — Tobler</b> (1993, presented in White 2015)<br>"
        "<code>v = 6 · e^(−3.5 · |S + 0.05|)</code> km/h<br>"
        "The <i>on-path</i> form: the × 0.6 factor Tobler gives for off-path travel is "
        "not applied, so cross-country routes come out optimistic.<br><br>"
        "<b>2 — Márquez-Pérez et al.</b> (2017)<br>"
        "<code>v = 4.8 · e^(−5.3 · |0.7·S + 0.03|)</code> km/h<br>"
        "Tobler recalibrated on GPS tracks from marked trails: slower, and it "
        "penalises slope more sharply.<br><br>"
        "<b>3 — Irmischer &amp; Clarke</b> (2017), <b>on-path male</b> variant<br>"
        "<code>v = 0.11 + e^(−(S%% + 5)² / 1800)</code> m/s, with "
        "<code>S%% = 100·S</code> and <code>1800 = 2·30²</code><br>"
        "The paper publishes four variants (male/female × on-path/off-path); this is "
        "the on-path male one. The 0.11 m/s term is a floor: the function never "
        "reaches zero speed however steep the ground.<br><br>"
        "<b>4 — Herzog</b> (2013), fitted to Minetti et al. (2002) — <b>energy, not "
        "time</b><br>"
        "<code>C(S) = 1337.8·S⁶ + 278.19·S⁵ − 517.39·S⁴ − 78.199·S³ + 93.419·S² + "
        "19.825·S + 1.64</code>, <code>cost = C(S) · dh</code> kJ/kg<br>"
        "The one function here that measures effort rather than duration; its "
        "minimum sits at about a 10.5% downhill, and rises on both sides because "
        "braking downhill costs energy too. Its costs are in <b>kilojoules per "
        "kilogram</b>, not hours, and cannot be compared with the other "
        "five.<br><br>"
        "<b>5, 6 — Campbell et al.</b> (2019), asymmetric Lorentz, 5th and 50th "
        "percentile<br>"
        "<code>v = c / (π·b·(1 + ((θ − a)/b)²)) + d + e·θ</code> m/s, with "
        "<code>θ</code> the slope in degrees<br>"
        "Fitted to 421,247 GPS activities from Strava. The <b>5th</b> percentile is "
        "ordinary hiking pace; the <b>50th</b> is the dataset's median and is a run, "
        "not a walk.<br><br>"
        "<b>Units.</b> Speed is km/h for 1–2 and m/s for 3, 5 and 6 (converted "
        "internally). Every function but Herzog returns a cost in <b>hours</b>, so "
        "those five cost surfaces are numerically comparable; Herzog's are not, and "
        "must never be added to or subtracted from them. The exported slope raster "
        "is in degrees for Tobler and Campbell and in percent for the others — that "
        "affects the raster only, never the computation.<br><br>"
        "Five of these six model <b>time</b>, not effort — none accounts for load or "
        "the cost of braking on a steep descent. Herzog models effort instead, and "
        "is the exception to both of those.");
}

QString preservePeaksHelpText()
{
    return QObject::tr(
        "<b>Preserve local peaks</b><br><br>"
        "Only does anything when the sample spacing is greater than 1, and off "
        "by default.<br><br>"
        "<b>The problem it solves.</b> With a spacing of N the interpolation "
        "keeps whichever cell happens to land on an N x N grid. On a smooth, "
        "dense input that is a fair way to generalise. On a thin, spiky one — a "
        "FETE density is the obvious case — it is not: the corridors are one or "
        "two cells wide and the junctions are single cells, so the grid keeps "
        "an arbitrary sample of them and drops the rest. Continuous corridors "
        "come out as strings of dots, and the highest values disappear "
        "altogether. With a spacing of 4 on a real density raster the maximum "
        "of the interpolated surface can fall to well under half the maximum of "
        "the input.<br><br>"
        "<b>What this does.</b> For every N x N block it also keeps the cell "
        "that actually holds the block's highest value. At most one extra "
        "sample per block, so the cost barely moves, and the peaks survive the "
        "subsampling instead of being decided by where the grid happened to "
        "fall.<br><br>"
        "<b>When to leave it off.</b> When the input really is a smooth field "
        "and you want a plain generalisation — adding the maxima then biases "
        "the surface upwards, because a block's maximum is not its typical "
        "value.");
}

QString costCorridorHelpText()
{
    return QObject::tr(
        "<b>Cost corridor</b><br><br>"
        "A least-cost path is one pixel wide, and that width is a lie about how "
        "certain it is. The corridor is the honest version: for every cell it "
        "asks what a detour through there would cost, as a percentage above the "
        "best route.<br><br>"
        "<code>excess(c) = (cost(origin→c) + cost(c→destination) − best) / best</code>"
        "<br><br>"
        "Zero along the optimal path itself. A <b>narrow</b> corridor means the "
        "terrain dictated the route and the line on the map is a real finding. "
        "A <b>wide</b> one means dozens of routes cost almost the same, and the "
        "single line is an artefact of the algorithm having to pick one — which "
        "is exactly the thing least-cost maps are most often over-read "
        "for.<br><br>"
        "<b>The width</b> sets how much extra cost still counts as being in the "
        "corridor. 5% keeps only what is nearly as cheap as the best route; 25% "
        "shows the broad band of plausible alternatives. Cells beyond it are "
        "written as nodata, so a GIS draws nothing there instead of stretching "
        "the palette across the whole map. The values inside are the excess "
        "percentages themselves, so you can re-threshold later without "
        "recomputing.<br><br>"
        "<b>What it costs.</b> One extra search per destination, roughly "
        "doubling the run for a single destination and more for many. It also "
        "has to be a genuinely separate search: walking uphill does not cost "
        "what walking downhill costs, so the journey <i>to</i> a destination is "
        "not the reverse of the journey <i>from</i> it, and the second half "
        "cannot be reused from the first.<br><br>"
        "Off by default.");
}

QString routeCompareHelpText()
{
    return QObject::tr(
        "<b>Comparing with a known route</b><br><br>"
        "A least-cost path always produces an answer, and the answer always "
        "looks convincing. The only way to find out whether the model is any "
        "good is to run it where the real route is already known and see how "
        "close it lands.<br><br>"
        "<b>What is measured.</b> Both lines are sampled at regular intervals, "
        "and for every sample the distance to the other line is computed. The "
        "result is reported as a distribution rather than one number, because a "
        "route can follow the real one for nine kilometres and then take the "
        "wrong side of a hill for one — and an average hides exactly that.<br><br>"
        "<b>Both directions matter.</b> <i>Computed → known</i> asks how much of "
        "what the model drew is real. <i>Known → computed</i> asks how much of "
        "the real route the model found. A short path lying on top of a long "
        "one scores well on the first and badly on the second, and only the "
        "pair of numbers tells you that.<br><br>"
        "The <b>Hausdorff distance</b> is the worst disagreement in either "
        "direction: a single pessimistic number, useful for comparing two "
        "models against the same known route.<br><br>"
        "<b>Tolerance</b> sets what counts as close, and the honest value is "
        "not arbitrary: use the positional accuracy of the known route itself. "
        "A road digitised from a 1:25,000 map is worth no better than about "
        "25 m; a GPS survey, a few metres. Setting it far below the accuracy of "
        "your reference data manufactures a disagreement that is really just "
        "map error.<br><br>"
        "<b>Requirements.</b> Both layers in the same projected CRS. Degrees are "
        "refused rather than reported, because the distances would not be in "
        "metres and would mean nothing.<br><br>"
        "The report is also written as a text file next to the known route.");
}

QString slopeCutoffHelpText()
{
    return QObject::tr(
        "<b>Slope cut-off</b><br><br>"
        "Above the angle you set, a move stops being expensive and becomes "
        "<b>impossible</b>: the engine removes it from the graph instead of "
        "pricing it. Off by default, because a limit nobody chose would change "
        "every result silently.<br><br>"
        "<b>It applies to a move, not to a cell.</b> A terrace can be entered "
        "from the side and not from below, and that is exactly what happens "
        "here: the cell stays available, the approach that was too steep does "
        "not. This is why the setting sits next to the cost function and not "
        "next to the barriers, which do remove whole cells.<br><br>"
        "<b>Uphill and downhill are separate</b> because they are not "
        "symmetric in practice: a slope that can be climbed slowly is often "
        "refused on the way down, where the risk is falling rather than "
        "tiring.<br><br>"
        "<b>What it is for.</b> Two things. Keeping a route out of ground no "
        "one would walk — around 30° for a loaded traveller, less on rock or "
        "scree. And keeping a cost function inside the range it was measured "
        "in: Herzog's polynomial is fitted to treadmill data up to about ±45%, "
        "roughly 24°, and Campbell's fit is calibrated below 30°. Past those "
        "angles the numbers are extrapolation, and the cut-off is the honest "
        "way to say so.<br><br>"
        "<b>What to watch for.</b> Set it too tight and a destination can "
        "become unreachable — the run then reports paths it could not compute "
        "rather than inventing one. If a result loses whole regions, the limit "
        "is the first thing to check.");
}

QString costModifiersHelpText()
{
    return QObject::tr(
        "<b>Cost modifiers</b><br><br>"
        "By default the cost of crossing a cell comes from the terrain alone — "
        "the slope between it and its neighbour, through the chosen cost "
        "function. Cost modifiers let you override that with knowledge the DEM "
        "does not hold: a river that has to be forded, a marsh, a wall, a road "
        "that makes walking easier, an area that may not be entered at all. "
        "Each feature carries a multiplier: above 1 the ground becomes dearer "
        "to cross, below 1 cheaper, and a very large value is effectively an "
        "obstacle.<br><br>"
        "<b>Why it takes longer</b><br>"
        "Three things are added to the run. The vector layer is first "
        "<i>rasterized</i>: every feature is drawn onto the DEM grid, widened "
        "by the polyline buffer, and combined with the modifier raster into a "
        "single multiplier surface — a preparation pass over the whole map "
        "before any path is computed. During the search itself, every step then "
        "reads that surface and multiplies the cost of entering the cell, work "
        "the engine skips entirely when no modifiers are in use.<br><br>"
        "The third cost is the largest, and it is not the arithmetic. Expensive "
        "ground makes the search consider detours it would otherwise dismiss, "
        "so it settles far more cells before it can be certain of the cheapest "
        "route. A multiplier of a few thousand, left as an ordinary cost, can "
        "force the search to expand almost the entire raster for every source "
        "point. That is what <i>impassable barriers</i> below is for: past the "
        "threshold a cell is removed from the search instead of being merely "
        "expensive, which is both what an obstacle means and much faster. "
        "Leaving that option on is strongly recommended.<br><br>"
        "The slowdown depends on how much of the map the modifiers touch: a few "
        "rivers cost little, a multiplier surface covering everything costs a "
        "great deal.");
}

QString costModifiersNoteText()
{
    return QObject::tr("— cost modifiers make the analysis slower");
}

// --------------------------------------------------------------------------
// Site-corridor coherence
// --------------------------------------------------------------------------

QString coherenceSurfaceHelpText()
{
    return QObject::tr(
        "<b>The FETE surface</b><br><br>"
        "The density raster a FETE run produced, either as it came out or after "
        "Natural Neighbour Interpolation. Its values are counts of paths, so "
        "they depend on how many source points were used — which is why nothing "
        "here uses those values directly. Every cell is first replaced by its "
        "<b>percentile rank</b> within this surface: \"busier than 97% of the "
        "map\" means the same thing on any raster, and two periods can be "
        "compared even when one was computed from ten times as many "
        "points.<br><br>"
        "The input raster must be in a <b>projected</b> coordinate system "
        "(i.e. a CRS in metres), because the radius below is in metres. A "
        "raster in degrees is refused rather than measured wrongly.<br><br>"
        "Cells with no data stay <b>missing</b> throughout. They are not zero: "
        "zero means \"measured, and nothing passes here\", which is a fact about "
        "the landscape, and counting the two together would move every rank. "
        "What missing data costs each site is reported as that site's "
        "<i>coverage</i>.<br><br>"
        "<b>Memory:</b> the surface is held in RAM twice over while it is "
        "worked on. For rasters beyond about 4000 × 4000 cells you might need "
        "to allocate additional RAM (e.g. 16 GB instead of 8 GB).");
}

QString coherenceSitesHelpText()
{
    return QObject::tr(
        "<b>The sites</b><br><br>"
        "Any point layer GDAL can read — shapefile, GeoPackage, GeoJSON, .kml "
        "— in the <b>same projected coordinate system</b> as the surface.<br><br>"
        "Each point is scored where it falls. Points that land outside the "
        "raster are counted, listed and excluded: if all of them do, the two "
        "layers are almost certainly in different coordinate systems, and that "
        "is said rather than left to be inferred from a strange answer.<br><br>"
        "A point that falls <i>inside</i> the raster but on a cell with no data "
        "is a different case and is kept: its distance to a corridor is "
        "geometry and is still valid, while its neighbourhood is only partly "
        "measured. Both counts appear in the summary.<br><br>"
        "The layer's own attributes are copied into the outputs, so a result "
        "can be read without going back to the source. A column of the layer "
        "whose name would collide with one of the tool's own is left out.");
}

QString coherenceRadiusHelpText()
{
    return QObject::tr(
        "<b>Radius</b><br><br>"
        "How far around each site the tool looks, in metres. Beside the box it "
        "is restated in <b>cells</b>, because that is what decides whether the "
        "measurement means anything: three cells is noise, three hundred is "
        "most of the map.<br><br>"
        "It governs two things and not a third:<br>"
        "• The <b>proximity index</b>, the share of the neighbourhood inside it "
        "that is high-traffic corridor;<br>"
        "• The <b>intensity index</b>, the weighted average of that same "
        "neighbourhood;<br>"
        "• The <b>distance in metres</b> — which it does not affect at all. "
        "That is deliberate, and it is why the whole of question 2 in the "
        "report (the median, the deciles, the histogram, the distance bands) "
        "can be compared between two runs that were given different radii.<br><br>"
        "<b>Default 250 m.</b> Raise it for a regional question, lower it to "
        "ask whether sites are literally on the route. If the answer changes a "
        "lot with the radius, turn on <i>Sensitivity</i> and report the whole "
        "curve rather than one number.");
}

QString coherenceThresholdHelpText()
{
    return QObject::tr(
        "<b>What counts as a corridor</b><br><br>"
        "Distances are measured to the nearest <i>corridor cell</i>, so this "
        "setting decides what the rest of the tool is measuring towards.<br><br>"
        "<b>Top percentage of the surface</b> — the busiest q% by rank. "
        "Comparable between datasets by construction, which is what a "
        "diachronic study needs. <b>Default 1%</b>: on a FETE surface the "
        "cells that carry real traffic are almost always inside the top "
        "percent, and often inside a tenth of it. Raise it to 5% to describe "
        "the road system as a whole; lower it to 0.1% for the main arteries "
        "only.<br><br>"
        "<b>Automatic (Otsu)</b> — the split that best separates two "
        "populations, computed on the logarithm of the values. The log matters: "
        "on the raw counts the histogram is a spike at zero with a very long "
        "tail, and the best two-class split is \"the spike\" against half the "
        "map. It reports which percentile it landed on, and warns if it "
        "selected more than a quarter of the surface or almost none of it — "
        "both mean the surface has no clean corridor/background split. Good for "
        "exploring one dataset, less good for comparing several, because "
        "\"corridor\" then means something slightly different in each.<br><br>"
        "<b>Cells at or above a value</b> — a raw count, for someone who knows "
        "what theirs mean.<br><br>"
        "The threshold that was actually used is always reported as a value, as "
        "a percentile, and as a share of the surface. Those three can disagree "
        "with what you asked for: on a sparse surface where 99% of cells are "
        "exactly zero, \"the top 1%\" cannot be cut anywhere but at the first "
        "non-zero value.");
}

QString coherenceNullHelpText()
{
    return QObject::tr(
        "<b>The null model — the part that makes the numbers mean something</b>"
        "<br><br>"
        "A median score of 64 says nothing on its own. It depends on how much "
        "of the map is corridor, on the radius, and on the shape of the study "
        "area: with a generous threshold and a generous radius, points thrown "
        "at random score well too.<br><br>"
        "So the same statistic is computed again on point sets that have no "
        "relationship with the corridors but share everything else — the same "
        "area, the same number of points. What is reported is where the real "
        "sample falls in that distribution: the <b>p-value</b> (with 999 sets "
        "the smallest is 0.001) and the <b>ratio</b> of observed to expected "
        "distance. A ratio of 0.5 means the sites are half as far from a "
        "corridor as chance would put them; that ratio is the number to carry "
        "between periods, because it is free of the units, the area and the "
        "size of the sample.<br><br>"
        "<b>The same pattern, moved as a block</b> (default) keeps the sites' "
        "own clustering: it asks whether this constellation of settlements, "
        "exactly as it is arranged, would be as close to the corridors if it "
        "were laid down elsewhere on the same map. Settlements cluster, and "
        "independent random points do not, so scattered points make the null "
        "too tight and every p-value too small.<br><br>"
        "<b>Scattered points</b> is the simpler alternative, and the tool falls "
        "back to it by itself — saying so — when the sites cover so much of the "
        "raster that no translation keeps them all on it.<br><br>"
        "<b>999 sets</b> is the usual choice. The distance test always gets all "
        "of them because it costs one lookup per point; the intensity test has "
        "to walk the disc around every point, so at a large radius it is given "
        "as many as the time allows and says how many it had.");
}

QString coherenceEcdfHelpText()
{
    return QObject::tr(
        "<b>Distance bands</b><br><br>"
        "<i>How many of the sites are near a corridor at all?</i> This is the "
        "first and most general question the analysis answers, and the table "
        "these distances produce is the answer: the share of sites within "
        "each distance of the nearest corridor cell.<br><br>"
        "The distances are fixed metres and not fractions of the radius, and "
        "that is the point of them. <b>Two runs can be laid side by side row "
        "for row</b> — two periods, two regions, two surfaces — whatever "
        "radius each was given, because none of these numbers depends on the "
        "radius.<br><br>"
        "Give them in metres, separated by commas; <b>0</b> means "
        "\"standing on a corridor cell\". Bands finer than one raster cell "
        "are dropped, because a raster cannot resolve them: on a 90 m grid a "
        "site is either on a corridor or at least 90 m away, so a 50 m band "
        "would only repeat the 0 m one.<br><br>"
        "Leave the box empty to use 0, 100, 250, 500, 1000 and 2500 m as default.");
}

QString coherenceSensitivityHelpText()
{
    return QObject::tr(
        "<b>Sensitivity to the radius</b><br><br>"
        "Runs the same analysis at several radii and prints one row each.<br><br>"
        "It costs very little, because the two expensive steps — ranking the "
        "surface and measuring every cell's distance to a corridor — do not "
        "depend on the radius and are done once.<br><br>"
        "It is worth turning on whenever the result will be shown to someone "
        "else, because it answers in advance the first question they will ask. "
        "<b>A relationship that holds across the whole range is a "
        "relationship; one that appears at a single radius is usually the "
        "radius.</b><br><br>"
        "Give the radii in metres, separated by commas. Anything narrower than "
        "two cells is skipped.");
}

QString coherenceEdgeHelpText()
{
    return QObject::tr(
        "<b>Edge guard</b><br><br>"
        "FETE under-counts near the boundary of the DEM: there is less ground "
        "on one side for routes to come from, so a corridor that really "
        "continues past the edge fades out before it. A site in that band is "
        "measured against a network that is missing some of its arms.<br><br>"
        "With this on, sites within one radius of the raster's edge are "
        "<b>flagged</b> — in the <i>near_edge</i> column and in the summary's "
        "counts — and the random point sets are kept out of the same band, so "
        "the comparison is not made against ground the surface describes "
        "poorly.<br><br>"
        "They are flagged, not removed: whether to drop them is a decision "
        "about your material, not about the arithmetic. Leave it on unless the "
        "raster comfortably surrounds the study area.");
}

QString coherenceHistogramScriptHelpText()
{
    return QObject::tr(
        "<b>Histogram script</b><br><br>"
        "Writes an R script next to the table that redraws the distance "
        "histogram from question 2 of the report as a <b>ggplot2</b> figure — "
        "the same bins and the same counts shown on screen, not a fresh "
        "binning of the raw distances. Run it with "
        "<code>Rscript name_histogram.R</code> (needs the ggplot2 package: "
        "<code>install.packages(\"ggplot2\")</code>, once) and it saves a PNG "
        "beside itself and prints the plot.<br><br>"
        "On by default: it costs a few lines of text next to the table you are "
        "already writing, and turns a block of ASCII bars into a figure that "
        "goes straight into a paper or a slide, styled to match this "
        "interface rather than R's own default grey.");
}

QString coherenceOutputHelpText()
{
    return QObject::tr(
        "<b>What is written</b><br><br>"
        "<b>A table (.csv)</b> with one row per site. For each site the "
        "following fields are generated:<br><br>"
        "1) The site's distance to the nearest corridor in metres "
        "(<i>dist_m</i>)<br>"
        "2) The share of the site's neighbourhood that is corridor "
        "(<i>prox_idx</i>)<br>"
        "3) That share against the whole surface's (<i>enrich</i>, where "
        "1.00 is chance)<br>"
        "4) How busy the ground around the site is (<i>inten_idx</i>, where "
        "50 is the average location)<br>"
        "5) The rank of the cell under the site<br>"
        "6) The site's coverage<br>"
        "7) Whether the site is near the edge, and a class — ON_CORRIDOR, "
        "NEAR_THIN, DIFFUSE or OFF.<br><br>"
        "Sites that fell outside the raster are in it too, marked as such, so "
        "nothing disappears silently.<br><br>"
        "<b>A point layer</b> with the same columns, ready to symbolise in "
        "QGIS. GeoPackage keeps full column names; Shapefile is offered for "
        "workflows that expect one, and truncates names to ten "
        "characters.<br><br>"
        "<b>The distance raster</b> (optional, and nearly free — it is computed "
        "anyway): every cell holds its distance in metres to the nearest "
        "corridor cell. It is the fastest way to see the catchment of the "
        "network, to read the score of a place you have <i>not</i> surveyed, "
        "and to notice that the threshold was set too generously.<br><br>"
        "<b>A summary (.txt)</b>: the same report shown on screen, so the "
        "supplementary data of a paper and the screen cannot disagree.<br><br>"
        "<b>A histogram script (.R)</b> (optional, see <i>Histogram script</i> "
        "above): the same distance histogram as a ggplot2 figure, not an "
        "ASCII one.<br><br>"
        "The parameters are not repeated on every row — they are in the "
        "summary. When the run finishes, the distance raster and the scored "
        "sites are opened in the Viewer together, and the sites are "
        "<b>coloured by their score</b>: a plain ramp for the proximity "
        "index, and for the intensity index a ramp that breaks at 50 — the "
        "score the average location gets — so above and below it read as two "
        "different things. <b>Click a site</b> on the map and its whole row, including "
        "the columns your own layer brought with it, opens in a panel at the "
        "bottom right.");
}

// ------------------------------------------------------------ colour wheel

namespace {

constexpr int kWheelSide = 140;   // diameter of the disc
constexpr int kBarWidth  = 18;
constexpr int kWheelGap  = 12;
constexpr double kPi     = 3.14159265358979323846;   // M_PI is not portable

// The disc and the value bar beside it — the one part of the popup that has
// to be painted by hand, because a saturation that falls off towards the
// centre is two gradients composed, and QSS cannot express that. Everything
// else around it (the frame, the title, the slider) is an ordinary styled
// widget; only this rectangle is not.
//
// No Q_OBJECT: it has no signals of its own and calls its owner directly
// through a plain callback, the same reason PauseMark and the other
// file-local widgets near it do not carry one either — a class local to this
// file has no moc output to put one in.
class ColourWheelCanvas : public QWidget
{
public:
    ColourWheelCanvas(const QColor &initial, QWidget *parent)
        : QWidget(parent)
    {
        m_colour = initial.isValid() ? initial : QColor(Qt::red);
        int h = 0, s = 0, v = 0;
        m_colour.getHsv(&h, &s, &v);
        // getHsv answers -1 for an achromatic colour; the disc has to point
        // somewhere, and red is where every one of these starts.
        m_hue = h < 0 ? 0 : h;
        m_sat = s;
        m_val = v;
        setFixedSize(kBarWidth + kWheelGap + kWheelSide, kWheelSide);
        setCursor(Qt::CrossCursor);
    }

    // Called on every change, continuously while the pointer is down — the
    // point of picking a colour against a map is seeing it against the map,
    // not confirming a swatch and finding out afterwards.
    std::function<void(const QColor &)> onChange;

protected:
    void paintEvent(QPaintEvent *) override
    {
        rebuildWheel();
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing, true);

        const QColor edge = ThemeManager::mapped("#3a414b");

        // --- value bar: the chosen hue at full brightness, down to black ---
        const QRect bar = barRect();
        QLinearGradient g(bar.topLeft(), bar.bottomLeft());
        g.setColorAt(0.0, QColor::fromHsv(m_hue, m_sat, 255));
        g.setColorAt(1.0, QColor::fromHsv(m_hue, m_sat, 0));
        p.setPen(Qt::NoPen);
        p.setBrush(g);
        p.drawRect(bar);
        p.setPen(QPen(edge, 1));
        p.setBrush(Qt::NoBrush);
        p.drawRect(bar);

        // The marker rides the bar as a full-width tick, drawn in both black
        // and white so it survives either end of the gradient.
        const int markY = bar.top()
            + int(std::lround((1.0 - m_val / 255.0) * (bar.height() - 1)));
        p.setPen(QPen(Qt::black, 3));
        p.drawLine(bar.left(), markY, bar.right(), markY);
        p.setPen(QPen(Qt::white, 1));
        p.drawLine(bar.left(), markY, bar.right(), markY);

        // --- the disc ---
        const QRect wheel = wheelRect();
        p.drawImage(wheel.topLeft(), m_wheel);

        // --- the picked point, as a ringed dot ---
        const double r = kWheelSide / 2.0;
        const double angle = m_hue * kPi / 180.0;
        const double dist = (m_sat / 255.0) * r;
        const QPointF at(wheel.left() + r + std::cos(angle) * dist,
                         wheel.top() + r - std::sin(angle) * dist);
        p.setBrush(Qt::NoBrush);
        p.setPen(QPen(Qt::black, 3));
        p.drawEllipse(at, 5.0, 5.0);
        p.setPen(QPen(Qt::white, 1.5));
        p.drawEllipse(at, 5.0, 5.0);
        p.setPen(QPen(Qt::white, 1.5));
        p.drawEllipse(at, 1.5, 1.5);
    }

    void mousePressEvent(QMouseEvent *event) override
    {
        const QPoint pos = event->position().toPoint();
        m_grab = barRect().adjusted(-4, -4, 4, 4).contains(pos) ? Grab::Bar : Grab::Wheel;
        applyAt(pos);
        event->accept();
    }

    void mouseMoveEvent(QMouseEvent *event) override
    {
        if (m_grab != Grab::None)
            applyAt(event->position().toPoint());
        event->accept();
    }

    void mouseReleaseEvent(QMouseEvent *event) override
    {
        m_grab = Grab::None;
        event->accept();
    }

private:
    enum class Grab { None, Wheel, Bar };

    QRect barRect() const { return QRect(0, 0, kBarWidth, kWheelSide); }
    QRect wheelRect() const { return QRect(kBarWidth + kWheelGap, 0, kWheelSide, kWheelSide); }

    // The disc, at the current value. Built into an image rather than painted
    // with a conical gradient because saturation has to fall off towards the
    // middle as well, and that is two gradients Qt cannot compose — and
    // because the result is then cached: the only thing that invalidates it
    // is a change of value.
    void rebuildWheel()
    {
        if (m_wheelValue == m_val && m_wheel.width() == kWheelSide)
            return;
        m_wheelValue = m_val;

        m_wheel = QImage(kWheelSide, kWheelSide, QImage::Format_ARGB32_Premultiplied);
        m_wheel.fill(Qt::transparent);

        const double r = kWheelSide / 2.0;
        for (int y = 0; y < kWheelSide; ++y) {
            auto *line = reinterpret_cast<QRgb *>(m_wheel.scanLine(y));
            const double dy = y + 0.5 - r;
            for (int x = 0; x < kWheelSide; ++x) {
                const double dx = x + 0.5 - r;
                const double dist = std::sqrt(dx * dx + dy * dy);
                if (dist > r)
                    continue;
                double hue = std::atan2(-dy, dx) * 180.0 / kPi;
                if (hue < 0.0)
                    hue += 360.0;
                const int sat = int(std::lround(std::min(1.0, dist / r) * 255.0));
                QColor c = QColor::fromHsv(int(hue) % 360, sat, m_val);
                // One pixel of feathering at the rim, so the disc does not end
                // in a staircase. Premultiplied, because the image is.
                const double edge = r - dist;
                if (edge < 1.0)
                    c.setAlpha(int(std::lround(std::max(0.0, edge) * 255.0)));
                line[x] = qPremultiply(c.rgba());
            }
        }
    }

    void applyAt(const QPoint &pos)
    {
        if (m_grab == Grab::Bar) {
            const QRect bar = barRect();
            const double t = double(pos.y() - bar.top()) / std::max(1, bar.height() - 1);
            m_val = int(std::lround((1.0 - std::clamp(t, 0.0, 1.0)) * 255.0));
        } else if (m_grab == Grab::Wheel) {
            const QRect wheel = wheelRect();
            const double r = kWheelSide / 2.0;
            const double dx = pos.x() - (wheel.left() + r);
            const double dy = pos.y() - (wheel.top() + r);
            double hue = std::atan2(-dy, dx) * 180.0 / kPi;
            if (hue < 0.0)
                hue += 360.0;
            m_hue = int(hue) % 360;
            // Clamped, not ignored: a drag that leaves the disc should pin the
            // colour to the rim rather than stop responding.
            m_sat = int(std::lround(std::min(1.0, std::sqrt(dx * dx + dy * dy) / r) * 255.0));
        } else {
            return;
        }

        m_colour = QColor::fromHsv(m_hue, m_sat, m_val);
        update();
        if (onChange)
            onChange(m_colour);
    }

    QColor m_colour;
    int m_hue = 0;          // 0-359; kept separately because a grey has no hue
    int m_sat = 255;
    int m_val = 255;
    Grab m_grab = Grab::None;
    QImage m_wheel;         // rebuilt on resize and on a change of value
    int m_wheelValue = -1;  // the value m_wheel was built for
};

} // namespace

// The full popup: the wheel and its value bar, a size slider, a title and a
// close button. A plain child of the window it opens over rather than a
// window of its own — the same reason and the same pattern as HelpPopup
// above, and it is what lets the QSlider inside pick up the application
// stylesheet like any other control on the page, rather than risking the
// black rectangle a real top-level popup was not guaranteed to avoid.
//
// Dismissed by its own close button, by Escape, or by a click anywhere else
// — an application-wide event filter, not Qt::Popup's own outside-click grab,
// because that grab was not reliably reaching this widget from on top of a
// QGraphicsView doing its own mouse handling underneath.
class ColourWheel : public QFrame
{
public:
    static void showFor(QWidget *anchor, const QPoint &globalPos,
                        const QColor &initialColour, int initialSizePercent,
                        std::function<void(const QColor &)> onColour,
                        std::function<void(int)> onSizePercent)
    {
        QWidget *host = anchor ? anchor->window() : nullptr;
        if (!host)
            return;
        // One at a time, the same rule HelpPopup keeps: opening a second
        // layer's wheel replaces the first rather than stacking on top of it.
        if (s_open)
            s_open->close();

        auto *popup = new ColourWheel(host, initialColour, initialSizePercent,
                                      std::move(onColour), std::move(onSizePercent));
        s_open = popup;
        popup->place(host->mapFromGlobal(globalPos));
        popup->show();
        popup->raise();
    }

private:
    static QPointer<ColourWheel> s_open;

    ColourWheel(QWidget *host, const QColor &initialColour, int initialSizePercent,
               std::function<void(const QColor &)> onColour,
               std::function<void(int)> onSizePercent)
        : QFrame(host)
    {
        // Its own name: visually the same kind of thing as the Overlays panel
        // it opens from (a small pane floating over the map), styled the same
        // way in theme.qss, but named separately because the two can be open
        // — well, visible — at once and must not be confused by a selector.
        setObjectName(QStringLiteral("ColourWheelPanel"));
        setAttribute(Qt::WA_DeleteOnClose, true);

        auto *layout = new QVBoxLayout(this);
        layout->setContentsMargins(12, 8, 12, 10);
        layout->setSpacing(8);

        auto *head = new QHBoxLayout;
        head->setSpacing(6);
        auto *title = new QLabel(QObject::tr("Colour"), this);
        title->setObjectName(QStringLiteral("CardTitle"));
        head->addWidget(title, 1);
        auto *close = new QToolButton(this);
        close->setObjectName(QStringLiteral("TourClose"));
        close->setText(QStringLiteral("✕"));
        close->setCursor(Qt::PointingHandCursor);
        close->setAutoRaise(true);
        QObject::connect(close, &QToolButton::clicked, this, &QWidget::close);
        head->addWidget(close, 0, Qt::AlignTop);
        layout->addLayout(head);

        auto *canvas = new ColourWheelCanvas(initialColour, this);
        canvas->onChange = std::move(onColour);
        layout->addWidget(canvas, 0, Qt::AlignHCenter);

        // The feature-size slider, styled and sized like the Viewer's own
        // Opacity control — the same kind of "how much of this" question,
        // answered the same way.
        auto *sizeRow = new QHBoxLayout;
        sizeRow->setSpacing(8);
        auto *sizeLabel = new QLabel(QObject::tr("Size"), this);
        sizeLabel->setObjectName(QStringLiteral("HintLabel"));
        sizeRow->addWidget(sizeLabel);
        auto *slider = new QSlider(Qt::Horizontal, this);
        slider->setRange(25, 300);
        slider->setValue(std::clamp(initialSizePercent, 25, 300));
        slider->setFixedWidth(kBarWidth + kWheelGap + kWheelSide);
        slider->setFixedHeight(22);
        guardWheel(slider);
        auto onSize = std::make_shared<std::function<void(int)>>(std::move(onSizePercent));
        QObject::connect(slider, &QSlider::valueChanged, this, [onSize](int v) {
            if (*onSize)
                (*onSize)(v);
        });
        sizeRow->addWidget(slider, 1);
        layout->addLayout(sizeRow);

        adjustSize();

        // Installed on the application, not on the host: the next press may
        // land on any widget at all, and every one of them dismisses this —
        // see the class comment for why this is used instead of relying on
        // Qt::Popup's own grab.
        qApp->installEventFilter(this);
    }

    // Under the pointer, kept inside the window on both axes — the same rule
    // HelpPopup follows, and for the same reason: this application is one
    // window, not a desktop of screen space to spill a popup onto.
    void place(const QPoint &hostPos)
    {
        QWidget *host = parentWidget();
        if (!host)
            return;
        const int x = qBound(8, hostPos.x(), qMax(8, host->width() - width() - 8));
        const int y = qBound(8, hostPos.y(), qMax(8, host->height() - height() - 8));
        move(x, y);
    }

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        Q_UNUSED(watched);
        if (event->type() == QEvent::MouseButtonPress) {
            // Global screen coordinates, not which widget technically caught
            // the press. The wheel's disc and value bar are hand-painted on a
            // single canvas widget and the slider is a real QSlider, so a
            // press meant for either can be reported as landing on all sorts
            // of different objects along the way — checking against this
            // panel's own on-screen rectangle instead answers the one
            // question that actually matters and cannot be got wrong by an
            // ancestry chain that turns out not to be what it looked like.
            const auto *me = static_cast<QMouseEvent *>(event);
            const QRect onScreen(mapToGlobal(QPoint(0, 0)), size());
            if (!onScreen.contains(me->globalPosition().toPoint()))
                close();
        } else if (event->type() == QEvent::KeyPress) {
            const auto *key = static_cast<QKeyEvent *>(event);
            if (key->key() == Qt::Key_Escape)
                close();
        }
        return QFrame::eventFilter(watched, event);
    }
};

QPointer<ColourWheel> ColourWheel::s_open;

void pickColour(QWidget *anchor, const QPoint &globalPos, const QColor &initialColour,
                int initialSizePercent,
                const std::function<void(const QColor &)> &onColour,
                const std::function<void(int)> &onSizePercent)
{
    ColourWheel::showFor(anchor, globalPos, initialColour, initialSizePercent,
                         onColour, onSizePercent);
}

QIcon makeWindowIcon(WindowGlyph glyph, const QColor &color, int size)
{
    QPixmap pm(size * 2, size * 2);
    pm.setDevicePixelRatio(2.0);
    pm.fill(Qt::transparent);

    QPainter painter(&pm);
    painter.setRenderHint(QPainter::Antialiasing, true);
    QPen pen(color);
    pen.setWidthF(1.3);
    pen.setCapStyle(Qt::FlatCap);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);

    // A ~10 px glyph inside a 20 px box, which is what the Windows title bar
    // itself uses; a tighter inset made these read as half-size next to any
    // other application.
    const qreal s = size;
    const qreal m = s * 0.25;          // inset from the icon box
    const QRectF box(m, m, s - 2 * m, s - 2 * m);

    switch (glyph) {
    case WindowGlyph::Minimise:
        painter.drawLine(QPointF(box.left(), box.center().y()),
                         QPointF(box.right(), box.center().y()));
        break;
    case WindowGlyph::Maximise:
        painter.drawRect(box);
        break;
    case WindowGlyph::Restore: {
        // Back sheet peeking out of the top-right of the front one.
        const qreal off = box.width() * 0.26;
        painter.drawRect(box.adjusted(0, off, -off, 0));
        QPainterPath back;
        back.moveTo(box.left() + off, box.top() + off);
        back.lineTo(box.left() + off, box.top());
        back.lineTo(box.right(), box.top());
        back.lineTo(box.right(), box.bottom() - off);
        back.lineTo(box.right() - off, box.bottom() - off);
        painter.drawPath(back);
        break;
    }
    case WindowGlyph::Close:
        painter.drawLine(box.topLeft(), box.bottomRight());
        painter.drawLine(box.topRight(), box.bottomLeft());
        break;
    }
    painter.end();
    return QIcon(pm);
}

} // namespace TrajectaUi
