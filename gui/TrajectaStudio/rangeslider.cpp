#include "rangeslider.h"

#include "thememanager.h"

#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>

#include <algorithm>
#include <cmath>

namespace {
// Theme colors (kept in sync with theme.qss).
// Read from the active palette on every repaint rather than baked in: as
// constants these stayed dark-theme teal on a light theme, which is why the
// filter track and its histogram were the only part of the Viewer that did not
// follow the theme. Each argument is a colour that exists in theme.qss, so the
// palette map has a translation for it.
inline QColor kTrack()       { return ThemeManager::mapped("#22272f"); }
inline QColor kTrackBorder() { return ThemeManager::mapped("#333a44"); }
inline QColor kHistogram()   { return ThemeManager::mapped("#3a414b"); }
inline QColor kActive()      { return ThemeManager::mapped("#7ea8a0"); }
inline QColor kActiveDim()   { return ThemeManager::mapped("#2f3a38"); }
inline QColor kHandle()      { return ThemeManager::mapped("#e4e7ec"); }

constexpr int kHandleRadius = 7;
constexpr int kTrackHeight = 6;
constexpr int kHistHeight = 26;
} // namespace

RangeSlider::RangeSlider(QWidget *parent)
    : QWidget(parent)
{
    setCursor(Qt::PointingHandCursor);
    setMinimumWidth(160);
}

void RangeSlider::setRange(double lower, double upper)
{
    m_lower = std::clamp(lower, 0.0, 1.0);
    m_upper = std::clamp(upper, m_lower, 1.0);
    update();
}

void RangeSlider::setHistogram(const QVector<float> &bins)
{
    m_bins = bins;
    update();
}

QSize RangeSlider::sizeHint() const
{
    return {240, kHistHeight + kHandleRadius * 2 + 10};
}

QSize RangeSlider::minimumSizeHint() const
{
    return {160, kHistHeight + kHandleRadius * 2 + 10};
}

QRect RangeSlider::trackRect() const
{
    const int y = kHistHeight + 4;
    return {kHandleRadius, y, width() - kHandleRadius * 2, kTrackHeight};
}

double RangeSlider::posToValue(int x) const
{
    const QRect r = trackRect();
    if (r.width() <= 0)
        return 0.0;
    return std::clamp((x - r.left()) / double(r.width()), 0.0, 1.0);
}

int RangeSlider::valueToPos(double v) const
{
    const QRect r = trackRect();
    return r.left() + int(std::lround(v * r.width()));
}

void RangeSlider::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    const QRect track = trackRect();
    const int loX = valueToPos(m_lower);
    const int hiX = valueToPos(m_upper);

    // Histogram, with the in-range part highlighted.
    if (!m_bins.isEmpty()) {
        const QRect hist(track.left(), 0, track.width(), kHistHeight);
        const double binW = hist.width() / double(m_bins.size());
        for (int i = 0; i < m_bins.size(); ++i) {
            const int h = int(std::lround(m_bins.at(i) * hist.height()));
            if (h <= 0)
                continue;
            const int x0 = hist.left() + int(std::floor(i * binW));
            const int x1 = hist.left() + int(std::floor((i + 1) * binW));
            const QRect bar(x0, hist.bottom() - h + 1,
                            std::max(1, x1 - x0), h);
            const bool inRange = bar.center().x() >= loX && bar.center().x() <= hiX;
            p.fillRect(bar, inRange ? kActiveDim() : kHistogram());
        }
    }

    // Track base + selected span.
    p.setPen(QPen(kTrackBorder(), 1));
    p.setBrush(kTrack());
    p.drawRoundedRect(track, 3, 3);
    p.setPen(Qt::NoPen);
    p.setBrush(kActive());
    p.drawRoundedRect(QRect(QPoint(loX, track.top()),
                            QPoint(hiX, track.bottom())), 3, 3);

    // Handles.
    const int cy = track.center().y() + 1;
    for (const int cx : {loX, hiX}) {
        p.setBrush(kHandle());
        p.setPen(QPen(kActive(), 2));
        p.drawEllipse(QPoint(cx, cy), kHandleRadius - 1, kHandleRadius - 1);
    }
}

void RangeSlider::mousePressEvent(QMouseEvent *event)
{
    if (event->button() != Qt::LeftButton)
        return;
    const int loX = valueToPos(m_lower);
    const int hiX = valueToPos(m_upper);
    const int x = event->pos().x();
    // Grab the nearest handle; ties go to the one that can still move.
    if (std::abs(x - loX) < std::abs(x - hiX))
        m_grab = Grab::Lower;
    else if (std::abs(x - loX) > std::abs(x - hiX))
        m_grab = Grab::Upper;
    else
        m_grab = x < loX ? Grab::Lower : Grab::Upper;
    mouseMoveEvent(event);
}

void RangeSlider::mouseMoveEvent(QMouseEvent *event)
{
    if (m_grab == Grab::None)
        return;
    const double v = posToValue(event->pos().x());
    if (m_grab == Grab::Lower)
        m_lower = std::min(v, m_upper);
    else
        m_upper = std::max(v, m_lower);
    update();
    emit rangeChanged(m_lower, m_upper);
}

void RangeSlider::mouseReleaseEvent(QMouseEvent *)
{
    m_grab = Grab::None;
}
