#pragma once

#include <QVector>
#include <QWidget>

// Dual-handle horizontal slider used for raster value filtering. The track
// background shows a (log-scaled) histogram of the data so thresholds like
// "top 5%" can be picked by eye. Values are normalized to [0, 1]; the owner
// maps them to real data units.
class RangeSlider : public QWidget
{
    Q_OBJECT

public:
    explicit RangeSlider(QWidget *parent = nullptr);

    double lowerValue() const { return m_lower; }
    double upperValue() const { return m_upper; }
    void setRange(double lower, double upper);   // does not emit rangeChanged
    void setHistogram(const QVector<float> &bins);

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

signals:
    void rangeChanged(double lower, double upper);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    enum class Grab { None, Lower, Upper };

    double posToValue(int x) const;
    int valueToPos(double v) const;
    QRect trackRect() const;

    double m_lower = 0.0;
    double m_upper = 1.0;
    QVector<float> m_bins;   // normalized 0..1 heights
    Grab m_grab = Grab::None;
};
