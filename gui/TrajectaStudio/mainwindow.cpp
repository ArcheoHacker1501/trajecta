#include "mainwindow.h"

#include "batchpage.h"
#include "postbatchpage.h"

#include "advancedsettingsdialog.h"
#include "coherence.h"
#include "confirmdialog.h"
#include "reportform.h"
#include "consoleview.h"
#include "gdalapi.h"
#include "neighbourhood.h"   // kMin/kMax for the custom neighbours box
#include "pathpicker.h"
#include "routecompare.h"
#include "smoothcombobox.h"
#include "systeminfo.h"
#include "thememanager.h"
#include "uiwidgets.h"
#include "walkthrough.h"
#include "viewerpage.h"

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QDoubleSpinBox>
#include <QEventLoop>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMap>
#include <QMenu>
#include <QStyledItemDelegate>
#include <QTextDocumentFragment>
#include <QMimeData>
#include <QDragEnterEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QMessageBox>
#include <QPainterPath>
#include <QPalette>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QScreen>
#include <QFont>
#include <QFontMetrics>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QStyle>
#include <QTextBrowser>
#include <QThread>
#include <QTimer>
#include <QToolButton>
#include <QTransform>
#include <QUrl>
#include <QVBoxLayout>
#include <QVariantAnimation>
#include <QWindow>

#include <cmath>
#include <tuple>
#include <vector>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dwmapi.h>
#include <windowsx.h>   // GET_X_LPARAM / GET_Y_LPARAM
#endif

namespace {

const char *kVectorFilter =
    "Vector files (*.shp *.geojson *.json *.kml *.gml *.xml *.csv);;All files (*)";
const char *kRasterFilter = "GeoTIFF (*.tif *.tiff);;All files (*)";
const char *kProjectUrl = "https://github.com/ArcheoHacker1501/trajecta";
// The author's GitHub page, which is where the button on the About screen
// goes: the repository is one of the things on it, and someone who follows a
// mark labelled "GitHub" is looking for the person as often as the project.
const char *kGithubUrl = "https://github.com/ArcheoHacker1501";
// The author's ORCID iD. Beside the GitHub mark on the About screen and not
// buried in the citation line: in the fields Trajecta is written for, this is
// the identifier that ties the software to a person unambiguously, which a
// username cannot do.
const char *kOrcidUrl = "https://orcid.org/0009-0005-7692-3527";

// Stamped beside the stored batch, and checked when it is read back. Its whole
// job is to let loadSettings() tell a job it can reason about from one written
// by a build that kept no record of whether the batch had been run. Raise it
// whenever that reasoning changes again.
constexpr int kBatchJobVersion = 2;
// Same reasoning as kBatchJobVersion, for the post-processing batch page.
constexpr int kPostBatchJobVersion = 1;

// One height for every log canvas, from TrajectaUi: the single-run panels, the
// batch and the two post-processing tools all read the same size, so moving
// between them does not change how much transcript is in front of you.
constexpr int kLogCanvasHeight = TrajectaUi::kLogCanvasHeight;

// A QSpinBox sizes itself from its own range — the widest of minimum and
// maximum, in the current style and font — so the single-run CPU-threads and
// RAM fields, ranged to what *this* machine actually has, come out narrower
// than the batch card's, ranged generously to 1024 threads and 1 TB so a
// chunk file saved on a bigger machine is never clipped (see batchpage.cpp).
// Widening the single-run box to match is only a display question, not a
// validation one: its own setRange() stays untouched, and this throwaway
// probe — never shown, never parented — exists purely to ask the style what
// width the batch card's wider range would need, then hand that back as a
// floor.
int batchSpinBoxWidth(int minValue, int maxValue, const QString &suffix = QString())
{
    QSpinBox probe;
    probe.setRange(minValue, maxValue);
    if (!suffix.isEmpty())
        probe.setSuffix(suffix);
    return probe.sizeHint().width();
}

// Renders a menu that has never been opened into a still picture, and reports
// where each of its groups sits inside it.
//
// The tour cannot open the real menu. A QMenu is a top-level popup: it takes a
// mouse and keyboard grab, paints above the overlay, stays clickable, and eats
// the click meant for the tour's own Continue button. A picture of it has none
// of those properties and cannot drift out of date either, because it is made
// from the same QActions the menu is made of — add a theme tomorrow and the
// picture grows a row.
//
// WA_DontShowOnScreen is what makes it possible: the widget goes through the
// whole show and layout path, so it is arranged and painted exactly as it
// would be, but no native window is ever mapped and therefore no grab is taken.
// `highlight`/`highlightRect` are for the one entry that wants its own
// caption rather than sharing its group's: filled in the same pass, from the
// same laid-out menu, so it is never a stale geometry from a previous show.
QPixmap renderMenuPicture(QMenu *menu, QVector<QRect> *groups,
                          QAction *highlight = nullptr, QRect *highlightRect = nullptr)
{
    if (!menu)
        return QPixmap();

    menu->setAttribute(Qt::WA_DontShowOnScreen, true);
    menu->show();
    menu->resize(menu->sizeHint());
    const QPixmap picture = menu->grab();

    if (highlight && highlightRect)
        *highlightRect = menu->actionGeometry(highlight);

    if (groups) {
        // The group headings are the only disabled entries in the menu, which
        // makes them a reliable way to find the boundaries without hard-coding
        // how many themes or fonts there happen to be.
        QVector<QAction *> headings;
        const QList<QAction *> actions = menu->actions();
        for (QAction *a : actions) {
            if (!a->isSeparator() && !a->isEnabled() && !a->text().isEmpty())
                headings.append(a);
        }
        for (int i = 0; i < headings.size(); ++i) {
            const QRect head = menu->actionGeometry(headings.at(i));
            const int bottom = (i + 1 < headings.size())
                                   ? menu->actionGeometry(headings.at(i + 1)).top() - 4
                                   : menu->height() - 4;
            if (head.isValid() && bottom > head.top())
                groups->append(QRect(head.left(), head.top(), head.width(), bottom - head.top()));
        }
    }

    menu->hide();
    menu->setAttribute(Qt::WA_DontShowOnScreen, false);
    return picture;
}

QString formatElapsed(qint64 ms)
{
    const qint64 secs = ms / 1000;
    return QStringLiteral("%1:%2:%3")
        .arg(secs / 3600)
        .arg((secs % 3600) / 60, 2, 10, QLatin1Char('0'))
        .arg(secs % 60, 2, 10, QLatin1Char('0'));
}

bool isValidFileName(const QString &name)
{
    static const QRegularExpression bad(QStringLiteral("[\\\\/:*?\"<>|]"));
    return !name.trimmed().isEmpty() && !bad.match(name).hasMatch();
}

bool isAscii(const QString &s)
{
    for (const QChar &c : s)
        if (c.unicode() > 127)
            return false;
    return true;
}

using TrajectaUi::guardWheel;

// A gear drawn rather than shipped: it has to take the colour of whichever
// palette is active, and a vector path stays crisp at any device pixel ratio.
// Eight rounded teeth on a disc, with the hub punched out.
// `angleDeg` spins it in place; the eight teeth make it 45-degree symmetric, so
// any multiple of 45 lands on an orientation identical to the one it left.
QIcon makeGearIcon(const QColor &color, int size, qreal angleDeg = 0.0)
{
    QPixmap pm(size * 2, size * 2);   // 2x, then let Qt scale it down
    pm.setDevicePixelRatio(2.0);
    pm.fill(Qt::transparent);

    const qreal r = size / 2.0;
    QPainterPath gear;
    for (int i = 0; i < 8; ++i) {
        QPainterPath tooth;
        tooth.addRoundedRect(QRectF(-r * 0.155, -r * 0.99, r * 0.31, r * 0.42),
                             r * 0.07, r * 0.07);
        QTransform rot;
        rot.rotate(i * 45.0);
        gear = gear.united(rot.map(tooth));
    }
    QPainterPath disc;
    disc.addEllipse(QPointF(0, 0), r * 0.66, r * 0.66);
    gear = gear.united(disc);
    QPainterPath hub;
    hub.addEllipse(QPointF(0, 0), r * 0.27, r * 0.27);
    gear = gear.subtracted(hub);

    QPainter painter(&pm);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.translate(size / 2.0, size / 2.0);
    if (angleDeg != 0.0)
        painter.rotate(angleDeg);
    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    painter.drawPath(gear);
    painter.end();

    return QIcon(pm);
}

// GitHub's mark, taken from the SVG in the resources and dyed the colour the
// caller asks for. Not drawn by hand like the gear: this one is somebody's
// trademark and has exactly one correct shape, so it is shipped as artwork —
// but it still has to follow the palette, and the file is a single solid
// silhouette, so a source-in fill over its alpha is all the recolouring it
// needs. Read at twice the target size and marked 2x, like every other icon
// here, so it stays sharp on a high-DPI screen.
QIcon makeGithubIcon(const QColor &color, int size)
{
    QPixmap art(QStringLiteral(":/assets/github.svg"));
    if (art.isNull())
        return QIcon();
    QPixmap pm = art.scaled(size * 2, size * 2, Qt::KeepAspectRatio,
                            Qt::SmoothTransformation);
    QPainter painter(&pm);
    painter.setCompositionMode(QPainter::CompositionMode_SourceIn);
    painter.fillRect(pm.rect(), color);
    painter.end();
    pm.setDevicePixelRatio(2.0);
    return QIcon(pm);
}

// The ORCID iD mark. No colour argument, unlike makeGithubIcon() above: this
// one arrives already coloured and stays that way. A monochrome ORCID icon
// dyed to match the theme would be a different mark, and the point of showing
// it is that it is recognised on sight.
QIcon makeOrcidIcon(int size)
{
    QPixmap art(QStringLiteral(":/assets/orcid.svg"));
    if (art.isNull())
        return QIcon();
    QPixmap pm = art.scaled(size * 2, size * 2, Qt::KeepAspectRatio,
                            Qt::SmoothTransformation);
    pm.setDevicePixelRatio(2.0);
    return QIcon(pm);
}

// Minimise / maximise / restore / close: TrajectaUi::makeWindowIcon() in
// uiwidgets.cpp, shared with FramelessDialog's own close button so the mark
// that closes a dialog and the one that closes the whole application are the
// same drawing.
using TrajectaUi::WindowGlyph;
using TrajectaUi::makeWindowIcon;

// ---------------------------------------------------------------------------
// GuideBrowser — the Guide page, with figures that follow the window width.
//
// QTextDocument has no notion of a relative image width: an <img> is laid out
// at exactly the pixel width written into the tag, so fixed widths force a
// horizontal scrollbar as soon as the window is narrower than the figures.
// The page is therefore re-emitted from a template whenever the viewport width
// changes, with the figure widths derived from the space actually available.
//
// Figures are also re-sampled to their new size once the resize settles:
// QTextBrowser scales images with a fast, non-smooth filter that leaves
// photographic figures visibly soft, the more so on HiDPI screens where the
// document lays out in logical pixels and the result is upscaled again.
// Between the two, each source is kept at the largest size the page can show
// (bounded, so the full-resolution JPEGs are not held decoded in memory).
// ---------------------------------------------------------------------------
class GuideBrowser : public QTextBrowser
{
public:
    explicit GuideBrowser(QWidget *parent = nullptr)
        : QTextBrowser(parent)
    {
        // Reserving the vertical scrollbar keeps the viewport width constant.
        // Letting it come and go would feed back into the layout: re-emitting
        // the page changes its height, the scrollbar appears, the viewport
        // narrows, the figures shrink, the page gets shorter, and round again.
        // The guide is far taller than any window, so it is always needed.
        setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        // Refuses drops for the same reason the log and the map canvas do: a
        // text view accepts them by default and discards them when read-only,
        // which stops Qt's search for a handler dead. See MapView's constructor.
        setAcceptDrops(false);
        viewport()->setAcceptDrops(false);
        m_resample = new QTimer(this);
        m_resample->setSingleShot(true);
        m_resample->setInterval(150);
        connect(m_resample, &QTimer::timeout, this, [this] { relayout(true); });
    }

    // `name` is the token used as <img src="guide:name"> in the template.
    // `fullWidth` marks a figure spanning the text column; the others share a
    // row with a second figure and get half of it.
    void addFigure(const QString &name, const QString &path, bool fullWidth)
    {
        m_figures.push_back({name, path, QImage(), fullWidth, 0});
    }

    // The template carries the %HALF% and %FULL% width placeholders.
    void setTemplate(const QString &html) { m_template = html; }

    // False until the page has been opened once, which is also to say: until
    // its document has been parsed and its figures decoded. A theme change
    // asks, because re-parsing a page nobody has looked at yet would spend the
    // cost the laziness exists to avoid — and would spend it for nothing,
    // since the colours are read at parse time and the first opening will pick
    // up the new ones anyway.
    bool isReady() const { return m_ready; }

protected:
    void showEvent(QShowEvent *event) override
    {
        QTextBrowser::showEvent(event);
        // Decoding and rescaling the figures costs a noticeable fraction of a
        // second, so it waits until the page is actually opened.
        if (!m_ready) {
            m_ready = true;
            relayout(true);
        }
    }

    void resizeEvent(QResizeEvent *event) override
    {
        QTextBrowser::resizeEvent(event);
        if (m_ready && event->size().width() != event->oldSize().width()) {
            relayout(false);      // immediate, so the page tracks the drag
            m_resample->start();  // crisp re-sampling once the drag settles
        }
    }

private:
    struct Figure {
        QString name;
        QString path;
        QImage master;       // source, capped at kMaxFigure * dpr
        bool fullWidth;
        int renderedWidth;   // device pixels currently registered
    };

    static constexpr int kMinFigure = 200;
    static constexpr int kMaxFigure = 1400;  // logical px, also caps `master`
    static constexpr int kCellSpacing = 10;  // cellspacing of the figure tables
    static constexpr int kCellPadding = 10;  // td padding from the page stylesheet

    // Width the document body gets, once its own margins are taken out.
    int contentWidth() const
    {
        return viewport()->width() - 2 * qRound(document()->documentMargin()) - 6;
    }

    // A figure spanning the text column sits in a one-cell table: two
    // spacings and two paddings come off the content width.
    int fullFigureWidth() const
    {
        return qBound(kMinFigure,
                      contentWidth() - 2 * kCellSpacing - 2 * kCellPadding,
                      kMaxFigure);
    }

    // Two figures side by side: three spacings and four paddings, halved.
    int halfFigureWidth() const
    {
        return qBound(kMinFigure,
                      (contentWidth() - 3 * kCellSpacing - 4 * kCellPadding) / 2,
                      kMaxFigure);
    }

    void registerScaled(Figure &figure, int logicalWidth)
    {
        if (figure.master.isNull()) {
            const QImage src(figure.path);
            if (src.isNull())
                return;
            const int cap = qRound(kMaxFigure * devicePixelRatioF());
            figure.master = src.width() > cap
                                ? src.scaledToWidth(cap, Qt::SmoothTransformation)
                                : src;
        }
        // Registered at (display width x device pixel ratio) and referenced at
        // that same logical width, the figure is drawn 1:1 with no further
        // resampling by the text layout.
        const int deviceWidth =
            qMin(qRound(logicalWidth * devicePixelRatioF()), figure.master.width());
        if (deviceWidth == figure.renderedWidth)
            return;
        document()->addResource(
            QTextDocument::ImageResource,
            QUrl(QLatin1String("guide:") + figure.name),
            figure.master.scaledToWidth(deviceWidth, Qt::SmoothTransformation));
        figure.renderedWidth = deviceWidth;
    }

public:
    // Public so a palette change can force the page to be re-emitted: the
    // heading and link colours live in the template's own <style> block, which
    // the application stylesheet cannot reach.
    void relayout(bool resample, bool force = false)
    {
        if (m_template.isEmpty())
            return;
        // Ignore sub-pixel-ish jitter; only a real width change is worth a
        // full re-parse of the document.
        if (!resample && !force && qAbs(contentWidth() - m_lastWidth) < 8)
            return;
        const int full = fullFigureWidth();
        const int half = halfFigureWidth();

        if (resample) {
            for (Figure &figure : m_figures)
                registerScaled(figure, figure.fullWidth ? full : half);
        }

        // setHtml re-parses the document from scratch, so hold the reading
        // position: the content height changes with the figure size, which
        // makes a proportional restore the right one.
        QScrollBar *bar = verticalScrollBar();
        const double fraction =
            bar->maximum() > 0 ? double(bar->value()) / bar->maximum() : 0.0;

        QString html = m_template;
        html.replace(QLatin1String("%HALF%"), QString::number(half));
        html.replace(QLatin1String("%FULL%"), QString::number(full));
        html.replace(QLatin1String("%H2%"), ThemeManager::mapped("#a8d0c8").name());
        html.replace(QLatin1String("%H3%"), ThemeManager::mapped("#d3a25e").name());
        html.replace(QLatin1String("%LINK%"), ThemeManager::mapped("#7ea8a0").name());
        setHtml(html);

        bar->setValue(qRound(fraction * bar->maximum()));
        m_lastWidth = contentWidth();
    }

private:
    QString m_template;
    std::vector<Figure> m_figures;
    QTimer *m_resample = nullptr;
    int m_lastWidth = -1;
    bool m_ready = false;
};

// The About page was folded into the Guide's Overview, which is where a
// colophon belongs and which leaves one fewer tab in the bar. The page itself
// is still built and still compiles — flip this to true and it comes back,
// with its tab, exactly as it was.
constexpr bool kShowAboutTab = false;

// Where the Guide sits in the stack. Named because switchPage() has to
// recognise it to send the Guide home on arrival, and a bare 3 there would be
// the kind of number that survives a reordering and quietly means the wrong
// page afterwards.
constexpr int kGuidePageIndex = 3;

// Indents a row's icon+text when it is a child inside a collapsible group
// (see GuideNav::addChildItem) — QListWidget has no tree structure and no
// per-item padding of its own, and the plain style used elsewhere in the app
// has no indent to borrow, so this is the one row property a delegate has to
// supply by hand.
class GuideNavDelegate : public QStyledItemDelegate
{
public:
    using QStyledItemDelegate::QStyledItemDelegate;
    static constexpr int kIndentRole = Qt::UserRole + 2;

    void paint(QPainter *painter, const QStyleOptionViewItem &option,
               const QModelIndex &index) const override
    {
        QStyleOptionViewItem opt(option);
        if (index.data(kIndentRole).toBool())
            opt.rect.adjust(20, 0, 0, 0);
        QStyledItemDelegate::paint(painter, opt, index);
    }
};

// The Guide's page list.
//
// A QListWidget can colour the row it is on, but it cannot *move* anything:
// Qt stylesheets have no transitions, so the selection would jump. The one
// piece of motion in the navigation is therefore painted here — a rounded
// accent that slides from the old row to the new one — over a stylesheet that
// leaves the selected row's own background alone.
//
// Two rows, Processing and Post-processing, are collapsible groups rather
// than pages: clicking one reveals or hides the page rows nested under it
// and rotates a small chevron to show which state it is in — the same
// disclosure Quarto's own sidebar uses for its sections
// (https://quarto.org/docs/websites/). A group carries no page of its own
// (kPageRole -1), which is why toggling one is handled in mousePressEvent
// rather than left to fall through to the normal click machinery: Qt would
// otherwise make the header itself "current" for the instant between press
// and release, sliding the mark onto an empty row and back.
//
// No Q_OBJECT and no new property: this translation unit has no moc pass, so
// every animation here drives a plain member through QVariantAnimation's
// valueChanged rather than through the property system, and the "a page was
// chosen" notification is a std::function rather than a signal.
class GuideNav : public QListWidget
{
public:
    // A plain page link carries its target page index here; a group header
    // carries -1 and puts its key in kGroupRole instead, so a click can tell
    // "open this page" from "toggle this group" apart without a parallel
    // array that could fall out of sync with the list.
    static constexpr int kPageRole = Qt::UserRole;
    static constexpr int kGroupRole = Qt::UserRole + 1;

    explicit GuideNav(QWidget *parent = nullptr)
        : QListWidget(parent)
    {
        setObjectName(QStringLiteral("GuideSidebar"));
        setFrameShape(QFrame::NoFrame);
        setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        setFocusPolicy(Qt::NoFocus);
        setUniformItemSizes(true);
        setItemDelegate(new GuideNavDelegate(this));

        m_anim = new QVariantAnimation(this);
        m_anim->setDuration(190);
        m_anim->setEasingCurve(QEasingCurve::OutCubic);
        connect(m_anim, &QVariantAnimation::valueChanged, this,
                [this](const QVariant &v) {
                    m_markTop = v.toReal();
                    viewport()->update();
                });
        connect(this, &QListWidget::currentRowChanged, this,
                [this](int) { slideToCurrent(true); });
        connect(this, &QListWidget::itemClicked, this, [this](QListWidgetItem *it) {
            const int page = it->data(kPageRole).toInt();
            if (page >= 0) {
                m_activePageItem = it;
                if (onPageChosen)
                    onPageChosen(page);
            }
        });
    }

    // A plain top-level row that opens a page directly (Overview, Credits,
    // References, About).
    void addPageItem(const QString &label, int pageIndex)
    {
        auto *it = new QListWidgetItem(label, this);
        it->setData(kPageRole, pageIndex);
    }

    // A collapsible section header (Processing, Post-processing): no page of
    // its own, just the rows that follow it.
    void addGroupHeader(const QString &label, const QString &group)
    {
        auto *it = new QListWidgetItem(label, this);
        it->setData(kPageRole, -1);
        it->setData(kGroupRole, group);
        m_headerItem.insert(group, it);
        m_groupOpen.insert(group, false);
        m_groupAngle.insert(group, 0.0);
    }

    // A page-linking row nested under `group`, hidden until the group opens.
    void addChildItem(const QString &label, int pageIndex, const QString &group)
    {
        auto *it = new QListWidgetItem(label, this);
        it->setData(kPageRole, pageIndex);
        it->setData(kGroupRole, group);
        it->setData(GuideNavDelegate::kIndentRole, true);
        it->setHidden(true);
        m_children[group].append(it);
    }

    // Selects the row for `pageIndex`, opening its group first if it has one
    // and is not already open — used whenever MainWindow changes the page
    // from somewhere other than a click here (the walkthrough, arriving at
    // the tab, the --guide-page test hook).
    void selectPage(int pageIndex)
    {
        for (auto git = m_children.constBegin(); git != m_children.constEnd(); ++git) {
            for (QListWidgetItem *child : git.value()) {
                if (child->data(kPageRole).toInt() == pageIndex
                    && !m_groupOpen.value(git.key()))
                    setGroupOpen(git.key(), true, false);
            }
        }
        for (int i = 0; i < count(); ++i) {
            QListWidgetItem *it = item(i);
            if (it->data(kPageRole).toInt() == pageIndex) {
                m_activePageItem = it;
                setCurrentItem(it);
                return;
            }
        }
    }

    // Called once the rows exist and again whenever the page changes from
    // somewhere other than a click, so the mark never lags behind the
    // selection it is supposed to be.
    void slideToCurrent(bool animate)
    {
        const QModelIndex idx = currentIndex();
        if (!idx.isValid())
            return;
        const QRect r = visualRect(idx);
        m_markHeight = r.height();
        m_anim->stop();
        if (!animate || !m_marked) {
            m_marked = true;
            m_markTop = r.top();
            viewport()->update();
            return;
        }
        m_anim->setStartValue(m_markTop);
        m_anim->setEndValue(qreal(r.top()));
        m_anim->start();
    }

    // Set once by MainWindow right after construction: what a click on a
    // page-linking row should do. A group header handles its own click and
    // never reaches this.
    std::function<void(int)> onPageChosen;

protected:
    // QAbstractScrollArea routes the viewport's paint events here, so this
    // painter targets the viewport and the mark lands under the rows the base
    // class draws immediately afterwards.
    void paintEvent(QPaintEvent *event) override
    {
        if (m_marked && m_markHeight > 0) {
            QPainter p(viewport());
            p.setRenderHint(QPainter::Antialiasing, true);
            const QRectF r(2.0, m_markTop, viewport()->width() - 4.0, m_markHeight);
            p.setPen(Qt::NoPen);
            p.setBrush(ThemeManager::mapped("#1e2a28"));
            p.drawRoundedRect(r, 8.0, 8.0);
            // The bar at the leading edge, in the accent: it is what the eye
            // follows while the fill slides.
            p.setBrush(ThemeManager::mapped("#7ea8a0"));
            p.drawRoundedRect(QRectF(r.left(), r.top() + 6.0, 3.0,
                                     r.height() - 12.0),
                              1.5, 1.5);
        }
        QListWidget::paintEvent(event);

        // The chevrons, drawn last so they sit over the row the base class
        // just painted, not under it.
        QPainter cp(viewport());
        cp.setRenderHint(QPainter::Antialiasing, true);
        cp.setPen(Qt::NoPen);
        const QColor stroke = ThemeManager::mapped("#b6bec9");
        for (auto it = m_headerItem.constBegin(); it != m_headerItem.constEnd(); ++it) {
            const QRect r = visualItemRect(it.value());
            if (r.isEmpty())
                continue;
            drawChevron(cp, r, stroke, m_groupAngle.value(it.key()));
        }
    }

    void resizeEvent(QResizeEvent *event) override
    {
        QListWidget::resizeEvent(event);
        slideToCurrent(false);
    }

    // Intercepted rather than left to the normal click path: a group header
    // must toggle without ever becoming the view's "current" item (see the
    // class comment above for why), which means it has to be stopped before
    // QListWidget's own mousePressEvent gets to decide that for us.
    void mousePressEvent(QMouseEvent *event) override
    {
        if (event->button() == Qt::LeftButton) {
            if (QListWidgetItem *it = itemAt(event->pos())) {
                if (it->data(kPageRole).toInt() < 0) {
                    const QString group = it->data(kGroupRole).toString();
                    if (!group.isEmpty())
                        setGroupOpen(group, !m_groupOpen.value(group), true);
                    return;
                }
            }
        }
        QListWidget::mousePressEvent(event);
    }

private:
    void drawChevron(QPainter &p, const QRect &row, const QColor &color, qreal angle) const
    {
        // Right-pointing at rest, rotating to point down as the group opens.
        p.save();
        p.translate(QPointF(row.right() - 18.0, row.center().y()));
        p.rotate(angle);
        QPainterPath path;
        path.moveTo(-3.0, -4.5);
        path.lineTo(4.0, 0.0);
        path.lineTo(-3.0, 4.5);
        path.closeSubpath();
        p.setBrush(color);
        p.drawPath(path);
        p.restore();
    }

    void setGroupOpen(const QString &group, bool open, bool animate)
    {
        m_groupOpen[group] = open;
        for (QListWidgetItem *child : m_children.value(group))
            child->setHidden(!open);

        const qreal target = open ? 90.0 : 0.0;
        if (!animate) {
            m_groupAngle[group] = target;
            slideToCurrent(false);
            viewport()->update();
            return;
        }
        // A fresh animation per toggle rather than a shared one: two groups
        // can be mid-toggle at once, and each needs its own start value.
        auto *anim = new QVariantAnimation(this);
        anim->setDuration(190);
        anim->setEasingCurve(QEasingCurve::OutCubic);
        anim->setStartValue(m_groupAngle.value(group));
        anim->setEndValue(target);
        connect(anim, &QVariantAnimation::valueChanged, this,
                [this, group](const QVariant &v) {
                    m_groupAngle[group] = v.toReal();
                    viewport()->update();
                });
        connect(anim, &QVariantAnimation::finished, this,
                [this] { slideToCurrent(false); });
        anim->start(QAbstractAnimation::DeleteWhenStopped);
        slideToCurrent(false);
    }

    QVariantAnimation *m_anim = nullptr;
    qreal m_markTop = 0.0;
    int m_markHeight = 0;
    bool m_marked = false;
    QListWidgetItem *m_activePageItem = nullptr;

    QMap<QString, QListWidgetItem *> m_headerItem;
    QMap<QString, QVector<QListWidgetItem *>> m_children;
    QMap<QString, bool> m_groupOpen;
    QMap<QString, qreal> m_groupAngle;
};

// Both live in uiwidgets.cpp so the batch page presents the same setting the
// same way; kept as local names so the call sites below stay unqualified.
using TrajectaUi::makeFieldLabel;
using TrajectaUi::makeHelpDot;

// Large memory pages now live in largepages.h as a widget of their own, used
// in exactly one place — the Advanced settings dialog — with every run reading
// the setting through largePagesRequested() rather than through a widget of
// its own. See advancedsettingsdialog.h.

} // namespace

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(tr("Trajecta Studio"));
    resize(1240, 800);

    // The native title bar is replaced by the app's own top bar, which carries
    // the minimise/maximise/close buttons.
    //
    // Deliberately NOT Qt::FramelessWindowHint: that turns the window into a
    // WS_POPUP, and Windows only plays the minimise/restore animations for
    // WS_OVERLAPPEDWINDOW. The window therefore stays a perfectly ordinary
    // framed window — keeping the animations, Aero Snap, the drop shadow and
    // the taskbar-aware maximise — and the non-client area is removed instead
    // by answering WM_NCCALCSIZE in nativeEvent(). Dragging and edge resizing
    // are handed back to the window manager (startSystemMove, WM_NCHITTEST)
    // rather than reimplemented.
    setAttribute(Qt::WA_Hover, true);

    // Dropped files are accepted by the window as well as by the Viewer.
    //
    // Not redundant. Which widget a drag is offered to is decided by walking up
    // from whatever sits under the cursor until something accepts drops, and a
    // window that accepts nothing is the difference between that walk ending
    // somewhere useful and it ending nowhere. It also registers the drop site on
    // the native window at the moment the window is built, rather than as a
    // side effect of a page being constructed later.
    setAcceptDrops(true);

#ifdef Q_OS_WIN
    {
        const HWND hwnd = reinterpret_cast<HWND>(winId());

        // Qt still marks the window WS_POPUP, and a popup is exactly what the
        // shell refuses to animate on minimise/restore. Spelling out a plain
        // overlapped window puts the animations, and the taskbar's own
        // minimise/restore behaviour, back.
        LONG_PTR style = GetWindowLongPtr(hwnd, GWL_STYLE);
        style &= ~static_cast<LONG_PTR>(WS_POPUP);
        style |= WS_OVERLAPPEDWINDOW;
        SetWindowLongPtr(hwnd, GWL_STYLE, style);
        SetWindowPos(hwnd, nullptr, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
                         | SWP_FRAMECHANGED);

        // With the frame logically gone, this is what still draws the shadow
        // and the Windows 11 rounded outline around the window.
        const MARGINS shadow{ 0, 0, 0, 1 };
        DwmExtendFrameIntoClientArea(hwnd, &shadow);

        // Let a drag from File Explorer reach this window even when Trajecta is
        // running as administrator.
        //
        // Windows refuses messages sent from a lower integrity level to a
        // higher one (UIPI), and a drag is carried by messages: Explorer runs
        // as the user, an elevated Trajecta does not, and the drop is blocked
        // before any Qt code sees it. What the user gets is the "no entry"
        // cursor over the Viewer and no explanation anywhere.
        //
        // This is not a corner case here — the large-pages setup ends by
        // telling the user to start Trajecta as administrator, so the feature
        // that needs elevation would silently take drag & drop away. These
        // three messages are the documented exception list for exactly this
        // (ChangeWindowMessageFilterEx); WM_COPYGLOBALDATA is the one OLE uses
        // to hand over the dragged data, and has no name in the SDK headers.
        // On a process that is not elevated there is no barrier and the calls
        // do nothing.
        const UINT kCopyGlobalData = 0x0049;
        for (UINT message : { UINT(WM_DROPFILES), UINT(WM_COPYDATA), kCopyGlobalData })
            ChangeWindowMessageFilterEx(hwnd, message, MSGFLT_ALLOW, nullptr);
    }
#endif

    m_runner = new TrajectaRunner(this);

    auto *central = new QWidget(this);
    // Named so a theme can make it transparent and let a background image on
    // the window show through: the stylesheet gives every QWidget an opaque
    // background, which would otherwise hide it completely (see Washi).
    central->setObjectName(QStringLiteral("CentralArea"));
    auto *rootLayout = new QVBoxLayout(central);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    QWidget *topBar = buildTopBar();
    rootLayout->addWidget(topBar);

    m_pages = new QStackedWidget(central);
    // The run panel now lives at the bottom of the setup page, so there is no
    // separate "Processing" page any more — the setup page is it.
    m_pages->addWidget(buildSetupPage());
    m_pages->addWidget(buildPostPage());
    m_viewer = new ViewerPage(this);
    m_pages->addWidget(m_viewer);
    m_pages->addWidget(buildGuidePage());
    // About is the last page on purpose: leaving it out shifts no index, so
    // every showPage(0..3) in the walkthrough and in the CLI hooks keeps
    // meaning what it meant.
    if (kShowAboutTab)
        m_pages->addWidget(buildAboutPage());
    rootLayout->addWidget(m_pages, 1);

    rootLayout->addWidget(buildStatusBar());

    setCentralWidget(central);

    // The window may never get narrow enough to clip the brand or one of the
    // navigation tabs. The top bar's size hint is exactly the width it needs
    // with nothing compressed — its trailing stretch contributes zero — so it
    // is measured rather than hard-coded, and stays right when the font, the
    // screen DPI or a translation changes the width of the tab labels.
    // 1000 x 680 remains the floor for the pages below it.
    topBar->ensurePolished();
    setMinimumSize(qMax(1000, topBar->sizeHint().width()), 680);

    // Runner wiring. Everything goes through m_activeUi so the output of a
    // FETE/LCPA run lands on the "Run & results" panel and the output of an
    // NNI run lands on the "Post-processing" panel.
    connect(m_runner, &TrajectaRunner::consoleOutput, this, [this](const QString &raw) {
        m_activeUi->console->appendChunk(raw);
    });
    connect(m_runner, &TrajectaRunner::consoleErrorLine, this, [this](const QString &line) {
        const QString t = line.trimmed();
        const QColor color = t.startsWith(QLatin1String("ERROR"), Qt::CaseInsensitive)
                                 ? QColor(0xff, 0x6b, 0x6b)
                                 : QColor(0xff, 0xd1, 0x66);
        m_activeUi->console->appendMarker(line, color);
    });
    connect(m_runner, &TrajectaRunner::answerSent, this, [this](const QString &a) {
        m_activeUi->console->appendMarker(QStringLiteral("  ▸ %1").arg(a),
                                          QColor(0x8a, 0x97, 0xa5));
    });
    connect(m_runner, &TrajectaRunner::progressChanged, this, [this](double pct) {
        // The bar starts each run in busy mode (range 0,0); the first real
        // percentage from the engine switches it to determinate.
        if (m_activeUi->progress->maximum() == 0)
            m_activeUi->progress->setRange(0, 1000);
        m_activeUi->progress->setValue(int(pct * 10.0));
        m_runPercent = pct;
        refreshRunTicker();
    });
    connect(m_runner, &TrajectaRunner::statusChanged, this, [this](const QString &s) {
        m_activeUi->phase->setText(s);
    });
    connect(m_runner, &TrajectaRunner::finished,
            this, &MainWindow::onRunFinished);
    connect(m_runner, &TrajectaRunner::pauseStateChanged,
            this, &MainWindow::onPauseStateChanged);

    m_elapsedTimer = new QTimer(this);
    m_elapsedTimer->setInterval(1000);
    connect(m_elapsedTimer, &QTimer::timeout, this, [this] {
        m_activeUi->elapsed->setText(formatElapsed(m_elapsed.elapsed() - m_pausedMs));
        // The ticker's estimate is a function of time as well as of progress,
        // so it has to be recomputed on the clock and not only when the engine
        // reports — on the long preparation phases it reports nothing at all.
        refreshRunTicker();
    });

    // The additional/total cost surface names only make sense when cost
    // modifiers are enabled: keep them disabled otherwise.
    auto syncModifierNames = [this] {
        const bool on = m_modifiersGroup->isChecked();
        m_additionalNameLabel->setEnabled(on);
        m_additionalNameEdit->setEnabled(on);
        m_totalNameLabel->setEnabled(on);
        m_totalNameEdit->setEnabled(on);
    };
    connect(m_modifiersGroup, &QGroupBox::toggled, this, syncModifierNames);

    loadSettings();
    syncModifierNames();
    updateModeUi();
    updateEnvironmentStatus();
    refreshModeCardWidths();
    alignLabelColumns();
    switchPage(0);

    // Queued, not called here: the window has to be on screen before a modal
    // dialog is put in front of it, or the dialog appears over nothing.
    QTimer::singleShot(0, this, [this] {
        offerCrashRecovery();
        // Second, and only if the first had nothing to say: an unfinished
        // analysis is about the user's own work and comes first.
        offerWalkthroughOnFirstRun();
    });
}

// ---------------------------------------------------------------------------
// Top bar (brand + horizontal navigation tabs)
// ---------------------------------------------------------------------------

QWidget *MainWindow::buildTopBar()
{
    auto *bar = new QFrame(this);
    bar->setObjectName(QStringLiteral("TopBar"));
    bar->setFixedHeight(60);
    m_topBar = bar;
    // Drag-to-move and double-click-to-maximise, the two things the native
    // title bar used to provide.
    bar->installEventFilter(this);

    auto *layout = new QHBoxLayout(bar);
    layout->setContentsMargins(20, 0, 20, 0);
    layout->setSpacing(10);

    auto *logo = new QLabel(bar);
    QPixmap pm(QStringLiteral(":/assets/logo.png"));
    logo->setPixmap(pm.scaled(34, 34, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    logo->setAlignment(Qt::AlignVCenter);
    layout->addWidget(logo);

    auto *title = new QLabel(QStringLiteral("TRAJECTA STUDIO"), bar);
    title->setObjectName(QStringLiteral("TopBarTitle"));
    // The brand must never be squeezed. A QLabel's default policy lets a
    // layout shrink it below its sizeHint, and a theme with a wider face or
    // extra letter spacing (Washi, Neon Circuit) then clips it to
    // "TRAJECTA STU". Fixed keeps it at whatever width its text needs.
    title->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    layout->addWidget(title);

    // "v" only here: applicationVersion() also feeds the About page ("Version
    // 1.0.0") and the basemap User-Agent ("TrajectaStudio/1.0.0"), where the
    // prefix would read wrong.
    auto *version = new QLabel(QStringLiteral("v%1").arg(QApplication::applicationVersion()),
                               bar);
    version->setObjectName(QStringLiteral("TopBarVersion"));
    layout->addWidget(version);

    layout->addSpacing(24);

    QStringList navNames = {tr("Processing"), tr("Post-processing"),
                            tr("Viewer"), tr("Guide")};
    if (kShowAboutTab)
        navNames << tr("About");
    auto *group = new QButtonGroup(bar);
    group->setExclusive(true);
    for (int i = 0; i < navNames.size(); ++i) {
        auto *btn = new QPushButton(navNames.at(i), bar);
        btn->setObjectName(QStringLiteral("TabButton"));
        btn->setCheckable(true);
        btn->setCursor(Qt::PointingHandCursor);
        group->addButton(btn, i);
        layout->addWidget(btn);
        m_navButtons.append(btn);
    }
    connect(group, &QButtonGroup::idClicked, this, [this](int id) {
        m_pages->setCurrentIndex(id);
    });

    layout->addStretch(1);

    // ----- Appearance (gear) -----
    m_gearButton = new QToolButton(bar);
    m_gearButton->setObjectName(QStringLiteral("GearButton"));
    m_gearButton->setCursor(Qt::PointingHandCursor);
    m_gearButton->setIconSize(QSize(20, 20));
    m_gearButton->setIcon(makeGearIcon(ThemeManager::mapped("#99a1ac"), 20));
    m_gearButton->setToolTip(tr("Appearance"));
    m_gearButton->setPopupMode(QToolButton::InstantPopup);

    auto *menu = new QMenu(m_gearButton);
    auto *header = menu->addAction(tr("UI Theme"));
    header->setEnabled(false);
    menu->addSeparator();
    auto *themeGroup = new QActionGroup(menu);
    themeGroup->setExclusive(true);
    const QVector<ThemeManager::Theme> &themes = ThemeManager::themes();
    for (int i = 0; i < themes.size(); ++i) {
        QAction *action = menu->addAction(themes.at(i).name);
        action->setCheckable(true);
        action->setChecked(i == ThemeManager::current());
        themeGroup->addAction(action);
        m_themeActions.append(action);
        connect(action, &QAction::triggered, this, [this, i] { applyTheme(i); });
    }

    // ----- UI font, chosen independently of the palette -----
    menu->addSeparator();
    auto *fontHeader = menu->addAction(tr("UI Font"));
    fontHeader->setEnabled(false);
    menu->addSeparator();
    auto *fontGroup = new QActionGroup(menu);
    fontGroup->setExclusive(true);
    const QVector<ThemeManager::FontChoice> &uiFonts = ThemeManager::fonts();
    for (int i = 0; i < uiFonts.size(); ++i) {
        QAction *action = menu->addAction(uiFonts.at(i).name);
        action->setCheckable(true);
        action->setChecked(i == ThemeManager::currentFont());
        // Draw each entry in the font it selects, so the choice is visible
        // before making it.
        if (!uiFonts.at(i).family.isEmpty())
            action->setFont(QFont(uiFonts.at(i).family));
        fontGroup->addAction(action);
        m_fontActions.append(action);
        connect(action, &QAction::triggered, this, [this, i] { applyUiFont(i); });
    }

    // ----- Automatic saving of a running analysis -----
    menu->addSeparator();
    auto *saveHeader = menu->addAction(tr("Automatic saving"));
    saveHeader->setEnabled(false);
    menu->addSeparator();

    const Checkpoint::Settings cp = Checkpoint::settings();
    m_autosaveAction = menu->addAction(tr("Auto-save"));
    m_autosaveAction->setCheckable(true);
    m_autosaveAction->setChecked(cp.enabled);
    m_autosaveAction->setToolTip(
        tr("Writes the state of a FETE run to disk so it can be resumed after a "
           "crash, a power cut or a deliberate shutdown. On by default, every "
           "30 minutes: the writing costs seconds, and losing a run that has "
           "been going for days costs days."));
    connect(m_autosaveAction, &QAction::toggled, this, [this](bool on) {
        Checkpoint::Settings s = Checkpoint::settings();
        s.enabled = on;
        Checkpoint::setSettings(s);
        refreshAutosaveMenu();
    });

    // The interval, as a handful of choices rather than a spin box: a menu
    // cannot hold one, and these are the intervals anyone actually wants.
    m_autosaveIntervalMenu = menu->addMenu(tr("Save every"));
    auto *intervalGroup = new QActionGroup(m_autosaveIntervalMenu);
    intervalGroup->setExclusive(true);
    for (int minutes : {5, 15, 30, 60, 120}) {
        QAction *a = m_autosaveIntervalMenu->addAction(
            minutes < 60 ? tr("%1 minutes").arg(minutes)
                         : tr("%n hour(s)", nullptr, minutes / 60));
        a->setCheckable(true);
        a->setChecked(minutes == cp.minutes);
        a->setData(minutes);
        intervalGroup->addAction(a);
        m_autosaveIntervalActions.append(a);
        connect(a, &QAction::triggered, this, [this, minutes] {
            Checkpoint::Settings s = Checkpoint::settings();
            s.minutes = minutes;
            Checkpoint::setSettings(s);
            refreshAutosaveMenu();
        });
    }

    m_autosaveFolderAction = menu->addAction(tr("Select auto-save folder…"));
    connect(m_autosaveFolderAction, &QAction::triggered,
            this, &MainWindow::chooseAutosaveFolder);
    refreshAutosaveMenu();

    // ----- Advanced settings, below auto-saving -----
    // No disabled heading of its own, unlike the three groups above: a single
    // entry does not need a caption to say what it is, and leaving it out
    // keeps it inside the auto-save group as far as the walkthrough's picture
    // of this menu is concerned (see renderMenuPicture()). Kept as a member
    // regardless, so that same walkthrough step can still give it a caption
    // of its own — see "2 — the gear" in buildWalkthrough().
    menu->addSeparator();
    m_advancedSettingsAction = menu->addAction(tr("Advanced settings…"));
    connect(m_advancedSettingsAction, &QAction::triggered, this, &MainWindow::showAdvancedSettings);

    m_gearButton->setMenu(menu);
    layout->addWidget(m_gearButton);

    // ----- Window controls, in place of the native title bar -----
    layout->addSpacing(10);
    const QColor glyph = ThemeManager::mapped("#99a1ac");

    m_minButton = new QToolButton(bar);
    m_minButton->setObjectName(QStringLiteral("WindowButton"));
    m_minButton->setIcon(makeWindowIcon(WindowGlyph::Minimise, glyph, 20));
    m_minButton->setToolTip(tr("Minimise"));
    connect(m_minButton, &QToolButton::clicked, this, &QWidget::showMinimized);

    m_maxButton = new QToolButton(bar);
    m_maxButton->setObjectName(QStringLiteral("WindowButton"));
    m_maxButton->setToolTip(tr("Maximise"));
    connect(m_maxButton, &QToolButton::clicked, this, &MainWindow::toggleMaximised);

    m_closeButton = new QToolButton(bar);
    m_closeButton->setObjectName(QStringLiteral("WindowCloseButton"));
    m_closeButton->setIcon(makeWindowIcon(WindowGlyph::Close, glyph, 20));
    m_closeButton->setToolTip(tr("Close"));
    connect(m_closeButton, &QToolButton::clicked, this, &QWidget::close);

    for (QToolButton *b : { m_minButton, m_maxButton, m_closeButton }) {
        b->setIconSize(QSize(20, 20));
        b->setCursor(Qt::ArrowCursor);
        b->setFocusPolicy(Qt::NoFocus);
        layout->addWidget(b);
    }
    refreshWindowButtons();

    return bar;
}

void MainWindow::toggleMaximised()
{
    if (isMaximized())
        showNormal();
    else
        showMaximized();
}

// The maximise button doubles as restore, and its glyph has to say which.
void MainWindow::refreshWindowButtons()
{
    if (!m_maxButton)
        return;
    const QColor glyph = ThemeManager::mapped("#99a1ac");
    const bool max = isMaximized();
    m_maxButton->setIcon(makeWindowIcon(
        max ? WindowGlyph::Restore : WindowGlyph::Maximise, glyph, 20));
    m_maxButton->setToolTip(max ? tr("Restore") : tr("Maximise"));
    if (m_minButton)
        m_minButton->setIcon(makeWindowIcon(WindowGlyph::Minimise, glyph, 20));
    if (m_closeButton)
        m_closeButton->setIcon(makeWindowIcon(WindowGlyph::Close, glyph, 20));
}

#ifdef Q_OS_WIN
bool MainWindow::nativeEvent(const QByteArray &eventType, void *message,
                             qintptr *result)
{
    if (eventType != "windows_generic_MSG")
        return QMainWindow::nativeEvent(eventType, message, result);

    auto *msg = static_cast<MSG *>(message);
    if (!msg)
        return QMainWindow::nativeEvent(eventType, message, result);

    // Collapse the non-client area into the client area: the window keeps its
    // real frame (hence the animations and snap) but stops drawing a title bar
    // and borders over it.
    if (msg->message == WM_NCCALCSIZE && msg->wParam) {
        auto *params = reinterpret_cast<NCCALCSIZE_PARAMS *>(msg->lParam);
        if (IsZoomed(msg->hwnd)) {
            // Maximised, Windows oversizes the window by the frame on every
            // side. Left as is, the content would bleed off-screen and cover
            // the taskbar, so the frame is trimmed back off here.
            const int padded = GetSystemMetrics(SM_CXPADDEDBORDER);
            const int cx = GetSystemMetrics(SM_CXSIZEFRAME) + padded;
            const int cy = GetSystemMetrics(SM_CYSIZEFRAME) + padded;
            params->rgrc[0].left += cx;
            params->rgrc[0].right -= cx;
            params->rgrc[0].top += cy;
            params->rgrc[0].bottom -= cy;
        }
        *result = 0;
        return true;
    }

    if (msg->message != WM_NCHITTEST)
        return QMainWindow::nativeEvent(eventType, message, result);

    // A maximised or full-screen window has no edges to drag.
    if (isMaximized() || isFullScreen())
        return false;

    // lParam is in physical pixels; the widget geometry is in logical ones.
    const qreal dpr = devicePixelRatioF();
    const QPoint globalPhys(GET_X_LPARAM(msg->lParam), GET_Y_LPARAM(msg->lParam));
    const QPoint local = mapFromGlobal(
        QPoint(qRound(globalPhys.x() / dpr), qRound(globalPhys.y() / dpr)));

    const int border = 6;
    const bool left = local.x() >= 0 && local.x() < border;
    const bool right = local.x() < width() && local.x() >= width() - border;
    const bool top = local.y() >= 0 && local.y() < border;
    const bool bottom = local.y() < height() && local.y() >= height() - border;

    LRESULT hit = 0;
    if (top && left)          hit = HTTOPLEFT;
    else if (top && right)    hit = HTTOPRIGHT;
    else if (bottom && left)  hit = HTBOTTOMLEFT;
    else if (bottom && right) hit = HTBOTTOMRIGHT;
    else if (left)            hit = HTLEFT;
    else if (right)           hit = HTRIGHT;
    else if (top)             hit = HTTOP;
    else if (bottom)          hit = HTBOTTOM;
    else                      return false;   // not an edge: let Qt have it

    *result = hit;
    return true;
}
#endif

void MainWindow::changeEvent(QEvent *event)
{
    QMainWindow::changeEvent(event);
    if (event->type() == QEvent::WindowStateChange)
        refreshWindowButtons();
}

bool MainWindow::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == m_topBar) {
        // Only the bar's own background drags the window: presses that landed
        // on a tab, the gear or a window button are delivered to those widgets
        // and never reach here.
        if (event->type() == QEvent::MouseButtonPress) {
            auto *me = static_cast<QMouseEvent *>(event);
            if (me->button() == Qt::LeftButton && windowHandle()) {
                windowHandle()->startSystemMove();   // keeps Aero Snap
                return true;
            }
        } else if (event->type() == QEvent::MouseButtonDblClick) {
            auto *me = static_cast<QMouseEvent *>(event);
            if (me->button() == Qt::LeftButton) {
                toggleMaximised();
                return true;
            }
        }
    }
    return QMainWindow::eventFilter(watched, event);
}


void MainWindow::applyTheme(int index)
{
    ThemeManager::apply(index);

    for (int i = 0; i < m_themeActions.size(); ++i)
        m_themeActions.at(i)->setChecked(i == index);
    if (m_gearButton)
        m_gearButton->setIcon(
            makeGearIcon(ThemeManager::mapped("#99a1ac"), 20));
    if (m_githubButton)
        m_githubButton->setIcon(makeGithubIcon(ThemeManager::mapped("#e4e7ec"), 20));
    refreshWindowButtons();
    // A palette may bring its own typeface, and the mode cards are sized from
    // the width of their captions in it.
    refreshModeCardWidths();
    alignLabelColumns();

    // Everything painted outside the stylesheet has to be told.
    updateEnvironmentStatus();
    if (m_viewer)
        m_viewer->applyTheme();
    if (m_runUi.console)
        m_runUi.console->applyTheme();
    if (m_postUi.console)
        m_postUi.console->applyTheme();
    if (m_cmpConsole)
        m_cmpConsole->applyTheme();
    if (m_batchPage)
        m_batchPage->applyTheme();
    if (m_postBatchPage)
        m_postBatchPage->applyTheme();
    // Every page of the guide, not just the one on screen: a theme change
    // re-dyes the whole application at once, and a page that kept the old
    // colours until it was next opened would be a visible seam. Each browser
    // is only ever a GuideBrowser, but the type is file-local so it cannot be
    // named in the header and carries no Q_OBJECT for qobject_cast to key on
    // — hence findChildren of the base type and a static_cast.
    if (m_guidePages) {
        const QList<QTextBrowser *> browsers =
            m_guidePages->findChildren<QTextBrowser *>();
        for (QTextBrowser *b : browsers) {
            auto *guide = static_cast<GuideBrowser *>(b);
            if (guide->isReady())
                guide->relayout(false, true);
        }
    }
    if (m_guideNav)
        m_guideNav->viewport()->update();
}

void MainWindow::applyUiFont(int index)
{
    // The font lives in the stylesheet, so changing it means rebuilding the
    // sheet for the theme currently in use — which is exactly what applyTheme
    // does, including repainting everything drawn outside the stylesheet.
    ThemeManager::setFont(index);
    for (int i = 0; i < m_fontActions.size(); ++i)
        m_fontActions.at(i)->setChecked(i == index);
    applyTheme(ThemeManager::current());
}

void MainWindow::showAdvancedSettings()
{
    // Built once and kept, not recreated on every open: a QDialog's own child
    // widgets — which sidebar row was last selected — are then still there
    // the next time the gear menu asks for it.
    if (!m_advancedSettings)
        m_advancedSettings = new AdvancedSettingsDialog(this);
    TrajectaUi::centreOnScreen(*m_advancedSettings, this);
    m_advancedSettings->exec();
}

// ---------------------------------------------------------------------------
// Bottom status bar (environment indicators + locate actions)
// ---------------------------------------------------------------------------

QWidget *MainWindow::buildStatusBar()
{
    auto *bar = new QFrame(this);
    bar->setObjectName(QStringLiteral("StatusBar"));
    bar->setFixedHeight(36);
    // Kept for the walkthrough, which lights the whole strip. Lighting the four
    // controls instead left whatever was added later — the two "?" badges —
    // outside the light, which is the kind of thing nobody remembers to update.
    m_statusBar = bar;

    auto *layout = new QHBoxLayout(bar);
    layout->setContentsMargins(20, 0, 20, 0);
    layout->setSpacing(10);

    m_engineStatus = new QLabel(bar);
    m_engineStatus->setObjectName(QStringLiteral("EnvStatus"));
    layout->addWidget(m_engineStatus);

    auto *divider = new QLabel(QStringLiteral("•"), bar);
    divider->setObjectName(QStringLiteral("StatusDivider"));
    layout->addWidget(divider);

    m_gdalStatus = new QLabel(bar);
    m_gdalStatus->setObjectName(QStringLiteral("EnvStatus"));
    layout->addWidget(m_gdalStatus);

    // Deliberately not in this layout: it centres itself on the bar instead, so
    // it sits in the middle of the window and not in the middle of whatever
    // space the four controls happen to leave. It is hidden until something is
    // actually running.
    m_ticker = new RunTicker(bar);

    layout->addStretch(1);

    // Both carry a "?" of their own. These two are the only controls in the
    // application that are about Trajecta's own installation rather than about
    // an analysis, and on a machine where the indicators are green there is
    // nothing on screen to say what they would be for.
    auto *locateEngineBtn = new QPushButton(tr("Locate engine..."), bar);
    m_locateEngineButton = locateEngineBtn;
    locateEngineBtn->setObjectName(QStringLiteral("LinkButton"));
    locateEngineBtn->setCursor(Qt::PointingHandCursor);
    connect(locateEngineBtn, &QPushButton::clicked, this, &MainWindow::locateEngine);
    layout->addWidget(locateEngineBtn);
    layout->addWidget(makeHelpDot(
        tr("Points Trajecta Studio at the computing engine — trajecta.exe, the "
           "program that does the actual work while this window watches it.\n\n"
           "You do not normally need this: the installer puts the engine next "
           "to the interface and it is found on its own. Use it when the "
           "indicator on the left says the engine is missing, which happens if "
           "you are running the interface from a folder of its own, if the file "
           "was moved or removed — an antivirus quarantine is the usual "
           "reason — or if you keep several builds and want this window to "
           "drive a particular one.\n\n"
           "The choice is remembered, so it has to be made once."),
        bar));

    auto *locateGdalBtn = new QPushButton(tr("Locate GDAL folder..."), bar);
    m_locateGdalButton = locateGdalBtn;
    locateGdalBtn->setObjectName(QStringLiteral("LinkButton"));
    locateGdalBtn->setCursor(Qt::PointingHandCursor);
    connect(locateGdalBtn, &QPushButton::clicked, this, &MainWindow::locateGdal);
    layout->addWidget(locateGdalBtn);
    layout->addWidget(makeHelpDot(
        tr("Points Trajecta Studio at GDAL, the library that reads and writes "
           "geospatial files. Without it no DEM can be opened, the Viewer stays "
           "empty and no result can be written.\n\n"
           "As with the engine, the installed copy is found on its own. Use "
           "this when the indicator says GDAL is missing, or when you would "
           "rather Trajecta used the GDAL that comes with an installation you "
           "already have — QGIS or OSGeo4W.\n\n"
           "What is asked for is the folder holding the DLLs, usually the "
           "<b>bin</b> folder of that installation: it must contain files "
           "named gdal*.dll, and a folder without them is refused rather than "
           "accepted and found wanting later."),
        bar));

    return bar;
}

void MainWindow::switchPage(int index)
{
    // Arriving at the Guide from another tab puts it back on its Overview.
    // A guide is a place you come to with a question, not a document you were
    // half way through: being dropped back where you last stopped reading is
    // disorienting, and the page you actually want is one click from home.
    // Only on arrival — moving around inside the Guide leaves it alone.
    const bool arriving = m_pages && m_pages->currentIndex() != index;
    if (arriving && index == kGuidePageIndex && m_guideNav)
        showGuideSection(0);

    m_pages->setCurrentIndex(index);
    if (index >= 0 && index < m_navButtons.size())
        m_navButtons.at(index)->setChecked(true);
}

// ---------------------------------------------------------------------------
// Cards helper
// ---------------------------------------------------------------------------

QWidget *MainWindow::makeCard(const QString &title, const QString &subtitle, QWidget *content,
                              const QString &titleHelp)
{
    auto *card = new QFrame(this);
    card->setObjectName(QStringLiteral("Card"));

    auto *layout = new QVBoxLayout(card);
    layout->setContentsMargins(18, 14, 18, 16);
    layout->setSpacing(6);

    auto *titleLabel = new QLabel(title, card);
    titleLabel->setObjectName(QStringLiteral("CardTitle"));
    if (titleHelp.isEmpty())
        layout->addWidget(titleLabel);
    else
        layout->addWidget(TrajectaUi::withHelpDot(titleLabel, titleHelp));

    if (!subtitle.isEmpty()) {
        auto *subtitleLabel = new QLabel(subtitle, card);
        subtitleLabel->setObjectName(QStringLiteral("CardSubtitle"));
        subtitleLabel->setWordWrap(true);
        layout->addWidget(subtitleLabel);
    }

    layout->addSpacing(4);
    layout->addWidget(content);
    return card;
}

// ---------------------------------------------------------------------------
// Setup page
// ---------------------------------------------------------------------------

QWidget *MainWindow::buildSetupPage()
{
    auto *page = new QWidget(this);
    auto *pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(0, 0, 0, 0);

    auto *scroll = new QScrollArea(page);
    m_setupScroll = scroll;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);

    auto *inner = new QWidget(scroll);
    auto *layout = new QVBoxLayout(inner);
    layout->setContentsMargins(28, 24, 28, 24);
    layout->setSpacing(14);

    // ----- Analysis type -----
    // The very first choice, and the one everything below answers to: one
    // analysis, or a batch of them. Single analysis shows the tool card right
    // under this one; Batch processing replaces everything below this card
    // with the batch page. Kept apart from which tool runs (FETE or LCPA)
    // because they are different questions — asking both on one card, with
    // Batch as a third option beside FETE and LCPA, was confusing once the
    // batch page itself grew hardware, chunks and a run bar of its own: the
    // page read as one long form with an unrelated third path through it.
    {
        auto *content = new QWidget(inner);
        auto *row = new QHBoxLayout(content);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(12);

        m_modeSingle = new QPushButton(
            tr("Single analysis\n"
               "Set up and run one FETE or LCPA computation"),
            content);
        m_modeSingle->setToolTip(
            tr("The tool card below chooses FETE or LCPA; everything else on "
               "the page is that one analysis."));
        m_modeBatch = new QPushButton(
            tr("Batch processing\n"
               "Runs many analyses in a row, unattended"),
            content);
        m_modeBatch->setToolTip(
            tr("Queues several FETE or LCPA runs and executes them one after "
               "another: one row per analysis, grouped into chunks that share an "
               "algorithm and a set of cost modifiers."));
        m_modeSingle->setProperty("mode", QStringLiteral("single"));
        m_modeBatch->setProperty("mode", QStringLiteral("batch"));
        for (QPushButton *b : {m_modeSingle, m_modeBatch}) {
            b->setObjectName(QStringLiteral("ModeCard"));
            b->setCheckable(true);
            b->setCursor(Qt::PointingHandCursor);
            b->setMinimumHeight(72);
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            row->addWidget(b, 1);
        }
        auto *typeGroup = new QButtonGroup(content);
        typeGroup->setExclusive(true);
        typeGroup->addButton(m_modeSingle);
        typeGroup->addButton(m_modeBatch);
        m_modeSingle->setChecked(true);
        connect(typeGroup, &QButtonGroup::buttonClicked, this,
                [this](QAbstractButton *) { updateModeUi(); });

        m_cardAnalysisType = makeCard(
            tr("Analysis type"),
            tr("Choose whether to run one analysis, or queue several as a batch."),
            content);
        layout->addWidget(m_cardAnalysisType);
    }

    // ----- Processing tool: mode, then the hardware it runs with -----
    // One card now, not two: the tool is chosen and the hardware it runs on
    // is set in the same place, the way the batch card on this page already
    // asks both questions together.
    {
        auto *content = new QWidget(inner);
        auto *cardLayout = new QVBoxLayout(content);
        cardLayout->setContentsMargins(0, 0, 0, 0);
        // Same 12 as the equivalent card on the Batch processing page (see
        // batchpage.cpp): the gap between the mode row and the hardware row
        // below it is otherwise the one visible difference between the two.
        cardLayout->setSpacing(12);

        auto *modeRow = new QWidget(content);
        auto *row = new QHBoxLayout(modeRow);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(12);

        m_modeFete = new QPushButton(
            tr("FETE — From Everywhere To Everywhere\n"
               "Models general mobility across the landscape"),
            content);
        m_modeFete->setToolTip(
            tr("Computes least-cost paths between every pair of sample points and "
               "accumulates them into a path-density raster: natural movement "
               "corridors and accessibility patterns."));
        m_modeLcpa = new QPushButton(
            tr("LCPA — Least-Cost Path Analysis\n"
               "Computes optimal routes from origin to destination(s)"),
            content);
        m_modeLcpa->setToolTip(
            tr("Computes the optimal routes from a single origin point to one or "
               "more destinations: paths raster and polyline shapefile."));
        // Each card carries its own name for the stylesheet: selected, the
        // two fill with the same colour, so what says which analysis is in
        // front of you is which card is filled, not which shade it wears —
        // see theme.qss for how to bring the per-mode colours back.
        m_modeFete->setProperty("mode", QStringLiteral("fete"));
        m_modeLcpa->setProperty("mode", QStringLiteral("lcpa"));
        for (QPushButton *b : {m_modeFete, m_modeLcpa}) {
            b->setObjectName(QStringLiteral("ModeCard"));
            b->setCheckable(true);
            b->setCursor(Qt::PointingHandCursor);
            b->setMinimumHeight(72);
            // The width comes from the text, in refreshModeCardWidths() — it
            // has to be redone whenever the theme or the UI font changes, and
            // the stylesheet has not been applied to these buttons yet.
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            row->addWidget(b, 1);
        }
        auto *modeGroup = new QButtonGroup(content);
        modeGroup->setExclusive(true);
        modeGroup->addButton(m_modeFete);
        modeGroup->addButton(m_modeLcpa);
        m_modeFete->setChecked(true);
        connect(modeGroup, &QButtonGroup::buttonClicked, this,
                [this](QAbstractButton *) { updateModeUi(); });
        cardLayout->addWidget(modeRow);

        // ----- Hardware resources, folded into this same card -----
        // Threads and RAM in one horizontal row — the same disposition as the
        // hardware row on the Batch processing card (see batchpage.cpp), so
        // the two fields look and behave the same wherever they appear.
        auto *hw = new QWidget(content);
        auto *hwRow = new QHBoxLayout(hw);
        hwRow->setContentsMargins(0, 0, 0, 0);
        hwRow->setSpacing(10);

        const int maxThreads = qMax(1, QThread::idealThreadCount());
        const int recommendedThreads = qMax(1, maxThreads - 4);
        const qint64 totalRam = SystemInfo::totalRamMb();
        // Never a share of what is installed: the engine needs the same modest
        // amount on every machine. On a small one the ceiling cannot exceed
        // what is there, hence the qMin.
        const int recommendedRam =
            int(qMin<qint64>(SystemInfo::kRecommendedRamMb, totalRam));

        hwRow->addWidget(makeFieldLabel(
            tr("CPU threads"),
            tr("Number of parallel CPU threads used for the computation. "
               "Keeping a few cores free preserves system responsiveness."),
            hw));
        m_threadsSpin = new QSpinBox(hw);
        m_threadsSpin->setRange(1, maxThreads);
        m_threadsSpin->setValue(recommendedThreads);
        m_threadsSpin->setMinimumWidth(batchSpinBoxWidth(1, 1024));
        hwRow->addWidget(m_threadsSpin);

        hwRow->addWidget(makeFieldLabel(
            tr("Maximum RAM"),
            tr("Memory ceiling used for raster processing. The analysis needs "
               "far less than most machines have: at least %1 MB of RAM is "
               "recommended, and raising the ceiling further does not make "
               "the computation any faster.")
                   .arg(SystemInfo::kRecommendedRamMb)
               + TrajectaUi::ramHeadroomNote(),
            hw));
        m_ramSpin = new QSpinBox(hw);
        m_ramSpin->setRange(512, int(totalRam));
        m_ramSpin->setSingleStep(512);
        m_ramSpin->setSuffix(QStringLiteral(" MB"));
        m_ramSpin->setValue(recommendedRam);
        m_ramSpin->setMinimumWidth(batchSpinBoxWidth(256, 1024 * 1024, QStringLiteral(" MB")));
        hwRow->addWidget(m_ramSpin);
        auto *ramHint = new QLabel(
            tr("at least %1 MB of RAM is recommended")
                .arg(SystemInfo::kRecommendedRamMb),
            hw);
        ramHint->setObjectName(QStringLiteral("HintLabel"));
        hwRow->addWidget(ramHint);
        hwRow->addStretch(1);
        cardLayout->addWidget(hw);

        auto *optionsRow = new QWidget(content);
        auto *optionsLayout = new QHBoxLayout(optionsRow);
        optionsLayout->setContentsMargins(0, 0, 0, 0);
        optionsLayout->setSpacing(24);

        m_verboseCheck = new QCheckBox(
            tr("Detailed debug output (verbose console log)"), optionsRow);
        optionsLayout->addWidget(TrajectaUi::withHelpDot(
            m_verboseCheck,
            tr("Prints detailed diagnostic messages in the console log. "
               "Useful for troubleshooting and bug reports.")));

        // On by default, unlike the debug log: this one is not for diagnosing
        // Trajecta, it is for being able to say later what produced a result.
        m_manifestCheck = new QCheckBox(
            tr("Write a run manifest next to the results"), optionsRow);
        m_manifestCheck->setChecked(true);
        optionsLayout->addWidget(
            TrajectaUi::withHelpDot(m_manifestCheck, TrajectaUi::manifestHelpText()));
        optionsLayout->addStretch(1);
        cardLayout->addWidget(optionsRow);

        m_cardMode = makeCard(tr("Tool selection"),
                              tr("Choose the analysis tool to use."), content);
        layout->addWidget(m_cardMode);
    }

    // ----- Input data -----
    {
        auto *content = new QWidget(inner);
        auto *grid = new QGridLayout(content);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);
        m_labelColumns.append({grid, 0});

        int r = 0;
        auto addRow = [&](const QString &label, const QString &help, QWidget *w) -> QWidget * {
            QWidget *l = makeFieldLabel(label, help, content);
            grid->addWidget(l, r, 0);
            grid->addWidget(w, r, 1);
            ++r;
            return l;
        };

        m_demPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                     tr("Select the DEM (GeoTIFF)"),
                                     QString::fromLatin1(kRasterFilter), content);
        m_demPicker->setPlaceholder(tr("Digital Elevation Model (.tif), georeferenced"));
        addRow(tr("DEM raster"),
               tr("DEM in GeoTIFF format. Must be "
                  "georeferenced with a defined coordinate reference system; "
                  "terrain slope and movement costs are derived from it."),
               m_demPicker);

        m_pointsSourceCombo = new SmoothComboBox(content);
        m_pointsSourceCombo->addItem(tr("Import from a file"), 0);
        m_pointsSourceCombo->addItem(tr("Generate from the DEM"), 1);
        m_pointsSourceLabel =
            addRow(tr("Sample points source"),
                   tr("Either use your own point layer, or let Trajecta build one "
                      "from the DEM. A generated layer is written to the output "
                      "folder as a shapefile and then used as the input of the "
                      "analysis, so the exact input is saved on disk."),
                   m_pointsSourceCombo);

        m_pointsPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                        tr("Select the sample points file"),
                                        QString::fromLatin1(kVectorFilter), content);
        m_pointsPicker->setPlaceholder(tr("Point locations (.shp, .geojson, .kml, .gml, .csv)"));
        m_pointsLabel = addRow(tr("Sample points"),
                               tr("Known locations (e.g. sites, findspots). FETE computes "
                                  "least-cost paths between every pair of these points and "
                                  "accumulates them into the density raster."),
                               m_pointsPicker);

        // ----- Generated sample points -----
        // Spans both grid columns: while it is hidden the layout collapses and
        // the card looks exactly like it did before this option existed.
        {
            m_generateGroup = new QGroupBox(tr("Generated sample points"), content);
            auto *gGrid = new QGridLayout(m_generateGroup);
            gGrid->setContentsMargins(8, 28, 8, 10);
            gGrid->setHorizontalSpacing(12);
            gGrid->setVerticalSpacing(14);
            gGrid->setColumnStretch(1, 1);
            m_labelColumns.append({gGrid, 0});

            int gr = 0;
            auto addGenRow = [&](const QString &label, QWidget *w,
                                 const QString &tip) -> QWidget * {
                QWidget *l = makeFieldLabel(label, tip, m_generateGroup);
                gGrid->addWidget(l, gr, 0);
                gGrid->addWidget(w, gr, 1);
                ++gr;
                return l;
            };

            m_genDensityCombo = new SmoothComboBox(m_generateGroup);
            m_genDensityCombo->addItem(tr("Point spacing (one point every N cells)"), 0);
            m_genDensityCombo->addItem(tr("Target number of points"), 1);
            // Stating how many points you want is the question people actually
            // have; the spacing is an implementation detail of the grid.
            m_genDensityCombo->setCurrentIndex(1);
            addGenRow(tr("Density given as"), m_genDensityCombo,
                      tr("Either state the spacing directly, or state how many points "
                         "you want and let Trajecta derive the spacing from the number "
                         "of usable DEM cells."));

            m_genSpacingSpin = new QSpinBox(m_generateGroup);
            m_genSpacingSpin->setRange(1, 100000);
            m_genSpacingSpin->setValue(10);
            m_genSpacingSpin->setSuffix(tr(" cell(s)"));
            m_genSpacingLabel =
                addGenRow(tr("Point spacing"), m_genSpacingSpin,
                          tr("One point every N rows and every N columns, so the count "
                             "falls with the square of N. 1 means one point per DEM "
                             "cell, which is only realistic on very small rasters due "
                             "to large processing time."));

            m_genTargetSpin = new QSpinBox(m_generateGroup);
            m_genTargetSpin->setRange(2, 100000000);
            m_genTargetSpin->setValue(5000);
            m_genTargetSpin->setSingleStep(500);
            m_genTargetSpin->setSuffix(tr(" points"));
            m_genTargetLabel =
                addGenRow(tr("Target points"), m_genTargetSpin,
                          tr("How many points you would like. The spacing is rounded to "
                             "a whole number of cells, so the actual count lands near "
                             "the target rather than exactly on it."));

            m_genArrangementCombo = new SmoothComboBox(m_generateGroup);
            m_genArrangementCombo->addItem(tr("Regular grid"), 0);
            m_genArrangementCombo->addItem(tr("Stratified random"), 1);
            addGenRow(tr("Arrangement"), m_genArrangementCombo,
                      tr("A regular grid places every point at the same offset inside "
                         "its block. Stratified random picks one random cell per block "
                         "instead, which keeps the same density but removes the "
                         "regularity that a grid imposes on the result."));

            m_genSeedSpin = new QSpinBox(m_generateGroup);
            m_genSeedSpin->setRange(0, 2147483647);
            m_genSeedSpin->setValue(1);
            m_genSeedLabel =
                addGenRow(tr("Random seed"), m_genSeedSpin,
                          tr("The same seed always produces the same points, so a run "
                             "can be reproduced exactly."));

            m_genEdgeSpin = new QSpinBox(m_generateGroup);
            m_genEdgeSpin->setRange(0, 100000);
            m_genEdgeSpin->setValue(0);
            m_genEdgeSpin->setSuffix(tr(" cell(s)"));
            addGenRow(tr("Edge buffer"), m_genEdgeSpin,
                      tr("Keeps points this far from the DEM border. A source near the "
                         "edge has fewer directions to move in, and any route that would "
                         "leave the raster is invisible to the analysis, so the density "
                         "there is biased inwards. The connectivity radius is only 1-3 "
                         "cells; a margin that meaningfully reduces the edge bias has to "
                         "be much wider, and costs you the band it excludes."));

            m_genNameEdit = new QLineEdit(QStringLiteral("sample_points"), m_generateGroup);
            addGenRow(tr("Layer name"), m_genNameEdit,
                      tr("Name of the shapefile written into the output folder, without "
                         "extension. It is created before the analysis starts and is "
                         "what the analysis reads."));

            m_genPreviewLabel = new QLabel(m_generateGroup);
            m_genPreviewLabel->setObjectName(QStringLiteral("HintLabel"));
            m_genPreviewLabel->setWordWrap(true);
            gGrid->addWidget(m_genPreviewLabel, gr, 1);
            ++gr;

            // Write the layer now and look at it in the Viewer, before
            // committing to a full analysis. The run afterwards consumes this
            // very file (see generationKey()), so what was inspected is what
            // gets processed.
            auto *genRow = new QWidget(m_generateGroup);
            auto *genRowLayout = new QHBoxLayout(genRow);
            genRowLayout->setContentsMargins(0, 4, 0, 0);
            genRowLayout->setSpacing(12);
            m_genPointsButton = new QPushButton(tr("Generate points"), genRow);
            m_genPointsButton->setObjectName(QStringLiteral("SecondaryRunButton"));
            m_genPointsButton->setCursor(Qt::PointingHandCursor);
            m_genPointsButton->setMinimumSize(210, 38);
            m_genPointsButton->setToolTip(
                tr("Write the point layer now and open it in the Viewer, without "
                   "running the analysis. The analysis then reuses this exact "
                   "file as long as the parameters above stay unchanged."));
            connect(m_genPointsButton, &QPushButton::clicked,
                    this, &MainWindow::startPointsRun);
            // Right-aligned, so it sits where "Run analysis" does at the foot
            // of the form; the status line fills the space to its left.
            m_genStatusLabel = new QLabel(genRow);
            m_genStatusLabel->setObjectName(QStringLiteral("HintLabel"));
            m_genStatusLabel->setWordWrap(true);
            genRowLayout->addWidget(m_genStatusLabel, 1);
            genRowLayout->addWidget(m_genPointsButton);
            gGrid->addWidget(genRow, gr, 0, 1, 2);

            grid->addWidget(m_generateGroup, r, 0, 1, 2);
            ++r;
        }

        m_originPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                        tr("Select the origin file (exactly 1 point)"),
                                        QString::fromLatin1(kVectorFilter), content);
        m_originPicker->setPlaceholder(tr("Origin location — must contain exactly 1 point"));
        m_originLabel = addRow(tr("Origin"),
                               tr("Vector file containing exactly one point: the starting "
                                  "location of the least-cost routes."),
                               m_originPicker);

        m_destinationsPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                              tr("Select the destinations file (1+ points)"),
                                              QString::fromLatin1(kVectorFilter), content);
        m_destinationsPicker->setPlaceholder(tr("Destination locations — one or more points"));
        m_destinationsLabel = addRow(tr("Destinations"),
                                     tr("Vector file with one or more points: the target(s) the "
                                        "optimal route(s) is/are computed to."),
                                     m_destinationsPicker);

        m_outputDirPicker = new PathPicker(PathPicker::Kind::Directory,
                                           tr("Select the output folder"), QString(), content);
        m_outputDirPicker->setPlaceholder(tr("Folder where all results will be written"));
        addRow(tr("Output folder"),
               tr("Folder where every result file (rasters and shapefiles) "
                  "will be written."),
               m_outputDirPicker);

        m_cardInput = makeCard(
            tr("Input data"),
            tr("The DEM and every vector file must share the same coordinate "
               "reference system, and all points must fall inside the DEM extent."),
            content);
        layout->addWidget(m_cardInput);
    }

    // ----- Cost modifiers -----
    {
        m_modifiersGroup = new QGroupBox(tr("Use cost modifiers in this analysis"), inner);
        m_modifiersGroup->setCheckable(true);
        m_modifiersGroup->setChecked(false);

        // Row 0 *is* the title band: the note lives in it, beside the title
        // rather than under it. For the two to share a line the row has to
        // start where the title does, and the title is drawn from the top of
        // the margin box (`subcontrol-origin: margin`) while the content starts
        // below `margin-top` — so both of those come off, here and only here.
        // The first real row then begins under the pair.
        m_modifiersGroup->setStyleSheet(QStringLiteral("QGroupBox { margin-top: 0px; }"));
        auto *grid = new QGridLayout(m_modifiersGroup);
        grid->setContentsMargins(8, 0, 8, 10);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);
        m_labelColumns.append({grid, 0});

        int r = 0;
        // The description lives only on the "?" badge, not on the input itself.
        auto addRow = [&](const QString &label, QWidget *w, const QString &tip) {
            QWidget *l = makeFieldLabel(label, tip, m_modifiersGroup);
            grid->addWidget(l, r, 0);
            grid->addWidget(w, r, 1);
            ++r;
        };

        // What the switch costs, said beside the switch itself rather than
        // underneath it, where it read as a caption for the first field.
        {
            QWidget *note = TrajectaUi::makeGroupNote(m_modifiersGroup,
                                                      TrajectaUi::costModifiersNoteText(),
                                                      TrajectaUi::costModifiersHelpText());
            note->setMinimumHeight(m_modifiersGroup->fontMetrics().height() + 8);
            // Left, not right: it belongs against the switch it qualifies. The
            // note indents itself past the title (TitleFollower).
            grid->addWidget(note, r, 0, 1, 2, Qt::AlignLeft | Qt::AlignVCenter);
        }
        ++r;

        m_costVectorPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                            tr("Select the cost modifiers vector file"),
                                            QString::fromLatin1(kVectorFilter),
                                            m_modifiersGroup);
        m_costVectorPicker->setOptional(true);
        m_costVectorPicker->setPlaceholder(tr("Polylines with a 'cost' attribute — leave empty to skip"));
        addRow(tr("Vector modifiers"), m_costVectorPicker,
               tr("Polyline features (rivers, walls, restricted areas...) with a float "
                  "'cost' field holding the traversal multiplier (e.g. 2.0 = twice as "
                  "costly, 999999 = obstacle)."));

        m_polylineBufferSpin = new QSpinBox(m_modifiersGroup);
        m_polylineBufferSpin->setRange(0, 10);
        m_polylineBufferSpin->setValue(2);
        m_polylineBufferSpin->setSuffix(tr(" cell(s) per side"));
        addRow(tr("Polyline buffer"), m_polylineBufferSpin,
               tr("Widens rasterized polylines so the search cannot 'jump' across them. "
                  "2 cells per side is safe for 16-connectivity. For larger "
                  "connectivity grids (e.g. 32, 64...) the buffer has to be "
                  "increased."));

        m_costRasterPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                            tr("Select the cost modifiers raster"),
                                            QString::fromLatin1(kRasterFilter),
                                            m_modifiersGroup);
        m_costRasterPicker->setOptional(true);
        m_costRasterPicker->setPlaceholder(tr("Multiplier raster aligned with the DEM — leave empty to skip"));
        addRow(tr("Raster modifiers"), m_costRasterPicker,
               tr("GeoTIFF with the same size as the DEM. Cell values multiply the "
                  "traversal cost (1.0 = unchanged, 2.0 = double cost)."));

        auto *barrierWidget = new QWidget(m_modifiersGroup);
        auto *barrierRow = new QHBoxLayout(barrierWidget);
        barrierRow->setContentsMargins(0, 0, 0, 0);
        barrierRow->setSpacing(10);
        m_barrierCheck = new QCheckBox(tr("Multipliers above"), barrierWidget);
        m_barrierCheck->setChecked(true);
        m_barrierSpin = new QDoubleSpinBox(barrierWidget);
        m_barrierSpin->setRange(0.1, 1000000000.0);
        m_barrierSpin->setDecimals(1);
        m_barrierSpin->setValue(1000.0);
        auto *barrierTail = new QLabel(tr("are impassable barriers"), barrierWidget);
        barrierRow->addWidget(m_barrierCheck);
        barrierRow->addWidget(m_barrierSpin);
        barrierRow->addWidget(barrierTail);
        barrierRow->addStretch(1);
        connect(m_barrierCheck, &QCheckBox::toggled, m_barrierSpin, &QWidget::setEnabled);
        addRow(tr("Barrier threshold"), barrierWidget,
               tr("Cells whose multiplier reaches the threshold are excluded from "
                  "movement entirely. Recommended when obstacles use very large "
                  "multipliers; disabling it can slow the computation dramatically."));

        // Extra breathing room between the card subtitle and the checkable
        // group title.
        auto *groupHolder = new QWidget(inner);
        auto *holderLayout = new QVBoxLayout(groupHolder);
        holderLayout->setContentsMargins(0, 0, 0, 0);
        holderLayout->addSpacing(10);
        holderLayout->addWidget(m_modifiersGroup);

        m_cardModifiers = makeCard(
            tr("Cost modifiers (optional)"),
            tr("Increase traversal costs over specific features such as rivers, "
               "restricted areas or difficult terrain."),
            groupHolder);
        layout->addWidget(m_cardModifiers);
    }

    // ----- Algorithm -----
    {
        auto *content = new QWidget(inner);
        auto *grid = new QGridLayout(content);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);
        m_labelColumns.append({grid, 0});

        int r = 0;
        // The description lives only on the "?" badge, not on the input itself.
        auto addRow = [&](const QString &label, QWidget *w, const QString &tip) {
            QWidget *l = makeFieldLabel(label, tip, content);
            grid->addWidget(l, r, 0);
            grid->addWidget(w, r, 1);
            ++r;
        };

        // The presets plus a free number. The spin box only appears once
        // "Custom" is chosen, so the common case still reads as one control.
        QWidget *neighboursRow = new QWidget(content);
        auto *neighboursLayout = new QHBoxLayout(neighboursRow);
        neighboursLayout->setContentsMargins(0, 0, 0, 0);
        neighboursLayout->setSpacing(8);

        m_neighboursCombo = new SmoothComboBox(neighboursRow);
        // The number alone. The old captions ("knight moves", "extended")
        // described the shape of the template rather than what choosing it does,
        // and the "?" beside the field explains the trade-off properly.
        for (int n : {8, 16, 24, 32, 64})
            m_neighboursCombo->addItem(QString::number(n), n);
        m_neighboursCombo->addItem(tr("Custom…"), 0);
        m_neighboursCombo->setCurrentIndex(1);
        neighboursLayout->addWidget(m_neighboursCombo, 1);

        m_neighboursCustom = new QSpinBox(neighboursRow);
        m_neighboursCustom->setRange(neighbourhood::kMin, neighbourhood::kMax);
        m_neighboursCustom->setValue(48);
        m_neighboursCustom->setSuffix(tr(" directions"));
        m_neighboursCustom->setVisible(false);
        neighboursLayout->addWidget(m_neighboursCustom);

        connect(m_neighboursCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this] { refreshNeighboursCustom(); });
        // Snap on the way out of the box rather than on every keystroke:
        // typing "64" would otherwise be corrected to 8 after the first digit.
        connect(m_neighboursCustom, &QSpinBox::editingFinished, this, [this] {
            m_neighboursCustom->setValue(
                TrajectaUi::snapNeighbourCount(m_neighboursCustom->value()));
        });

        addRow(tr("Neighbours"), neighboursRow, TrajectaUi::neighboursHelpText());

        m_costFunctionCombo = new SmoothComboBox(content);
        m_costFunctionCombo->addItem(tr("Modified Tobler's Hiking Function (White 2015)"), 1);
        m_costFunctionCombo->addItem(tr("Márquez-Pérez et al. (2017)"), 2);
        m_costFunctionCombo->addItem(tr("Irmischer & Clarke (2017) — on-path, male"), 3);
        m_costFunctionCombo->addItem(tr("Herzog (2013) — energy, kJ/kg"), 4);
        m_costFunctionCombo->addItem(tr("Campbell et al. (2019) — 5th percentile"), 5);
        m_costFunctionCombo->addItem(tr("Campbell et al. (2019) — 50th percentile"), 6);
        addRow(tr("Cost function"), m_costFunctionCombo, TrajectaUi::costFunctionHelpText());

        // Herzog answers a different question from the others, and its outputs
        // are in different units. Saying so where the choice is made costs one
        // line and prevents an energy raster being read as hours.
        m_costUnitsNote = new QLabel(content);
        m_costUnitsNote->setObjectName(QStringLiteral("FieldHint"));
        m_costUnitsNote->setWordWrap(true);
        grid->addWidget(m_costUnitsNote, r, 1);
        ++r;
        connect(m_costFunctionCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this] { refreshCostUnitsNote(); });

        // --- Slope cut-off ---
        QWidget *slopeCapRow = new QWidget(content);
        auto *slopeCapLayout = new QHBoxLayout(slopeCapRow);
        slopeCapLayout->setContentsMargins(0, 0, 0, 0);
        slopeCapLayout->setSpacing(8);
        m_slopeCapCheck = new QCheckBox(tr("Refuse moves steeper than"), slopeCapRow);
        m_slopeCapCheck->setChecked(false);
        slopeCapLayout->addWidget(m_slopeCapCheck);
        m_slopeCapUp = new QSpinBox(slopeCapRow);
        m_slopeCapUp->setRange(1, 89);
        m_slopeCapUp->setValue(30);
        m_slopeCapUp->setSuffix(tr("° uphill"));
        slopeCapLayout->addWidget(m_slopeCapUp);
        m_slopeCapDown = new QSpinBox(slopeCapRow);
        m_slopeCapDown->setRange(1, 89);
        m_slopeCapDown->setValue(30);
        m_slopeCapDown->setSuffix(tr("° downhill"));
        slopeCapLayout->addWidget(m_slopeCapDown);
        slopeCapLayout->addWidget(TrajectaUi::makeHelpDot(TrajectaUi::slopeCutoffHelpText(), slopeCapRow));
        slopeCapLayout->addStretch(1);
        grid->addWidget(slopeCapRow, r, 1);
        ++r;
        connect(m_slopeCapCheck, &QCheckBox::toggled, this, [this](bool on) {
            m_slopeCapUp->setEnabled(on);
            m_slopeCapDown->setEnabled(on);
        });
        m_slopeCapUp->setEnabled(false);
        m_slopeCapDown->setEnabled(false);

        m_smoothingSpin = new QSpinBox(content);
        m_smoothingSpin->setRange(0, 10);
        m_smoothingSpin->setValue(0);
        m_smoothingSpin->setSuffix(tr(" cell(s) per side"));
        addRow(tr("Path smoothing buffer"), m_smoothingSpin,
               tr("Buffer applied around computed paths when accumulating the "
                  "result, to simulate larger paths. 0 keeps the raw "
                  "single-cell paths."));

        m_cardAlgorithm = makeCard(tr("Algorithm"), QString(), content);
        layout->addWidget(m_cardAlgorithm);
    }

    // ----- Output file names -----
    {
        auto *content = new QWidget(inner);
        auto *grid = new QGridLayout(content);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);
        grid->setColumnStretch(3, 1);
        // Two label columns here: both get the page's width, so the two halves
        // of the card are mirror images of each other.
        m_labelColumns.append({grid, 0});
        m_labelColumns.append({grid, 2});

        int r = 0, c = 0;
        // Clearing an optional name makes the engine skip that file entirely;
        // the placeholder says so, so an empty box never looks like a mistake.
        auto addName = [&](const QString &label, const QString &defaultName,
                           const QString &tip, bool optional = true) {
            QWidget *l = makeFieldLabel(
                label, optional ? tip + QLatin1Char(' ')
                                      + tr("Leave empty to skip this output.")
                                : tip,
                content);
            auto *e = new QLineEdit(defaultName, content);
            if (optional)
                e->setPlaceholderText(tr("not saved"));
            grid->addWidget(l, r, c);
            grid->addWidget(e, r, c + 1);
            c += 2;
            if (c >= 4) {
                c = 0;
                ++r;
            }
            return std::make_pair(l, e);
        };

        std::tie(std::ignore, m_slopeNameEdit) =
            addName(tr("Slope raster"), QStringLiteral("slope"),
                    tr("Terrain slope computed from the DEM."));
        std::tie(std::ignore, m_costNameEdit) =
            addName(tr("Base cost surface"), QStringLiteral("cost_surface"),
                    tr("Cost surface derived from slope and the cost function."));
        std::tie(m_additionalNameLabel, m_additionalNameEdit) =
            addName(tr("Additional cost"), QStringLiteral("cost_surface_additional"),
                    tr("Rasterized cost-modifier polylines. Editable only when "
                       "cost modifiers are enabled."));
        std::tie(m_totalNameLabel, m_totalNameEdit) =
            addName(tr("Total cost surface"), QStringLiteral("cost_surface_total"),
                    tr("Base cost multiplied by the modifiers. Editable only when "
                       "cost modifiers are enabled."));
        // Paths raster + shapefile are created before the density field so
        // that, in LCPA mode, they share the same grid row.
        std::tie(m_pathRasterNameLabel, m_pathRasterNameEdit) =
            addName(tr("Paths raster"), QStringLiteral("raster_lcps"),
                    tr("Main LCPA result: raster marking the optimal routes."));
        std::tie(m_pathLinesNameLabel, m_pathLinesNameEdit) =
            addName(tr("Paths shapefile"), QStringLiteral("LCPS_vectors"),
                    tr("Main LCPA result: polyline shapefile of the optimal routes. "
                       "Each line carries its origin and destination, the total cost "
                       "with its unit, the planimetric length in metres and the number "
                       "of cells it crosses."));
        std::tie(m_corridorNameLabel, m_corridorNameEdit) =
            addName(tr("Corridor raster"), QStringLiteral("cost_corridor"),
                    TrajectaUi::costCorridorHelpText());
        // The corridor is a second, optional answer to the same question the
        // paths answer, so it lives with them: a checkbox that turns it on and
        // a width that stays visible while it is off, so the cost of switching
        // it on is readable before you do.
        QWidget *corridorRow = new QWidget(content);
        auto *corridorLayout = new QHBoxLayout(corridorRow);
        corridorLayout->setContentsMargins(0, 0, 0, 0);
        corridorLayout->setSpacing(8);
        m_corridorCheck = new QCheckBox(tr("Also compute the cost corridor"), corridorRow);
        m_corridorCheck->setChecked(false);
        corridorLayout->addWidget(m_corridorCheck);
        m_corridorWidthSpin = new QDoubleSpinBox(corridorRow);
        m_corridorWidthSpin->setRange(1.0, 500.0);
        m_corridorWidthSpin->setDecimals(1);
        m_corridorWidthSpin->setValue(10.0);
        m_corridorWidthSpin->setSuffix(tr("% above the optimum"));
        m_corridorWidthSpin->setEnabled(false);
        guardWheel(m_corridorWidthSpin);
        corridorLayout->addWidget(m_corridorWidthSpin);
        corridorLayout->addWidget(
            TrajectaUi::makeHelpDot(TrajectaUi::costCorridorHelpText(), corridorRow));
        corridorLayout->addStretch(1);
        // The card lays its name fields out in two label/field pairs per row;
        // this one is a sentence, so it starts a fresh row and spans it.
        if (c != 0) { c = 0; ++r; }
        grid->addWidget(corridorRow, r, 0, 1, 4);
        ++r;
        connect(m_corridorCheck, &QCheckBox::toggled, this, [this](bool on) {
            m_corridorWidthSpin->setEnabled(on);
            m_corridorNameLabel->setEnabled(on);
            m_corridorNameEdit->setEnabled(on);
        });
        m_corridorRow = corridorRow;

        // The one output that cannot be skipped: without it a FETE run writes
        // nothing at all.
        std::tie(m_densityNameLabel, m_densityNameEdit) =
            addName(tr("Density raster"), QStringLiteral("FETE_density"),
                    tr("Main FETE result: accumulated path-usage density. "
                       "Required."),
                    false);

        m_cardOutputs = makeCard(
            tr("Output files"),
            tr("Names only, without extension — everything is written inside the "
               "output folder."),
            content);
        layout->addWidget(m_cardOutputs);
    }

    // ----- Run bar -----
    {
        // A bare layout, not a QWidget wrapper: a wrapper inherits the opaque
        // base background and shows up as a slab behind the button on any
        // theme whose pages are transparent.
        auto *row = new QHBoxLayout;
        // Same 4 px top and bottom: this row's own gap from the card above
        // it and the run panel below it, on top of the 14 px every card on
        // this page already keeps from its neighbours, comes out equal
        // either way — the same balance every other floating run button on
        // the Processing and Post-processing pages keeps (see
        // buildPostPage() for the other three).
        row->setContentsMargins(0, 4, 0, 4);
        row->addStretch(1);

        m_runButton = new QPushButton(tr("▶  Run analysis"), inner);
        m_runButton->setObjectName(QStringLiteral("RunButton"));
        m_runButton->setCursor(Qt::PointingHandCursor);
        m_runButton->setMinimumSize(220, 46);
        connect(m_runButton, &QPushButton::clicked, this, &MainWindow::startRun);
        row->addWidget(m_runButton);

        layout->addLayout(row);
    }

    // ----- Live run: progress, console, results -----
    // This was a page of its own ("Processing"). It belongs directly under the
    // button that starts the run: the run is the end of the form, not a
    // different part of the application.
    m_runPanel = buildRunPanel(
        m_runUi, inner, tr("Configure the analysis and press “Run analysis”."),
        nullptr, true);
    layout->addWidget(m_runPanel);

    layout->addStretch(1);
    scroll->setWidget(inner);
    pageLayout->addWidget(scroll);

    // Scrolling the form must never silently change values.
    for (QWidget *w : std::initializer_list<QWidget *>{
             m_polylineBufferSpin, m_barrierSpin, m_neighboursCombo,
             m_neighboursCustom, m_slopeCapUp, m_slopeCapDown,
             m_costFunctionCombo, m_smoothingSpin, m_threadsSpin, m_ramSpin,
             m_pointsSourceCombo, m_genDensityCombo, m_genSpacingSpin,
             m_genTargetSpin, m_genArrangementCombo, m_genSeedSpin,
             m_genEdgeSpin})
        guardWheel(w);

    // Sample point generation: the source selector drives the whole group,
    // and every parameter that changes the resulting count refreshes the
    // preview line. None of this runs while the points come from a file.
    connect(m_pointsSourceCombo, &QComboBox::currentIndexChanged,
            this, &MainWindow::updatePointsSourceUi);
    connect(m_genDensityCombo, &QComboBox::currentIndexChanged,
            this, &MainWindow::updatePointsSourceUi);
    connect(m_genArrangementCombo, &QComboBox::currentIndexChanged,
            this, &MainWindow::updatePointsSourceUi);
    connect(m_genSpacingSpin, &QSpinBox::valueChanged,
            this, &MainWindow::updateGeneratedPointsPreview);
    connect(m_genTargetSpin, &QSpinBox::valueChanged,
            this, &MainWindow::updateGeneratedPointsPreview);
    connect(m_genSeedSpin, &QSpinBox::valueChanged,
            this, &MainWindow::updateGeneratedPointsStatus);
    connect(m_genEdgeSpin, &QSpinBox::valueChanged,
            this, &MainWindow::updateGeneratedPointsPreview);
    connect(m_genNameEdit, &QLineEdit::textChanged,
            this, &MainWindow::updateGeneratedPointsStatus);
    connect(m_outputDirPicker, &PathPicker::pathChanged,
            this, &MainWindow::updateGeneratedPointsStatus);
    connect(m_demPicker, &PathPicker::pathChanged, this, [this] {
        if (m_pointsSourceCombo->currentIndex() == 1)
            updateGeneratedPointsPreview();
    });

    // Everything added after the mode card is the single-run form. Collecting
    // it here, rather than naming each card, keeps batch mode's show/hide to
    // one loop and means a new card is picked up automatically.
    for (int i = 1; i < layout->count(); ++i) {
        QLayoutItem *item = layout->itemAt(i);
        if (QWidget *w = item->widget()) {
            m_singleRunCards.append(w);
        } else if (QLayout *nested = item->layout()) {
            // The run bar is deliberately a bare layout rather than a widget
            // (see above), so its button has to be picked up from inside it.
            for (int j = 0; j < nested->count(); ++j)
                if (QWidget *w = nested->itemAt(j)->widget())
                    m_singleRunCards.append(w);
        }
    }

    m_batchPage = new BatchPage(inner);
    m_batchPage->setVisible(false);
    // Right after the Analysis type card, not merely "before the trailing
    // stretch": the run bar a few cards down is a bare QHBoxLayout (see
    // above), and a bare layout's isEmpty() does not follow its widgets going
    // hidden the way a QWidgetItem's does — it stayed "non-empty" to this
    // layout even with every widget inside it hidden, so the outer spacing
    // counted it as a real item and the gap above the batch page came out a
    // few pixels taller than the gap above the single-run tool card. Sitting
    // immediately after item 0 leaves nothing but a single well-behaved
    // QWidgetItem between them in either mode, so the two gaps are the same
    // spacing() with nothing left to round.
    layout->insertWidget(1, m_batchPage);
    // A batch owns the engine for as long as it runs, so the single-run button
    // must not be able to start a second one alongside it.
    connect(m_batchPage, &BatchPage::runningChanged, this, [this](bool running) {
        // Every button that would put a second engine on the machine, not just
        // the obvious one: the two on this page and the two on Post-processing
        // are all served by the same engine, and a batch owns it outright.
        m_runButton->setEnabled(!running);
        if (m_genPointsButton)
            m_genPointsButton->setEnabled(!running);
        if (m_runInterpButton)
            m_runInterpButton->setEnabled(!running);
        if (m_cmpButton)
            m_cmpButton->setEnabled(!running);
        refreshRunTicker();
        // Recorded the moment a batch starts, and deliberately outliving the
        // process: loadSettings() reads it at the next start-up to decide
        // whether the page comes back or starts clean. Written here rather
        // than on the way out because a run that ends in a crash is exactly
        // the case that must not be forgotten.
        if (running)
            QSettings().setValue(QStringLiteral("batch/processed"), true);
    });
    // The same two actions as the single-run panel, and the same code behind
    // them: a batch checkpoint and a FETE checkpoint are the same file, and
    // resumeFromCheckpoint() already tells the two sessions apart.
    // Everything the ticker shows about a batch comes from the page; this is
    // the page saying one of those things has moved.
    connect(m_batchPage, &BatchPage::tickerChanged,
            this, &MainWindow::refreshRunTicker);
    connect(m_batchPage, &BatchPage::exportCheckpointRequested,
            this, &MainWindow::exportCheckpointCopy);
    connect(m_batchPage, &BatchPage::importCheckpointRequested,
            this, &MainWindow::importCheckpointAndResume);
    // Only the chunks that asked for it send anything here, and none does by
    // default: a thirty-row batch would otherwise fill the Viewer's layer list
    // and hold thirty files open.
    connect(m_batchPage, &BatchPage::viewerLayersReady, this,
            [this](const QStringList &rasters, const QStringList &vectors) {
        if (!m_viewer)
            return;
        for (const QString &path : rasters)
            m_viewer->registerRaster(QFileInfo(path).completeBaseName(), path, false);
        for (const QString &path : vectors)
            m_viewer->registerVectorOverlay(QFileInfo(path).completeBaseName(), path);
    });

    return page;
}

void MainWindow::loadBatchFile(const QString &path)
{
    if (!m_batchPage)
        return;
    selectMode(QStringLiteral("batch"));
    QString error;
    if (!m_batchPage->loadBatchFile(path, &error))
        qWarning("batch load failed: %s", qPrintable(error));
}

void MainWindow::triggerBatchRun()
{
    if (m_batchPage)
        m_batchPage->startBatchNow();
}

void MainWindow::loadPostBatchFile(const QString &path)
{
    if (!m_postBatchPage)
        return;
    showPage(1);
    selectPostMode(QStringLiteral("batch"));
    QString error;
    if (!m_postBatchPage->loadBatchFile(path, &error))
        qWarning("post-batch load failed: %s", qPrintable(error));
}

void MainWindow::triggerPostBatchRun()
{
    if (m_postBatchPage)
        m_postBatchPage->startBatchNow();
}

void MainWindow::selectMode(const QString &name)
{
    if (name == QLatin1String("batch")) {
        m_modeBatch->setChecked(true);
    } else {
        // FETE/LCPA and Single/Batch are two different button groups now (see
        // buildSetupPage()'s "Analysis type" card), so picking a tool no
        // longer implies Single the way it did when all three shared one
        // group; it has to be said explicitly.
        if (m_modeSingle)
            m_modeSingle->setChecked(true);
        if (name == QLatin1String("lcpa"))
            m_modeLcpa->setChecked(true);
        else
            m_modeFete->setChecked(true);
    }
    updateModeUi();
}

void MainWindow::updateModeUi()
{
    // Batch replaces the tool card and the single-run form wholesale: none of
    // the per-field visibility below applies to it.
    const bool batch = m_modeBatch && m_modeBatch->isChecked();
    if (m_cardMode)
        m_cardMode->setVisible(!batch);
    for (QWidget *w : std::as_const(m_singleRunCards))
        w->setVisible(!batch);
    if (m_batchPage)
        m_batchPage->setVisible(batch);
    if (batch)
        return;

    const bool fete = m_modeFete->isChecked();

    m_pointsSourceLabel->setVisible(fete);
    m_pointsSourceCombo->setVisible(fete);
    m_originLabel->setVisible(!fete);
    m_originPicker->setVisible(!fete);
    m_destinationsLabel->setVisible(!fete);
    m_destinationsPicker->setVisible(!fete);

    m_densityNameLabel->setVisible(fete);
    m_densityNameEdit->setVisible(fete);
    m_pathRasterNameLabel->setVisible(!fete);
    m_pathRasterNameEdit->setVisible(!fete);
    m_pathLinesNameLabel->setVisible(!fete);
    m_pathLinesNameEdit->setVisible(!fete);
    m_corridorRow->setVisible(!fete);
    m_corridorNameLabel->setVisible(!fete);
    m_corridorNameEdit->setVisible(!fete);

    // Point generation exists in FETE mode only; LCPA always takes its origin
    // and destinations from files.
    updatePointsSourceUi();
}

// Import and generation are mutually exclusive. In import mode the generation
// group is hidden AND disabled, no parameter of it is read when the run
// starts, and the engine is never even asked the generation questions.
void MainWindow::updatePointsSourceUi()
{
    const bool fete = m_modeFete->isChecked();
    const bool generate = fete && m_pointsSourceCombo->currentIndex() == 1;

    m_pointsLabel->setVisible(fete && !generate);
    m_pointsPicker->setVisible(fete && !generate);

    m_generateGroup->setVisible(generate);
    m_generateGroup->setEnabled(generate);
    if (!generate)
        return;

    const bool byTarget = m_genDensityCombo->currentIndex() == 1;
    m_genSpacingLabel->setVisible(!byTarget);
    m_genSpacingSpin->setVisible(!byTarget);
    m_genTargetLabel->setVisible(byTarget);
    m_genTargetSpin->setVisible(byTarget);

    const bool random = m_genArrangementCombo->currentIndex() == 1;
    m_genSeedLabel->setVisible(random);
    m_genSeedSpin->setVisible(random);

    updateGeneratedPointsPreview();
}

// Live estimate of what the generation will produce. The engine derives the
// spacing from the share of usable DEM cells measured on a decimated read;
// the same rule is applied here (see kGenPreviewDim in main_fete.cpp), so the
// spacing shown is the spacing the engine will use.
void MainWindow::updateGeneratedPointsPreview()
{
    // Asks what the form says, not what is on screen: this also runs while the
    // window is still being built from the saved settings, when nothing is
    // visible yet.
    if (!m_modeFete->isChecked() || m_pointsSourceCombo->currentIndex() != 1)
        return;

    const QString dem = m_demPicker->path();
    if (dem.isEmpty() || !QFileInfo::exists(dem)) {
        m_genCachedDem.clear();
        m_genValidFraction = -1.0;
        m_genPreviewLabel->setText(
            tr("Select a DEM to see how many points will be generated."));
        updateGeneratedPointsStatus();
        return;
    }

    if (dem != m_genCachedDem) {
        m_genCachedDem = dem;
        m_genValidFraction = -1.0;
        m_genDemWidth = m_genDemHeight = 0;

        GdalApi &api = GdalApi::instance();
        ensureGdalLoaded();
        // Opened and closed immediately: a retained handle would stop the
        // engine from rewriting the DEM's folder on the next run.
        if (api.isLoaded()) {
            const QByteArray utf8 = dem.toUtf8();
            GDALDatasetH ds = api.OpenEx(utf8.constData(), GdalApi::OF_Raster,
                                         nullptr, nullptr, nullptr);
            if (ds) {
                const int w = api.GetRasterXSize(ds);
                const int h = api.GetRasterYSize(ds);
                GDALRasterBandH band = api.GetRasterBand(ds, 1);
                if (w > 0 && h > 0 && band) {
                    constexpr int kPreviewDim = 1000;  // matches the engine
                    const double scale = qMin(1.0, double(kPreviewDim) / qMax(w, h));
                    const int dw = qMax(1, int(qRound(w * scale)));
                    const int dh = qMax(1, int(qRound(h * scale)));
                    QVector<float> buf(qsizetype(dw) * dh);
                    int hasNoData = 0;
                    const double noData = api.GetRasterNoDataValue(band, &hasNoData);
                    if (api.RasterIO(band, GdalApi::ReadFlag, 0, 0, w, h,
                                     buf.data(), dw, dh, GdalApi::Float32,
                                     0, 0) == 0) {
                        qsizetype good = 0;
                        for (const float v : std::as_const(buf)) {
                            if (!std::isnan(v) && v < 9999.0f
                                && !(hasNoData && double(v) == noData))
                                ++good;
                        }
                        m_genValidFraction = double(good) / buf.size();
                        m_genDemWidth = w;
                        m_genDemHeight = h;
                    }
                }
                api.Close(ds);
            }
        }
    }

    if (m_genValidFraction < 0.0) {
        m_genPreviewLabel->setText(
            tr("Point count unknown — GDAL could not read the DEM here. The "
               "engine computes it itself when the analysis starts."));
        updateGeneratedPointsStatus();
        return;
    }

    // The buffer takes a band off every side; only what is left can hold
    // points, which is also how the engine derives the spacing from a target.
    const int buffer = m_genEdgeSpin->value();
    const int innerW = m_genDemWidth - 2 * buffer;
    const int innerH = m_genDemHeight - 2 * buffer;
    if (innerW < 1 || innerH < 1) {
        m_genPreviewLabel->setText(
            tr("An edge buffer of %1 cells leaves no room on a %2 × %3 DEM.")
                .arg(buffer).arg(m_genDemWidth).arg(m_genDemHeight));
        updateGeneratedPointsStatus();
        return;
    }

    const double validCells = m_genValidFraction * double(innerW) * double(innerH);
    int spacing = m_genSpacingSpin->value();
    if (m_genDensityCombo->currentIndex() == 1) {
        spacing = qMax(1, int(qRound(std::sqrt(validCells / m_genTargetSpin->value()))));
    }
    const qint64 rows = (innerH + spacing - 1) / spacing;
    const qint64 cols = (innerW + spacing - 1) / spacing;
    const qint64 estimate = qint64(m_genValidFraction * double(rows * cols));

    QString text = tr("About %L1 points — one every %2 cell(s) over the %L3 "
                      "usable cells of a %4 × %5 DEM.")
                       .arg(estimate)
                       .arg(spacing)
                       .arg(qint64(validCells))
                       .arg(m_genDemWidth)
                       .arg(m_genDemHeight);
    if (buffer > 0) {
        text += QLatin1Char('\n')
                + tr("A %1-cell band along each border is left empty.").arg(buffer);
    }
    if (estimate >= 200000) {
        text += QLatin1Char('\n')
                + tr("FETE cost grows with the square of the point count: this "
                     "many points will take a very long time.");
    } else if (estimate >= 50000) {
        text += QLatin1Char('\n')
                + tr("That is a lot of points — expect a long run.");
    }
    m_genPreviewLabel->setText(text);
    updateGeneratedPointsStatus();
}

// Everything that decides what the generated layer contains and where it goes.
// Two runs with the same key produce the same file, which is what makes reusing
// a previewed layer safe rather than merely convenient.
QString MainWindow::generationKey() const
{
    return QStringList{
        QDir::toNativeSeparators(m_demPicker->path()),
        QDir::toNativeSeparators(m_outputDirPicker->path()),
        m_genNameEdit->text().trimmed(),
        QString::number(m_genDensityCombo->currentIndex()),
        QString::number(m_genSpacingSpin->value()),
        QString::number(m_genTargetSpin->value()),
        QString::number(m_genArrangementCombo->currentIndex()),
        QString::number(m_genSeedSpin->value()),
        QString::number(m_genEdgeSpin->value()),
    }.join(QLatin1Char('|'));
}

void MainWindow::updateGeneratedPointsStatus()
{
    if (!m_genStatusLabel)
        return;
    if (m_previewedPointsPath.isEmpty()) {
        m_genStatusLabel->setText(
            tr("Optional: write the layer and inspect it in the Viewer first."));
        return;
    }
    const bool fresh = m_previewedPointsKey == generationKey()
                       && QFileInfo::exists(m_previewedPointsPath);
    if (fresh) {
        m_genStatusLabel->setText(
            tr("✓ %1 is on disk — the analysis will use this exact file.")
                .arg(QFileInfo(m_previewedPointsPath).fileName()));
    } else {
        m_genStatusLabel->setText(
            tr("Parameters changed since the last generation — the analysis "
               "will write the layer again."));
    }
}

// ---------------------------------------------------------------------------
// Run panels ("Run & results" and "Post-processing" share the same layout)
// ---------------------------------------------------------------------------

// The pair of links that sit at the right-hand end of every log row: FETE and
// LCPA share one run panel, post-processing has its own, and the batch page
// asks for a third. All three do exactly the same thing, so they are built
// here rather than three times over.
// The two checkpoint actions, as buttons rather than as links.
//
// They began as links on the log row, which was a mistake of category: they do
// something — they write a folder full of state, or take one over — and the
// only other things on that row that do anything are buttons. As links they
// also sat above the run controls, where they read as part of the transcript
// rather than as part of what you can do to the run.
//
// They belong on the button row, and to the *left* of it, separated from
// "Open output folder" by a gap: those three are about this run in this
// window, and these two are about a file on disk that outlives it.
void MainWindow::buildCheckpointButtons(QHBoxLayout *row, QWidget *parent)
{
    auto *save = new QPushButton(tr("Save a copy of the checkpoint..."), parent);
    // Kept so the walkthrough can point at the pair. Only the analysis panel
    // asks for them, so there is exactly one of each to remember.
    m_ckptSaveButton = save;
    save->setObjectName(QStringLiteral("PrimaryButton"));
    save->setCursor(Qt::PointingHandCursor);
    save->setToolTip(tr(
        "Writes a copy of the state of the analysis in progress to a folder of "
        "your choosing. The run is not affected, and the copy can be resumed "
        "later, here or on another machine.\n\n"
        "There is something to copy only once auto-save has written its first "
        "checkpoint."));
    connect(save, &QPushButton::clicked, this, &MainWindow::exportCheckpointCopy);
    // The same badge the batch page carries, worded for a single analysis: the
    // two buttons are the same two buttons, and a user who reads the batch one
    // and then finds nothing here would reasonably conclude they do something
    // else in FETE and LCPA. They do not.
    row->addWidget(TrajectaUi::makeHelpDot(
        tr("<b>The two checkpoint buttons</b><br><br>"
           "A checkpoint is the state of the analysis, written to disk while it "
           "runs, from which the engine can carry on instead of starting again. "
           "On a run measured in days that is the difference between a power cut "
           "costing an hour and costing the week.<br><br>"
           "<b>Save a copy of the checkpoint…</b> writes the state of the "
           "analysis in progress to a folder you choose. The run carries on "
           "regardless — this only copies. There is something to copy once "
           "auto-save has written its first checkpoint, so not in the first "
           "minutes of a run.<br><br>"
           "<b>Resume from a checkpoint file…</b> picks an interrupted analysis "
           "up again, here or on another machine: the engine carries on from "
           "where it stopped rather than from its first source point."),
        parent));
    row->addWidget(save);

    auto *load = new QPushButton(tr("Resume from a checkpoint file..."), parent);
    m_ckptLoadButton = load;
    load->setObjectName(QStringLiteral("PrimaryButton"));
    load->setCursor(Qt::PointingHandCursor);
    load->setToolTip(tr(
        "Picks up an interrupted analysis from a checkpoint saved earlier, "
        "instead of waiting for Trajecta to offer it at the next start."));
    connect(load, &QPushButton::clicked, this, &MainWindow::importCheckpointAndResume);
    row->addWidget(load);
}

QWidget *MainWindow::buildRunPanel(RunUi &ui, QWidget *parent,
                                   const QString &idlePhrase, QWidget *leadingButton,
                                   bool withCheckpointLinks)
{
    // A card, like every other section: as a bare widget it painted its own
    // opaque background and read as a loose strip rather than part of the page.
    auto *panel = new QFrame(parent);
    panel->setObjectName(QStringLiteral("Card"));
    auto *layout = new QVBoxLayout(panel);
    layout->setContentsMargins(18, 16, 18, 18);
    layout->setSpacing(12);

    // Status row
    auto *statusRow = new QHBoxLayout;
    statusRow->setSpacing(12);

    ui.chip = new QLabel(tr("IDLE"), panel);
    ui.chip->setObjectName(QStringLiteral("StateChip"));
    ui.chip->setProperty("state", QStringLiteral("idle"));
    ui.chip->setAlignment(Qt::AlignCenter);
    // Minimums, not fixed sizes: a wider typeface from the gear menu has to be
    // able to make the chip taller instead of being cut off inside it.
    ui.chip->setMinimumHeight(26);
    ui.chip->setMinimumWidth(110);
    statusRow->addWidget(ui.chip);

    ui.phase = new QLabel(idlePhrase, panel);
    ui.phase->setObjectName(QStringLiteral("PhaseLabel"));
    statusRow->addWidget(ui.phase, 1);

    ui.elapsed = new QLabel(QStringLiteral("0:00:00"), panel);
    ui.elapsed->setObjectName(QStringLiteral("ElapsedLabel"));
    statusRow->addWidget(ui.elapsed);

    layout->addLayout(statusRow);

    ui.progress = new ActivityBar(panel);
    ui.progress->setRange(0, 1000);
    ui.progress->setValue(0);
    ui.progress->setTextVisible(true);
    ui.progress->setFormat(QStringLiteral("%p%"));
    ui.progress->setFixedHeight(20);
    layout->addWidget(ui.progress);

    // The engine's own output, folded away by default. Most runs need nothing
    // from it: the progress bar and the summary say what happened. It matters
    // when something goes wrong, or when you want to watch the thing work, so
    // it is one click away rather than absent.
    auto *logHandle = new QToolButton(panel);
    logHandle->setObjectName(QStringLiteral("LogHandle"));
    logHandle->setCursor(Qt::PointingHandCursor);
    logHandle->setAutoRaise(true);
    logHandle->setCheckable(true);
    logHandle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    logHandle->setArrowType(Qt::RightArrow);
    logHandle->setText(tr("Engine log"));

    auto *logRow = new QHBoxLayout;
    logRow->setSpacing(14);
    logRow->addWidget(logHandle);
    logRow->addStretch(1);
    layout->addLayout(logRow);

    ui.console = new ConsoleView(panel);
    ui.console->setVisible(false);
    // A log you have to scroll a line at a time is a log nobody reads, so
    // opening it hands over several screens of transcript at once; how many is
    // kLogCanvasHeight's business. Set here rather than on the caller's side so
    // both run panels get it — only the post-processing one used to have a
    // height of its own.
    ui.console->setMinimumHeight(kLogCanvasHeight);
    layout->addWidget(ui.console, 1);

    ui.logHandle = logHandle;
    // The widgets are captured by value: `ui` is a reference to a member owned
    // by MainWindow, but capturing the reference itself would tie the lambda to
    // this call's argument rather than to the widgets it is about.
    ConsoleView *const console = ui.console;
    connect(logHandle, &QToolButton::toggled, this, [logHandle, console](bool open) {
        // The arrow is the state: pointing right is folded, pointing down is
        // open, which is the convention every file tree uses.
        logHandle->setArrowType(open ? Qt::DownArrow : Qt::RightArrow);
        console->setVisible(open);
    });

    // Summary card (hidden until a run finishes)
    ui.summaryCard = new QFrame(panel);
    ui.summaryCard->setObjectName(QStringLiteral("Card"));
    auto *summaryLayout = new QVBoxLayout(ui.summaryCard);
    summaryLayout->setContentsMargins(18, 12, 18, 12);
    ui.summaryTitle = new QLabel(ui.summaryCard);
    ui.summaryTitle->setObjectName(QStringLiteral("CardTitle"));
    ui.summaryBody = new QLabel(ui.summaryCard);
    ui.summaryBody->setObjectName(QStringLiteral("SummaryBody"));
    ui.summaryBody->setWordWrap(true);
    ui.summaryBody->setTextInteractionFlags(Qt::TextSelectableByMouse);
    summaryLayout->addWidget(ui.summaryTitle);
    summaryLayout->addWidget(ui.summaryBody);
    ui.summaryCard->setVisible(false);
    layout->addWidget(ui.summaryCard);

    // Buttons
    auto *buttonRow = new QHBoxLayout;
    buttonRow->setSpacing(10);

    if (leadingButton)
        buttonRow->addWidget(leadingButton);

    buttonRow->addStretch(1);

    // The saved state: keeping a copy of an unfinished run, and picking one
    // back up. Until these existed the only way into an interrupted analysis
    // was the prompt at the next start — which meant the state could only ever
    // be resumed on the machine that made it, once, and only if nothing had
    // cleared it in between.
    //
    // Not on the post-processing panel: an interpolation runs in minutes and
    // has no state to save, so offering to save one would be a lie.
    if (withCheckpointLinks) {
        buildCheckpointButtons(buttonRow, panel);
        // The gap that says these two are a different kind of thing from the
        // three that follow. Wide enough to read as a division rather than as
        // uneven spacing.
        buttonRow->addSpacing(28);
    }

    ui.openFolderButton = new QPushButton(tr("Open output folder"), panel);
    ui.openFolderButton->setObjectName(QStringLiteral("PrimaryButton"));
    ui.openFolderButton->setCursor(Qt::PointingHandCursor);
    ui.openFolderButton->setEnabled(false);
    connect(ui.openFolderButton, &QPushButton::clicked, this, &MainWindow::openOutputFolder);
    buttonRow->addWidget(ui.openFolderButton);

    ui.pauseButton = new QPushButton(tr("Pause"), panel);
    ui.pauseButton->setObjectName(QStringLiteral("PrimaryButton"));
    // The two bars, to answer the ▶ on Run analysis. Set here and taken off
    // again only while the label says "▶ Resume" (see setPauseUi).
    TrajectaUi::setPauseMark(ui.pauseButton, true);
    ui.pauseButton->setCursor(Qt::PointingHandCursor);
    ui.pauseButton->setEnabled(false);
    ui.pauseButton->setToolTip(
        tr("Freezes the computation and releases the CPU. The engine's memory "
           "stays allocated: you can sleep or hibernate the PC, but shutting "
           "it down loses the run."));
    connect(ui.pauseButton, &QPushButton::clicked, this, [this] {
        if (m_runner->isPaused())
            m_runner->resume();
        else
            m_runner->pause();
    });
    buttonRow->addWidget(ui.pauseButton);

    ui.cancelButton = new QPushButton(tr("Cancel run"), panel);
    ui.cancelButton->setObjectName(QStringLiteral("DangerButton"));
    ui.cancelButton->setCursor(Qt::PointingHandCursor);
    ui.cancelButton->setEnabled(false);
    connect(ui.cancelButton, &QPushButton::clicked, this, [this] {
        if (m_runner->isRunning()
            && TrajectaUi::confirm(this, tr("Cancel run"),
                                   tr("Stop the running analysis?"))) {
            m_runner->cancel();
        }
    });
    buttonRow->addWidget(ui.cancelButton);

    layout->addLayout(buttonRow);
    return panel;
}

// The comparison's result panel. Deliberately the same card as buildRunPanel's,
// down to the object names, so the two post-processing tools look and behave
// alike; what it leaves out is everything that describes a subprocess. There is
// no progress bar because the work is one synchronous pass, and no Pause or
// Cancel because there is nothing to signal.
QWidget *MainWindow::buildComparePanel(QWidget *parent)
{
    auto *panel = new QFrame(parent);
    panel->setObjectName(QStringLiteral("Card"));
    auto *layout = new QVBoxLayout(panel);
    layout->setContentsMargins(18, 16, 18, 18);
    layout->setSpacing(12);

    auto *statusRow = new QHBoxLayout;
    statusRow->setSpacing(12);

    m_cmpChip = new QLabel(tr("IDLE"), panel);
    m_cmpChip->setObjectName(QStringLiteral("StateChip"));
    m_cmpChip->setProperty("state", QStringLiteral("idle"));
    m_cmpChip->setAlignment(Qt::AlignCenter);
    m_cmpChip->setMinimumHeight(26);
    m_cmpChip->setMinimumWidth(110);
    statusRow->addWidget(m_cmpChip);

    m_cmpPhase = new QLabel(tr("Choose the two layers and press “Run analysis”."), panel);
    m_cmpPhase->setObjectName(QStringLiteral("PhaseLabel"));
    m_cmpPhase->setWordWrap(true);
    statusRow->addWidget(m_cmpPhase, 1);

    layout->addLayout(statusRow);

    // Folded away by default, as on the run panels. Here it holds what was
    // read rather than what an engine printed — the CRS of each layer, the
    // feature counts, the sampling — which is where a surprising answer is
    // usually explained.
    m_cmpLogHandle = new QToolButton(panel);
    m_cmpLogHandle->setObjectName(QStringLiteral("LogHandle"));
    m_cmpLogHandle->setCursor(Qt::PointingHandCursor);
    m_cmpLogHandle->setAutoRaise(true);
    m_cmpLogHandle->setCheckable(true);
    m_cmpLogHandle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_cmpLogHandle->setArrowType(Qt::RightArrow);
    m_cmpLogHandle->setText(tr("Comparison log"));
    layout->addWidget(m_cmpLogHandle, 0, Qt::AlignLeft);

    m_cmpConsole = new ConsoleView(panel);
    m_cmpConsole->setVisible(false);
    m_cmpConsole->setMinimumHeight(kLogCanvasHeight);
    layout->addWidget(m_cmpConsole, 1);

    QToolButton *const handle = m_cmpLogHandle;
    ConsoleView *const console = m_cmpConsole;
    connect(handle, &QToolButton::toggled, this, [handle, console](bool open) {
        handle->setArrowType(open ? Qt::DownArrow : Qt::RightArrow);
        console->setVisible(open);
    });

    m_cmpSummaryCard = new QFrame(panel);
    m_cmpSummaryCard->setObjectName(QStringLiteral("Card"));
    auto *summaryLayout = new QVBoxLayout(m_cmpSummaryCard);
    summaryLayout->setContentsMargins(18, 12, 18, 12);
    m_cmpSummaryTitle = new QLabel(m_cmpSummaryCard);
    m_cmpSummaryTitle->setObjectName(QStringLiteral("CardTitle"));
    m_cmpResult = new QLabel(m_cmpSummaryCard);
    m_cmpResult->setObjectName(QStringLiteral("SummaryBody"));
    m_cmpResult->setWordWrap(true);
    m_cmpResult->setTextInteractionFlags(Qt::TextSelectableByMouse);
    summaryLayout->addWidget(m_cmpSummaryTitle);
    summaryLayout->addWidget(m_cmpResult);
    m_cmpSummaryCard->setVisible(false);
    layout->addWidget(m_cmpSummaryCard);

    m_cmpPanel = panel;
    return panel;
}

// The coherence tool's result panel: the same card, chip, foldable log and
// summary as the comparison's, for the same reasons.
QWidget *MainWindow::buildCoherencePanel(QWidget *parent)
{
    auto *panel = new QFrame(parent);
    panel->setObjectName(QStringLiteral("Card"));
    auto *layout = new QVBoxLayout(panel);
    layout->setContentsMargins(18, 16, 18, 18);
    layout->setSpacing(12);

    auto *statusRow = new QHBoxLayout;
    statusRow->setSpacing(12);

    m_cohChip = new QLabel(tr("IDLE"), panel);
    m_cohChip->setObjectName(QStringLiteral("StateChip"));
    m_cohChip->setProperty("state", QStringLiteral("idle"));
    m_cohChip->setAlignment(Qt::AlignCenter);
    m_cohChip->setMinimumHeight(26);
    m_cohChip->setMinimumWidth(110);
    statusRow->addWidget(m_cohChip);

    m_cohPhase = new QLabel(
        tr("Choose a surface and a point layer, then press “Run analysis”."), panel);
    m_cohPhase->setObjectName(QStringLiteral("PhaseLabel"));
    m_cohPhase->setWordWrap(true);
    statusRow->addWidget(m_cohPhase, 1);
    layout->addLayout(statusRow);

    m_cohLogHandle = new QToolButton(panel);
    m_cohLogHandle->setObjectName(QStringLiteral("LogHandle"));
    m_cohLogHandle->setCursor(Qt::PointingHandCursor);
    m_cohLogHandle->setAutoRaise(true);
    m_cohLogHandle->setCheckable(true);
    m_cohLogHandle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_cohLogHandle->setArrowType(Qt::RightArrow);
    m_cohLogHandle->setText(tr("Scoring log"));
    layout->addWidget(m_cohLogHandle, 0, Qt::AlignLeft);

    m_cohConsole = new ConsoleView(panel);
    m_cohConsole->setVisible(false);
    m_cohConsole->setMinimumHeight(kLogCanvasHeight);
    layout->addWidget(m_cohConsole, 1);

    QToolButton *const handle = m_cohLogHandle;
    ConsoleView *const console = m_cohConsole;
    connect(handle, &QToolButton::toggled, this, [handle, console](bool open) {
        handle->setArrowType(open ? Qt::DownArrow : Qt::RightArrow);
        console->setVisible(open);
    });

    m_cohSummaryCard = new QFrame(panel);
    m_cohSummaryCard->setObjectName(QStringLiteral("Card"));
    auto *summaryLayout = new QVBoxLayout(m_cohSummaryCard);
    summaryLayout->setContentsMargins(18, 12, 18, 12);
    m_cohSummaryTitle = new QLabel(m_cohSummaryCard);
    m_cohSummaryTitle->setObjectName(QStringLiteral("CardTitle"));
    m_cohResult = new QLabel(m_cohSummaryCard);
    m_cohResult->setObjectName(QStringLiteral("SummaryBody"));
    m_cohResult->setWordWrap(true);
    m_cohResult->setTextInteractionFlags(Qt::TextSelectableByMouse);
    summaryLayout->addWidget(m_cohSummaryTitle);
    summaryLayout->addWidget(m_cohResult);
    m_cohSummaryCard->setVisible(false);
    layout->addWidget(m_cohSummaryCard);

    m_cohPanel = panel;
    return panel;
}

// Each mode card is made wide enough for its own caption to read in full. The
// three do not have to fit on screen at once — the setup page scrolls sideways
// — but a card must never truncate its own text.
//
// Measured from the button's *live* font, not from theme.qss's 15px: a palette
// may bring its own type (Washi's serif, the terminal look of Neon Circuit) and
// the same words are then a different width. Hence the call from applyTheme()
// and applyUiFont() as well as from the constructor.
void MainWindow::refreshModeCardWidths()
{
    for (QPushButton *b : {m_modeSingle, m_modeFete, m_modeLcpa, m_modeBatch}) {
        if (!b)
            continue;
        // ensurePolished() applies the stylesheet, so fontMetrics() reports the
        // font the button will actually be drawn with.
        b->ensurePolished();
        const QFontMetrics fm = b->fontMetrics();
        int textWidth = 0;
        const QStringList lines = b->text().split(QLatin1Char('\n'));
        for (const QString &line : lines)
            textWidth = qMax(textWidth, fm.horizontalAdvance(line));
        b->setMinimumWidth(textWidth + 2 * (16 + 1) + 12);   // padding + border
    }
}

// One label column for the whole page, so that every field — and every "?"
// beside it — begins at the same x whichever card it belongs to.
//
// The width is the widest label of any registered column, measured from the
// widgets themselves once the stylesheet has been applied to them: a palette
// can bring its own typeface, which is why this is redone on every theme and
// font change rather than computed once.
void MainWindow::alignLabelColumns()
{
    // A grid sitting inside a group box starts a few pixels in. What has to
    // line up is the right edge of the label column — where the badges and the
    // fields are — so that indent is counted in when measuring and taken back
    // out when the width is applied.
    auto indentOf = [](QGridLayout *grid) {
        int l = 0, t = 0, r = 0, b = 0;
        grid->getContentsMargins(&l, &t, &r, &b);
        return l;
    };

    int widest = 0;
    for (const auto &lc : std::as_const(m_labelColumns)) {
        QGridLayout *grid = lc.first;
        const int indent = indentOf(grid);
        for (int i = 0; i < grid->count(); ++i) {
            int row = 0, col = 0, rowSpan = 0, colSpan = 0;
            grid->getItemPosition(i, &row, &col, &rowSpan, &colSpan);
            // Only the labels themselves: anything spanning into the field
            // column is a note or a sub-panel as wide as the card, and its
            // width says nothing about how much room the labels need.
            if (col != lc.second || colSpan != 1)
                continue;
            if (QWidget *w = grid->itemAt(i)->widget()) {
                w->ensurePolished();
                widest = qMax(widest, w->sizeHint().width() + indent);
            }
        }
    }
    if (widest <= 0)
        return;
    for (const auto &lc : std::as_const(m_labelColumns))
        lc.first->setColumnMinimumWidth(lc.second,
                                        qMax(0, widest - indentOf(lc.first)));
}

// ---------------------------------------------------------------------------
// Automatic saving (gear menu) and crash recovery
// ---------------------------------------------------------------------------

void MainWindow::refreshAutosaveMenu()
{
    if (!m_autosaveAction)
        return;
    const Checkpoint::Settings s = Checkpoint::settings();
    m_autosaveAction->setChecked(s.enabled);
    // The interval and the folder mean nothing while it is off, and a menu that
    // offers settings for something switched off invites the wrong conclusion.
    if (m_autosaveIntervalMenu)
        m_autosaveIntervalMenu->setEnabled(s.enabled);
    if (m_autosaveFolderAction) {
        m_autosaveFolderAction->setEnabled(s.enabled);
        const QString dir = Checkpoint::activeDir();
        m_autosaveFolderAction->setToolTip(QDir::toNativeSeparators(dir));
    }
    for (QAction *a : std::as_const(m_autosaveIntervalActions))
        a->setChecked(a->data().toInt() == s.minutes);
}

void MainWindow::chooseAutosaveFolder()
{
    Checkpoint::Settings s = Checkpoint::settings();
    const QString start = Checkpoint::activeDir();
    const QString picked = QFileDialog::getExistingDirectory(
        this, tr("Where should a running analysis be saved?"), start);
    if (picked.isEmpty())
        return;
    s.dir = QDir::toNativeSeparators(picked);
    Checkpoint::setSettings(s);
    refreshAutosaveMenu();
}

// A session file left behind means the previous run never reached an orderly
// end. Offered once, at start-up, and only when there is actually something to
// resume from.
void MainWindow::offerCrashRecovery()
{
    const QString dir = Checkpoint::activeDir();
    const Checkpoint::Session session = Checkpoint::readSession(dir);
    if (!session.valid)
        return;
    const Checkpoint::Info info = Checkpoint::latest(dir);
    if (!info.found && !session.batch) {
        // A single run interrupted before its first save has no state at all,
        // so there is nothing to offer. Clear the marker and say nothing.
        Checkpoint::clearSession(dir);
        return;
    }

    // A batch stopped between two rows has no engine state, but the rows it had
    // already finished are still done and only the rest need running.
    const QString progress =
        !info.found
            ? tr("The row it stopped on will start again from the beginning.")
            : (info.sources > 0
                   ? tr("%1 of %2 source points were finished.")
                         .arg(info.nextSource).arg(info.sources)
                   : tr("%1 source points were finished.").arg(info.nextSource));
    const QString saved = info.found ? tr("Last saved: %1").arg(info.modified)
                                     : QString();

    // Two questions, and "Cancel" on the second goes back to the first — hence
    // the loop rather than two nested calls.
    for (;;) {
        const QString opening =
            session.deliberate
                ? tr("An analysis was interrupted before it finished.")
                : tr("An unexpected shutdown was detected.");
        if (TrajectaUi::confirm(
                this, tr("Unfinished analysis"),
                tr("%1\n\n%2\n%3\n%4\n"
                   "Do you want to resume the last process?")
                    .arg(opening)
                    .arg(session.label.isEmpty() ? tr("A FETE analysis") : session.label)
                    .arg(progress)
                    .arg(saved),
                tr("Resume"), tr("No"))) {
            resumeFromCheckpoint(session);
            return;
        }

        // "No" is not the end of it: the saved state is about to be thrown
        // away, and that has to be said out loud before it happens.
        bool backToFirst = false;
        while (!backToFirst) {
            const int choice = TrajectaUi::choose(
                this, tr("Unfinished analysis"),
                tr("If you do not continue this process now, it will be deleted "
                   "from memory."),
                {tr("Cancel"), tr("Continue"), tr("Save process...")});
            if (choice == 0) {
                backToFirst = true;   // ask the first question again
            } else if (choice == 1) {
                Checkpoint::discard(dir);
                return;
            } else {
                const QString target = QFileDialog::getExistingDirectory(
                    this, tr("Where should the unfinished process be kept?"));
                if (target.isEmpty())
                    continue;   // dialog dismissed: ask again rather than delete
                QString error;
                if (Checkpoint::exportTo(dir, target, &error)) {
                    QMessageBox::information(
                        this, tr("Process saved"),
                        tr("The unfinished analysis was moved to:\n%1\n\nTo resume "
                           "it later, point \"Folder for saved analyses\" at that "
                           "folder and restart Trajecta Studio.")
                            .arg(QDir::toNativeSeparators(target)));
                    return;
                }
                QMessageBox::warning(this, tr("Process not saved"), error);
            }
        }
    }
}

void MainWindow::offerWalkthroughOnFirstRun()
{
    // Asked once per user profile, ever. Reinstalling does not bring it back —
    // the key lives in the profile, not beside the executable — and the first
    // start of the version that introduced the tour is where a user of long
    // standing meets it. That is how the feature announces itself.
    static const QString kOfferedKey = QStringLiteral("ui/walkthroughOffered");
    const QStringList args = QCoreApplication::arguments();
    // --offer-tour (hidden, testing): show the offer whatever the state of the
    // key and whatever else is on the command line, and — the point of it —
    // without writing the key, so a screenshot run cannot use up the one
    // question a real user gets asked.
    const bool forced = args.contains(QStringLiteral("--offer-tour"));

    QSettings settings;
    if (!forced && settings.value(kOfferedKey, false).toBool())
        return;

    // Any switch at all means the application is being driven rather than
    // used: the hidden hooks exist for screenshots and scripted checks, and a
    // modal in front of one of those makes it a picture of a dialog. The rule
    // is deliberately "is there a --switch" rather than a list of the ones that
    // exist today — a list would have to be remembered every time a new hook is
    // added, and the one thing certain about that is that it would not be.
    if (!forced) {
        for (int i = 1; i < args.size(); ++i) {
            if (args.at(i).startsWith(QLatin1String("--")))
                return;
        }
    }

    // An unfinished analysis was offered a moment ago and the user is dealing
    // with it, or has just resumed it. Deliberately without writing the key:
    // the question is asked again at the first quiet start.
    if (!forced
        && ((m_runner && m_runner->isRunning())
            || (m_batchPage && m_batchPage->isRunning())
            || Checkpoint::readSession(Checkpoint::activeDir()).valid)) {
        return;
    }

    // Written the moment the choice is made, not after the second dialog:
    // closing that one with the X must not make the question come back.
    if (!forced) {
        settings.setValue(kOfferedKey, true);
        settings.sync();
    }

    // A little taller than an ordinary confirmation: this one is a greeting
    // with a question at the end of it, not a question.
    if (TrajectaUi::confirm(
            this, tr("Welcome to Trajecta Studio"),
            tr("There is a short guided tour of the interface: what each page "
               "is for, what every setting does, and how a result is read.\n\n"
               "Trajecta Studio will be maximised, so that every screen of the "
               "tour has room to be shown whole.\n\n"
               "Nothing you have set up is changed, and the tour can be stopped "
               "at any point."),
            tr("Start walkthrough"), tr("Not now"), 40,
            TrajectaUi::Fill::Accept)) {
        startWalkthroughMaximised();
        return;
    }

    TrajectaUi::notify(
        this, tr("It stays available"),
        tr("The walkthrough can be started whenever you like: open the "
           "<b>Guide</b> page and click the <b>tutorial</b> link near the top."));
}

void MainWindow::exportCheckpointCopy()
{
    const QString dir = Checkpoint::activeDir();
    const Checkpoint::Info info = Checkpoint::latest(dir);
    const Checkpoint::Session session = Checkpoint::readSession(dir);
    if (!info.found && !session.valid) {
        const Checkpoint::Settings cp = Checkpoint::settings();
        TrajectaUi::notify(
            this, tr("Nothing to save yet"),
            cp.enabled
                ? tr("Auto-save is on, but it has not written a checkpoint yet. "
                     "The first one is written %1 minutes into a FETE run.")
                      .arg(cp.minutes)
                : tr("There is no saved state to copy.\n\nAuto-save is off. Turn "
                     "it on from the gear menu before a long FETE run, and "
                     "Trajecta will keep its progress on disk."));
        return;
    }

    const QString target = QFileDialog::getExistingDirectory(
        this, tr("Where should the copy be kept?"), m_lastOutputDir);
    if (target.isEmpty())
        return;

    QString error;
    if (!Checkpoint::copyTo(dir, target, &error)) {
        TrajectaUi::notify(this, tr("Copy not made"), error);
        return;
    }
    // What was copied, in the terms the user thinks in: how far the analysis
    // had got. A folder full of .tckpt files says nothing on its own.
    const QString progress =
        info.found ? (info.sources > 0
                          ? tr("%1 of %2 source points were finished.")
                                .arg(info.nextSource).arg(info.sources)
                          : tr("%1 source points were finished.").arg(info.nextSource))
                   : tr("The batch had finished the rows before the current one.");
    TrajectaUi::notify(
        this, tr("Copy saved"),
        tr("%1\n\nA copy was written to:\n%2\n\nUse \"Resume from a checkpoint "
           "file...\" to pick it up again.")
            .arg(progress, QDir::toNativeSeparators(target)));
}

void MainWindow::importCheckpointAndResume()
{
    if (m_runner && m_runner->isRunning()) {
        TrajectaUi::notify(
            this, tr("An analysis is already running"),
            tr("Wait for the current run to finish, or cancel it, before "
               "resuming a saved one."));
        return;
    }
    if (m_batchPage && m_batchPage->isRunning()) {
        TrajectaUi::notify(
            this, tr("A batch is already running"),
            tr("Wait for the batch to finish, or stop it, before resuming a "
               "saved analysis."));
        return;
    }

    // The checkpoint itself is what the user was told to look for, so that is
    // what the dialog asks for; the session file that has to travel with it
    // lives in the same folder and is found from there.
    const QString file = QFileDialog::getOpenFileName(
        this, tr("Open a saved checkpoint"), m_lastOutputDir,
        tr("Trajecta checkpoint (*.tckpt);;All files (*)"));
    if (file.isEmpty())
        return;

    const QString sourceDir = QFileInfo(file).absolutePath();
    Checkpoint::Session session = Checkpoint::readSession(sourceDir);
    if (!session.valid) {
        TrajectaUi::notify(
            this, tr("Cannot resume from this file"),
            tr("The folder holding this checkpoint has no session.json beside "
               "it.\n\nThat file records which DEM, which points and which "
               "settings the run used — without it the checkpoint is a block "
               "of numbers with nothing to attach it to. Copy the whole folder, "
               "not the checkpoint alone."));
        return;
    }

    const QString dir = Checkpoint::activeDir();
    // Anything already there belongs to a different run and is about to be
    // overwritten. Say so first: it may be the state of an analysis the user
    // has not resumed yet.
    const Checkpoint::Session existing = Checkpoint::readSession(dir);
    if (existing.valid
        && QFileInfo(sourceDir).canonicalFilePath() != QFileInfo(dir).canonicalFilePath()) {
        if (!TrajectaUi::confirm(
                this, tr("Replace the state now in memory?"),
                tr("Trajecta is already holding an unfinished analysis (%1).\n\n"
                   "Loading this file replaces it, and the one being replaced "
                   "is lost.")
                    .arg(existing.label.isEmpty() ? tr("a FETE analysis")
                                                  : existing.label),
                tr("Replace"), tr("Cancel"))) {
            return;
        }
    }

    QString error;
    if (QFileInfo(sourceDir).canonicalFilePath() != QFileInfo(dir).canonicalFilePath()
        && !Checkpoint::importFrom(sourceDir, dir, &error)) {
        TrajectaUi::notify(this, tr("Cannot resume from this file"), error);
        return;
    }

    session = Checkpoint::readSession(dir);
    const Checkpoint::Info info = Checkpoint::latest(dir);
    if (!session.valid || (!info.found && !session.batch)) {
        TrajectaUi::notify(
            this, tr("Cannot resume from this file"),
            tr("The checkpoint was copied but could not be read back. It may "
               "have been written by a different version of Trajecta."));
        return;
    }

    const QString progress =
        info.found ? (info.sources > 0
                          ? tr("%1 of %2 source points were finished.")
                                .arg(info.nextSource).arg(info.sources)
                          : tr("%1 source points were finished.").arg(info.nextSource))
                   : tr("The row it stopped on will start again from the beginning.");
    if (!TrajectaUi::confirm(
            this, tr("Resume this analysis?"),
            tr("%1\n%2\n%3\n\nThe run continues from where it stopped.")
                .arg(session.label.isEmpty() ? tr("A FETE analysis") : session.label,
                     progress,
                     info.found ? tr("Last saved: %1").arg(info.modified) : QString()),
            tr("Resume"), tr("Cancel"))) {
        return;
    }
    resumeFromCheckpoint(session);
}

void MainWindow::resumeFromCheckpoint(const Checkpoint::Session &session)
{
    const QString dir = Checkpoint::activeDir();
    const Checkpoint::Info info = Checkpoint::latest(dir);
    // A single run has nothing to resume without engine state; a batch does —
    // the rows it had already finished.
    if (!info.found && !session.batch)
        return;

    if (session.batch && session.isPostBatch && m_postBatchPage) {
        // None of NNI, Compare or Coherence can pick up mid-chunk (see
        // PostBatchController's class comment) — the chunk it stopped on
        // always restarts from its own beginning — so there is no checkpoint
        // path to hand over, only which chunk to resume at.
        m_postBatchPage->resumeJob(session.job, session.queueIndex);
        showPage(1);
        selectPostMode(QStringLiteral("batch"));
        return;
    }
    if (session.batch && m_batchPage) {
        // The interrupted row resumes from its checkpoint, and the rows after
        // it run normally. Handing the whole job back to the batch page keeps
        // one code path for both.
        m_batchPage->resumeJob(session.job, session.queueIndex, info.path);
        selectMode(QStringLiteral("batch"));
        return;
    }

    TrajectaRunner::Parameters params = Checkpoint::fromJson(session.params);
    // The engine and GDAL are located afresh: this may be a different machine,
    // or a reinstall.
    const TrajectaRunner::Parameters env = currentEnvironment();
    params.exePath = env.exePath;
    params.gdalBinDir = env.gdalBinDir;
    params.projDataDir = env.projDataDir;
    params.gdalDataDir = env.gdalDataDir;
    params.workingDir = env.workingDir;
    params.resumeCheckpoint = info.path;
    const Checkpoint::Settings cp = Checkpoint::settings();
    params.checkpointEnabled = cp.enabled;
    params.checkpointMinutes = cp.minutes;
    params.checkpointDir = dir;

    if (params.exePath.isEmpty()) {
        QMessageBox::warning(this, tr("Engine not found"),
                             tr("The analysis cannot be resumed because "
                                "trajecta.exe was not found. Use \"Locate "
                                "engine...\" and try again from the gear menu."));
        return;
    }
    m_lastOutputDir = params.outputDir;
    beginRun(params);
    // Even if automatic saving has since been switched off, the state this run
    // was resumed from belongs to it and has to be cleared when it ends —
    // otherwise the same recovery would be offered again after a clean finish.
    m_lastRunCheckpointDir = dir;
}

// Brings the run panel into view on the setup page. Called when a run starts,
// which used to be a page change and now has to be a scroll.
void MainWindow::revealRunPanel()
{
    switchPage(0);
    if (!m_setupScroll || !m_runPanel)
        return;
    // Queued: the panel may still be hidden or unlaid-out when a run starts
    // right after a mode change, and ensureWidgetVisible would aim at the
    // wrong place.
    QTimer::singleShot(0, this, [this] {
        if (m_setupScroll && m_runPanel && m_runPanel->isVisible())
            m_setupScroll->ensureWidgetVisible(m_runPanel, 0, 0);
    });
}

// ---------------------------------------------------------------------------
// Post-processing page (NNI)
// ---------------------------------------------------------------------------

QWidget *MainWindow::buildPostPage()
{
    auto *page = new QWidget(this);
    auto *pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(0, 0, 0, 0);
    pageLayout->setSpacing(0);

    auto *scroll = new QScrollArea(page);
    m_postScroll = scroll;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);

    auto *inner = new QWidget(scroll);
    auto *layout = new QVBoxLayout(inner);
    layout->setContentsMargins(28, 24, 28, 12);
    layout->setSpacing(14);

    // Asked before which tool, the same way — and for the same reason — as
    // the Processing page's own "Analysis type" card: Single analysis shows
    // the tool card right below; Batch processing replaces everything below
    // this card with the post-processing batch page.
    {
        auto *content = new QWidget(inner);
        auto *row = new QHBoxLayout(content);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(12);

        m_postModeSingle = new QPushButton(
            tr("Single analysis\n"
               "Run one NNI, comparison or coherence pass"),
            content);
        m_postModeSingle->setToolTip(
            tr("The tool card below chooses NNI, Compare or Coherence; "
               "everything else on the page is that one analysis."));
        // Same title and subtitle as Processing's own Batch processing card —
        // it is the same feature, offered in a second place — with a tooltip
        // adjusted for what actually differs here: no shared algorithm or
        // cost modifiers, because a post-processing chunk already is one
        // analysis rather than a group of rows (see postbatchmodel.h).
        m_postModeBatch = new QPushButton(
            tr("Batch processing\n"
               "Runs many analyses in a row, unattended"),
            content);
        m_postModeBatch->setToolTip(
            tr("Queues several NNI, comparison or coherence runs and executes "
               "them one after another: one chunk per analysis, of whichever "
               "of the three tools is selected."));
        m_postModeSingle->setProperty("mode", QStringLiteral("single"));
        m_postModeBatch->setProperty("mode", QStringLiteral("postbatch"));
        for (QPushButton *b : {m_postModeSingle, m_postModeBatch}) {
            b->setObjectName(QStringLiteral("ModeCard"));
            b->setCheckable(true);
            b->setCursor(Qt::PointingHandCursor);
            b->setMinimumHeight(72);
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            row->addWidget(b, 1);
        }
        auto *postTypeGroup = new QButtonGroup(content);
        postTypeGroup->setExclusive(true);
        postTypeGroup->addButton(m_postModeSingle);
        postTypeGroup->addButton(m_postModeBatch);
        m_postModeSingle->setChecked(true);
        connect(postTypeGroup, &QButtonGroup::buttonClicked, this,
                [this](QAbstractButton *) { updatePostModeUi(); });

        m_postCardAnalysisType = makeCard(
            tr("Analysis type"),
            tr("Choose whether to run one analysis, or queue several as a batch."),
            content);
        layout->addWidget(m_postCardAnalysisType);
    }

    // Two tools that have nothing to do with each other share this page, so it
    // opens the same way the Processing page does: pick what you are doing
    // first, and only that tool's settings are on screen. Showing both at once
    // made the page read as one long form with an unrelated half.
    {
        auto *content = new QWidget(inner);
        auto *cardLayout = new QVBoxLayout(content);
        cardLayout->setContentsMargins(0, 0, 0, 0);
        // Same 12 as the equivalent card on the Post-processing batch page
        // (see postbatchpage.cpp) — see the identical note on the Processing
        // page's tool card.
        cardLayout->setSpacing(12);

        auto *modeRow = new QWidget(content);
        auto *row = new QHBoxLayout(modeRow);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(12);

        m_postModeNni = new QPushButton(
            tr("NNI — Natural Neighbour Interpolation\n"
               "Turns a sparse density raster into a continuous surface"),
            content);
        m_postModeNni->setToolTip(
            tr("Discrete Sibson interpolation over the cells of a density raster."));
        m_postModeCompare = new QPushButton(
            tr("Compare with a known route\n"
               "Measures how closely a computed route follows a real one"),
            content);
        m_postModeCompare->setToolTip(
            tr("Geometric comparison between two vector layers: distances in both "
               "directions, and how much of each line runs close to the other."));
        m_postModeCoherence = new QPushButton(
            tr("Site-corridor coherence\n"
               "Scores how well a set of sites sits on the predicted corridors"),
            content);
        m_postModeCoherence->setToolTip(
            tr("Per-site distance to the nearest corridor and intensity of the "
               "surrounding movement, with a test against random point sets."));

        // Their own two colours, not the analysis modes': a card filled with
        // FETE's green on one page and NNI's on the other taught the wrong
        // thing about what the colour means.
        m_postModeNni->setProperty("mode", QStringLiteral("nni"));
        m_postModeCompare->setProperty("mode", QStringLiteral("compare"));
        m_postModeCoherence->setProperty("mode", QStringLiteral("coherence"));
        for (QPushButton *b : {m_postModeNni, m_postModeCompare, m_postModeCoherence}) {
            b->setObjectName(QStringLiteral("ModeCard"));
            b->setCheckable(true);
            b->setCursor(Qt::PointingHandCursor);
            b->setMinimumHeight(72);
            b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
            row->addWidget(b, 1);
        }
        auto *postGroup = new QButtonGroup(content);
        postGroup->setExclusive(true);
        postGroup->addButton(m_postModeNni);
        postGroup->addButton(m_postModeCompare);
        postGroup->addButton(m_postModeCoherence);
        m_postModeNni->setChecked(true);
        connect(postGroup, &QButtonGroup::buttonClicked, this,
                [this](QAbstractButton *) { updatePostModeUi(); });
        cardLayout->addWidget(modeRow);

        // ----- Hardware resources, folded into this same card, NNI only -----
        // Compare and Coherence run inside the interface, not as a trajecta.exe
        // process, and finish in a moment — there is nothing here for them to
        // configure. NNI is the one post-processing tool that is a real engine
        // run, so it is the one that shows this box — see updatePostModeUi().
        // Threads and RAM in one horizontal row — the same disposition as the
        // hardware row on the Batch processing card (see batchpage.cpp), so
        // the two fields look and behave the same wherever they appear.
        m_postHardwareBox = new QWidget(content);
        auto *postHwLayout = new QVBoxLayout(m_postHardwareBox);
        postHwLayout->setContentsMargins(0, 0, 0, 0);
        // 12, not the wider 16 this carried before: the manifest checkbox
        // sits the same distance below the CPU/RAM row here as it does in
        // the batch chunk's equivalent hardware block (postbatchpage.cpp).
        postHwLayout->setSpacing(12);

        auto *hwRow = new QHBoxLayout;
        hwRow->setSpacing(10);

        const int maxThreads = qMax(1, QThread::idealThreadCount());
        const int recommendedThreads = qMax(1, maxThreads - 4);
        const qint64 totalRam = SystemInfo::totalRamMb();
        const int recommendedRam =
            int(qMin<qint64>(SystemInfo::kRecommendedRamMb, totalRam));

        hwRow->addWidget(makeFieldLabel(
            tr("CPU threads"),
            tr("Number of parallel CPU threads used for the interpolation. "
               "Keeping a few cores free preserves system responsiveness."),
            m_postHardwareBox));
        m_postThreadsSpin = new QSpinBox(m_postHardwareBox);
        m_postThreadsSpin->setRange(1, maxThreads);
        m_postThreadsSpin->setValue(recommendedThreads);
        m_postThreadsSpin->setMinimumWidth(batchSpinBoxWidth(1, 1024));
        hwRow->addWidget(m_postThreadsSpin);

        hwRow->addWidget(makeFieldLabel(
            tr("Maximum RAM"),
            tr("Memory ceiling used for the interpolation. At least %1 MB "
               "of RAM is recommended, and raising the ceiling further does "
               "not make the computation any faster.")
                   .arg(SystemInfo::kRecommendedRamMb)
               + TrajectaUi::ramHeadroomNote(),
            m_postHardwareBox));
        m_postRamSpin = new QSpinBox(m_postHardwareBox);
        m_postRamSpin->setRange(512, int(totalRam));
        m_postRamSpin->setSingleStep(512);
        m_postRamSpin->setSuffix(QStringLiteral(" MB"));
        m_postRamSpin->setValue(recommendedRam);
        m_postRamSpin->setMinimumWidth(batchSpinBoxWidth(256, 1024 * 1024, QStringLiteral(" MB")));
        hwRow->addWidget(m_postRamSpin);
        auto *ramHint = new QLabel(
            tr("at least %1 MB of RAM is recommended")
                .arg(SystemInfo::kRecommendedRamMb),
            m_postHardwareBox);
        ramHint->setObjectName(QStringLiteral("HintLabel"));
        hwRow->addWidget(ramHint);
        hwRow->addStretch(1);
        postHwLayout->addLayout(hwRow);

        // On by default, like everywhere else it appears: not a diagnostic
        // tool, but the record of what produced a result.
        m_postManifestCheck = new QCheckBox(
            tr("Write a run manifest next to the results"), m_postHardwareBox);
        m_postManifestCheck->setChecked(true);
        auto *optionsRow = new QHBoxLayout;
        optionsRow->setSpacing(24);
        optionsRow->addWidget(
            TrajectaUi::withHelpDot(m_postManifestCheck, TrajectaUi::manifestHelpText()));
        optionsRow->addStretch(1);
        postHwLayout->addLayout(optionsRow);

        cardLayout->addWidget(m_postHardwareBox);

        m_cardPostTool = makeCard(tr("Tool selection"),
                                  tr("Choose the analysis tool to use."),
                                  content);
        layout->addWidget(m_cardPostTool);
    }

    {
        auto *content = new QWidget(inner);
        auto *grid = new QGridLayout(content);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);

        int r = 0;
        auto addRow = [&](const QString &label, const QString &help, QWidget *w) {
            grid->addWidget(makeFieldLabel(label, help, content), r, 0);
            grid->addWidget(w, r, 1);
            ++r;
        };

        m_interpInputPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                             tr("Select the density raster (GeoTIFF)"),
                                             QString::fromLatin1(kRasterFilter), content);
        m_interpInputPicker->setPlaceholder(
            tr("FETE density raster (.tif) — filled automatically after a FETE run"));
        addRow(tr("Density raster"),
               tr("The raster to interpolate, typically the FETE density output. "
                  "Cells at or above the sample threshold become the sample points "
                  "of the interpolation."),
               m_interpInputPicker);

        m_interpOutputDirPicker = new PathPicker(PathPicker::Kind::Directory,
                                                 tr("Select the output folder"),
                                                 QString(), content);
        m_interpOutputDirPicker->setPlaceholder(tr("Folder where the interpolated raster will be written"));
        addRow(tr("Output folder"),
               tr("Folder where the interpolated GeoTIFF will be written."),
               m_interpOutputDirPicker);

        m_interpThresholdSpin = new QDoubleSpinBox(content);
        m_interpThresholdSpin->setRange(0.0, 1e9);
        m_interpThresholdSpin->setDecimals(1);
        m_interpThresholdSpin->setValue(1.0);
        addRow(tr("Sample threshold"),
               tr("Cells with a value at or above this threshold are used as "
                  "sample points. 1 keeps every non-zero density cell."),
               m_interpThresholdSpin);

        m_interpSpacingSpin = new QSpinBox(content);
        m_interpSpacingSpin->setRange(1, 1000);
        m_interpSpacingSpin->setValue(4);
        m_interpSpacingSpin->setSuffix(tr(" cell(s)"));
        addRow(tr("Sample spacing"),
               tr("Samples are taken every N cells on a regular grid. 1 uses "
                  "every qualifying cell — on a dense raster that reproduces "
                  "the input almost unchanged. Larger values (4-10) generalize "
                  "the surface, keeping only the broad corridor structure. A "
                  "value between 2 and 6 is usually recommended for best "
                  "results. 4 is the default."),
               m_interpSpacingSpin);

        // Only meaningful once the spacing is throwing cells away.
        m_interpPeaksCheck = new QCheckBox(tr("Preserve local peaks"), content);
        m_interpPeaksCheck->setChecked(false);
        QWidget *peaksRow = TrajectaUi::withHelpDot(m_interpPeaksCheck,
                                                    TrajectaUi::preservePeaksHelpText());
        grid->addWidget(peaksRow, r, 1);
        ++r;
        connect(m_interpSpacingSpin, QOverload<int>::of(&QSpinBox::valueChanged),
                this, [this](int v) { m_interpPeaksCheck->setEnabled(v > 1); });
        m_interpPeaksCheck->setEnabled(m_interpSpacingSpin->value() > 1);

        m_interpRadiusSpin = new QSpinBox(content);
        m_interpRadiusSpin->setRange(0, 100000);
        m_interpRadiusSpin->setValue(0);
        m_interpRadiusSpin->setSuffix(tr(" cell(s)"));
        m_interpRadiusSpin->setSpecialValueText(tr("unlimited"));
        addRow(tr("Max search radius"),
               tr("Limits how far the interpolation reaches into areas without "
                  "samples; beyond it, cells take the value of their nearest "
                  "sample. Unlimited is the classic natural neighbour behaviour; "
                  "a cap keeps large, mostly-empty rasters fast."),
               m_interpRadiusSpin);

        m_interpNameEdit = new QLineEdit(QStringLiteral("FETE_density_NNI"), content);
        addRow(tr("Output filename"),
               tr("Name of the interpolated raster, without extension."),
               m_interpNameEdit);

        m_postNniCard = makeCard(
            tr("Natural Neighbour Interpolation (NNI)"), QString(), content,
            tr("Turns the sparse FETE density raster into a smooth, continuous "
               "surface with discrete Sibson interpolation: each cell receives "
               "the weighted average of the sample cells whose influence area "
               "it would claim. The result keeps the sample values exactly and "
               "transitions smoothly between corridors. NNI helps creating more "
               "realistic movements corridors as well as improving "
               "visualization of the results."));
        layout->addWidget(m_postNniCard);
    }

    // --- Compare with a known route ---
    // Not an interpolation, so it gets a card of its own. It runs in the
    // interface rather than in the engine: it is pure geometry over two vector
    // layers, needs no DEM and no cost surface, and finishes in a moment.
    {
        auto *content = new QWidget(inner);
        auto *grid = new QGridLayout(content);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);
        m_labelColumns.append({grid, 0});

        int r = 0;
        auto addRow = [&](const QString &label, const QString &help, QWidget *w) {
            grid->addWidget(makeFieldLabel(label, help, content), r, 0);
            grid->addWidget(w, r, 1);
            ++r;
        };

        m_cmpComputedPicker = new PathPicker(
            PathPicker::Kind::ExistingFile,
            tr("Select the computed routes (vector)"),
            QString::fromLatin1(kVectorFilter), content);
        m_cmpComputedPicker->setPlaceholder(
            tr("Usually the LCPA paths shapefile produced by a run"));
        addRow(tr("Computed routes"),
               tr("The routes the model produced — normally the paths shapefile "
                  "from an LCPA run, but any line layer will do."),
               m_cmpComputedPicker);

        m_cmpKnownPicker = new PathPicker(
            PathPicker::Kind::ExistingFile,
            tr("Select the known route (vector)"),
            QString::fromLatin1(kVectorFilter), content);
        m_cmpKnownPicker->setPlaceholder(
            tr("The real route: a surveyed road, a historic track, a mapped path"));
        addRow(tr("Known route"),
               tr("The route you are testing the model against. Both layers must "
                  "be in the same projected CRS."),
               m_cmpKnownPicker);

        m_cmpToleranceSpin = new QDoubleSpinBox(content);
        m_cmpToleranceSpin->setRange(1.0, 100000.0);
        m_cmpToleranceSpin->setDecimals(0);
        m_cmpToleranceSpin->setValue(100.0);
        m_cmpToleranceSpin->setSuffix(tr(" m"));
        guardWheel(m_cmpToleranceSpin);
        addRow(tr("Tolerance"), TrajectaUi::routeCompareHelpText(), m_cmpToleranceSpin);

        m_postCompareCard = makeCard(
            tr("Compare with a known route"), QString(), content,
            tr("Measures how closely the computed routes follow a route that is "
               "actually known — a surveyed road, a historic track or any "
               "other known path. Without this "
               "step a least-cost model can only ever agree with itself; with "
               "it, the model can be shown to be wrong, which is what makes "
               "agreement worth anything."));
        layout->addWidget(m_postCompareCard);
    }

    // --- Site-corridor coherence ---
    // The third tool, and the one that asks the question the FETE was computed
    // for: do the sites sit on the movement the surface predicts? Like the
    // comparison it runs here rather than in the engine.
    {
        auto *content = new QWidget(inner);
        auto *grid = new QGridLayout(content);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);
        m_labelColumns.append({grid, 0});

        int r = 0;
        auto addRow = [&](const QString &label, const QString &help, QWidget *w) {
            grid->addWidget(makeFieldLabel(label, help, content), r, 0);
            grid->addWidget(w, r, 1);
            ++r;
        };

        m_cohRasterPicker = new PathPicker(
            PathPicker::Kind::ExistingFile, tr("Select the FETE surface (raster)"),
            QString::fromLatin1(kRasterFilter), content);
        m_cohRasterPicker->setPlaceholder(
            tr("The FETE density raster, raw or interpolated with NNI"));
        addRow(tr("FETE surface"), TrajectaUi::coherenceSurfaceHelpText(),
               m_cohRasterPicker);

        m_cohPointsPicker = new PathPicker(
            PathPicker::Kind::ExistingFile, tr("Select the sites (point layer)"),
            QString::fromLatin1(kVectorFilter), content);
        m_cohPointsPicker->setPlaceholder(
            tr("The sites to score, in the same projected CRS as the raster"));
        addRow(tr("Sites"), TrajectaUi::coherenceSitesHelpText(), m_cohPointsPicker);

        {
            auto *box = new QWidget(content);
            auto *row = new QHBoxLayout(box);
            row->setContentsMargins(0, 0, 0, 0);
            row->setSpacing(10);
            m_cohRadiusSpin = new QDoubleSpinBox(box);
            m_cohRadiusSpin->setRange(1.0, 1000000.0);
            m_cohRadiusSpin->setDecimals(0);
            m_cohRadiusSpin->setValue(250.0);
            m_cohRadiusSpin->setSuffix(tr(" m"));
            guardWheel(m_cohRadiusSpin);
            row->addWidget(m_cohRadiusSpin);
            // How many cells that is. A radius means nothing until you know
            // whether it is three cells or three hundred.
            m_cohCellNote = new QLabel(box);
            m_cohCellNote->setObjectName(QStringLiteral("HintLabel"));
            row->addWidget(m_cohCellNote, 1);
            addRow(tr("Radius"), TrajectaUi::coherenceRadiusHelpText(), box);
            connect(m_cohRadiusSpin, &QDoubleSpinBox::valueChanged, this,
                    [this] { refreshCoherenceUi(); });
            connect(m_cohRasterPicker, &PathPicker::pathChanged, this,
                    [this] { refreshCoherenceUi(); });
        }

        {
            auto *box = new QWidget(content);
            auto *row = new QHBoxLayout(box);
            row->setContentsMargins(0, 0, 0, 0);
            row->setSpacing(10);
            m_cohThresholdCombo = new SmoothComboBox(box);
            m_cohThresholdCombo->addItem(tr("Top percentage of the surface"), 0);
            m_cohThresholdCombo->addItem(tr("Automatic (Otsu, on log values)"), 1);
            m_cohThresholdCombo->addItem(tr("Cells at or above a value"), 2);
            row->addWidget(m_cohThresholdCombo, 1);
            m_cohThresholdSpin = new QDoubleSpinBox(box);
            m_cohThresholdSpin->setRange(0.001, 1000000000.0);
            // Two decimals in per-cent mode: three made "1,000 %" out of one
            // per cent on a locale that separates decimals with a comma, which
            // reads as a thousand.
            m_cohThresholdSpin->setDecimals(2);
            m_cohThresholdSpin->setValue(1.0);
            m_cohThresholdSpin->setSuffix(tr(" %"));
            guardWheel(m_cohThresholdSpin);
            row->addWidget(m_cohThresholdSpin);
            addRow(tr("Corridor"), TrajectaUi::coherenceThresholdHelpText(), box);
            connect(m_cohThresholdCombo, &QComboBox::currentIndexChanged, this,
                    [this] { refreshCoherenceUi(); });
        }

        {
            auto *box = new QWidget(content);
            auto *row = new QHBoxLayout(box);
            row->setContentsMargins(0, 0, 0, 0);
            row->setSpacing(10);
            m_cohSensCheck = new QCheckBox(tr("Also report other radii"), box);
            row->addWidget(m_cohSensCheck);
            m_cohSensEdit = new QLineEdit(QStringLiteral("100, 250, 500, 1000"), box);
            row->addWidget(m_cohSensEdit, 1);
            addRow(tr("Sensitivity"), TrajectaUi::coherenceSensitivityHelpText(), box);
            connect(m_cohSensCheck, &QCheckBox::toggled, this,
                    [this] { refreshCoherenceUi(); });
        }

        m_cohEcdfEdit = new QLineEdit(QStringLiteral("0, 100, 250, 500, 1000, 2500"),
                                      content);
        addRow(tr("Distance bands"), TrajectaUi::coherenceEcdfHelpText(),
               m_cohEcdfEdit);

        {
            // Comes after the distance bands, not before: it tests those same
            // distances against chance, so the choice it depends on has to be
            // on the page above it.
            auto *box = new QWidget(content);
            auto *row = new QHBoxLayout(box);
            row->setContentsMargins(0, 0, 0, 0);
            row->setSpacing(10);
            m_cohNullCheck = new QCheckBox(tr("Test against random point sets"), box);
            m_cohNullCheck->setChecked(true);
            row->addWidget(m_cohNullCheck);
            m_cohNullModeCombo = new SmoothComboBox(box);
            m_cohNullModeCombo->addItem(tr("the same pattern, moved as a block"), 0);
            m_cohNullModeCombo->addItem(tr("scattered points"), 1);
            row->addWidget(m_cohNullModeCombo, 1);
            m_cohRepsSpin = new QSpinBox(box);
            m_cohRepsSpin->setRange(99, 9999);
            m_cohRepsSpin->setValue(999);
            m_cohRepsSpin->setPrefix(tr("× "));
            guardWheel(m_cohRepsSpin);
            row->addWidget(m_cohRepsSpin);
            addRow(tr("Null model"), TrajectaUi::coherenceNullHelpText(), box);
            connect(m_cohNullCheck, &QCheckBox::toggled, this,
                    [this] { refreshCoherenceUi(); });
        }

        m_cohEdgeCheck = new QCheckBox(
            tr("Flag sites within one radius of the raster's edge"), content);
        m_cohEdgeCheck->setChecked(true);
        addRow(tr("Edge guard"), TrajectaUi::coherenceEdgeHelpText(), m_cohEdgeCheck);

        m_cohRScriptCheck = new QCheckBox(
            tr("Write an R script (ggplot2) for the distance histogram"), content);
        m_cohRScriptCheck->setChecked(true);
        addRow(tr("Histogram script"), TrajectaUi::coherenceHistogramScriptHelpText(),
               m_cohRScriptCheck);

        m_cohOutPicker = new PathPicker(PathPicker::Kind::Directory,
                                        tr("Select the output folder"),
                                        QString(), content);
        m_cohOutPicker->setPlaceholder(tr("Folder where the scored sites will be written"));
        addRow(tr("Output folder"),
               tr("Where the table, the layer and the distance raster are "
                  "written. The folder is created if it does not exist."),
               m_cohOutPicker);

        {
            auto *box = new QWidget(content);
            auto *row = new QHBoxLayout(box);
            row->setContentsMargins(0, 0, 0, 0);
            row->setSpacing(10);
            m_cohPrefixEdit = new QLineEdit(QStringLiteral("coherence"), box);
            // Stretch factor 1 with the dropdown and checkbox left at the
            // default 0: Qt hands 100% of the row's surplus width to the
            // only item with a nonzero stretch, so the prefix fills the
            // line while the other two stay pinned at their natural size.
            row->addWidget(m_cohPrefixEdit, 1);
            m_cohVectorCombo = new SmoothComboBox(box);
            m_cohVectorCombo->addItem(tr("GeoPackage"), 0);
            m_cohVectorCombo->addItem(tr("Shapefile"), 1);
            row->addWidget(m_cohVectorCombo);
            m_cohRasterCheck = new QCheckBox(tr("Distance raster"), box);
            m_cohRasterCheck->setChecked(true);
            row->addWidget(TrajectaUi::withHelpDot(
                m_cohRasterCheck,
                tr("Every cell holds its distance in metres to the nearest "
                   "corridor cell. Optional, and nearly free to add — it is "
                   "computed anyway. The fastest way to see the catchment of "
                   "the network, to read the score of a place not surveyed, "
                   "and to notice a threshold set too generously.")));
            addRow(tr("Outputs"), TrajectaUi::coherenceOutputHelpText(), box);
        }

        m_postCoherenceCard = makeCard(
            tr("Site-corridor coherence"), QString(), content,
            tr("This feature was created to assess if the settlements where "
               "people lived sit on the routes the model predicts. Coherence "
               "analysis answers this question in four steps, from the most "
               "general down:<br><br>"
               "1) How many sites are located near a corridor at all?<br>"
               "2) How far are settlements from the corridors?<br>"
               "3) How much corridor lies around the settlements?<br>"
               "4) How busy are the corridors located in the vicinity of the "
               "settlements?<br><br>"
               "Coherence analysis is built so that it allows to compare — "
               "for example — different periods or regions even when their "
               "FETE rasters were computed from different numbers of "
               "points."));
        layout->addWidget(m_postCoherenceCard);
    }

    // Run bar. Each tool has its own, below its own parameters: this one starts
    // the interpolation and travels with it.
    {
        m_postRunRow = new QWidget(inner);
        // A widget rather than a bare layout because the whole row is shown and
        // hidden with the tool; the object name is what keeps it from painting
        // the window colour across the page — see QWidget#RunRow in theme.qss.
        m_postRunRow->setObjectName(QStringLiteral("RunRow"));
        auto *row = new QHBoxLayout(m_postRunRow);
        // Same 4 px top and bottom as every other floating run button on
        // this page and on Processing — see the matching comment on
        // buildSetupPage()'s own run bar.
        row->setContentsMargins(0, 4, 0, 4);
        row->addStretch(1);

        // "Run analysis", not "Run interpolation": all four floating run
        // buttons across Processing and Post-processing say the same thing
        // now — which tool it starts is already the one card above it.
        m_runInterpButton = new QPushButton(tr("▶  Run analysis"), m_postRunRow);
        m_runInterpButton->setObjectName(QStringLiteral("RunButton"));
        m_runInterpButton->setCursor(Qt::PointingHandCursor);
        m_runInterpButton->setMinimumSize(220, 46);
        connect(m_runInterpButton, &QPushButton::clicked, this, &MainWindow::startInterpRun);
        row->addWidget(m_runInterpButton);

        layout->addWidget(m_postRunRow);
    }

    // Live run panel: now in the same scrolling column as the parameters,
    // instead of a separately-scrolled block pinned below. No extra
    // addSpacing() ahead of it any more: the run row's own bottom margin
    // already keeps the same gap every other floating run button on this
    // page and on Processing does, and the extra 10 px here on top of that
    // was the one place that gap actually came out different.
    m_postRunPanel = buildRunPanel(
        m_postUi, inner,
        tr("Set the parameters and press “Run analysis”."),
        nullptr, false);
    layout->addWidget(m_postRunPanel);

    // The comparison's own start button, below its parameters and filled in the
    // same colour as "Run analysis" — it is the action the page exists for, and
    // inside the card it read as one more field.
    {
        m_cmpRunRow = new QWidget(inner);
        m_cmpRunRow->setObjectName(QStringLiteral("RunRow"));
        auto *row = new QHBoxLayout(m_cmpRunRow);
        // Same 4 px top and bottom as every other floating run button — see
        // the matching comment on buildSetupPage()'s own run bar.
        row->setContentsMargins(0, 4, 0, 4);
        row->addStretch(1);

        // "Run analysis", not "Compare": all four floating run buttons
        // across Processing and Post-processing say the same thing now.
        m_cmpButton = new QPushButton(tr("▶  Run analysis"), m_cmpRunRow);
        m_cmpButton->setObjectName(QStringLiteral("RunButton"));
        m_cmpButton->setCursor(Qt::PointingHandCursor);
        m_cmpButton->setMinimumSize(220, 46);
        connect(m_cmpButton, &QPushButton::clicked, this, &MainWindow::runRouteComparison);
        row->addWidget(m_cmpButton);

        layout->addWidget(m_cmpRunRow);
    }

    layout->addWidget(buildComparePanel(inner));

    // The coherence tool's own start button and result panel.
    {
        m_cohRunRow = new QWidget(inner);
        m_cohRunRow->setObjectName(QStringLiteral("RunRow"));
        auto *row = new QHBoxLayout(m_cohRunRow);
        // Same 4 px top and bottom as every other floating run button — see
        // the matching comment on buildSetupPage()'s own run bar.
        row->setContentsMargins(0, 4, 0, 4);
        row->addStretch(1);

        // "Run analysis", not "Score the sites": all four floating run
        // buttons across Processing and Post-processing say the same thing
        // now.
        m_cohButton = new QPushButton(tr("▶  Run analysis"), m_cohRunRow);
        m_cohButton->setObjectName(QStringLiteral("RunButton"));
        m_cohButton->setCursor(Qt::PointingHandCursor);
        m_cohButton->setMinimumSize(220, 46);
        connect(m_cohButton, &QPushButton::clicked, this, &MainWindow::runCoherence);
        row->addWidget(m_cohButton);

        layout->addWidget(m_cohRunRow);
    }
    layout->addWidget(buildCoherencePanel(inner));

    layout->addStretch(1);

    // Everything added after the tool-selector card is one of the three
    // single-tool forms. Collected the same way the Processing page collects
    // its single-run form (see buildSetupPage()), so the BP card's show/hide
    // is one loop and a new card is picked up automatically. The trailing
    // stretch just above is a QSpacerItem, not a widget, so it is silently
    // skipped rather than collected.
    for (int i = 1; i < layout->count(); ++i) {
        QLayoutItem *item = layout->itemAt(i);
        if (QWidget *w = item->widget()) {
            m_postSingleRunCards.append(w);
        } else if (QLayout *nested = item->layout()) {
            for (int j = 0; j < nested->count(); ++j)
                if (QWidget *w = nested->itemAt(j)->widget())
                    m_postSingleRunCards.append(w);
        }
    }

    m_postBatchPage = new PostBatchPage(inner);
    m_postBatchPage->setVisible(false);
    // Right after the Analysis type card — same reasoning, and the same fix,
    // as the Processing page's m_batchPage just above.
    layout->insertWidget(1, m_postBatchPage);
    connect(m_postBatchPage, &PostBatchPage::runningChanged, this, [this](bool running) {
        // Every button that would put a second engine on the machine: a batch
        // (Processing or post-processing) owns the engine outright while it
        // runs, whether or not this particular batch is the one using it.
        m_runButton->setEnabled(!running);
        if (m_genPointsButton)
            m_genPointsButton->setEnabled(!running);
        if (m_runInterpButton)
            m_runInterpButton->setEnabled(!running);
        if (m_cmpButton)
            m_cmpButton->setEnabled(!running);
        if (m_cohButton)
            m_cohButton->setEnabled(!running);
        refreshRunTicker();
        if (running)
            QSettings().setValue(QStringLiteral("postbatch/processed"), true);
    });
    connect(m_postBatchPage, &PostBatchPage::tickerChanged,
            this, &MainWindow::refreshRunTicker);
    connect(m_postBatchPage, &PostBatchPage::exportCheckpointRequested,
            this, &MainWindow::exportCheckpointCopy);
    connect(m_postBatchPage, &PostBatchPage::importCheckpointRequested,
            this, &MainWindow::importCheckpointAndResume);
    connect(m_postBatchPage, &PostBatchPage::viewerLayersReady, this,
            [this](const QStringList &rasters, const QStringList &vectors) {
        if (!m_viewer)
            return;
        for (const QString &path : rasters)
            m_viewer->registerRaster(QFileInfo(path).completeBaseName(), path, false);
        for (const QString &path : vectors)
            m_viewer->registerVectorOverlay(QFileInfo(path).completeBaseName(), path);
    });
    m_postBatchPage->setGdalLoader([this] { return ensureGdalLoaded(); });

    // NNI is the tool the page opens on, so the other two — and the batch
    // page — start hidden.
    updatePostModeUi();
    refreshCoherenceUi();

    scroll->setWidget(inner);
    pageLayout->addWidget(scroll, 1);

    for (QWidget *w : std::initializer_list<QWidget *>{
             m_interpThresholdSpin, m_interpSpacingSpin, m_interpRadiusSpin})
        guardWheel(w);

    return page;
}

// ---------------------------------------------------------------------------
// Guide & About pages
// ---------------------------------------------------------------------------

QWidget *MainWindow::buildGuidePage()
{
    auto *page = new QWidget(this);
    auto *outer = new QHBoxLayout(page);
    outer->setContentsMargins(28, 24, 28, 24);
    outer->setSpacing(14);

    m_guideNav = new GuideNav(page);
    m_guideNav->setFixedWidth(216);
    outer->addWidget(m_guideNav);
    // Only ever a GuideNav, but the type is file-local and cannot be named in
    // the header — the same reason the guide's browsers are reached this way.
    // A local alias for the rest of this function, which is the only place
    // that ever needs the group/child-aware API.
    auto *nav = static_cast<GuideNav *>(m_guideNav);

    m_guidePages = new QStackedWidget(page);
    outer->addWidget(m_guidePages, 1);

    // The right-hand column: a caption over the sections of the page being
    // read. Both hidden together whenever the page has fewer than two
    // sections, because a caption over a list of one is not a way of getting
    // anywhere either — see showGuideSection().
    auto *tocPanel = new QFrame(page);
    tocPanel->setObjectName(QStringLiteral("GuideTocPanel"));
    tocPanel->setFrameShape(QFrame::NoFrame);
    tocPanel->setFixedWidth(190);
    auto *tocLayout = new QVBoxLayout(tocPanel);
    // Left inset matches what QListWidget#GuideToc used to carry itself
    // (padding: 4px 0px 4px 6px) — moved out here so the divider border and
    // the indent apply to the caption too, not just the list below it.
    tocLayout->setContentsMargins(6, 4, 0, 4);
    tocLayout->setSpacing(6);

    auto *tocTitle = new QLabel(tr("On this page"), tocPanel);
    tocTitle->setObjectName(QStringLiteral("GuideTocTitle"));
    tocLayout->addWidget(tocTitle);

    m_guideToc = new QListWidget(tocPanel);
    m_guideToc->setObjectName(QStringLiteral("GuideToc"));
    m_guideToc->setFrameShape(QFrame::NoFrame);
    m_guideToc->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_guideToc->setFocusPolicy(Qt::NoFocus);
    m_guideToc->setWordWrap(true);
    tocLayout->addWidget(m_guideToc, 1);

    m_guideTocPanel = tocPanel;
    outer->addWidget(m_guideTocPanel);

    // Overview first, and it is a page of widgets rather than a document: it
    // carries the logo, the version and the links that used to be the About
    // tab, which no rendering of HTML in a QTextBrowser would dress the same
    // way as the rest of the application.
    m_guidePages->addWidget(buildGuideOverviewPage());
    nav->addPageItem(tr("Overview"), m_guidePages->count() - 1);

    const QString guideHtml = QStringLiteral(R"HTML(
<style>
 h2 { color:%H2%; } h3 { color:%H3%; }
 a { color:%LINK%; }
 td, th { padding: 3px 10px; }
 p { text-align: justify; }
</style>
<!--nav:FETE-->
<h2>FETE — From Everywhere To Everywhere</h2>
<p>Trajecta provides two complementary workflows for modeling movement across terrain: FETE, described here, and LCPA on the next page. Both use anisotropic cost functions (e.g. Modified Tobler's Hiking Function, see Algorithm parameters) and support cost surface modifiers (e.g. waterbodies, terrain indexes).</p>
<p>From-Everywhere-To-Everywhere (FETE) is a GIS-based method initially conceptualized by White and Barber (2012). FETE allows to model probable movement corridors across a landscape without requiring predetermined origin and destination points as, instead, in Least-Cost Path Analysis (see next section). In this way, instead of calculating single paths between pre-selected points, FETE allows to model the general mobility characterizing a region. This is done by using a grid containing hundreds, thousands or even hundred of thousands regularly or randomly scattered points. The FETE algorithm implemented by Trajecta then calculates all the least-cost paths connecting every point to every other point of the grid. Based on all the LCPs calculated, overlap analysis is then computed and a density raster is created, with the same resolution as the input DEM. Each cell of the density raster contains a number. This number is the arithmetical sum of all the LCPs that cross that specific cell. The most crossed cells (i.e. those with highest values) represent the busiest and most travelled routes. Different color gradients can be used to display most probable paths among all calculated LCPs corridors. To compute all these LCPs, DEM or other elevation based data can be used to calculate slope which can then be transformed using cost functions (e.g. Tobler's Hiking Function). Additional costs can be added as for waterbodies or different types of terrain via raster or vector inputs.<br><br>

The density raster generated can then be used in different ways. For example, it can be compared to known routes or settlements in order to assess possible relationships between mobility across a region and settlement patterns.</p>

<table border="0" cellspacing="10" width="100%">
<tr>
<td align="center"><img src="guide:grid" width="%HALF%"></td>
<td align="center"><img src="guide:unfiltered" width="%HALF%"></td>
</tr>
<tr>
<td align="center"><i>Example of regular point grid and SRTM 30m DEM used as input for FETE computation.</i></td>
<td align="center"><i>Unfiltered FETE raster resulting from computation with Trajecta.</i></td>
</tr>
</table>

<table border="0" cellspacing="10" width="100%">
<tr><td align="center"><img src="guide:filtered" width="%FULL%"></td></tr>
<tr><td align="center"><i>Filtered FETE raster using only top 20% results.</i></td></tr>
</table>

<h3>Sample points</h3>
<p>In FETE mode the sample points can either be <b>imported from a file</b> (e.g. .shp, .geojson, .csv), or <b>directly generated from the DEM</b>.</p>
<p>Generation takes two parameters. The <b>density</b> is expressed either as a
<b>point spacing</b> (one point every N rows and every N columns, so the count falls
with the square of N) or as a <b>target number of points</b>, from which the spacing
is derived using the number of usable DEM cells. The <b>arrangement</b> is either a
<b>regular grid</b>, which puts every point at the same offset inside its block, or
<b>stratified random</b>, which picks one random cell per block: same density, none
of the regularity a grid imposes on the result. A stratified random layer is
reproducible from its <b>seed</b>.</p>
<p>Points are only placed on cells a path can actually cross, so NoData areas stay
empty. The layer is written to the output folder as a shapefile <i>before</i> the
analysis starts and is then read back as its input: what the run consumed is always
on disk, and it appears in the <b>Viewer</b> as a selectable overlay. The setup page
shows the resulting point count while you type, with a warning above roughly 50,000
points — FETE cost grows with the square of the number of points.</p>

<h3>Cost modifiers</h3>
<p>Modifiers let you make specific features (water bodies, restricted areas, specific types of terrain) more
expensive to cross. Vector modifiers are rasterized onto the DEM grid: every segment is
first clipped to the raster's bounds (Liang &amp; Barsky 1984), then walked cell by cell
with <b>Bresenham's line algorithm</b> (Bresenham 1965). The clipping is what keeps a layer
delivered in the wrong CRS from being merely wrong instead of also unbearably slow.
The <b>polyline buffer</b> widens rasterized lines so paths cannot
slip diagonally across them. The <b>barrier threshold</b> turns extreme multipliers
(e.g. 999999) into hard barriers: cells at or above the threshold are excluded from
movement, which also keeps the computation fast.</p>

<h3>Algorithm parameters</h3>
<p>Most of the following algorithm parameters are shared by both FETE and LCPA modes.</p>
<p><b>Neighbours</b> — connectivity of the search grid (8, 16, 24, 32, 64, or any
admissible number through <i>Custom</i>). Higher values allow finer path angles at the
price of speed. A connectivity radius of 16 (Knight's Move) is the usual choice.
See the <b>?</b> next to the field for which totals are admissible and why.</p>
<p><b>The search</b> &mdash; least-cost paths are found with <b>Dijkstra's algorithm</b>
(Dijkstra 1959) over the grid that connectivity defines, with no heuristic: what comes back
is the cheapest path that exists in that grid, not a good approximation of it. Because every
move is priced on its own and the price depends on the direction of travel, the graph is
<b>directed</b> &mdash; which is why A&rarr;B and B&rarr;A need not follow the same cells.</p>

<h3>How the cost of one cell-to-cell move is computed</h3>
<p>Every cost function in Trajecta is applied <b>to a single move between two cell
centres</b>, not to a cell in isolation. For each move the engine computes:</p>
<table>
<tr><td><code>dh</code></td><td>horizontal distance between the two cell centres, in metres —
from the neighbour offset and the DEM cell size, so a diagonal move is longer than an
orthogonal one</td></tr>
<tr><td><code>dz</code></td><td>elevation difference, <code>z(to) &minus; z(from)</code>, in metres —
<b>signed</b>: positive uphill, negative downhill</td></tr>
<tr><td><code>S</code></td><td>the slope of that move, <code>S = dz / dh</code> (a tangent, not an angle).
Where a formula below uses a percentage the engine passes <code>S &times; 100</code></td></tr>
</table>
<p>The cost function converts <code>S</code> into a walking speed <code>v</code>, and the cost of
the move is the time it takes:</p>
<p style="margin-left:20px"><code>cost = (dh / 1000) / v</code> &nbsp;&nbsp;→&nbsp;&nbsp; <b>hours</b>, when
<code>v</code> is in km/h</p>
<p>Because <code>S</code> keeps its sign, all three functions are <b>anisotropic</b>: going up a
slope and coming back down it cost different amounts, and A→B is not the same as B→A.
Any cost modifiers you supply multiply this base cost afterwards.</p>
<p>The <b>base cost surface</b> raster is <i>not</i> what the path search uses. It is a
summary for inspection: the mean of the move costs from each cell to all of its
neighbours. The search itself always uses the individual move costs.</p>

<h3>The cost functions, exactly as implemented</h3>
<p><b>1 &mdash; Modified Tobler's Hiking Function</b> (Tobler 1993; inverted into time as
described by White 2015)</p>
<p style="margin-left:20px"><code>v = 6 &middot; e<sup>&minus;3.5 &middot; |S + 0.05|</sup></code> &nbsp; km/h &nbsp;&rarr;&nbsp;
<code>cost = (dh/1000) / v</code> hours</p>
<p style="margin-left:20px">Fastest at <code>S = &minus;0.05</code>, i.e. a 5% downhill, where <code>v = 6</code> km/h;
on the flat 5.4 km/h. This is Tobler's <b>on-path</b> form: the ×&nbsp;0.6 factor that
Tobler suggests for <b>off-path</b> travel is <b>not</b> applied. If your route is
cross-country, expect this function to be optimistic by roughly that factor.</p>

<p><b>2 &mdash; M&aacute;rquez-P&eacute;rez et al. (2017)</b></p>
<p style="margin-left:20px"><code>v = 4.8 &middot; e<sup>&minus;5.3 &middot; |(0.7 &middot; S) + 0.03|</sup></code> &nbsp; km/h</p>
<p style="margin-left:20px">A recalibration of Tobler on GPS tracks from marked trails in Spanish natural
parks. Slower overall (4.8 instead of 6) and it penalises slope more sharply.
Fastest at <code>S &asymp; &minus;0.043</code>.</p>

<p><b>3 &mdash; Irmischer &amp; Clarke (2017)</b> — <b>on-path, male</b> variant</p>
<p style="margin-left:20px"><code>v = 0.11 + e<sup>&minus;(S% + 5)<sup>2</sup> / 1800</sup></code> &nbsp; m/s,
&nbsp; with <code>S% = 100 &middot; S</code> &nbsp; and &nbsp; <code>1800 = 2 &middot; 30<sup>2</sup></code></p>
<p style="margin-left:20px">The paper publishes four variants (male/female × on-path/off-path);
Trajecta implements the <b>on-path male</b> one. The others differ by a 0.67 factor on
the exponential and a +2 rather than +5 shift (off-path), and by an overall ×&nbsp;0.95
(female). Derived from GPS tracks of 200 cadets, so it includes way-finding time, which
is why it is slower than Tobler on the flat. The constant 0.11 m/s is a floor: this
function never reaches zero speed, however steep the ground.</p>
<p style="margin-left:20px"><i>Note.</i> Trajecta feeds this function the <b>signed</b> slope, so
the +5 shift makes it anisotropic with its peak at a 5% downhill, as the shift is meant
to express. Some other implementations pass <code>|S|</code> instead, which makes the function
symmetric and moves the peak to the flat; results are therefore not directly comparable
with those packages.</p>

<p><b>4 &mdash; Herzog (2013)</b>, fitted to Minetti et al. (2002) &mdash; <b>energy, not time</b></p>
<p style="margin-left:20px"><code>C(S) = 1337.8&middot;S<sup>6</sup> + 278.19&middot;S<sup>5</sup> &minus; 517.39&middot;S<sup>4</sup>
&minus; 78.199&middot;S<sup>3</sup> + 93.419&middot;S<sup>2</sup> + 19.825&middot;S + 1.64</code></p>
<p style="margin-left:20px"><code>cost = C(S) &middot; dh</code> &nbsp; &rarr; &nbsp; <b>kilojoules per kilogram</b> of walker</p>
<p style="margin-left:20px">The only function here that measures <b>effort</b> rather than duration.
Herzog fitted this sixth-degree polynomial to the treadmill measurements of Minetti et al.,
and it has the shape the data show and every speed model misses: the minimum sits at about a
<b>10.5% downhill</b>, and the curve rises on <i>both</i> sides &mdash; because braking down a
steep slope costs energy too. Tobler and the others simply get faster and faster downhill.</p>
<p style="margin-left:20px"><b>Read the units.</b> Every cost in a Herzog run &mdash; the cost
surfaces, the accumulated cost behind each path &mdash; is in kJ/kg, not hours. Those rasters
cannot be compared with, or added to, the output of any other function. Trajecta says so in the
run summary, in the manifest and under the selector, but the file itself carries no unit.</p>
<p style="margin-left:20px"><b>Range.</b> Minetti's data span roughly &plusmn;45% slope
(about &plusmn;24&deg;). Beyond that the polynomial is extrapolation: it stays positive and climbs
steeply, which is right in direction but is no longer a measurement. Use the slope cut-off below
to keep a run inside the calibrated range.</p>

<p><b>5, 6 &mdash; Campbell et al. (2019)</b>, asymmetric Lorentz, 5th and 50th percentile</p>
<p style="margin-left:20px"><code>v = c / (&pi;&middot;b&middot;(1 + ((&theta; &minus; a)/b)<sup>2</sup>)) + d + e&middot;&theta;</code>
&nbsp; m/s, with <code>&theta;</code> the slope in <b>degrees</b></p>
<table border="1" cellspacing="0" style="margin-left:20px">
<tr><th>Percentile</th><th>c</th><th>b</th><th>a</th><th>d</th><th>e</th></tr>
<tr><td>5th</td><td>36.813</td><td>14.041</td><td>&minus;1.527</td><td>0.320</td><td>&minus;0.00273</td></tr>
<tr><td>50th</td><td>63.660</td><td>10.064</td><td>&minus;2.171</td><td>0.628</td><td>&minus;0.00463</td></tr>
</table>
<p style="margin-left:20px">Fitted to <b>421,247 GPS activities</b> from 29,928 people recorded through
Strava &mdash; by far the largest empirical basis of any function here. The dataset mixes walking,
jogging and running, so the paper publishes one parameter set per percentile of the population
rather than a single average.</p>
<p style="margin-left:20px"><b>Which percentile to choose.</b> The authors recommend the
<b>5th</b> as representative of ordinary hiking: on the flat it gives about 1.15&nbsp;m/s
(4.1&nbsp;km/h), a normal walking pace. The <b>50th</b> is the median of the whole dataset and
reaches about 2.55&nbsp;m/s (9.2&nbsp;km/h) on the flat &mdash; that is a run, not a walk. Use it
only if fast movement is what you mean to model.</p>
<p style="margin-left:20px"><b>Range.</b> The fit is calibrated for slopes below 30&deg;; the paper
discarded steeper segments. The other percentiles (1st, 25th, 75th, 95th and the rest) exist in the
paper's supplementary material and can be added on request.</p>

<h3>Slope cut-off</h3>
<p>Off by default. When it is on, a move steeper than the limit you set is not expensive &mdash;
it is <b>impossible</b>, and the engine removes it from the graph. The limit applies to the
<b>move</b>, not to the cell: a terrace can still be entered from the side when the approach
from below is too steep, which is how real terrain behaves. Uphill and downhill are set
separately, because a slope that can be climbed slowly is often refused on the way down.</p>
<p>Two uses: keeping routes out of ground nobody would walk, and keeping a cost function inside
the range it was measured in (see Herzog and Campbell above). Set it too tight and a destination
can become unreachable &mdash; the run then reports the paths it could not compute rather than
inventing one.</p>

<h3>Units — what the numbers in the outputs actually mean</h3>
<br>
<table border="1" cellspacing="0">
<tr><th>Quantity</th><th>Unit</th><th>Notes</th></tr>
<tr><td>DEM elevation <code>z</code></td><td>metres</td><td>assumed; a DEM in feet gives slopes too small by 3.28×</td></tr>
<tr><td>Cell size, <code>dh</code></td><td>metres</td><td>taken from the DEM geotransform, so the CRS must be <b>projected</b>, never geographic degrees</td></tr>
<tr><td>Slope <code>S</code></td><td>dimensionless (m/m)</td><td>a tangent. <code>S = 1</code> is 45°, not 100°</td></tr>
<tr><td>Slope raster output</td><td><b>degrees</b> or <b>percent</b></td><td>degrees with Tobler and Campbell, percent with the others; stated in the run summary and in the manifest. This affects the <i>exported raster only</i> — the cost functions always receive <code>S = dz/dh</code></td></tr>
<tr><td>Speed <code>v</code></td><td>km/h (1&ndash;2), m/s (3, 5, 6)</td><td>converted internally; the m/s functions are multiplied by 3.6</td></tr>
<tr><td>Cost of one move</td><td><b>hours</b>, except Herzog: <b>kJ/kg</b></td><td>printed in every run summary and written into the manifest as <i>cost units</i></td></tr>
<tr><td>Base / additional / total cost surface</td><td>same as the move</td><td>mean over the neighbours of a cell — a summary, not what the search uses</td></tr>
<tr><td>Accumulated cost (internal)</td><td>same as the move</td><td>sum of move costs along the cheapest route found</td></tr>
<tr><td>FETE density raster</td><td><b>count</b> of paths</td><td>a pure integer count, not a cost and not a time</td></tr>
<tr><td>Cost modifiers</td><td><b>dimensionless multiplier</b></td><td>multiplies the base cost, so 2.0 means "twice as slow here"</td></tr>
</table>
<p>The five time-based functions return hours, so their cost surfaces are <b>numerically
comparable</b>: a cell at 0.5 means half an hour in all of them. <b>Herzog is not</b> — its
rasters are in kJ/kg and must never be compared with, subtracted from, or added to the
others. What is comparable to nothing at all is a cost surface against a density raster:
they measure different things.</p>
<p>None of the six models load carriage or ground surface. Herzog is the only one that
represents effort; the other five represent duration, and should not be described as a
measure of effort however intuitive that reading is.</p>
<p><b>Path smoothing buffer</b> — buffer in cells applied around each computed path
when accumulating results. This function allows to calculate wider paths, thus maximising the possibility that actual, historical path(s) were located inside the modeled high-traversability areas.</p>

<h3>Input requirements</h3>
<table border="0" cellspacing="0">
<tr><th align="left">Input</th><th align="left">Requirements</th></tr>
<tr><td><b>DEM</b></td><td>GeoTIFF (.tif/.tiff), georeferenced, with a CRS.</td></tr>
<tr><td><b>Sample points</b></td><td>.shp, .geojson/.json, .kml, .gml/.xml or .csv
    (coordinate columns named x/y, lon/lat or easting/northing) if imported; or
    generated directly from the DEM instead.</td></tr>
<tr><td><b>Vector modifiers</b> (optional)</td><td>Polylines with a float <b>cost</b> field
    holding the multiplier; for .csv the geometry must be in a WKT column.</td></tr>
<tr><td><b>Raster modifiers</b> (optional)</td><td>GeoTIFF with the same dimensions as the DEM;
    cell values are multipliers (1.0 = unchanged, 2.0 = double cost).</td></tr>
</table>

<h3>Outputs</h3>
<p>Slope raster, base cost surface, and — with modifiers — the additional and total
cost surfaces; the <b>path-density raster</b>; and the sample points shapefile when
the points were generated from the DEM.</p>
<p>Every run also writes a <b>run manifest</b> next to its results, unless the
option is turned off: a plain text record of the version, the inputs with their
content hashes, every setting, the hardware and the files produced.</p>

<!--nav:LCPA-->
<h2>LCPA — Least-Cost Path Analysis</h2>
<p>For a detailed introduction to Least-Cost Path Analysis (LCPA), see White (2015). LCPA is a spatial analysis method, typically implemented in GIS environments, that identifies the minimum cumulative-cost route between two points across a cost surface. Each cell of the raster grid represents the cost of traversing it – expressed in terms of physical effort, time, energy expenditure, or resistance to movement – calculated as a function of variables such as slope, land cover, hydrography, or other environmental and cultural factors relevant to the study context.</p>
<p>Algorithmically, the raster surface is treated as a weighted graph (cells as nodes, adjacencies as edges), and the problem is solved with shortest-path algorithms such as Dijkstra's or A* (A-star), which compute both the accumulated cost surface from the source and the optimal path to one or more destinations. A key distinction is between isotropic cost (equal in every direction) and anisotropic cost (direction-dependent, as with slope varying between ascent and descent – Tobler's Hiking Function being the classic example for pedestrian movement).</p>
<p>In archaeology, LCPA is widely used to reconstruct probable movement corridors, ancient route networks, or trade paths from digital elevation models, on the assumption that human movement tends to minimize effort. It should nonetheless be used cautiously when investigating ancient routes: LCPA inherently introduces a strong selection bias as it necessarily needs the user to select at least two points (one origin and at least one destination) to be connected. Importantly, the two points might have never been actually connected in ancient times. Consequently, this selection bias must always be taken into account and additional proofing of the results should be always provided.</p>
<p>To compute LCPs in Trajecta, DEM or other elevation based data can be used to calculate slope which can then be transformed using different cost functions (e.g. Modified Tobler's Hiking Function, Irmischer and Clarke 2017, Herzog 2013). Additional costs can be added as for waterbodies or terrain indexes using raster or vector input layers.</p>

<table border="0" cellspacing="10" width="100%">
<tr><td align="center"><img src="guide:lcpa" width="%FULL%"></td></tr>
<tr><td align="center"><i>Least-Cost Paths from single origin to multiple destinations calculated using
Trajecta and SRTM 30m DEM.</i></td></tr>
</table>

<h3>Cost modifiers</h3>
<p>Modifiers let you make specific features (water bodies, restricted areas, specific types of terrain) more
expensive to cross. Vector modifiers are rasterized onto the DEM grid: every segment is
first clipped to the raster's bounds (Liang &amp; Barsky 1984), then walked cell by cell
with <b>Bresenham's line algorithm</b> (Bresenham 1965). The clipping is what keeps a layer
delivered in the wrong CRS from being merely wrong instead of also unbearably slow.
The <b>polyline buffer</b> widens rasterized lines so paths cannot
slip diagonally across them. The <b>barrier threshold</b> turns extreme multipliers
(e.g. 999999) into hard barriers: cells at or above the threshold are excluded from
movement, which also keeps the computation fast.</p>

<h3>Algorithm parameters</h3>
<p>Most of the following algorithm parameters are shared by both FETE and LCPA modes.</p>
<p><b>Neighbours</b> — connectivity of the search grid (8, 16, 24, 32, 64, or any
admissible number through <i>Custom</i>). Higher values allow finer path angles at the
price of speed. A connectivity radius of 16 (Knight's Move) is the usual choice.
See the <b>?</b> next to the field for which totals are admissible and why.</p>
<p><b>The search</b> &mdash; least-cost paths are found with <b>Dijkstra's algorithm</b>
(Dijkstra 1959) over the grid that connectivity defines, with no heuristic: what comes back
is the cheapest path that exists in that grid, not a good approximation of it. Because every
move is priced on its own and the price depends on the direction of travel, the graph is
<b>directed</b> &mdash; which is why A&rarr;B and B&rarr;A need not follow the same cells.</p>

<h3>How the cost of one cell-to-cell move is computed</h3>
<p>Every cost function in Trajecta is applied <b>to a single move between two cell
centres</b>, not to a cell in isolation. For each move the engine computes:</p>
<table>
<tr><td><code>dh</code></td><td>horizontal distance between the two cell centres, in metres —
from the neighbour offset and the DEM cell size, so a diagonal move is longer than an
orthogonal one</td></tr>
<tr><td><code>dz</code></td><td>elevation difference, <code>z(to) &minus; z(from)</code>, in metres —
<b>signed</b>: positive uphill, negative downhill</td></tr>
<tr><td><code>S</code></td><td>the slope of that move, <code>S = dz / dh</code> (a tangent, not an angle).
Where a formula below uses a percentage the engine passes <code>S &times; 100</code></td></tr>
</table>
<p>The cost function converts <code>S</code> into a walking speed <code>v</code>, and the cost of
the move is the time it takes:</p>
<p style="margin-left:20px"><code>cost = (dh / 1000) / v</code> &nbsp;&nbsp;→&nbsp;&nbsp; <b>hours</b>, when
<code>v</code> is in km/h</p>
<p>Because <code>S</code> keeps its sign, all three functions are <b>anisotropic</b>: going up a
slope and coming back down it cost different amounts, and A→B is not the same as B→A.
Any cost modifiers you supply multiply this base cost afterwards.</p>
<p>The <b>base cost surface</b> raster is <i>not</i> what the path search uses. It is a
summary for inspection: the mean of the move costs from each cell to all of its
neighbours. The search itself always uses the individual move costs.</p>

<h3>The cost functions, exactly as implemented</h3>
<p><b>1 &mdash; Modified Tobler's Hiking Function</b> (Tobler 1993; inverted into time as
described by White 2015)</p>
<p style="margin-left:20px"><code>v = 6 &middot; e<sup>&minus;3.5 &middot; |S + 0.05|</sup></code> &nbsp; km/h &nbsp;&rarr;&nbsp;
<code>cost = (dh/1000) / v</code> hours</p>
<p style="margin-left:20px">Fastest at <code>S = &minus;0.05</code>, i.e. a 5% downhill, where <code>v = 6</code> km/h;
on the flat 5.4 km/h. This is Tobler's <b>on-path</b> form: the ×&nbsp;0.6 factor that
Tobler suggests for <b>off-path</b> travel is <b>not</b> applied. If your route is
cross-country, expect this function to be optimistic by roughly that factor.</p>

<p><b>2 &mdash; M&aacute;rquez-P&eacute;rez et al. (2017)</b></p>
<p style="margin-left:20px"><code>v = 4.8 &middot; e<sup>&minus;5.3 &middot; |(0.7 &middot; S) + 0.03|</sup></code> &nbsp; km/h</p>
<p style="margin-left:20px">A recalibration of Tobler on GPS tracks from marked trails in Spanish natural
parks. Slower overall (4.8 instead of 6) and it penalises slope more sharply.
Fastest at <code>S &asymp; &minus;0.043</code>.</p>

<p><b>3 &mdash; Irmischer &amp; Clarke (2017)</b> — <b>on-path, male</b> variant</p>
<p style="margin-left:20px"><code>v = 0.11 + e<sup>&minus;(S% + 5)<sup>2</sup> / 1800</sup></code> &nbsp; m/s,
&nbsp; with <code>S% = 100 &middot; S</code> &nbsp; and &nbsp; <code>1800 = 2 &middot; 30<sup>2</sup></code></p>
<p style="margin-left:20px">The paper publishes four variants (male/female × on-path/off-path);
Trajecta implements the <b>on-path male</b> one. The others differ by a 0.67 factor on
the exponential and a +2 rather than +5 shift (off-path), and by an overall ×&nbsp;0.95
(female). Derived from GPS tracks of 200 cadets, so it includes way-finding time, which
is why it is slower than Tobler on the flat. The constant 0.11 m/s is a floor: this
function never reaches zero speed, however steep the ground.</p>
<p style="margin-left:20px"><i>Note.</i> Trajecta feeds this function the <b>signed</b> slope, so
the +5 shift makes it anisotropic with its peak at a 5% downhill, as the shift is meant
to express. Some other implementations pass <code>|S|</code> instead, which makes the function
symmetric and moves the peak to the flat; results are therefore not directly comparable
with those packages.</p>

<p><b>4 &mdash; Herzog (2013)</b>, fitted to Minetti et al. (2002) &mdash; <b>energy, not time</b></p>
<p style="margin-left:20px"><code>C(S) = 1337.8&middot;S<sup>6</sup> + 278.19&middot;S<sup>5</sup> &minus; 517.39&middot;S<sup>4</sup>
&minus; 78.199&middot;S<sup>3</sup> + 93.419&middot;S<sup>2</sup> + 19.825&middot;S + 1.64</code></p>
<p style="margin-left:20px"><code>cost = C(S) &middot; dh</code> &nbsp; &rarr; &nbsp; <b>kilojoules per kilogram</b> of walker</p>
<p style="margin-left:20px">The only function here that measures <b>effort</b> rather than duration.
Herzog fitted this sixth-degree polynomial to the treadmill measurements of Minetti et al.,
and it has the shape the data show and every speed model misses: the minimum sits at about a
<b>10.5% downhill</b>, and the curve rises on <i>both</i> sides &mdash; because braking down a
steep slope costs energy too. Tobler and the others simply get faster and faster downhill.</p>
<p style="margin-left:20px"><b>Read the units.</b> Every cost in a Herzog run &mdash; the cost
surfaces, the accumulated cost behind each path &mdash; is in kJ/kg, not hours. Those rasters
cannot be compared with, or added to, the output of any other function. Trajecta says so in the
run summary, in the manifest and under the selector, but the file itself carries no unit.</p>
<p style="margin-left:20px"><b>Range.</b> Minetti's data span roughly &plusmn;45% slope
(about &plusmn;24&deg;). Beyond that the polynomial is extrapolation: it stays positive and climbs
steeply, which is right in direction but is no longer a measurement. Use the slope cut-off below
to keep a run inside the calibrated range.</p>

<p><b>5, 6 &mdash; Campbell et al. (2019)</b>, asymmetric Lorentz, 5th and 50th percentile</p>
<p style="margin-left:20px"><code>v = c / (&pi;&middot;b&middot;(1 + ((&theta; &minus; a)/b)<sup>2</sup>)) + d + e&middot;&theta;</code>
&nbsp; m/s, with <code>&theta;</code> the slope in <b>degrees</b></p>
<table border="1" cellspacing="0" style="margin-left:20px">
<tr><th>Percentile</th><th>c</th><th>b</th><th>a</th><th>d</th><th>e</th></tr>
<tr><td>5th</td><td>36.813</td><td>14.041</td><td>&minus;1.527</td><td>0.320</td><td>&minus;0.00273</td></tr>
<tr><td>50th</td><td>63.660</td><td>10.064</td><td>&minus;2.171</td><td>0.628</td><td>&minus;0.00463</td></tr>
</table>
<p style="margin-left:20px">Fitted to <b>421,247 GPS activities</b> from 29,928 people recorded through
Strava &mdash; by far the largest empirical basis of any function here. The dataset mixes walking,
jogging and running, so the paper publishes one parameter set per percentile of the population
rather than a single average.</p>
<p style="margin-left:20px"><b>Which percentile to choose.</b> The authors recommend the
<b>5th</b> as representative of ordinary hiking: on the flat it gives about 1.15&nbsp;m/s
(4.1&nbsp;km/h), a normal walking pace. The <b>50th</b> is the median of the whole dataset and
reaches about 2.55&nbsp;m/s (9.2&nbsp;km/h) on the flat &mdash; that is a run, not a walk. Use it
only if fast movement is what you mean to model.</p>
<p style="margin-left:20px"><b>Range.</b> The fit is calibrated for slopes below 30&deg;; the paper
discarded steeper segments. The other percentiles (1st, 25th, 75th, 95th and the rest) exist in the
paper's supplementary material and can be added on request.</p>

<h3>Slope cut-off</h3>
<p>Off by default. When it is on, a move steeper than the limit you set is not expensive &mdash;
it is <b>impossible</b>, and the engine removes it from the graph. The limit applies to the
<b>move</b>, not to the cell: a terrace can still be entered from the side when the approach
from below is too steep, which is how real terrain behaves. Uphill and downhill are set
separately, because a slope that can be climbed slowly is often refused on the way down.</p>
<p>Two uses: keeping routes out of ground nobody would walk, and keeping a cost function inside
the range it was measured in (see Herzog and Campbell above). Set it too tight and a destination
can become unreachable &mdash; the run then reports the paths it could not compute rather than
inventing one.</p>

<h3>Units — what the numbers in the outputs actually mean</h3>
<br>
<table border="1" cellspacing="0">
<tr><th>Quantity</th><th>Unit</th><th>Notes</th></tr>
<tr><td>DEM elevation <code>z</code></td><td>metres</td><td>assumed; a DEM in feet gives slopes too small by 3.28×</td></tr>
<tr><td>Cell size, <code>dh</code></td><td>metres</td><td>taken from the DEM geotransform, so the CRS must be <b>projected</b>, never geographic degrees</td></tr>
<tr><td>Slope <code>S</code></td><td>dimensionless (m/m)</td><td>a tangent. <code>S = 1</code> is 45°, not 100°</td></tr>
<tr><td>Slope raster output</td><td><b>degrees</b> or <b>percent</b></td><td>degrees with Tobler and Campbell, percent with the others; stated in the run summary and in the manifest. This affects the <i>exported raster only</i> — the cost functions always receive <code>S = dz/dh</code></td></tr>
<tr><td>Speed <code>v</code></td><td>km/h (1&ndash;2), m/s (3, 5, 6)</td><td>converted internally; the m/s functions are multiplied by 3.6</td></tr>
<tr><td>Cost of one move</td><td><b>hours</b>, except Herzog: <b>kJ/kg</b></td><td>printed in every run summary and written into the manifest as <i>cost units</i></td></tr>
<tr><td>Base / additional / total cost surface</td><td>same as the move</td><td>mean over the neighbours of a cell — a summary, not what the search uses</td></tr>
<tr><td>Accumulated cost (internal)</td><td>same as the move</td><td>sum of move costs along the cheapest route found</td></tr>
<tr><td>FETE density raster</td><td><b>count</b> of paths</td><td>a pure integer count, not a cost and not a time</td></tr>
<tr><td>Cost modifiers</td><td><b>dimensionless multiplier</b></td><td>multiplies the base cost, so 2.0 means "twice as slow here"</td></tr>
</table>
<p>The five time-based functions return hours, so their cost surfaces are <b>numerically
comparable</b>: a cell at 0.5 means half an hour in all of them. <b>Herzog is not</b> — its
rasters are in kJ/kg and must never be compared with, subtracted from, or added to the
others. What is comparable to nothing at all is a cost surface against a density raster:
they measure different things.</p>
<p>None of the six models load carriage or ground surface. Herzog is the only one that
represents effort; the other five represent duration, and should not be described as a
measure of effort however intuitive that reading is.</p>
<p><b>Path smoothing buffer</b> — buffer in cells applied around each computed path
when accumulating results. This function allows to calculate wider paths, thus maximising the possibility that actual, historical path(s) were located inside the modeled high-traversability areas.</p>

<h3>Input requirements</h3>
<table border="0" cellspacing="0">
<tr><th align="left">Input</th><th align="left">Requirements</th></tr>
<tr><td><b>DEM</b></td><td>GeoTIFF (.tif/.tiff), georeferenced, with a CRS.</td></tr>
<tr><td><b>Origin</b></td><td>Vector file with exactly one point (.shp, .geojson/.json,
    .kml, .gml/.xml or .csv): the starting location of the least-cost routes.</td></tr>
<tr><td><b>Destinations</b></td><td>Vector file with one or more points, same formats: the
    target(s) the optimal route(s) is/are computed to.</td></tr>
<tr><td><b>Vector modifiers</b> (optional)</td><td>Polylines with a float <b>cost</b> field
    holding the multiplier; for .csv the geometry must be in a WKT column.</td></tr>
<tr><td><b>Raster modifiers</b> (optional)</td><td>GeoTIFF with the same dimensions as the DEM;
    cell values are multipliers (1.0 = unchanged, 2.0 = double cost).</td></tr>
</table>

<h3>Outputs</h3>
<p>Slope raster, base cost surface, and — with modifiers — the additional and total
cost surfaces; the <b>paths raster</b> and the <b>paths polyline shapefile</b>, plus
the <b>cost corridor raster</b> when one was asked for.</p>
<p>Every run also writes a <b>run manifest</b> next to its results, unless the
option is turned off: a plain text record of the version, the inputs with their
content hashes, every setting, the hardware and the files produced.</p>

<!--nav:NNI interpolation-->
<h2>Post-processing: NNI — Natural Neighbour Interpolation</h2>
<p>The <b>Post-processing</b> page turns a FETE density raster into a smooth,
continuous surface using <b>discrete Sibson (natural neighbour) interpolation
(Sibson 1981; Park et al. 2006)</b>.
Cells at or above the <b>sample threshold</b> act as sample points; every other
cell receives the average of the samples whose influence area it would claim.
Sample values are preserved exactly. The optional <b>max search radius</b> caps
how far the interpolation reaches into empty areas (beyond it, cells take the
value of their nearest sample), which keeps large rasters fast. After a
successful FETE run the density raster is filled in automatically, allowing for direct post-processing.</p>

<table border="0" cellspacing="10" width="100%">
<tr>
<td align="center"><img src="guide:density" width="%HALF%"></td>
<td align="center"><img src="guide:nni" width="%HALF%"></td>
</tr>
<tr>
<td align="center"><i>FETE density raster generated with Trajecta.</i></td>
<td align="center"><i>The same FETE density raster after NNI.</i></td>
</tr>
</table>

<h3>Input requirements</h3>
<p>A <b>density raster</b> (GeoTIFF), typically the FETE output — the interface fills
this in automatically after a successful FETE run.</p>

<h3>Outputs</h3>
<p>The <b>interpolated raster</b>, written next to the density raster it was made
from and named after it. A run manifest is written alongside it too, unless the
option is turned off.</p>

<!--nav:Route comparison-->
<h2>Post-processing: comparison with a known route</h2>
<p>The second tool on the <b>Post-processing</b> page does not compute anything new: it
<b>measures a computed route against a route that is actually known</b> — a Roman road, a
drover's track, a surveyed path. This is the step that turns a least-cost path from an
illustration into a claim that can be wrong; without it a model can only ever agree with
itself.</p>
<p>It takes two vector layers of lines — normally the LCPA paths shapefile and the known
route — and a <b>tolerance</b> in metres, which is what you consider "close". Both layers
must be <b>projected and in the same CRS</b>: distances in degrees would be meaningless, so
they are refused rather than silently reported.</p>
<p>The report gives, in both directions, the <b>median</b>, the <b>90th percentile</b> and the
<b>maximum</b> distance from one line to the other, and the <b>share of each line that runs
within the tolerance</b> of the other. A distribution rather than a single number, because a
route can follow the real one closely for 9 km and then take the wrong side of a hill for
1 km — and an average hides exactly that. Both directions are needed too: a short computed
path lying on top of a long known one is close in one direction and far in the other. The
worst disagreement anywhere, the maximum of the two, is the <b>Hausdorff distance</b>
(Hausdorff 1914).</p>
<p>The comparison is also available from the command line
(<code>--compare-routes</code>), which makes a whole set of routes testable in one script.</p>

<h3>Input requirements</h3>
<p>Two <b>vector line layers</b> (.shp, .geojson/.json, .kml, .gml/.xml or .csv):
the computed routes — normally the LCPA paths shapefile — and the known route to
test them against. Both must be <b>projected and in the same CRS</b>.</p>

<h3>Outputs</h3>
<p>No raster and no layer — the output is the <b>report itself</b>, printed in the
panel and in the log, which can be copied into a table or a publication as it
stands.</p>

<!--nav:Site–corridor coherence-->
<h2>Post-processing: site&#8211;corridor coherence</h2>
<p>The third tool asks the question the FETE was computed for: <b>do the sites sit on the
movement the surface predicts?</b> It takes the FETE surface and a <b>point layer of
sites</b>, and gives every site a score, the sample a verdict, and — this is the part that
makes two periods comparable — a statement of how much of that could have happened by
chance.</p>

<h3>The four questions the tool answers</h3>
<p>In simple terms, the site-corridor coherence tool aims at answering four main questions:</p>
<ol>
<li><b>Are any of the sites near a corridor at all?</b> If almost none is, everything below is
noise and you can stop your analysis here.</li>
<li><b>How far are the sites from the corridors?</b> The first quantity: near is not a yes or no,
it is a distance. Two sites (e.g. site A and site B) can be equally considered 'near' to a
corridor if this is within a distance of — for example — 500 m. Nonetheless, this same corridor
might be 400 m from site A and only 40 m from site B. Clearly, this is a significant difference
that would be impossible to detect with a binary 'near/far' classification.</li>
<li><b>How much corridor is around the sites?</b> Two sites (e.g. site A and site B) at a same
distance from a route are not in the same place if one has a single thread nearby and the other a
whole braid. Site A might be near a single, thin corridor while Site B might be near several,
larger corridors. This makes a big difference when assessing site-corridor coherence. It is
important to know not only how many corridors are near a single site, but also how big these
corridors are.</li>
<li><b>How busy is the ground around the sites?</b> Not how much corridor, but how heavily
travelled it is. You can have sites near several or even a lot of corridors, but these corridors
might be only limitedly travelled. On the contrary, you can have a site with just a corridor in
its vicinity, but that corridor might be extremely busy.</li>
</ol>
<p>Every number below is built so that <b>two runs can be compared</b> &mdash; two periods, two
regions &mdash; even when the two FETE surfaces were computed from different numbers of points
or at different resolutions. That is the whole purpose of the tool, and it constrains how each
measure is defined.</p>

<h3>1. Are any of the sites near a corridor at all?</h3>
<p>The <b>distance bands</b> table: the share of sites within 0, 100, 250, 500, 1000 and 2500
metres of the nearest corridor cell &mdash; you can set your own list. "Within 0 m" means
standing on a corridor cell.</p>
<p>The distances are fixed metres and <b>not</b> fractions of the radius, and that is deliberate:
it means two runs can be laid side by side row for row whatever radius each was given. Bands
finer than one raster cell are dropped, because a raster cannot resolve them &mdash; on a 90 m
grid a site is either on a corridor or at least 90 m from one, so a 50 m band would only repeat
the 0 m one.</p>

<h3>2. How far are the sites from the corridors?</h3>
<p>Reported as the <b>median</b>, the <b>deciles</b> (p10, p25, p75, p90) and a small
<b>histogram</b>. The median is the middle site: half are closer, half are further.</p>
<p>Why not just the median? Because a single middle value can hide the shape of a sample. If half
the sites sit almost on the corridors and the other half are kilometres away, with almost nothing
in between, the median lands in the empty middle and describes <i>nobody</i>. That pattern is
called a <b>bimodal</b> distribution, and it is common and interesting: it usually means you have
two kinds of site rather than one. The deciles reveal it &mdash; a big jump between two
neighbouring figures is the gap between the two groups &mdash; and the histogram shows it
directly. <b>If the deciles and the histogram disagree with the median, believe them.</b></p>
<p>These figures do not depend on the radius at all. That is what makes them the ones to quote.</p>

<h3>3. How much corridor is around the sites?</h3>
<p>The <b>proximity index</b>: of the cells within the radius that have data, the percentage that
are corridor cells.</p>
<p>Why a percentage and not a count? Because a count is not comparable. The same piece of ground
holds nine times as many cells at 30 m resolution as at 90 m, so a site would score nine times
higher merely for being measured on a finer raster. A <b>share</b> cancels that out: if 8%
of the neighbourhood is corridor, it is 8% at either resolution.</p>
<p>Beside it is <b>enrichment</b>, which is the proximity index divided by the corridor's share of
the <i>whole</i> surface. This has a property worth understanding, because it is what lets you say
whether a number is large:</p>
<p style="margin-left:24px;"><i>A point dropped at random on the map has, on average, exactly the
surface's own share of corridor around it. So <b>enrichment 1.00 is chance, exactly</b> &mdash;
not approximately, and not "as estimated by a simulation". Enrichment 5.00 means five times as
much corridor as average ground. The higher the enrichment, the more the corridor around a site
stands out from the rest of the surface.</i></p>

<h3>4. How busy is the ground around the sites?</h3>
<p>The <b>intensity index</b>. Being surrounded by corridor is one thing; being surrounded by a
<i>heavily travelled</i> corridor is another. This measures the second.</p>
<p>It is built in three steps, and each removes a specific way of being wrong:</p>
<ol>
<li><b>Take the logarithm of every cell's value.</b> A FETE cell holds a count of paths, and those
counts are wildly uneven &mdash; most cells near zero, a few in the millions. If you simply
averaged them, one enormous cell would dominate its whole neighbourhood and the measure would stop
describing the area and start reporting its single busiest cell. On a logarithmic scale a cell ten
times busier counts more, but not ten times more, which is the behaviour we want.</li>
<li><b>Weight by distance.</b> Cells close to the site count fully, cells at the edge of the radius
count nothing, in a straight line between the two.</li>
<li><b>Convert the result to a percentile.</b> The same weighted average is measured at tens of
thousands of places across the surface, and the site is scored against that yardstick.</li>
</ol>
<p>That third step is what makes the number comparable. A surface built from a million source
points has path counts hundreds of times larger than one built from ten thousand &mdash; but if
every place on both maps is scaled by the same factor, the <i>ordering</i> is unchanged, so the
percentile is unchanged. <b>50 is the average location on the surface</b>, and 64 really does mean
"busier than 64% of this map".</p>

<h3>Means and medians: which figure to quote</h3>
<p>For questions 3 and 4 the report prints <b>both</b> a mean and a median, and here &mdash;
unusually &mdash; <b>the mean is the one to quote.</b> The reason is that both reference points are
statements about a mean: the expected share of corridor around a random point is exactly the
surface's share, and mid-ranks make the average percentile exactly 50. Neither holds for a
median.</p>
<p>You will usually find the medians much lower, often zero, and that is <b>normal, not a fault</b>.
Corridors are thin, linear and clustered, so on a typical FETE the majority of locations &mdash;
sites and random points alike &mdash; have no corridor within reach at all, which puts the median
at zero for both. A median enrichment of 0.00 beside a mean of 2.59 is exactly what a real
landscape looks like.</p>

<h3>What counts as a corridor</h3>
<p>Distances are measured to the nearest <b>corridor cell</b>, so this setting decides what
everything else is measured towards. The default is <b>the top 1%</b> of the surface by rank,
because on a FETE surface the cells carrying real traffic are almost always inside the top
per cent.</p>
<p><b>Use the percentage filter for anything comparative</b>, and it is the default for that
reason. Selecting the top q% by rank returns exactly q% of the valid cells in <i>every</i> dataset,
by construction &mdash; not approximately, and not by luck. That is what makes two surfaces
comparable at all. <b>Otsu's method</b> (Otsu 1979), computed on the logarithm of the values, finds
a threshold automatically on a single dataset; it reports which percentile it landed on and warns
when the surface has no clean split. A raw value can also be given, but a raw threshold is
<b>not</b> comparable between surfaces built from different numbers of points, because the values
themselves are on different scales.</p>
<p>The threshold that was actually used is always reported three ways &mdash; as a value, as a
percentile, and as a share of the surface. Those can differ from what was asked: on a sparse
surface where 99% of cells are exactly zero, "the top 1%" cannot be cut anywhere except at
the first non-zero value, and the report says so rather than pretending.</p>
<p>Once the corridor cells are known, <b>every</b> cell of the raster is given its distance to
the nearest one, in a single pass over the grid, with the separable distance transform of
Felzenszwalb &amp; Huttenlocher (2012). That distance is <b>exact</b> &mdash; not the few
per cent of error the usual two-pass chamfer approximations leave &mdash; and it is computed
once for the whole surface rather than once per site, which is what makes the distance raster
free and the sensitivity curve nearly free.</p>
<p>Cells with no data stay <b>missing</b> throughout &mdash; they are not zero. Zero means
"measured, and nothing passes here", which is a fact about the landscape; counting the two
together would move every rank in the raster. What missing data costs a site is reported as
that site's <b>coverage</b>: the fraction of its disc that had data. A site whose coverage is
low is measured on less ground than the others, and the summary counts how many are below a
half.</p>

<h3>Could this have happened by chance?</h3>
<p>A median distance of 118 m means nothing on its own. So the same median is computed again on
<b>999 point sets that have no relationship with the corridors</b> but share everything else
&mdash; the same area, the same number of points, and by default the same internal arrangement,
translated as a block, because settlements cluster and independent random points do not.</p>
<p>This is a <b>Monte Carlo significance test</b> (Hope 1968; Besag &amp; Diggle 1977). The
p-value is the rank of the observed statistic among the simulated ones, taken as
(1&nbsp;+&nbsp;the number of random sets at least as extreme)&nbsp;&divide;&nbsp;(999&nbsp;+&nbsp;1):
that added 1 is what keeps the test honest, and it is also why the smallest p the tool can
print is 0.001 &mdash; which means "the lowest this many replicates can resolve", not "one in
a thousand exactly". Moving the whole pattern as a block is the <b>random-shift</b> null
(Lotwick &amp; Silverman 1982). It holds the sites' own spacing and clustering fixed and asks
only whether their <i>position</i> relative to the corridors is special, which is a stricter
question than scattering independent points &mdash; that alternative, offered as
<i>scattered points</i>, is the complete spatial randomness of classical point pattern
analysis (Baddeley et al. 2015). Shifts that would push any site off the raster or onto
missing data are discarded; if the sites cover the surface so completely that too few shifts
survive, the tool says so in the log and falls back to scattered points, rather than testing
against a handful of nearly identical sets.</p>
<p><b>Only the distance is tested this way, and that is on purpose.</b> A distance has no natural
reference point &mdash; there is nothing to compare 118 m against without simulating it. The other
two measures already carry their own: enrichment is 1.00 under chance and the intensity index is
50, both exactly and by construction. Running a simulation to rediscover a number we already know
would only add a column of statistics to be misread.</p>
<p>What comes out is reported in metres rather than as a test statistic: <i>observed 118 m,
expected 240 m, with 95% of the random sets falling between 190 and 310 m</i>. Beside it is the
<b>ratio</b>, which is the figure to carry between periods: 0.5 means the sites are half as far
from a corridor as chance would put them, and unlike a raw score it does not depend on the units,
the area or the size of the sample. A sentence of the kind the tool is for: <i>"in the earlier
period the sites are 0.31&times; as far from the corridors as chance predicts; in the later one
0.87&times; &mdash; the relationship between settlement and natural routes weakens."</i></p>

<h3>Comparing two datasets</h3>
<p>Because this is what the tool is for, it is worth being explicit about what is and is not
required.</p>
<p><b>Not required:</b> the same resolution, the same number of FETE source points, or the same
extent. Every measure above is defined so as not to depend on any of them &mdash; that is why the
proximity index is a share rather than a count, why enrichment divides by the surface's own
corridor share, and why the intensity index ends in a percentile.</p>
<p><b>Worth knowing anyway:</b> a surface computed from few source points is a <i>noisier</i>
estimate of the same thing. The units are the same and the comparison is valid, but the
measurement carries more error, so small differences between two runs should not be pressed. And
one thing the intensity index deliberately cannot tell you: because every surface's average
location scores 50 by construction, it says "this site is busier than 64% of <i>its own</i>
region" &mdash; never "this region is busier than that one". That is a different question, and a
fragile one to ask of path counts.</p>

<h3>Sensitivity to the radius</h3>
<p>Turning this on repeats the analysis at several radii and prints one row each. It costs
little, because ranking the surface and measuring every cell's distance to a corridor
do not depend on the radius and are done once. It is worth having whenever the result will be
shown to someone else, because it answers their first question in advance: <b>a relationship
that holds across the whole range is really a relationship; one that appears at a single radius
is usually the radius and should not be taken for reference.</b> Questions 1 and 2 do not appear in that table at all &mdash; they do not
change with the radius, which is why they are the headline result.</p>

<h3>Input requirements</h3>
<p>A <b>FETE surface</b> (GeoTIFF), raw or interpolated with NNI, and a <b>point
layer of sites</b> (.shp, .geojson/.json, .kml, .gml/.xml or .csv) in the same
projected CRS as the raster.</p>

<h3>Outputs</h3>
<p>A <b>table (.csv)</b> with one row per site: <code>dist_m</code> (metres to the nearest
corridor), <code>prox_idx</code> (the proximity index), <code>enrich</code> (enrichment, 1.00 =
chance), <code>inten_idx</code> (the intensity index, 50 = the average location),
<code>rank_site</code>, <code>coverage</code>, an edge flag and a <code>class</code> &mdash;
ON_CORRIDOR, NEAR_THIN, DIFFUSE or OFF. A <b>point layer</b> (GeoPackage or shapefile) carries the
same columns plus the input layer's own attributes; the <b>distance raster</b> holds, in every
cell, its distance in metres to the nearest corridor &mdash; the quickest way to see the catchment
of the network and to notice a threshold set too generously; and a <b>summary (.txt)</b> identical
to the report on screen, glossary included, so that the supplementary data of a paper and the
screen cannot disagree. Sites that fell outside the raster appear in the table marked as such, so
nothing disappears silently.</p>
<p>Optionally, a <b>histogram script (.R)</b> redraws question 2's distance histogram as a
<b>ggplot2</b> figure &mdash; the same bins and counts shown on screen, not a fresh binning of the
raw distances, so the script and the report can never disagree about what the sample looks like.</p>
<p>When the run finishes the distance raster and the scored sites open in the <b>Viewer</b>
together, and the sites are drawn <b>coloured by their score</b> &mdash; a plain ramp for the
proximity index, and for the intensity index a ramp that breaks at 50, the score the average
location gets, so that above and below are two different colours rather than two shades of one.
<b>Clicking a site</b> opens a panel at the bottom right with that site's whole row: the scores,
the class, and the columns your own layer brought with it. Clicking another site replaces it;
the cross closes it. Lines answer too, which makes the same panel useful over a set of LCPA
routes.</p>
<p>This tool is also available from the command line (<code>--coherence</code>), with the same
options, which makes a study of a dozen periods a single script.</p>

<!--nav:Credits-->
<h2>Credits</h2>
<p>Trajecta uses several third-party software packages to work.</p>

<h3>GDAL</h3>
<p>The Trajecta engine relies on the <a href="https://gdal.org/en/stable/"><b>GDAL</b></a> geospatial libraries. They are
installed together with Trajecta and sit next to the engine, so there is nothing
to install separately and no PATH to configure. The status at the bottom of the
sidebar should read <b>GDAL ready</b> from the first launch.</p>
<p>Trajecta looks beside its own engine before it looks anywhere else, which is
what makes this dependable: an installed copy always uses the libraries it
shipped with, and cannot be disturbed by any other GDAL on the machine — a QGIS
or OSGeo4W install included, whether it is updated, moved or removed.</p>
<p>If the status is not green, the installation is incomplete or the program has
been moved by hand rather than installed. Reinstalling is the proper remedy; as
a stopgap, <b>Locate GDAL folder</b> in the sidebar accepts any folder holding
<code>gdal*.dll</code>.</p>

<h3>Qt6</h3>
<p><a href="https://www.qt.io/product/qt6/qml-book/ch17-qtcpp-qtcpp"><b>Qt6</b></a> is the cross-platform application framework Trajecta Studio's
entire graphical interface is built with — every window, button and widget in
the program, including this Guide, is drawn by it. It is a separate, independent
project from GDAL: Qt draws and runs the interface, GDAL reads and writes the
geospatial data behind it. Trajecta bundles the Qt libraries it needs, the same
way it bundles GDAL, so nothing has to be installed separately for either.</p>

<!--nav:License-->
<h2>License</h2>
<p>Trajecta is free software, distributed under the GNU General Public License,
version 3. The full text below is the license itself, reproduced verbatim.</p>

<div style="text-align:center;">
<p><b>GNU GENERAL PUBLIC LICENSE</b><br>Version 3, 29 June 2007</p>
</div>
<div style="text-align:center;">
<p>Copyright (C) 2007 Free Software Foundation, Inc. <a href="http://fsf.org/">http://fsf.org/</a> Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.</p>
</div>
<h3>Preamble</h3>
<p>The GNU General Public License is a free, copyleft license for software and other kinds of works.</p>
<p>The licenses for most software and other practical works are designed to take away your freedom to share and change the works.  By contrast, the GNU General Public License is intended to guarantee your freedom to share and change all versions of a program--to make sure it remains free software for all its users.  We, the Free Software Foundation, use the GNU General Public License for most of our software; it applies also to any other work released this way by its authors.  You can apply it to your programs, too.</p>
<p>When we speak of free software, we are referring to freedom, not price.  Our General Public Licenses are designed to make sure that you have the freedom to distribute copies of free software (and charge for them if you wish), that you receive source code or can get it if you want it, that you can change the software or use pieces of it in new free programs, and that you know you can do these things.</p>
<p>To protect your rights, we need to prevent others from denying you these rights or asking you to surrender the rights.  Therefore, you have certain responsibilities if you distribute copies of the software, or if you modify it: responsibilities to respect the freedom of others.</p>
<p>For example, if you distribute copies of such a program, whether gratis or for a fee, you must pass on to the recipients the same freedoms that you received.  You must make sure that they, too, receive or can get the source code.  And you must show them these terms so they know their rights.</p>
<p>Developers that use the GNU GPL protect your rights with two steps: (1) assert copyright on the software, and (2) offer you this License giving you legal permission to copy, distribute and/or modify it.</p>
<p>For the developers' and authors' protection, the GPL clearly explains that there is no warranty for this free software.  For both users' and authors' sake, the GPL requires that modified versions be marked as changed, so that their problems will not be attributed erroneously to authors of previous versions.</p>
<p>Some devices are designed to deny users access to install or run modified versions of the software inside them, although the manufacturer can do so.  This is fundamentally incompatible with the aim of protecting users' freedom to change the software.  The systematic pattern of such abuse occurs in the area of products for individuals to use, which is precisely where it is most unacceptable.  Therefore, we have designed this version of the GPL to prohibit the practice for those products.  If such problems arise substantially in other domains, we stand ready to extend this provision to those domains in future versions of the GPL, as needed to protect the freedom of users.</p>
<p>Finally, every program is threatened constantly by software patents. States should not allow patents to restrict development and use of software on general-purpose computers, but in those that do, we wish to avoid the special danger that patents applied to a free program could make it effectively proprietary.  To prevent this, the GPL assures that patents cannot be used to render the program non-free.</p>
<p>The precise terms and conditions for copying, distribution and modification follow.</p>
<p style="text-align:center;"><b>TERMS AND CONDITIONS</b></p>
<h3>0. Definitions.</h3>
<p>"This License" refers to version 3 of the GNU General Public License.</p>
<p>"Copyright" also means copyright-like laws that apply to other kinds of works, such as semiconductor masks.</p>
<p>"The Program" refers to any copyrightable work licensed under this License.  Each licensee is addressed as "you".  "Licensees" and "recipients" may be individuals or organizations.</p>
<p>To "modify" a work means to copy from or adapt all or part of the work in a fashion requiring copyright permission, other than the making of an exact copy.  The resulting work is called a "modified version" of the earlier work or a work "based on" the earlier work.</p>
<p>A "covered work" means either the unmodified Program or a work based on the Program.</p>
<p>To "propagate" a work means to do anything with it that, without permission, would make you directly or secondarily liable for infringement under applicable copyright law, except executing it on a computer or modifying a private copy.  Propagation includes copying, distribution (with or without modification), making available to the public, and in some countries other activities as well.</p>
<p>To "convey" a work means any kind of propagation that enables other parties to make or receive copies.  Mere interaction with a user through a computer network, with no transfer of a copy, is not conveying.</p>
<p>An interactive user interface displays "Appropriate Legal Notices" to the extent that it includes a convenient and prominently visible feature that (1) displays an appropriate copyright notice, and (2) tells the user that there is no warranty for the work (except to the extent that warranties are provided), that licensees may convey the work under this License, and how to view a copy of this License.  If the interface presents a list of user commands or options, such as a menu, a prominent item in the list meets this criterion.</p>
<h3>1. Source Code.</h3>
<p>The "source code" for a work means the preferred form of the work for making modifications to it.  "Object code" means any non-source form of a work.</p>
<p>A "Standard Interface" means an interface that either is an official standard defined by a recognized standards body, or, in the case of interfaces specified for a particular programming language, one that is widely used among developers working in that language.</p>
<p>The "System Libraries" of an executable work include anything, other than the work as a whole, that (a) is included in the normal form of packaging a Major Component, but which is not part of that Major Component, and (b) serves only to enable use of the work with that Major Component, or to implement a Standard Interface for which an implementation is available to the public in source code form.  A "Major Component", in this context, means a major essential component (kernel, window system, and so on) of the specific operating system (if any) on which the executable work runs, or a compiler used to produce the work, or an object code interpreter used to run it.</p>
<p>The "Corresponding Source" for a work in object code form means all the source code needed to generate, install, and (for an executable work) run the object code and to modify the work, including scripts to control those activities.  However, it does not include the work's System Libraries, or general-purpose tools or generally available free programs which are used unmodified in performing those activities but which are not part of the work.  For example, Corresponding Source includes interface definition files associated with source files for the work, and the source code for shared libraries and dynamically linked subprograms that the work is specifically designed to require, such as by intimate data communication or control flow between those subprograms and other parts of the work.</p>
<p>The Corresponding Source need not include anything that users can regenerate automatically from other parts of the Corresponding Source.</p>
<p>The Corresponding Source for a work in source code form is that same work.</p>
<h3>2. Basic Permissions.</h3>
<p>All rights granted under this License are granted for the term of copyright on the Program, and are irrevocable provided the stated conditions are met.  This License explicitly affirms your unlimited permission to run the unmodified Program.  The output from running a covered work is covered by this License only if the output, given its content, constitutes a covered work.  This License acknowledges your rights of fair use or other equivalent, as provided by copyright law.</p>
<p>You may make, run and propagate covered works that you do not convey, without conditions so long as your license otherwise remains in force.  You may convey covered works to others for the sole purpose of having them make modifications exclusively for you, or provide you with facilities for running those works, provided that you comply with the terms of this License in conveying all material for which you do not control copyright.  Those thus making or running the covered works for you must do so exclusively on your behalf, under your direction and control, on terms that prohibit them from making any copies of your copyrighted material outside their relationship with you.</p>
<p>Conveying under any other circumstances is permitted solely under the conditions stated below.  Sublicensing is not allowed; section 10 makes it unnecessary.</p>
<h3>3. Protecting Users' Legal Rights From Anti-Circumvention Law.</h3>
<p>No covered work shall be deemed part of an effective technological measure under any applicable law fulfilling obligations under article 11 of the WIPO copyright treaty adopted on 20 December 1996, or similar laws prohibiting or restricting circumvention of such measures.</p>
<p>When you convey a covered work, you waive any legal power to forbid circumvention of technological measures to the extent such circumvention is effected by exercising rights under this License with respect to the covered work, and you disclaim any intention to limit operation or modification of the work as a means of enforcing, against the work's users, your or third parties' legal rights to forbid circumvention of technological measures.</p>
<h3>4. Conveying Verbatim Copies.</h3>
<p>You may convey verbatim copies of the Program's source code as you receive it, in any medium, provided that you conspicuously and appropriately publish on each copy an appropriate copyright notice; keep intact all notices stating that this License and any non-permissive terms added in accord with section 7 apply to the code; keep intact all notices of the absence of any warranty; and give all recipients a copy of this License along with the Program.</p>
<p>You may charge any price or no price for each copy that you convey, and you may offer support or warranty protection for a fee.</p>
<h3>5. Conveying Modified Source Versions.</h3>
<p>You may convey a work based on the Program, or the modifications to produce it from the Program, in the form of source code under the terms of section 4, provided that you also meet all of these conditions:</p>
<p style="margin-left:24px;">a) The work must carry prominent notices stating that you modified it, and giving a relevant date.</p>
<p style="margin-left:24px;">b) The work must carry prominent notices stating that it is released under this License and any conditions added under section 7.  This requirement modifies the requirement in section 4 to "keep intact all notices".</p>
<p style="margin-left:24px;">c) You must license the entire work, as a whole, under this License to anyone who comes into possession of a copy.  This License will therefore apply, along with any applicable section 7 additional terms, to the whole of the work, and all its parts, regardless of how they are packaged.  This License gives no permission to license the work in any other way, but it does not invalidate such permission if you have separately received it.</p>
<p style="margin-left:24px;">d) If the work has interactive user interfaces, each must display Appropriate Legal Notices; however, if the Program has interactive interfaces that do not display Appropriate Legal Notices, your work need not make them do so.</p>
<p>A compilation of a covered work with other separate and independent works, which are not by their nature extensions of the covered work, and which are not combined with it such as to form a larger program, in or on a volume of a storage or distribution medium, is called an "aggregate" if the compilation and its resulting copyright are not used to limit the access or legal rights of the compilation's users beyond what the individual works permit.  Inclusion of a covered work in an aggregate does not cause this License to apply to the other parts of the aggregate.</p>
<h3>6. Conveying Non-Source Forms.</h3>
<p>You may convey a covered work in object code form under the terms of sections 4 and 5, provided that you also convey the machine-readable Corresponding Source under the terms of this License, in one of these ways:</p>
<p style="margin-left:24px;">a) Convey the object code in, or embodied in, a physical product (including a physical distribution medium), accompanied by the Corresponding Source fixed on a durable physical medium customarily used for software interchange.</p>
<p style="margin-left:24px;">b) Convey the object code in, or embodied in, a physical product (including a physical distribution medium), accompanied by a written offer, valid for at least three years and valid for as long as you offer spare parts or customer support for that product model, to give anyone who possesses the object code either (1) a copy of the Corresponding Source for all the software in the product that is covered by this License, on a durable physical medium customarily used for software interchange, for a price no more than your reasonable cost of physically performing this conveying of source, or (2) access to copy the Corresponding Source from a network server at no charge.</p>
<p style="margin-left:24px;">c) Convey individual copies of the object code with a copy of the written offer to provide the Corresponding Source.  This alternative is allowed only occasionally and noncommercially, and only if you received the object code with such an offer, in accord with subsection 6b.</p>
<p style="margin-left:24px;">d) Convey the object code by offering access from a designated place (gratis or for a charge), and offer equivalent access to the Corresponding Source in the same way through the same place at no further charge.  You need not require recipients to copy the Corresponding Source along with the object code.  If the place to copy the object code is a network server, the Corresponding Source may be on a different server (operated by you or a third party) that supports equivalent copying facilities, provided you maintain clear directions next to the object code saying where to find the Corresponding Source.  Regardless of what server hosts the Corresponding Source, you remain obligated to ensure that it is available for as long as needed to satisfy these requirements.</p>
<p style="margin-left:24px;">e) Convey the object code using peer-to-peer transmission, provided you inform other peers where the object code and Corresponding Source of the work are being offered to the general public at no charge under subsection 6d.</p>
<p>A separable portion of the object code, whose source code is excluded from the Corresponding Source as a System Library, need not be included in conveying the object code work.</p>
<p>A "User Product" is either (1) a "consumer product", which means any tangible personal property which is normally used for personal, family, or household purposes, or (2) anything designed or sold for incorporation into a dwelling.  In determining whether a product is a consumer product, doubtful cases shall be resolved in favor of coverage.  For a particular product received by a particular user, "normally used" refers to a typical or common use of that class of product, regardless of the status of the particular user or of the way in which the particular user actually uses, or expects or is expected to use, the product.  A product is a consumer product regardless of whether the product has substantial commercial, industrial or non-consumer uses, unless such uses represent the only significant mode of use of the product.</p>
<p>"Installation Information" for a User Product means any methods, procedures, authorization keys, or other information required to install and execute modified versions of a covered work in that User Product from a modified version of its Corresponding Source.  The information must suffice to ensure that the continued functioning of the modified object code is in no case prevented or interfered with solely because modification has been made.</p>
<p>If you convey an object code work under this section in, or with, or specifically for use in, a User Product, and the conveying occurs as part of a transaction in which the right of possession and use of the User Product is transferred to the recipient in perpetuity or for a fixed term (regardless of how the transaction is characterized), the Corresponding Source conveyed under this section must be accompanied by the Installation Information.  But this requirement does not apply if neither you nor any third party retains the ability to install modified object code on the User Product (for example, the work has been installed in ROM).</p>
<p>The requirement to provide Installation Information does not include a requirement to continue to provide support service, warranty, or updates for a work that has been modified or installed by the recipient, or for the User Product in which it has been modified or installed.  Access to a network may be denied when the modification itself materially and adversely affects the operation of the network or violates the rules and protocols for communication across the network.</p>
<p>Corresponding Source conveyed, and Installation Information provided, in accord with this section must be in a format that is publicly documented (and with an implementation available to the public in source code form), and must require no special password or key for unpacking, reading or copying.</p>
<h3>7. Additional Terms.</h3>
<p>"Additional permissions" are terms that supplement the terms of this License by making exceptions from one or more of its conditions. Additional permissions that are applicable to the entire Program shall be treated as though they were included in this License, to the extent that they are valid under applicable law.  If additional permissions apply only to part of the Program, that part may be used separately under those permissions, but the entire Program remains governed by this License without regard to the additional permissions.</p>
<p>When you convey a copy of a covered work, you may at your option remove any additional permissions from that copy, or from any part of it.  (Additional permissions may be written to require their own removal in certain cases when you modify the work.)  You may place additional permissions on material, added by you to a covered work, for which you have or can give appropriate copyright permission.</p>
<p>Notwithstanding any other provision of this License, for material you add to a covered work, you may (if authorized by the copyright holders of that material) supplement the terms of this License with terms:</p>
<p style="margin-left:24px;">a) Disclaiming warranty or limiting liability differently from the terms of sections 15 and 16 of this License; or</p>
<p style="margin-left:24px;">b) Requiring preservation of specified reasonable legal notices or author attributions in that material or in the Appropriate Legal Notices displayed by works containing it; or</p>
<p style="margin-left:24px;">c) Prohibiting misrepresentation of the origin of that material, or requiring that modified versions of such material be marked in reasonable ways as different from the original version; or</p>
<p style="margin-left:24px;">d) Limiting the use for publicity purposes of names of licensors or authors of the material; or</p>
<p style="margin-left:24px;">e) Declining to grant rights under trademark law for use of some trade names, trademarks, or service marks; or</p>
<p style="margin-left:24px;">f) Requiring indemnification of licensors and authors of that material by anyone who conveys the material (or modified versions of it) with contractual assumptions of liability to the recipient, for any liability that these contractual assumptions directly impose on those licensors and authors.</p>
<p>All other non-permissive additional terms are considered "further restrictions" within the meaning of section 10.  If the Program as you received it, or any part of it, contains a notice stating that it is governed by this License along with a term that is a further restriction, you may remove that term.  If a license document contains a further restriction but permits relicensing or conveying under this License, you may add to a covered work material governed by the terms of that license document, provided that the further restriction does not survive such relicensing or conveying.</p>
<p>If you add terms to a covered work in accord with this section, you must place, in the relevant source files, a statement of the additional terms that apply to those files, or a notice indicating where to find the applicable terms.</p>
<p>Additional terms, permissive or non-permissive, may be stated in the form of a separately written license, or stated as exceptions; the above requirements apply either way.</p>
<h3>8. Termination.</h3>
<p>You may not propagate or modify a covered work except as expressly provided under this License.  Any attempt otherwise to propagate or modify it is void, and will automatically terminate your rights under this License (including any patent licenses granted under the third paragraph of section 11).</p>
<p>However, if you cease all violation of this License, then your license from a particular copyright holder is reinstated (a) provisionally, unless and until the copyright holder explicitly and finally terminates your license, and (b) permanently, if the copyright holder fails to notify you of the violation by some reasonable means prior to 60 days after the cessation.</p>
<p>Moreover, your license from a particular copyright holder is reinstated permanently if the copyright holder notifies you of the violation by some reasonable means, this is the first time you have received notice of violation of this License (for any work) from that copyright holder, and you cure the violation prior to 30 days after your receipt of the notice.</p>
<p>Termination of your rights under this section does not terminate the licenses of parties who have received copies or rights from you under this License.  If your rights have been terminated and not permanently reinstated, you do not qualify to receive new licenses for the same material under section 10.</p>
<h3>9. Acceptance Not Required for Having Copies.</h3>
<p>You are not required to accept this License in order to receive or run a copy of the Program.  Ancillary propagation of a covered work occurring solely as a consequence of using peer-to-peer transmission to receive a copy likewise does not require acceptance.  However, nothing other than this License grants you permission to propagate or modify any covered work.  These actions infringe copyright if you do not accept this License.  Therefore, by modifying or propagating a covered work, you indicate your acceptance of this License to do so.</p>
<h3>10. Automatic Licensing of Downstream Recipients.</h3>
<p>Each time you convey a covered work, the recipient automatically receives a license from the original licensors, to run, modify and propagate that work, subject to this License.  You are not responsible for enforcing compliance by third parties with this License.</p>
<p>An "entity transaction" is a transaction transferring control of an organization, or substantially all assets of one, or subdividing an organization, or merging organizations.  If propagation of a covered work results from an entity transaction, each party to that transaction who receives a copy of the work also receives whatever licenses to the work the party's predecessor in interest had or could give under the previous paragraph, plus a right to possession of the Corresponding Source of the work from the predecessor in interest, if the predecessor has it or can get it with reasonable efforts.</p>
<p>You may not impose any further restrictions on the exercise of the rights granted or affirmed under this License.  For example, you may not impose a license fee, royalty, or other charge for exercise of rights granted under this License, and you may not initiate litigation (including a cross-claim or counterclaim in a lawsuit) alleging that any patent claim is infringed by making, using, selling, offering for sale, or importing the Program or any portion of it.</p>
<h3>11. Patents.</h3>
<p>A "contributor" is a copyright holder who authorizes use under this License of the Program or a work on which the Program is based.  The work thus licensed is called the contributor's "contributor version".</p>
<p>A contributor's "essential patent claims" are all patent claims owned or controlled by the contributor, whether already acquired or hereafter acquired, that would be infringed by some manner, permitted by this License, of making, using, or selling its contributor version, but do not include claims that would be infringed only as a consequence of further modification of the contributor version.  For purposes of this definition, "control" includes the right to grant patent sublicenses in a manner consistent with the requirements of this License.</p>
<p>Each contributor grants you a non-exclusive, worldwide, royalty-free patent license under the contributor's essential patent claims, to make, use, sell, offer for sale, import and otherwise run, modify and propagate the contents of its contributor version.</p>
<p>In the following three paragraphs, a "patent license" is any express agreement or commitment, however denominated, not to enforce a patent (such as an express permission to practice a patent or covenant not to sue for patent infringement).  To "grant" such a patent license to a party means to make such an agreement or commitment not to enforce a patent against the party.</p>
<p>If you convey a covered work, knowingly relying on a patent license, and the Corresponding Source of the work is not available for anyone to copy, free of charge and under the terms of this License, through a publicly available network server or other readily accessible means, then you must either (1) cause the Corresponding Source to be so available, or (2) arrange to deprive yourself of the benefit of the patent license for this particular work, or (3) arrange, in a manner consistent with the requirements of this License, to extend the patent license to downstream recipients.  "Knowingly relying" means you have actual knowledge that, but for the patent license, your conveying the covered work in a country, or your recipient's use of the covered work in a country, would infringe one or more identifiable patents in that country that you have reason to believe are valid.</p>
<p>If, pursuant to or in connection with a single transaction or arrangement, you convey, or propagate by procuring conveyance of, a covered work, and grant a patent license to some of the parties receiving the covered work authorizing them to use, propagate, modify or convey a specific copy of the covered work, then the patent license you grant is automatically extended to all recipients of the covered work and works based on it.</p>
<p>A patent license is "discriminatory" if it does not include within the scope of its coverage, prohibits the exercise of, or is conditioned on the non-exercise of one or more of the rights that are specifically granted under this License.  You may not convey a covered work if you are a party to an arrangement with a third party that is in the business of distributing software, under which you make payment to the third party based on the extent of your activity of conveying the work, and under which the third party grants, to any of the parties who would receive the covered work from you, a discriminatory patent license (a) in connection with copies of the covered work conveyed by you (or copies made from those copies), or (b) primarily for and in connection with specific products or compilations that contain the covered work, unless you entered into that arrangement, or that patent license was granted, prior to 28 March 2007.</p>
<p>Nothing in this License shall be construed as excluding or limiting any implied license or other defenses to infringement that may otherwise be available to you under applicable patent law.</p>
<h3>12. No Surrender of Others' Freedom.</h3>
<p>If conditions are imposed on you (whether by court order, agreement or otherwise) that contradict the conditions of this License, they do not excuse you from the conditions of this License.  If you cannot convey a covered work so as to satisfy simultaneously your obligations under this License and any other pertinent obligations, then as a consequence you may not convey it at all.  For example, if you agree to terms that obligate you to collect a royalty for further conveying from those to whom you convey the Program, the only way you could satisfy both those terms and this License would be to refrain entirely from conveying the Program.</p>
<h3>13. Use with the GNU Affero General Public License.</h3>
<p>Notwithstanding any other provision of this License, you have permission to link or combine any covered work with a work licensed under version 3 of the GNU Affero General Public License into a single combined work, and to convey the resulting work.  The terms of this License will continue to apply to the part which is the covered work, but the special requirements of the GNU Affero General Public License, section 13, concerning interaction through a network will apply to the combination as such.</p>
<h3>14. Revised Versions of this License.</h3>
<p>The Free Software Foundation may publish revised and/or new versions of the GNU General Public License from time to time.  Such new versions will be similar in spirit to the present version, but may differ in detail to address new problems or concerns.</p>
<p>Each version is given a distinguishing version number.  If the Program specifies that a certain numbered version of the GNU General Public License "or any later version" applies to it, you have the option of following the terms and conditions either of that numbered version or of any later version published by the Free Software Foundation.  If the Program does not specify a version number of the GNU General Public License, you may choose any version ever published by the Free Software Foundation.</p>
<p>If the Program specifies that a proxy can decide which future versions of the GNU General Public License can be used, that proxy's public statement of acceptance of a version permanently authorizes you to choose that version for the Program.</p>
<p>Later license versions may give you additional or different permissions.  However, no additional obligations are imposed on any author or copyright holder as a result of your choosing to follow a later version.</p>
<h3>15. Disclaimer of Warranty.</h3>
<p>THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.</p>
<h3>16. Limitation of Liability.</h3>
<p>IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.</p>
<h3>17. Interpretation of Sections 15 and 16.</h3>
<p>If the disclaimer of warranty and limitation of liability provided above cannot be given local legal effect according to their terms, reviewing courts shall apply local law that most closely approximates an absolute waiver of all civil liability in connection with the Program, unless a warranty or assumption of liability accompanies a copy of the Program in return for a fee.</p>
<p style="text-align:center;"><b>END OF TERMS AND CONDITIONS</b></p>
<h3>How to Apply These Terms to Your New Programs</h3>
<p>If you develop a new program, and you want it to be of the greatest possible use to the public, the best way to achieve this is to make it free software which everyone can redistribute and change under these terms.</p>
<p>To do so, attach the following notices to the program.  It is safest to attach them to the start of each source file to most effectively state the exclusion of warranty; and each file should have at least the "copyright" line and a pointer to where the full notice is found.</p>
<p style="margin-left:24px; font-family:Consolas,monospace;">    &lt;one line to give the program's name and a brief idea of what it does.&gt;<br>
    Copyright (C) &lt;year&gt;  &lt;name of author&gt;<br>
<br>
    This program is free software: you can redistribute it and/or modify<br>
    it under the terms of the GNU General Public License as published by<br>
    the Free Software Foundation, either version 3 of the License, or<br>
    (at your option) any later version.<br>
<br>
    This program is distributed in the hope that it will be useful,<br>
    but WITHOUT ANY WARRANTY; without even the implied warranty of<br>
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the<br>
    GNU General Public License for more details.<br>
<br>
    You should have received a copy of the GNU General Public License<br>
    along with this program.  If not, see &lt;http://www.gnu.org/licenses/&gt;.</p>
<p>Also add information on how to contact you by electronic and paper mail.</p>
<p>If the program does terminal interaction, make it output a short notice like this when it starts in an interactive mode:</p>
<p style="margin-left:24px; font-family:Consolas,monospace;">    &lt;program&gt;  Copyright (C) &lt;year&gt;  &lt;name of author&gt;<br>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.<br>
    This is free software, and you are welcome to redistribute it<br>
    under certain conditions; type `show c' for details.</p>
<p>The hypothetical commands `show w' and `show c' should show the appropriate parts of the General Public License.  Of course, your program's commands might be different; for a GUI interface, you would use an "about box".</p>
<p>You should also get your employer (if you work as a programmer) or school, if any, to sign a "copyright disclaimer" for the program, if necessary. For more information on this, and how to apply and follow the GNU GPL, see <a href="http://www.gnu.org/licenses/">http://www.gnu.org/licenses/</a>.</p>
<p>The GNU General Public License does not permit incorporating your program into proprietary programs.  If your program is a subroutine library, you may consider it more useful to permit linking proprietary applications with the library.  If this is what you want to do, use the GNU Lesser General Public License instead of this License.  But first, please read <a href="http://www.gnu.org/philosophy/why-not-lgpl.html">http://www.gnu.org/philosophy/why-not-lgpl.html</a>.</p>

<!--nav:References-->
<h2>References</h2>
<p>Baddeley, A., Rubak, E., &amp; Turner, R. (2015). <i>Spatial Point Patterns:
Methodology and Applications with R</i>. Chapman &amp; Hall/CRC.</p>
<p>Besag, J., &amp; Diggle, P. J. (1977). Simple Monte Carlo tests for spatial
pattern. <i>Journal of the Royal Statistical Society: Series C (Applied
Statistics)</i>, 26(3), 327&ndash;333.
<a href="https://doi.org/10.2307/2346974">doi:10.2307/2346974</a></p>
<p>Bresenham, J. E. (1965). Algorithm for computer control of a digital plotter.
<i>IBM Systems Journal</i>, 4(1), 25&ndash;30.
<a href="https://doi.org/10.1147/sj.41.0025">doi:10.1147/sj.41.0025</a></p>
<p>Campbell, M. J., Dennison, P. E., Butler, B. W., &amp; Page, W. G. (2019).
Using crowdsourced fitness tracker data to model the relationship between slope
and travel rates. <i>Applied Geography</i>, 106, 93&ndash;107.
<a href="https://doi.org/10.1016/j.apgeog.2019.03.008">doi:10.1016/j.apgeog.2019.03.008</a></p>
<p>Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
<i>Numerische Mathematik</i>, 1, 269&ndash;271.
<a href="https://doi.org/10.1007/BF01386390">doi:10.1007/BF01386390</a></p>
<p>Felzenszwalb, P. F., &amp; Huttenlocher, D. P. (2012). Distance transforms of
sampled functions. <i>Theory of Computing</i>, 8(19), 415&ndash;428.
<a href="https://doi.org/10.4086/toc.2012.v008a019">doi:10.4086/toc.2012.v008a019</a></p>
<p>Hausdorff, F. (1914). <i>Grundz&uuml;ge der Mengenlehre</i>. Veit &amp; Comp.</p>
<p>Herzog, I. (2013). The potential and limits of Optimal Path Analysis. In
A. Bevan &amp; M. Lake (eds.), <i>Computational Approaches to Archaeological
Spaces</i> (pp. 179&ndash;211). Left Coast Press.</p>
<p>Herzog, I. (2014). A review of case studies in archaeological least-cost
analysis. <i>Archeologia e Calcolatori</i>, 25, 223&ndash;239.</p>
<p>Hope, A. C. A. (1968). A simplified Monte Carlo significance test procedure.
<i>Journal of the Royal Statistical Society: Series B (Methodological)</i>,
30(3), 582&ndash;598.
<a href="https://doi.org/10.1111/j.2517-6161.1968.tb00759.x">doi:10.1111/j.2517-6161.1968.tb00759.x</a></p>
<p>Irmischer, I. J., &amp; Clarke, K. C. (2017). Measuring and modeling the
speed of human navigation. <i>Cartography and Geographic Information Science</i>,
45(2), 177&ndash;186.
<a href="https://doi.org/10.1080/15230406.2017.1292150">doi:10.1080/15230406.2017.1292150</a></p>
<p>Liang, Y.-D., &amp; Barsky, B. A. (1984). A new concept and method for line
clipping. <i>ACM Transactions on Graphics</i>, 3(1), 1&ndash;22.
<a href="https://doi.org/10.1145/357332.357333">doi:10.1145/357332.357333</a></p>
<p>Lotwick, H. W., &amp; Silverman, B. W. (1982). Methods for analysing spatial
processes of several types of points. <i>Journal of the Royal Statistical
Society: Series B (Methodological)</i>, 44(3), 406&ndash;413.
<a href="https://doi.org/10.1111/j.2517-6161.1982.tb01221.x">doi:10.1111/j.2517-6161.1982.tb01221.x</a></p>
<p>M&aacute;rquez-P&eacute;rez, J., Vallejo-Villalta, I., &amp;
&Aacute;lvarez-Francoso, J. I. (2017). Estimated travel time for walking trails
in natural areas. <i>Geografisk Tidsskrift&ndash;Danish Journal of Geography</i>,
117(1), 53&ndash;62.
<a href="https://doi.org/10.1080/00167223.2017.1316212">doi:10.1080/00167223.2017.1316212</a></p>
<p>Minetti, A. E., Moia, C., Roi, G. S., Susta, D., &amp; Ferretti, G. (2002).
Energy cost of walking and running at extreme uphill and downhill slopes.
<i>Journal of Applied Physiology</i>, 93(3), 1039&ndash;1046.
<a href="https://doi.org/10.1152/japplphysiol.01177.2001">doi:10.1152/japplphysiol.01177.2001</a></p>
<p>Otsu, N. (1979). A threshold selection method from gray-level histograms.
<i>IEEE Transactions on Systems, Man, and Cybernetics</i>, 9(1), 62&ndash;66.
<a href="https://doi.org/10.1109/TSMC.1979.4310076">doi:10.1109/TSMC.1979.4310076</a></p>
<p>Park, S. W., Linsen, L., Kreylos, O., Owens, J. D., &amp; Hamann, B. (2006).
Discrete Sibson interpolation. <i>IEEE Transactions on Visualization and
Computer Graphics</i>, 12(2), 243&ndash;253.
<a href="https://doi.org/10.1109/TVCG.2006.27">doi:10.1109/TVCG.2006.27</a></p>
<p>Sibson, R. (1981). A brief description of natural neighbour interpolation. In
V. Barnett (ed.), <i>Interpreting Multivariate Data</i> (pp. 21&ndash;36). Wiley.</p>
<p>Tobler, W. (1993). <i>Three presentations on geographical analysis and
modeling</i>. National Center for Geographic Information and Analysis,
Technical Report 93-1.</p>
<p>White, D. A. (2015). The Basics of Least Cost Analysis for Archaeological
Applications. <i>Advances in Archaeological Practice</i>, 3(4), 407&ndash;414.
<a href="https://doi.org/10.7183/2326-3768.3.4.407">doi:10.7183/2326-3768.3.4.407</a></p>
<p>White, D. A., &amp; Barber, S. B. (2012). Geospatial modeling of pedestrian
transportation networks: A case study from precolumbian Oaxaca, Mexico.
<i>Journal of Archaeological Science</i>, 39(8), 2684&ndash;2696.
<a href="https://doi.org/10.1016/j.jas.2012.04.017">doi:10.1016/j.jas.2012.04.017</a></p>
)HTML");

    // Every figure the guide owns. A page is given only the ones it actually
    // shows: decoding and rescaling them is the expensive part, and a reader
    // who never opens the LCPA page should never pay for its picture.
    struct GuideFigure { const char *token; const char *path; bool fullWidth; };
    static const GuideFigure kFigures[] = {
        {"grid",       ":/assets/guide/Grid_FETE.jpg",         false},
        {"unfiltered", ":/assets/guide/unfiltered_FETE.jpg",   false},
        {"filtered",   ":/assets/guide/filtered_FETE.jpg",     true },
        {"lcpa",       ":/assets/guide/LCPA.jpg",              true },
        {"density",    ":/assets/guide/FETE_density.jpg",      false},
        {"nni",        ":/assets/guide/FETE_density_NNI.jpg",  false},
    };

    // The pages are cut out of that one document at the <!--nav:--> markers,
    // rather than held as a dozen separate literals. It keeps the guide
    // editable as a single piece of prose, and it means adding a section adds
    // a page to the sidebar without touching any of this.
    const QString marker = QStringLiteral("<!--nav:");
    const int firstMark = guideHtml.indexOf(marker);
    // Whatever precedes the first marker is the shared <style> block, and
    // every page needs its own copy: they are separate documents.
    const QString prelude = firstMark >= 0 ? guideHtml.left(firstMark) : QString();

    for (int pos = firstMark; pos >= 0; ) {
        const int labelEnd = guideHtml.indexOf(QStringLiteral("-->"), pos);
        if (labelEnd < 0)
            break;
        const QString label =
            guideHtml.mid(pos + marker.size(), labelEnd - pos - marker.size());
        const int bodyStart = labelEnd + 3;
        const int next = guideHtml.indexOf(marker, bodyStart);
        QString body = next < 0 ? guideHtml.mid(bodyStart)
                                : guideHtml.mid(bodyStart, next - bodyStart);

        // Anchors for the right-hand column, injected here so the prose does
        // not have to carry them and cannot forget one.
        QStringList sections;
        QString anchored;
        anchored.reserve(body.size() + 256);
        int from = 0;
        while (true) {
            const int h3 = body.indexOf(QStringLiteral("<h3>"), from);
            if (h3 < 0) {
                anchored += body.mid(from);
                break;
            }
            const int close = body.indexOf(QStringLiteral("</h3>"), h3);
            if (close < 0) {
                anchored += body.mid(from);
                break;
            }
            const QString inner = body.mid(h3 + 4, close - h3 - 4);
            anchored += body.mid(from, h3 - from);
            anchored += QStringLiteral("<a name=\"s%1\"></a>").arg(sections.size());
            anchored += body.mid(h3, close + 5 - h3);
            // Through a document fragment rather than by stripping tags: the
            // headings carry entities (&mdash;, &amp;) and a sidebar entry
            // showing "&mdash;" would be its own small bug.
            sections << QTextDocumentFragment::fromHtml(inner).toPlainText().simplified();
            from = close + 5;
        }

        auto *browser = new GuideBrowser(m_guidePages);
        browser->setObjectName(QStringLiteral("GuideBrowser"));
        browser->setOpenLinks(false);
        browser->setOpenExternalLinks(false);
        connect(browser, &QTextBrowser::anchorClicked, this,
                &MainWindow::handleGuideLink);
        browser->setTemplate(prelude + anchored);
        for (const GuideFigure &f : kFigures) {
            if (anchored.contains(QLatin1String("guide:") + QLatin1String(f.token)))
                browser->addFigure(QString::fromLatin1(f.token),
                                   QString::fromLatin1(f.path), f.fullWidth);
        }
        m_guidePages->addWidget(browser);

        // FETE/LCPA nest under a "Processing" group; the three
        // post-processing tools nest under "Post-processing" — the group
        // headers themselves are inserted here, right before their first
        // child, since they own no page and so no <!--nav:--> marker of
        // their own. Every other page (Credits, References — Overview and
        // About are added outside this loop) stays a plain top-level row.
        const int pageIndex = m_guidePages->count() - 1;
        static const QString kProcessing = QStringLiteral("processing");
        static const QString kPostProcessing = QStringLiteral("postprocessing");
        if (label == QLatin1String("FETE")) {
            nav->addGroupHeader(tr("Processing"), kProcessing);
            nav->addChildItem(tr("FETE analysis"), pageIndex, kProcessing);
        } else if (label == QLatin1String("LCPA")) {
            nav->addChildItem(tr("LCPA analysis"), pageIndex, kProcessing);
        } else if (label == QLatin1String("NNI interpolation")) {
            nav->addGroupHeader(tr("Post-processing"), kPostProcessing);
            nav->addChildItem(label, pageIndex, kPostProcessing);
        } else if (label == QLatin1String("Route comparison")) {
            nav->addChildItem(label, pageIndex, kPostProcessing);
        } else if (label.startsWith(QLatin1String("Site"))) {
            // The marker spells this with the same en dash as the in-page
            // heading; the sidebar spells it with a plain hyphen instead.
            nav->addChildItem(tr("Site-corridor coherence"), pageIndex, kPostProcessing);
        } else {
            nav->addPageItem(label, pageIndex);
        }
        m_guideSections << sections;

        pos = next;
    }

    // About, last: the logo and the project links, kept off the page a
    // reader lands on but still one click away. buildAboutPage() is the same
    // widget the standalone About tab used to show — reused as-is rather
    // than rebuilt, since a stacked page and a top-level tab are laid out
    // the same way here.
    m_guidePages->addWidget(buildAboutPage());
    nav->addPageItem(tr("About"), m_guidePages->count() - 1);
    m_guideSections << QStringList();

    // One entry per page, Overview included, so the two lists stay in step.
    m_guideSections.prepend(QStringList());

    nav->onPageChosen = [this](int page) { showGuideSection(page); };
    connect(m_guideToc, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row < 0)
            return;
        if (auto *b = qobject_cast<QTextBrowser *>(m_guidePages->currentWidget()))
            b->scrollToAnchor(QStringLiteral("s%1").arg(row));
    });

    nav->selectPage(0);
    nav->slideToCurrent(false);
    return page;
}

// Opens the Guide on one of its pages, and refreshes the right-hand column to
// match. Kept in one place because three callers need it: the sidebar, the
// walkthrough, and returning to the tab.
void MainWindow::showGuideSection(int index)
{
    if (!m_guidePages || index < 0 || index >= m_guidePages->count())
        return;
    m_guidePages->setCurrentIndex(index);
    // selectPage() also opens the row's group first if it has one and is not
    // already open — a plain setCurrentRow(index) would leave a page reached
    // via the walkthrough or the test hook with its own nav row hidden.
    if (m_guideNav)
        static_cast<GuideNav *>(m_guideNav)->selectPage(index);

    m_guideToc->blockSignals(true);
    m_guideToc->clear();
    const QStringList sections = index < m_guideSections.size()
                                     ? m_guideSections.at(index) : QStringList();
    for (const QString &s : sections)
        m_guideToc->addItem(s);
    m_guideToc->setCurrentRow(-1);
    m_guideToc->blockSignals(false);
    m_guideTocPanel->setVisible(sections.size() > 1);
}

int MainWindow::guidePageCount() const
{
    return m_guidePages ? m_guidePages->count() : 0;
}

// The guide's own two link schemes. Shared by every page's browser, which is
// why it is a slot rather than a lambda repeated a dozen times.
void MainWindow::handleGuideLink(const QUrl &url)
{
    if (url.scheme() == QLatin1String("trajecta")) {
        const QString what = url.path().isEmpty() ? url.host() : url.path();
        if (what == QLatin1String("walkthrough")) {
            // Asked first. The link sits in the middle of a paragraph of
            // running text, so it can be reached by a stray click while
            // reading — and starting the tour takes over the whole window
            // and moves the page you were on.
            if (TrajectaUi::confirm(
                    this, tr("Start the walkthrough"),
                    tr("Take the guided tour of the interface?\n\n"
                       "Trajecta Studio will be maximised, so that every "
                       "screen of the tour has room to be shown whole.\n\n"
                       "Nothing you have set up is changed, and the tour "
                       "can be stopped at any point."),
                    tr("Start walkthrough"), tr("Not now"), 40,
                    TrajectaUi::Fill::Accept)) {
                startWalkthroughMaximised();
            }
        } else if (what == QLatin1String("report")) {
            // Straight in, with no confirmation: opening the form commits
            // the user to nothing, and Cancel is right there.
            TrajectaUi::showReportForm(this);
        }
        return;
    }
    QDesktopServices::openUrl(url);
}

// The Guide's home. Every arrival at the Guide tab lands here, so it carries
// the things a newcomer needs first — what this is, how to be walked through
// it, and what its core tools are. The logo and the project links live on
// their own About page, last in the sidebar.
//
// Widgets, not a document: a QLabel can be given a rich-text link that reaches
// handleGuideLink, which is all this page needs from a QTextBrowser.
QWidget *MainWindow::buildGuideOverviewPage()
{
    auto *scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setObjectName(QStringLiteral("GuideOverviewScroll"));

    auto *inner = new QWidget(scroll);
    inner->setObjectName(QStringLiteral("GuideOverviewHost"));
    auto *pageLayout = new QVBoxLayout(inner);
    pageLayout->setContentsMargins(0, 0, 0, 0);

    // One panel for the whole page, like the other guide pages, rather than
    // each label drawing its own background: on a theme with a picture behind
    // it, loose labels read as a stack of unrelated strips. Stretched to fill
    // the page rather than hugging the (short) text, so the panel is the same
    // height as the GuideBrowser pages either side of it in the sidebar —
    // #Card and #GuideBrowser already share one background/border/radius,
    // only the height used to differ.
    auto *card = new QFrame(inner);
    card->setObjectName(QStringLiteral("Card"));
    pageLayout->addWidget(card, 1);

    auto *layout = new QVBoxLayout(card);
    layout->setContentsMargins(28, 24, 28, 24);
    layout->setSpacing(8);

    // The logo and the project links moved to their own About page, last in
    // the sidebar — this is the page a reader lands on, and its job is to
    // say what Trajecta does, not who made it.
    auto *pageTitle = new QLabel(tr("Overview"), card);
    // Its own rule, not CardTitle: this stands in for the <h2> every other
    // guide page opens with, so it is styled to match — the same teal
    // (#a8d0c8, already in kDarkColors) and a size to match, not the small
    // bold used for a card's own section heading elsewhere in the app.
    pageTitle->setObjectName(QStringLiteral("GuideOverviewTitle"));
    pageTitle->setAlignment(Qt::AlignLeft);
    layout->addWidget(pageTitle);

    layout->addSpacing(10);

    // No inline colours: a hard-coded grey is invisible on a light theme and
    // is the one thing the palette mapping cannot reach. Everything here
    // inherits AboutBody, so it follows the theme like the rest of the text.
    auto addBody = [card, layout](const QString &html, Qt::Alignment align) {
        auto *body = new QLabel(card);
        body->setObjectName(QStringLiteral("AboutBody"));
        body->setAlignment(align);
        body->setWordWrap(true);
        body->setTextFormat(Qt::RichText);
        QPalette pal = body->palette();
        pal.setColor(QPalette::Link, QColor(0x7e, 0xa8, 0xa0));
        body->setPalette(pal);
        body->setText(html);
        layout->addWidget(body);
        return body;
    };

    // The overview prose. Ranged left, since this is something to read.
    auto *intro = addBody(
        tr("<p><b>Trajecta</b> is a free, open-source least-cost analysis (LCA) engine distributed under the "
           "GNU General Public License 3.0. Trajecta is shipped along with <b>Trajecta Studio</b>, a fully customized GUI (Graphic User Interface) specifically developed to provide a seamless and user-friendly experience to every type of user, even without prior experience. "
           "Trajecta and Trajecta Studio are primarily designed to be used by archaeologists, historians, and other researchers who need to model movement across landscapes to investigate spatial patterns in the Ancient World. Trajecta is completely written in C++ and Qt for fast and efficient computation. The source code of Trajecta is available on <a href=\"https://github.com/ArcheoHacker1501/trajecta\">GitHub</a>.</p>"

           "<p>For an introductory walkthrough on how to use Trajecta, you can launch this in-app "
           "<a href=\"trajecta:walkthrough\"><b>tutorial</b></a>. You can also click on the <b>?</b> badge beside any field to get "
           "information about the selected parameter or function. For additional details on Trajecta's features, you can also refer to the specific pages in this <b>Guide</b> section.</p>"
           "<p>At its core, Trajecta models movement across a landscape with "
           "<b>FETE (From-Everywhere-To-Everywhere)</b> and <b>LCPA (Least-Cost Path Analysis)</b>, refines and checks the result with "
           "<b>NNI (Natural Neighbour Interpolation)</b> and the <b>route comparison tool</b>, and tests it "
           "against real settlement patterns with the <b>site&#8211;corridor "
           "coherence tool</b>. Finally, the built in <b>Viewer</b> offers a simple platform to visualize the results of the computations directly in-app.</p>"

           "<p>The list on the left opens the rest of the <b>Guide</b>, and each tool is described in detail. For contacts and information about the author, please refer to the <b>About</b> section.</p>"

           "<p>Trajecta was inspired and made possible thanks to the previous work of many scholars from different fields. All the references and sources used to develop Trajecta are listed in the <b>References</b> section of this Guide.</p><br><br><br>"

           "<p style=\"text-align:center; font-size:16px;\"><b>IMPORTANT</b>: Be patient, this software is currently under development and can "
           "contain bugs or errors! For bug reporting, problems during the "
           "installation, or to suggest improvements or additional features to "
           "be included in Trajecta, please use this "
           "<a href=\"trajecta:report\"><b>report form</b></a>.</p>"),
        Qt::AlignLeft | Qt::AlignTop);
    // Not setOpenExternalLinks: these two are the guide's own schemes and have
    // to reach handleGuideLink like every other link in the Guide does.
    connect(intro, &QLabel::linkActivated, this,
            [this](const QString &href) { handleGuideLink(QUrl(href)); });

    // Now that the card is stretched to the full page height (above), an
    // explicit stretch has to claim the leftover space itself — otherwise
    // Qt hands it to the body label, which has no stretch factor of its own
    // but is still the only item here free to grow, and its default vertical
    // centring then floats the text away from the title instead of leaving
    // it pinned underneath.
    layout->addStretch(1);

    scroll->setWidget(inner);
    return scroll;
}

QWidget *MainWindow::buildAboutPage()
{
    auto *page = new QWidget(this);
    auto *pageLayout = new QVBoxLayout(page);
    // 0, not the 28,24,28,24 this carried when About was only ever a
    // standalone top-level tab: as a Guide page it sits inside
    // buildGuidePage()'s own outer margin of the same size, and the two
    // stacked would inset the panel twice, leaving it visibly smaller than
    // every GuideBrowser page beside it. If kShowAboutTab ever puts this back
    // on the top bar as its own tab, this margin is the first thing to
    // revisit — every other top-level page (buildSetupPage(), buildPostPage())
    // is 0 here too, insetting further in, so this now matches that norm.
    pageLayout->setContentsMargins(0, 0, 0, 0);

    // One panel for the whole page, like the Guide, rather than each label
    // drawing its own background: on a theme with a picture behind it, loose
    // labels read as a stack of unrelated strips. Stretched to the full page
    // like Overview's, so the two widget-based Guide pages and the
    // GuideBrowser ones all present the same size panel.
    auto *card = new QFrame(page);
    card->setObjectName(QStringLiteral("Card"));
    pageLayout->addWidget(card, 1);

    auto *layout = new QVBoxLayout(card);
    layout->setContentsMargins(28, 24, 28, 24);
    layout->setSpacing(8);

    // Logo centered in the page, title/version/credits below it.
    layout->addStretch(1);

    auto *logo = new QLabel(card);
    QPixmap pm(QStringLiteral(":/assets/logo.png"));
    logo->setPixmap(pm.scaled(175, 175, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    logo->setAlignment(Qt::AlignHCenter);
    layout->addWidget(logo, 0, Qt::AlignHCenter);

    auto *title = new QLabel(QStringLiteral("TRAJECTA STUDIO"), card);
    title->setObjectName(QStringLiteral("AboutTitle"));
    title->setAlignment(Qt::AlignHCenter);
    layout->addWidget(title);

    auto *version = new QLabel(tr("Version %1").arg(QApplication::applicationVersion()), card);
    version->setObjectName(QStringLiteral("CardSubtitle"));
    version->setAlignment(Qt::AlignHCenter);
    layout->addWidget(version);

    // No inline colours: a hard-coded grey is invisible on a light theme and
    // is the one thing the palette mapping cannot reach. Everything here
    // inherits AboutBody, so it follows the theme like the rest of the text.
    auto addBody = [card, layout](const QString &html) {
        auto *body = new QLabel(card);
        body->setObjectName(QStringLiteral("AboutBody"));
        body->setAlignment(Qt::AlignHCenter);
        body->setWordWrap(true);
        body->setOpenExternalLinks(true);
        body->setTextFormat(Qt::RichText);
        QPalette pal = body->palette();
        pal.setColor(QPalette::Link, QColor(0x7e, 0xa8, 0xa0));
        body->setPalette(pal);
        body->setText(html);
        layout->addWidget(body);
        return body;
    };

    addBody(tr("<p><b>Trajecta</b> is a software by <b>Stefano Aprà</b></p>"));

    // The project's home, as the mark people actually look for rather than as
    // a line of blue URL. A button and not a rich-text link: an icon beside
    // the word is the whole point, and a QLabel would have to be handed the
    // artwork as an <img> at a fixed pixel size that no theme could recolour.
    m_githubButton = new QPushButton(tr("GitHub"), card);
    m_githubButton->setObjectName(QStringLiteral("GithubButton"));
    m_githubButton->setCursor(Qt::PointingHandCursor);
    m_githubButton->setIconSize(QSize(20, 20));
    m_githubButton->setIcon(makeGithubIcon(ThemeManager::mapped("#e4e7ec"), 20));
    m_githubButton->setToolTip(QString::fromLatin1(kGithubUrl));
    connect(m_githubButton, &QPushButton::clicked, this, [] {
        QDesktopServices::openUrl(QUrl(QString::fromLatin1(kGithubUrl)));
    });

    // The ORCID iD, styled as the same kind of object as the GitHub mark
    // — same object name, so one stylesheet rule dresses both and they read as
    // a pair rather than as two unrelated buttons.
    auto *orcidButton = new QPushButton(tr("ORCID"), card);
    orcidButton->setObjectName(QStringLiteral("GithubButton"));
    orcidButton->setCursor(Qt::PointingHandCursor);
    orcidButton->setIconSize(QSize(20, 20));
    orcidButton->setIcon(makeOrcidIcon(20));
    orcidButton->setToolTip(QString::fromLatin1(kOrcidUrl));
    connect(orcidButton, &QPushButton::clicked, this, [] {
        QDesktopServices::openUrl(QUrl(QString::fromLatin1(kOrcidUrl)));
    });

    // Side by side, and centred as a pair: the stretches belong to the row, so
    // the two stay together in the middle instead of each finding its own
    // centre. Not kept as a member like the GitHub button, which needs finding
    // again on every theme change to be re-dyed — this icon never changes.
    auto *linkRow = new QHBoxLayout;
    linkRow->setSpacing(10);
    linkRow->addStretch(1);
    linkRow->addWidget(m_githubButton);
    linkRow->addWidget(orcidButton);
    linkRow->addStretch(1);
    layout->addLayout(linkRow);

    addBody(tr(
        "<p>If you use Trajecta in your research, please cite:<br/>"
        "<i><a href=\"https://isaw.nyu.edu/people/students/stefano-apra\">Stefano Aprà, Ph.D. candidate — Institute for the Study of the Ancient World at New York University</a></i></p>"));

    layout->addStretch(1);
    return page;
}

// ---------------------------------------------------------------------------
// Environment discovery
// ---------------------------------------------------------------------------

QString MainWindow::engineExePath() const
{
    QSettings settings;
    const QString overridePath = settings.value(QStringLiteral("env/enginePath")).toString();
    if (!overridePath.isEmpty() && QFileInfo::exists(overridePath))
        return overridePath;

    const QString envPath = qEnvironmentVariable("TRAJECTA_ENGINE");
    if (!envPath.isEmpty() && QFileInfo::exists(envPath))
        return envPath;

    const QString local = QCoreApplication::applicationDirPath()
                          + QStringLiteral("/trajecta.exe");
    if (QFileInfo::exists(local))
        return local;

    return QString();
}

bool MainWindow::dirHasGdal(const QString &dir)
{
    if (dir.isEmpty() || !QDir(dir).exists())
        return false;
    return !QDir(dir).entryList({QStringLiteral("gdal*.dll")}, QDir::Files).isEmpty();
}

// Nothing loads GDAL at start-up, and for a long time the Viewer page was the
// only thing that ever did. The DEM preview and the known-route comparison are
// both reachable without going near it, and would otherwise report a missing
// library on a machine where it is installed and working.
bool MainWindow::ensureGdalLoaded()
{
    GdalApi &api = GdalApi::instance();
    if (api.isLoaded())
        return true;
    const GdalEnvironment env = detectGdalEnvironment();
    api.load(gdalDllDirs(), env.projData, env.gdalData);
    return api.isLoaded();
}

MainWindow::GdalEnvironment MainWindow::detectGdalEnvironment() const
{
    GdalEnvironment result;

    // Candidate "roots" that may carry the PROJ/GDAL data folders
    // (an OSGeo4W root looks like <root>/bin, <root>/share/proj,
    //  <root>/apps/gdal/share/gdal).
    QStringList roots;
    auto addRootForBin = [&roots](const QString &binDir) {
        const QString root = QFileInfo(binDir).absolutePath();
        if (!roots.contains(root))
            roots.append(root);
    };

    // 1) DLLs sitting next to the engine take precedence: nothing to inject.
    const QString engine = engineExePath();
    if (!engine.isEmpty() && dirHasGdal(QFileInfo(engine).absolutePath())) {
        result.found = true;
        roots.append(QFileInfo(engine).absolutePath());
    }

    // 2) Folder chosen by the user through "Locate GDAL folder".
    const QString overrideDir =
        QSettings().value(QStringLiteral("env/gdalBinDir")).toString();
    if (!result.found && dirHasGdal(overrideDir)) {
        result.found = true;
        result.binDir = overrideDir;
    }
    if (dirHasGdal(overrideDir))
        addRootForBin(overrideDir);

    // 3) Already reachable through the user's PATH?
    const QString pathValue = QProcessEnvironment::systemEnvironment()
                                  .value(QStringLiteral("PATH"));
    const QStringList pathDirs = pathValue.split(QDir::listSeparator(), Qt::SkipEmptyParts);
    for (const QString &dir : pathDirs) {
        if (dirHasGdal(dir)) {
            result.found = true;  // no PATH injection needed
            addRootForBin(dir);
            break;
        }
    }

    // 4) Standard OSGeo4W locations.
    const QStringList standardBins = {
        QStringLiteral("C:/OSGeo4W/bin"),
        QStringLiteral("C:/OSGeo4W64/bin"),
        QStringLiteral("D:/OSGeo4W/bin"),
        QStringLiteral("D:/OSGeo4W64/bin"),
    };
    for (const QString &dir : standardBins) {
        if (dirHasGdal(dir)) {
            if (!result.found) {
                result.found = true;
                result.binDir = dir;
            }
            addRootForBin(dir);
        }
    }

    // Locate the GDAL data folder in the discovered roots.
    for (const QString &root : roots) {
        if (!result.gdalData.isEmpty())
            break;
        for (const QString &sub : {QStringLiteral("/apps/gdal/share/gdal"),
                                   QStringLiteral("/share/gdal")}) {
            if (QDir(root + sub).exists()) {
                result.gdalData = root + sub;
                break;
            }
        }
    }

    // PROJ data folder. The PROJ library and proj.db are versioned together:
    // the library rejects a database whose layout predates what it expects, and
    // says so on every CRS lookup ("proj.db contains DATABASE.LAYOUT.VERSION.
    // MINOR = 2 whereas a number >= 4 is expected"). An OSGeo4W root upgraded
    // over several years easily ends up with DLLs years ahead of its own
    // proj.db, so every database on the machine is considered and the most
    // recently built one wins: a newer proj.db is accepted by an older PROJ,
    // never the other way round.
    QStringList projCandidates;
    for (const QString &root : roots)
        projCandidates << root + QStringLiteral("/share/proj");
    // Standalone QGIS is not a GDAL root we would load DLLs from, but it ships
    // a current PROJ database of its own.
    for (const QString &programs : {QStringLiteral("C:/Program Files"),
                                    QStringLiteral("C:/Program Files (x86)")}) {
        const QStringList installs =
            QDir(programs).entryList({QStringLiteral("QGIS*")}, QDir::Dirs);
        for (const QString &name : installs)
            projCandidates << programs + QLatin1Char('/') + name
                                  + QStringLiteral("/share/proj");
    }
    QDateTime newestDb;
    for (const QString &dir : projCandidates) {
        const QFileInfo db(dir + QStringLiteral("/proj.db"));
        if (!db.exists())
            continue;
        if (result.projData.isEmpty() || db.lastModified() > newestDb) {
            result.projData = dir;
            newestDb = db.lastModified();
        }
    }

    return result;
}

// Just the environment half of the run parameters: where the engine is and
// where its libraries live. Everything else is filled in by the caller.
TrajectaRunner::Parameters MainWindow::currentEnvironment() const
{
    TrajectaRunner::Parameters env;
    env.exePath = engineExePath();
    const GdalEnvironment g = detectGdalEnvironment();
    env.gdalBinDir = g.binDir;
    env.projDataDir = g.projData;
    env.gdalDataDir = g.gdalData;
    const QString workDir =
        QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(workDir);
    env.workingDir = workDir;
    return env;
}

void MainWindow::updateEnvironmentStatus()
{
    const QString engine = engineExePath();

    // The batch page drives the engine itself, so it gets the same environment
    // the single-run form resolves — refreshed here, which is where every
    // change to it (Locate engine, Locate GDAL folder) already lands.
    if (m_batchPage)
        m_batchPage->setEnvironment(currentEnvironment());
    if (m_postBatchPage)
        m_postBatchPage->setEnvironment(currentEnvironment());
    // A widget stylesheet replaces the rule the app sheet had for #EnvStatus, so
    // the size has to be repeated here or these two labels end up a size apart
    // from the "Locate..." links beside them on the same row.
    auto paintStatus = [](QLabel *label, const char *darkHex) {
        label->setStyleSheet(QStringLiteral("color:%1; font-size:13px;")
                                 .arg(ThemeManager::mapped(darkHex).name()));
    };

    if (engine.isEmpty()) {
        m_engineStatus->setText(tr("⚠ Engine not found"));
        paintStatus(m_engineStatus, "#cf7f7f");
        m_engineStatus->setToolTip(tr("trajecta.exe was not found next to the "
                                      "interface. Use \"Locate engine\" below."));
    } else {
        m_engineStatus->setText(tr("✓ Engine ready"));
        paintStatus(m_engineStatus, "#7fb08a");
        m_engineStatus->setToolTip(QDir::toNativeSeparators(engine));
    }

    const GdalEnvironment gdal = detectGdalEnvironment();
    if (!gdal.found) {
        m_gdalStatus->setText(tr("⚠ GDAL not found"));
        paintStatus(m_gdalStatus, "#cf7f7f");
        m_gdalStatus->setToolTip(tr("GDAL is installed with Trajecta, so this "
                                    "normally means the installation is "
                                    "incomplete: reinstalling is the remedy. To "
                                    "carry on meanwhile, use \"Locate GDAL "
                                    "folder\" below."));
    } else if (gdal.binDir.isEmpty()) {
        m_gdalStatus->setText(tr("✓ GDAL ready"));
        paintStatus(m_gdalStatus, "#7fb08a");
        m_gdalStatus->setToolTip(tr("GDAL libraries are already reachable (bundled "
                                    "with the engine or on the system PATH)."));
    } else {
        m_gdalStatus->setText(tr("✓ GDAL detected"));
        paintStatus(m_gdalStatus, "#7fb08a");
        m_gdalStatus->setToolTip(tr("Using %1 (added to PATH automatically for the engine).")
                                     .arg(QDir::toNativeSeparators(gdal.binDir)));
    }

    configureViewerGdal();
}

// Every folder that may hold gdal*.dll, most specific first. GdalApi::load()
// searches these and nothing else — in particular it does not fall back to the
// process PATH — so this list is the only thing standing between the
// application and "GDAL is not available" on a machine that has it.
//
// Note what this is *not*: GdalEnvironment::binDir is the folder to inject into
// the engine subprocess's PATH, and is deliberately left empty when GDAL is
// already reachable there. Using it alone to find the DLLs is how the known-
// route comparison came to fail on exactly the machines best set up for GDAL.
QStringList MainWindow::gdalDllDirs() const
{
    QStringList dirs;
    const QString engine = engineExePath();
    if (!engine.isEmpty())
        dirs << QFileInfo(engine).absolutePath();
    const QString overrideDir =
        QSettings().value(QStringLiteral("env/gdalBinDir")).toString();
    if (!overrideDir.isEmpty())
        dirs << overrideDir;
    const GdalEnvironment gdal = detectGdalEnvironment();
    if (!gdal.binDir.isEmpty())
        dirs << gdal.binDir;
    dirs << QStringLiteral("C:/OSGeo4W/bin") << QStringLiteral("C:/OSGeo4W64/bin")
         << QStringLiteral("D:/OSGeo4W/bin") << QStringLiteral("D:/OSGeo4W64/bin");
    const QStringList pathDirs =
        QProcessEnvironment::systemEnvironment()
            .value(QStringLiteral("PATH"))
            .split(QDir::listSeparator(), Qt::SkipEmptyParts);
    for (const QString &dir : pathDirs) {
        if (dirHasGdal(dir)) {
            dirs << dir;
            break;
        }
    }
    dirs.removeDuplicates();
    return dirs;
}

void MainWindow::configureViewerGdal()
{
    if (!m_viewer)
        return;
    const GdalEnvironment gdal = detectGdalEnvironment();
    m_viewer->configureGdal(gdalDllDirs(), gdal.projData, gdal.gdalData);
}

void MainWindow::viewerLoadFile(const QString &path)
{
    switchPage(2);   // Viewer
    // What the file is decides where it goes, and GDAL is the one that knows:
    // the extension list this used to keep could only ever name the formats
    // somebody had thought of.
    m_viewer->openAnyFile(path);
}

void MainWindow::locateEngine()
{
    const QString chosen = QFileDialog::getOpenFileName(
        this, tr("Locate trajecta.exe"), QString(),
        QStringLiteral("Trajecta engine (trajecta.exe);;Executables (*.exe)"));
    if (chosen.isEmpty())
        return;
    QSettings().setValue(QStringLiteral("env/enginePath"), chosen);
    updateEnvironmentStatus();
}

void MainWindow::locateGdal()
{
    const QString chosen = QFileDialog::getExistingDirectory(
        this, tr("Select your OSGeo4W\\bin folder (contains gdal*.dll)"));
    if (chosen.isEmpty())
        return;
    if (!dirHasGdal(chosen)) {
        QMessageBox::warning(this, tr("GDAL not found"),
                             tr("No gdal*.dll was found in:\n%1")
                                 .arg(QDir::toNativeSeparators(chosen)));
        return;
    }
    QSettings().setValue(QStringLiteral("env/gdalBinDir"), chosen);
    updateEnvironmentStatus();
}

// ---------------------------------------------------------------------------
// Validation & parameter collection
// ---------------------------------------------------------------------------

QString MainWindow::validationError() const
{
    const bool fete = m_modeFete->isChecked();
    const bool generate = fete && m_pointsSourceCombo->currentIndex() == 1;

    if (!m_demPicker->isSatisfied())
        return tr("Select an existing DEM file (.tif).");
    if (fete && !generate && !m_pointsPicker->isSatisfied())
        return tr("Select an existing sample points file.");
    if (generate && !isValidFileName(m_genNameEdit->text()))
        return tr("The generated points layer needs a name without "
                  "\\ / : * ? \" < > | characters.");
    if (generate && !isAscii(m_genNameEdit->text()))
        return tr("The generated points layer name contains accented or "
                  "non-Latin characters, which the engine cannot handle "
                  "reliably. Use plain ASCII.");
    if (!fete && !m_originPicker->isSatisfied())
        return tr("Select an existing origin file (exactly 1 point).");
    if (!fete && !m_destinationsPicker->isSatisfied())
        return tr("Select an existing destinations file.");
    if (m_outputDirPicker->path().isEmpty())
        return tr("Choose an output folder for the results.");

    if (m_modifiersGroup->isChecked()) {
        if (!m_costVectorPicker->isSatisfied())
            return tr("The cost modifiers vector file does not exist.");
        if (!m_costRasterPicker->isSatisfied())
            return tr("The cost modifiers raster file does not exist.");
    }

    // An empty name tells the engine not to write that output at all. The
    // intermediate rasters may all be skipped; the main result may not, or the
    // run would compute for hours and leave nothing behind.
    if (fete && !isValidFileName(m_densityNameEdit->text()))
        return tr("The density raster is the result of a FETE run: its name "
                  "cannot be empty or contain \\ / : * ? \" < > | characters.");
    if (!fete && m_pathRasterNameEdit->text().trimmed().isEmpty()
        && m_pathLinesNameEdit->text().trimmed().isEmpty())
        return tr("An LCPA run must save at least one of the paths raster and "
                  "the paths shapefile. Leave only one of the two names empty.");

    QStringList optionalNames = {m_slopeNameEdit->text(), m_costNameEdit->text()};
    if (m_modifiersGroup->isChecked() && !m_costVectorPicker->path().isEmpty())
        optionalNames << m_additionalNameEdit->text() << m_totalNameEdit->text();
    if (!fete)
        optionalNames << m_pathRasterNameEdit->text() << m_pathLinesNameEdit->text();
    for (const QString &name : optionalNames) {
        if (!name.trimmed().isEmpty() && !isValidFileName(name))
            return tr("Output file names cannot contain "
                      "\\ / : * ? \" < > | characters.");
    }

    QStringList paths = {m_demPicker->path(), m_outputDirPicker->path()};
    if (fete && !generate)
        paths << m_pointsPicker->path();
    else if (!fete)
        paths << m_originPicker->path() << m_destinationsPicker->path();
    if (m_modifiersGroup->isChecked())
        paths << m_costVectorPicker->path() << m_costRasterPicker->path();
    for (const QString &p : paths) {
        if (!isAscii(p))
            return tr("The path \"%1\" contains accented or non-Latin characters, "
                      "which the engine cannot handle reliably. Please move or "
                      "rename the file.")
                .arg(QDir::toNativeSeparators(p));
    }

    return QString();
}

TrajectaRunner::Parameters MainWindow::collectParameters() const
{
    TrajectaRunner::Parameters p;
    p.mode = m_modeFete->isChecked() ? TrajectaRunner::Mode::Fete
                                     : TrajectaRunner::Mode::Lcpa;
    p.verbose = m_verboseCheck->isChecked();
    p.writeManifest = m_manifestCheck->isChecked();
    p.maxThreads = m_threadsSpin->value();
    p.maxRamMb = m_ramSpin->value();
    // The global switch behind Advanced settings, already gated on the
    // privilege actually being in this session's token.
    p.largePages = largePagesRequested();

    p.demPath = m_demPicker->path();
    p.pointsPath = m_pointsPicker->path();
    p.originPath = m_originPicker->path();
    p.destinationsPath = m_destinationsPicker->path();
    p.outputDir = m_outputDirPicker->path();

    // Point generation is FETE-only and mutually exclusive with importing a
    // file. While it is off, not one of its widgets is read and the runner
    // answers the source question with "import", so the engine follows the
    // exact same path it always has.
    p.generatePoints = p.mode == TrajectaRunner::Mode::Fete
                       && m_pointsSourceCombo->currentIndex() == 1;
    if (p.generatePoints) {
        p.genByTargetCount = m_genDensityCombo->currentIndex() == 1;
        p.genSpacing = m_genSpacingSpin->value();
        p.genTargetCount = m_genTargetSpin->value();
        p.genRandom = m_genArrangementCombo->currentIndex() == 1;
        p.genSeed = m_genSeedSpin->value();
        p.genEdgeBuffer = m_genEdgeSpin->value();
        p.genLayerName = m_genNameEdit->text().trimmed();
        // The engine reads the layer it just wrote, so this is also the path
        // the analysis consumes.
        p.pointsPath = QDir(p.outputDir)
                           .filePath(p.genLayerName + QStringLiteral(".shp"));

        // "Generate points only" already wrote this layer and the user has
        // looked at it. Nothing that shapes it has changed since, so hand the
        // engine that file instead of asking it to write an identical one:
        // the run then provably consumes what was inspected, and the console
        // transcript shows the path it read.
        if (!m_previewedPointsPath.isEmpty()
            && m_previewedPointsKey == generationKey()
            && QFileInfo::exists(m_previewedPointsPath)) {
            p.generatePoints = false;
            p.pointsPath = m_previewedPointsPath;
        }
    }

    const bool modifiers = m_modifiersGroup->isChecked()
                           && (!m_costVectorPicker->path().isEmpty()
                               || !m_costRasterPicker->path().isEmpty());
    p.useCostModifiers = modifiers;
    if (modifiers) {
        p.costVectorPath = m_costVectorPicker->path();
        p.polylineBufferRadius = m_polylineBufferSpin->value();
        p.costRasterPath = m_costRasterPicker->path();
        p.barrierThreshold = m_barrierCheck->isChecked() ? m_barrierSpin->value() : 0.0;
    }

    p.neighbours = selectedNeighbours();
    p.costFunction = m_costFunctionCombo->currentData().toInt();
    p.slopeCutoffEnabled = m_slopeCapCheck->isChecked();
    p.maxSlopeUpDeg = m_slopeCapUp->value();
    p.maxSlopeDownDeg = m_slopeCapDown->value();
    p.smoothingBufferRadius = m_smoothingSpin->value();

    p.slopeName = m_slopeNameEdit->text().trimmed();
    p.costName = m_costNameEdit->text().trimmed();
    p.additionalCostName = m_additionalNameEdit->text().trimmed();
    p.totalCostName = m_totalNameEdit->text().trimmed();
    p.densityName = m_densityNameEdit->text().trimmed();
    p.pathRasterName = m_pathRasterNameEdit->text().trimmed();
    p.pathLinesName = m_pathLinesNameEdit->text().trimmed();
    p.costCorridor = m_corridorCheck->isChecked();
    p.corridorWidthPercent = m_corridorWidthSpin->value();
    p.corridorName = m_corridorNameEdit->text().trimmed();

    p.exePath = engineExePath();
    const GdalEnvironment gdal = detectGdalEnvironment();
    p.gdalBinDir = gdal.binDir;
    p.projDataDir = gdal.projData;
    p.gdalDataDir = gdal.gdalData;

    // trajecta writes small fete_config.txt / lcpa_config.txt files into its
    // working directory: point it to a writable per-user location.
    const QString workDir =
        QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(workDir);
    p.workingDir = workDir;

    return p;
}

// ---------------------------------------------------------------------------
// Run lifecycle
// ---------------------------------------------------------------------------

void MainWindow::startRun()
{
    if (refuseIfEngineBusy())
        return;

    const QString error = validationError();
    if (!error.isEmpty()) {
        QMessageBox::warning(this, tr("Check the configuration"), error);
        return;
    }

    // Offer to create the output folder if needed.
    const QString outDir = m_outputDirPicker->path();
    if (!QDir(outDir).exists()) {
        const auto answer = QMessageBox::question(
            this, tr("Create output folder"),
            tr("The output folder does not exist:\n%1\n\nCreate it now?")
                .arg(QDir::toNativeSeparators(outDir)));
        if (answer != QMessageBox::Yes)
            return;
        if (!QDir().mkpath(outDir)) {
            QMessageBox::critical(this, tr("Error"),
                                  tr("Could not create the output folder."));
            return;
        }
    }

    TrajectaRunner::Parameters params = collectParameters();

    if (params.exePath.isEmpty()) {
        QMessageBox::critical(
            this, tr("Engine not found"),
            tr("trajecta.exe was not found.\n\nUse \"Locate engine...\" in the "
               "sidebar to select it manually."));
        return;
    }

    saveSettings();
    m_lastOutputDir = outDir;
    beginRun(params);
}

// "Generate points only": writes the layer through the engine (mode 4) and
// opens it in the Viewer. Deliberately not a GUI-side reimplementation — the
// engine runs the very same generate_sample_points() a FETE run would, so the
// file inspected here is the file an analysis produces from these parameters.
void MainWindow::startPointsRun()
{
    if (engineBusy()) {
        refuseIfEngineBusy();
        // …and show the run it is talking about, which is half the answer.
        if (m_activeUi == &m_postUi)
            switchPage(1);
        else
            revealRunPanel();
        return;
    }

    if (!m_demPicker->isSatisfied()) {
        QMessageBox::warning(this, tr("Check the configuration"),
                             tr("Select an existing DEM file (.tif)."));
        return;
    }
    const QString outDir = m_outputDirPicker->path();
    if (outDir.isEmpty()) {
        QMessageBox::warning(this, tr("Check the configuration"),
                             tr("Choose an output folder for the point layer."));
        return;
    }
    if (!isValidFileName(m_genNameEdit->text()) || !isAscii(m_genNameEdit->text())) {
        QMessageBox::warning(this, tr("Check the configuration"),
                             tr("The generated points layer needs a plain ASCII "
                                "name without \\ / : * ? \" < > | characters."));
        return;
    }
    for (const QString &p : {m_demPicker->path(), outDir}) {
        if (!isAscii(p)) {
            QMessageBox::warning(this, tr("Check the configuration"),
                                 tr("The path \"%1\" contains accented or non-Latin "
                                    "characters, which the engine cannot handle "
                                    "reliably. Please move or rename the file.")
                                     .arg(QDir::toNativeSeparators(p)));
            return;
        }
    }
    if (!QDir(outDir).exists()) {
        const auto answer = QMessageBox::question(
            this, tr("Create output folder"),
            tr("The output folder does not exist:\n%1\n\nCreate it now?")
                .arg(QDir::toNativeSeparators(outDir)));
        if (answer != QMessageBox::Yes)
            return;
        if (!QDir().mkpath(outDir)) {
            QMessageBox::critical(this, tr("Error"),
                                  tr("Could not create the output folder."));
            return;
        }
    }

    TrajectaRunner::Parameters params = collectParameters();
    params.mode = TrajectaRunner::Mode::Points;
    // collectParameters() may have swapped in an already-generated layer;
    // this button always writes a fresh one.
    params.generatePoints = true;
    params.pointsPath =
        QDir(outDir).filePath(params.genLayerName + QStringLiteral(".shp"));

    if (params.exePath.isEmpty()) {
        QMessageBox::critical(
            this, tr("Engine not found"),
            tr("trajecta.exe was not found.\n\nUse \"Locate engine...\" in the "
               "sidebar to select it manually."));
        return;
    }

    saveSettings();
    m_lastOutputDir = outDir;
    beginRun(params);
}

void MainWindow::startInterpRun()
{
    if (engineBusy()) {
        refuseIfEngineBusy();
        // …and show the run it is talking about, which is half the answer.
        if (m_activeUi == &m_postUi)
            switchPage(1);
        else
            revealRunPanel();
        return;
    }

    const QString input = m_interpInputPicker->path();
    if (input.isEmpty() || !m_interpInputPicker->isSatisfied()) {
        QMessageBox::warning(this, tr("Check the configuration"),
                             tr("Select an existing density raster (.tif) to interpolate."));
        return;
    }
    const QString outDir = m_interpOutputDirPicker->path();
    if (outDir.isEmpty()) {
        QMessageBox::warning(this, tr("Check the configuration"),
                             tr("Choose an output folder for the interpolated raster."));
        return;
    }
    if (!isValidFileName(m_interpNameEdit->text())) {
        QMessageBox::warning(this, tr("Check the configuration"),
                             tr("Output file names cannot be empty or contain "
                                "\\ / : * ? \" < > | characters."));
        return;
    }
    for (const QString &p : {input, outDir}) {
        if (!isAscii(p)) {
            QMessageBox::warning(this, tr("Check the configuration"),
                                 tr("The path \"%1\" contains accented or non-Latin "
                                    "characters, which the engine cannot handle "
                                    "reliably. Please move or rename the file.")
                                     .arg(QDir::toNativeSeparators(p)));
            return;
        }
    }
    if (!QDir(outDir).exists()) {
        const auto answer = QMessageBox::question(
            this, tr("Create output folder"),
            tr("The output folder does not exist:\n%1\n\nCreate it now?")
                .arg(QDir::toNativeSeparators(outDir)));
        if (answer != QMessageBox::Yes)
            return;
        if (!QDir().mkpath(outDir)) {
            QMessageBox::critical(this, tr("Error"),
                                  tr("Could not create the output folder."));
            return;
        }
    }

    TrajectaRunner::Parameters params = collectParameters();
    params.mode = TrajectaRunner::Mode::Interp;
    params.outputDir = outDir;
    params.interpInputRaster = input;
    params.interpThreshold = m_interpThresholdSpin->value();
    params.interpSampleSpacing = m_interpSpacingSpin->value();
    params.interpPreservePeaks = m_interpPeaksCheck->isChecked()
                                 && m_interpSpacingSpin->value() > 1;
    params.interpMaxRadius = m_interpRadiusSpin->value();
    params.interpOutputName = m_interpNameEdit->text().trimmed();

    // NNI has its own hardware-resources card, so the Processing page's
    // choices (read into `params` above by collectParameters()) are replaced
    // rather than reused — the two tools are configured independently.
    // Verbose is forced off: NNI has no card for it, and inheriting whatever
    // the Processing page's checkbox happens to hold would be a silent,
    // unrelated side effect of a choice made on a different page.
    if (m_postThreadsSpin)
        params.maxThreads = m_postThreadsSpin->value();
    if (m_postRamSpin)
        params.maxRamMb = m_postRamSpin->value();
    params.largePages = largePagesRequested();
    if (m_postManifestCheck)
        params.writeManifest = m_postManifestCheck->isChecked();
    params.verbose = false;

    if (params.exePath.isEmpty()) {
        QMessageBox::critical(
            this, tr("Engine not found"),
            tr("trajecta.exe was not found.\n\nUse \"Locate engine...\" in the "
               "sidebar to select it manually."));
        return;
    }

    saveSettings();
    m_lastOutputDir = outDir;
    beginRun(params);
}

void MainWindow::beginRun(const TrajectaRunner::Parameters &paramsIn)
{
    TrajectaRunner::Parameters params = paramsIn;

    // Automatic saving: only FETE has a propagation phase long enough to be
    // worth resuming, and only that phase is what a checkpoint holds.
    const Checkpoint::Settings cp = Checkpoint::settings();
    const QString cpDir = Checkpoint::activeDir();
    const bool wantCheckpoints = cp.enabled && !cpDir.isEmpty()
                                 && params.mode == TrajectaRunner::Mode::Fete;
    if (wantCheckpoints) {
        // The engine clears the folder as it starts, so anything saved there
        // from an analysis the user stopped and meant to come back to is about
        // to go. Said out loud rather than done quietly: that state can be
        // days of computation.
        if (!confirmOverwritingSavedProcess(cpDir))
            return;
        params.checkpointEnabled = true;
        params.checkpointMinutes = cp.minutes;
        params.checkpointDir = cpDir;
        // The session file records what the checkpoint cannot: which analysis
        // this is. Its presence at the next start-up is the crash signal.
        Checkpoint::Session session;
        session.batch = false;
        session.params = Checkpoint::toJson(params);
        session.label = tr("FETE — %1").arg(QFileInfo(params.demPath).fileName());
        Checkpoint::writeSession(cpDir, session);
        m_lastRunCheckpointDir = cpDir;
    } else {
        params.checkpointEnabled = false;
        params.checkpointDir.clear();
        m_lastRunCheckpointDir.clear();
    }

    // One engine at a time: the "Run batch" button goes, the page does not.
    // Disabling the whole page also greyed out every field and every button on
    // it, which stopped the user preparing the next batch while this run went —
    // and preparing it is precisely what there is time for meanwhile.
    if (m_batchPage)
        m_batchPage->setStartAllowed(false);
    if (m_postBatchPage)
        m_postBatchPage->setStartAllowed(false);

    m_lastRunMode = params.mode;
    m_lastDensityPath = (params.mode == TrajectaRunner::Mode::Fete)
        ? QDir(params.outputDir).filePath(params.densityName + QStringLiteral(".tif"))
        : QString();

    // Snapshot the output files this run should produce; those that exist
    // when it succeeds get registered with the Viewer page. Ordered so the
    // mode's main product comes last (it becomes the auto-selected layer).
    m_pendingOutputs.clear();
    m_pendingVector.clear();
    m_pendingPoints.clear();
    const QDir outDir(params.outputDir);
    auto addTif = [&](const QString &label, const QString &name) {
        if (!name.isEmpty())
            m_pendingOutputs.append(
                {label, outDir.filePath(name + QStringLiteral(".tif"))});
    };
    if (params.mode == TrajectaRunner::Mode::Interp) {
        addTif(tr("NNI surface"), params.interpOutputName);
    } else if (params.mode == TrajectaRunner::Mode::Points) {
        // Nothing but the point layer is produced. The DEM comes along as the
        // raster to draw it over — an overlay needs a layer underneath it.
        m_pendingOutputs.append({tr("Input DEM"), params.demPath});
        m_pendingPoints = params.pointsPath;
    } else {
        m_pendingOutputs.append({tr("Input DEM"), params.demPath});
        addTif(tr("Slope"), params.slopeName);
        addTif(tr("Cost surface"), params.costName);
        if (params.useCostModifiers) {
            addTif(tr("Additional cost"), params.additionalCostName);
            addTif(tr("Total cost"), params.totalCostName);
        }
        if (params.mode == TrajectaRunner::Mode::Fete) {
            addTif(tr("FETE density"), params.densityName);
            // Either the engine writes it during this run, or the run consumes
            // the layer "Generate points only" already wrote.
            if (params.generatePoints || params.pointsPath == m_previewedPointsPath)
                m_pendingPoints = params.pointsPath;
        } else {
            addTif(tr("LCPA paths raster"), params.pathRasterName);
            m_pendingVector =
                outDir.filePath(params.pathLinesName + QStringLiteral(".shp"));
        }
    }

    // NNI runs live on the Post-processing panel, FETE/LCPA on Run & results.
    const bool interp = params.mode == TrajectaRunner::Mode::Interp;
    m_activeUi = interp ? &m_postUi : &m_runUi;
    RunUi &ui = *m_activeUi;

    ui.console->clearAll();
    ui.console->appendMarker(tr("— Launching %1")
                                 .arg(QDir::toNativeSeparators(params.exePath)),
                             QColor(0x6e, 0xa8, 0xfe));
    if (!params.gdalBinDir.isEmpty()) {
        ui.console->appendMarker(tr("— GDAL added to PATH from %1")
                                     .arg(QDir::toNativeSeparators(params.gdalBinDir)),
                                 QColor(0x6e, 0xa8, 0xfe));
    }
    if (!params.projDataDir.isEmpty()) {
        ui.console->appendMarker(tr("— PROJ database: %1")
                                     .arg(QDir::toNativeSeparators(params.projDataDir)),
                                 QColor(0x6e, 0xa8, 0xfe));
    }

    // The engine deletes and recreates its output rasters; Windows refuses to
    // delete a file the Viewer still holds open from a previous run.
    if (m_viewer)
        m_viewer->releaseFiles();
    // Busy (indeterminate) until the engine's first progress bar arrives:
    // the long preparation phases (DEM read, slope, cost surface) emit no
    // percentage, and a bar frozen at 0% reads as a hang.
    ui.progress->setRange(0, 0);
    ui.summaryCard->setVisible(false);
    ui.openFolderButton->setEnabled(false);
    ui.pauseButton->setEnabled(true);
    ui.pauseButton->setText(tr("Pause"));
    TrajectaUi::setPauseMark(ui.pauseButton, true);
    ui.cancelButton->setEnabled(true);
    m_runButton->setEnabled(false);
    m_runInterpButton->setEnabled(false);
    m_genPointsButton->setEnabled(false);
    ui.chip->setText(tr("RUNNING"));
    ui.chip->setProperty("state", QStringLiteral("running"));
    ui.chip->style()->unpolish(ui.chip);
    ui.chip->style()->polish(ui.chip);
    ui.phase->setText(tr("Starting..."));
    m_elapsed.start();
    m_pausedMs = 0;
    m_pauseClock.invalidate();
    ui.elapsed->setText(QStringLiteral("0:00:00"));
    m_elapsedTimer->start();

    // What the status bar will call this run. Decided here because the runner
    // does not keep its parameters, and by the time anyone reads the ticker the
    // page that started the run may be three pages away.
    switch (params.mode) {
    case TrajectaRunner::Mode::Fete:   m_runKind = tr("FETE"); break;
    case TrajectaRunner::Mode::Lcpa:   m_runKind = tr("LCPA"); break;
    case TrajectaRunner::Mode::Interp: m_runKind = tr("NNI"); break;
    case TrajectaRunner::Mode::Points: m_runKind = tr("Sample points"); break;
    }
    m_runHardware = tr("Hardware: %1 threads · %2 MB")
                        .arg(params.maxThreads)
                        .arg(params.maxRamMb);
    m_runPercent = -1.0;
    refreshRunTicker();

    if (interp)
        switchPage(1);          // Post-processing has its own page still
    else
        revealRunPanel();       // …the FETE/LCPA one is the tail of this page
    m_runner->start(params);
}

bool MainWindow::engineBusy() const
{
    return m_runner->isRunning() || (m_batchPage && m_batchPage->isRunning())
          || (m_postBatchPage && m_postBatchPage->isRunning());
}

bool MainWindow::refuseIfEngineBusy()
{
    if (!engineBusy())
        return false;
    // A statement, not a question: there is nothing here for the user to
    // decide. What it has to do is say *why* the button did nothing, which is
    // the part a disabled button cannot say — and, on a machine with the
    // Post-processing page open over a batch, the reason is not obvious.
    TrajectaUi::notify(
        this, tr("Another run is in progress"),
        tr("A computation is already running.\n\n"
           "Wait for it to finish, or stop it, before starting a new one — "
           "Trajecta runs one analysis at a time so that the whole machine is "
           "available to it."));
    return true;
}

// The walkthrough's demonstration of the ticker. `on` fills it with a plausible
// run; `off` hands it back to refreshRunTicker(), which shows whatever is
// really happening — usually nothing.
//
// A real run always wins: someone who starts the tour while a batch is going
// must keep seeing the batch, not an invented FETE.
void MainWindow::setTourTicker(bool on)
{
    if (!m_ticker)
        return;
    if (!on || engineBusy()) {
        refreshRunTicker();
        return;
    }
    RunTicker::State s;
    s.active = true;
    s.kind = tr("FETE");
    s.percent = 63.4;
    // No chunk line: this demonstration is a single run, and a batch's line
    // would be an invention on top of an invention.
    s.hardware = tr("Hardware: %1 threads · %2 MB")
                     .arg(m_threadsSpin ? m_threadsSpin->value() : 8)
                     .arg(m_ramSpin ? m_ramSpin->value() : 8192);
    s.remaining = tr("Time left: about 40 min");
    m_ticker->setState(s);
}

void MainWindow::refreshRunTicker()
{
    if (!m_ticker)
        return;

    RunTicker::State s;
    // The batch is asked first: it owns the engine while it runs, and its own
    // page knows things about it — which chunk, which row — that this side
    // could only guess at.
    if (m_batchPage && m_batchPage->isRunning()) {
        s = m_batchPage->tickerState();
    } else if (m_postBatchPage && m_postBatchPage->isRunning()) {
        s = m_postBatchPage->tickerState();
    } else if (m_runner->isRunning()) {
        s.active = true;
        s.paused = m_runner->isPaused();
        s.kind = m_runKind;
        s.percent = m_runPercent;
        s.hardware = m_runHardware;
        s.remaining = TrajectaUi::timeLeftText(
            m_elapsed.isValid() ? m_elapsed.elapsed() - m_pausedMs : 0, m_runPercent);
    }
    m_ticker->setState(s);
}

void MainWindow::onRunFinished(TrajectaRunner::Outcome outcome, const QString &report)
{
    RunUi &ui = *m_activeUi;

    m_elapsedTimer->stop();
    ui.elapsed->setText(formatElapsed(m_elapsed.elapsed() - m_pausedMs));
    ui.pauseButton->setEnabled(false);
    ui.pauseButton->setText(tr("Pause"));
    TrajectaUi::setPauseMark(ui.pauseButton, true);
    ui.cancelButton->setEnabled(false);
    m_runButton->setEnabled(true);
    m_runInterpButton->setEnabled(true);
    m_genPointsButton->setEnabled(true);
    if (m_batchPage)
        m_batchPage->setStartAllowed(true);
    if (m_postBatchPage)
        m_postBatchPage->setStartAllowed(true);
    // Nothing owns the engine any more, so the ticker empties itself.
    refreshRunTicker();

    // Only this run's own state is touched. A checkpoint sitting in the folder
    // because the user imported a saved process, or because a batch put it
    // there, is none of this run's business.
    if (m_lastRunCheckpointDir.isEmpty()) {
        // nothing to do
    } else if (outcome == TrajectaRunner::Outcome::Cancelled) {
        // A run the user stopped keeps its checkpoint: stopping in the evening
        // and carrying on in the morning is the whole point of the feature.
        Checkpoint::Session session = Checkpoint::readSession(m_lastRunCheckpointDir);
        if (session.valid && !session.batch) {
            session.deliberate = true;
            Checkpoint::writeSession(m_lastRunCheckpointDir, session);
        }
    } else {
        // Finished, or refused by the engine: nothing worth keeping.
        Checkpoint::discard(m_lastRunCheckpointDir);
        m_lastRunCheckpointDir.clear();
    }

    const bool success = outcome == TrajectaRunner::Outcome::Success;
    const bool pointsOnly = m_lastRunMode == TrajectaRunner::Mode::Points;
    // Leave busy mode in every case; a failed run parks the bar at zero.
    if (ui.progress->maximum() == 0)
        ui.progress->setRange(0, 1000);
    if (!success)
        ui.progress->setValue(0);

    if (success) {
        ui.chip->setText(tr("COMPLETED"));
        ui.chip->setProperty("state", QStringLiteral("success"));
        ui.phase->setText(pointsOnly ? tr("Sample points generated.")
                                     : tr("Analysis completed successfully."));
        ui.summaryTitle->setText(pointsOnly ? tr("✓ Sample points generated")
                                            : tr("✓ Analysis completed"));
        ui.openFolderButton->setEnabled(true);
        ui.console->appendMarker(pointsOnly
                                     ? tr("— Sample points generated.")
                                     : tr("— Analysis completed successfully."),
                                 QColor(0x5f, 0xd0, 0x68));
    } else if (outcome == TrajectaRunner::Outcome::Cancelled) {
        ui.chip->setText(tr("CANCELLED"));
        ui.chip->setProperty("state", QStringLiteral("idle"));
        ui.phase->setText(tr("The analysis was cancelled."));
        ui.summaryTitle->setText(tr("Analysis cancelled"));
        ui.console->appendMarker(tr("— Analysis cancelled by the user."),
                                 QColor(0xff, 0xd1, 0x66));
    } else {
        ui.chip->setText(tr("FAILED"));
        ui.chip->setProperty("state", QStringLiteral("failed"));
        ui.phase->setText(tr("The analysis did not complete."));
        ui.summaryTitle->setText(tr("✖ Analysis failed"));
        ui.console->appendMarker(tr("— Analysis failed."),
                                 QColor(0xff, 0x6b, 0x6b));
    }
    ui.chip->style()->unpolish(ui.chip);
    ui.chip->style()->polish(ui.chip);

    ui.summaryBody->setText(
        QStringLiteral("<pre style='white-space:pre-wrap; font-family:Consolas, "
                       "monospace; margin:0;'>%1</pre>")
            .arg(report.toHtmlEscaped()));
    ui.summaryCard->setVisible(true);

    // A fresh FETE density raster is the natural input of the NNI
    // post-processing page: prefill it.
    if (success && m_lastRunMode == TrajectaRunner::Mode::Fete
        && !m_lastDensityPath.isEmpty() && QFileInfo::exists(m_lastDensityPath)) {
        m_interpInputPicker->setPath(m_lastDensityPath);
        if (m_interpOutputDirPicker->path().isEmpty())
            m_interpOutputDirPicker->setPath(m_lastOutputDir);
        const QString cur = m_interpNameEdit->text().trimmed();
        if (cur.isEmpty() || cur.endsWith(QLatin1String("_NNI")))
            m_interpNameEdit->setText(
                QFileInfo(m_lastDensityPath).completeBaseName() + QStringLiteral("_NNI"));
    }

    // Feed the Viewer page. The last existing raster in m_pendingOutputs is
    // the mode's main product and becomes the auto-selected layer.
    if (success) {
        QString primaryLabel, primaryPath;
        for (const PendingOutput &out : std::as_const(m_pendingOutputs)) {
            if (QFileInfo::exists(out.path)) {
                m_viewer->registerRaster(out.label, out.path, false);
                primaryLabel = out.label;
                primaryPath = out.path;
            }
        }
        if (!primaryPath.isEmpty())
            m_viewer->registerRaster(primaryLabel, primaryPath, true);
        if (!m_pendingVector.isEmpty() && QFileInfo::exists(m_pendingVector))
            m_viewer->registerVectorOverlay(tr("LCPA paths"), m_pendingVector);
        if (!m_pendingPoints.isEmpty() && QFileInfo::exists(m_pendingPoints)) {
            m_viewer->registerVectorOverlay(tr("Sample points"), m_pendingPoints);
            if (pointsOnly) {
                // Remember what was written and under which parameters, so the
                // analysis can hand the engine this very file.
                m_previewedPointsPath = m_pendingPoints;
                m_previewedPointsKey = generationKey();
            }
        }
    }
    if (pointsOnly)
        updateGeneratedPointsStatus();
    m_pendingOutputs.clear();
    m_pendingVector.clear();
    m_pendingPoints.clear();

    // The whole point of a points-only run is to look at the result: land on
    // the Viewer instead of leaving the user on a console that has nothing
    // more to say.
    if (success && pointsOnly)
        switchPage(2);   // Viewer
}

void MainWindow::onPauseStateChanged(bool paused)
{
    RunUi &ui = *m_activeUi;

    if (paused) {
        m_pauseClock.start();
        m_elapsedTimer->stop();
        ui.pauseButton->setText(tr("▶ Resume"));
        // The bars go with the word: what this button now offers is play.
        TrajectaUi::setPauseMark(ui.pauseButton, false);
        ui.chip->setText(tr("PAUSED"));
        ui.chip->setProperty("state", QStringLiteral("paused"));
        ui.phase->setText(tr("Paused — CPU released, memory still allocated."));
        ui.console->appendMarker(tr("— Analysis paused."), QColor(0x6e, 0xa8, 0xfe));
    } else {
        if (m_pauseClock.isValid()) {
            m_pausedMs += m_pauseClock.elapsed();
            m_pauseClock.invalidate();
        }
        ui.pauseButton->setText(tr("Pause"));
        TrajectaUi::setPauseMark(ui.pauseButton, true);
        if (m_runner->isRunning()) {
            m_elapsedTimer->start();
            ui.chip->setText(tr("RUNNING"));
            ui.chip->setProperty("state", QStringLiteral("running"));
            ui.phase->setText(tr("Resumed."));
            ui.console->appendMarker(tr("— Analysis resumed."), QColor(0x6e, 0xa8, 0xfe));
        }
    }
    ui.chip->style()->unpolish(ui.chip);
    ui.chip->style()->polish(ui.chip);
    refreshRunTicker();
}

void MainWindow::openOutputFolder()
{
    if (!m_lastOutputDir.isEmpty())
        QDesktopServices::openUrl(QUrl::fromLocalFile(m_lastOutputDir));
}

void MainWindow::triggerRun()
{
    startRun();
}

void MainWindow::triggerInterpRun()
{
    startInterpRun();
}

void MainWindow::triggerPointsRun(bool thenAnalysis)
{
    if (thenAnalysis) {
        auto *link = new QMetaObject::Connection;
        *link = connect(m_runner, &TrajectaRunner::finished, this,
                        [this, link](TrajectaRunner::Outcome outcome, const QString &) {
                            disconnect(*link);
                            delete link;
                            if (outcome == TrajectaRunner::Outcome::Success)
                                QTimer::singleShot(0, this, &MainWindow::startRun);
                        });
    }
    startPointsRun();
}

void MainWindow::showPage(int index)
{
    switchPage(qBound(0, index, m_pages->count() - 1));
}

void MainWindow::scrollSetupToEnd(double fraction)
{
    // Scrolls whichever long page is showing, so --page X --scroll-end reaches
    // into the setup form or the guide alike. The guide is found by what it
    // is (the page holding a QTextBrowser), not by a hard-coded tab index.
    QAbstractScrollArea *area = m_setupScroll;
    if (m_pages && m_pages->currentWidget()) {
        // The guide is a stack of documents now, so it has to be asked for the
        // one on screen: findChild() would hand back whichever was built first,
        // and scrolling a page nobody is looking at is a silent no-op that
        // reads as a broken switch.
        if (m_guidePages && m_pages->currentWidget() == m_pages->widget(kGuidePageIndex)) {
            if (auto *browser = qobject_cast<QTextBrowser *>(m_guidePages->currentWidget()))
                area = browser;
        } else if (auto *browser = m_pages->currentWidget()->findChild<QTextBrowser *>()) {
            area = browser;
        }
    }
    if (area) {
        QScrollBar *bar = area->verticalScrollBar();
        bar->setValue(qRound(qBound(0.0, fraction, 1.0) * bar->maximum()));
    }
}

void MainWindow::openComboForTest(int index)
{
    QWidget *page = m_pages ? m_pages->currentWidget() : nullptr;
    if (!page)
        return;
    const QList<QComboBox *> combos = page->findChildren<QComboBox *>();
    if (index >= 0 && index < combos.size() && combos.at(index)->isVisible())
        combos.at(index)->showPopup();

}

void MainWindow::dropOnViewerForTest(const QStringList &paths)
{
    if (!m_viewer || paths.isEmpty())
        return;
    showPage(2);

    QMimeData *mime = new QMimeData;
    QList<QUrl> urls;
    for (const QString &p : paths)
        urls << QUrl::fromLocalFile(p);
    mime->setUrls(urls);

    // Sent to the *window*, not to a widget.
    //
    // This matters more than it looks. Windows hands the drag to the QWindow,
    // and it is QWidgetWindow — not the widget — that decides which widget will
    // receive it, remembers that choice for the whole gesture, and forgets it
    // again on the drop. Sending the two events straight to a widget skipped
    // that machinery entirely, which is why this test passed for months while
    // the real thing was broken: everything the test exercised was working.
    QApplication::processEvents();          // the page has to be laid out first
    QWindow *win = windowHandle();
    if (!win) {
        qWarning("drop test: the window has no handle");
        delete mime;
        return;
    }
    const QPoint global = m_viewer->mapToGlobal(
        QPoint(m_viewer->width() / 2, m_viewer->height() / 2));
    const QPointF inWindow = mapFromGlobal(global);

    QDragEnterEvent enter(inWindow.toPoint(), Qt::CopyAction, mime,
                          Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(win, &enter);
    qInfo("drop test: DragEnter accepted=%d", int(enter.isAccepted()));

    QDragMoveEvent move(inWindow.toPoint(), Qt::CopyAction, mime,
                        Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(win, &move);

    QDropEvent drop(inWindow, Qt::CopyAction, mime, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(win, &drop);
    qInfo("drop test: Drop accepted=%d", int(drop.isAccepted()));
    delete mime;
}

void MainWindow::setProgressForTest(int percent, bool paused)
{
    if (!m_runUi.progress)
        return;
    m_runUi.progress->setValue(
        qBound(0, percent, 100) * m_runUi.progress->maximum() / 100);
    revealRunPanel();

    // The status bar's ticker too, driven directly: it normally follows a real
    // engine, and photographing it would otherwise mean starting one.
    if (m_ticker) {
        RunTicker::State s;
        s.active = true;
        s.paused = paused;
        s.kind = tr("FETE");
        s.percent = qBound(0, percent, 100);
        s.chunks = tr("Chunk 1 of 1 · row 1 of 1");
        s.hardware = tr("Hardware: %1 threads · %2 MB")
                         .arg(m_threadsSpin->value()).arg(m_ramSpin->value());
        s.remaining = TrajectaUi::timeLeftText(600000, s.percent);
        m_ticker->setState(s);
    }
}

// Files dropped on the window, rather than on the Viewer or on a path field.
// Both of those accept drops themselves and are found first; what arrives here
// is a drop on the top bar, the status bar, or the empty part of a page.
void MainWindow::dragEnterEvent(QDragEnterEvent *event)
{
    if (!event->mimeData() || !event->mimeData()->hasUrls())
        return;
    for (const QUrl &url : event->mimeData()->urls()) {
        if (!url.toLocalFile().isEmpty()) {
            event->acceptProposedAction();
            return;
        }
    }
}

void MainWindow::dropEvent(QDropEvent *event)
{
    if (!m_viewer || !event->mimeData())
        return;
    QStringList files;
    for (const QUrl &url : event->mimeData()->urls()) {
        const QString local = url.toLocalFile();
        if (!local.isEmpty())
            files << local;
    }
    if (files.isEmpty())
        return;
    event->acceptProposedAction();
    // Straight to the Viewer, and shown: a file dropped on the window can only
    // sensibly mean "look at this", and leaving it loaded on a page the user
    // cannot see would look like nothing had happened.
    showPage(2);
    QStringList refused;
    for (const QString &path : files) {
        QString error;
        if (!m_viewer->openAnyFile(path, &error))
            refused << tr("• %1 — %2").arg(QFileInfo(path).fileName(), error.simplified());
    }
    if (!refused.isEmpty()) {
        TrajectaUi::notify(this, tr("Some files could not be opened"),
                           refused.join(QLatin1Char('\n')), QString(), 60);
    }
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    // Quitting mid-run throws the run away: the engine writes its rasters only
    // once the computation is over, so a run that has been going for hours
    // leaves nothing behind. Hence a confirmation, through the themed dialog
    // used everywhere else rather than QMessageBox — see confirmdialog.h.
    // A paused run counts as running; cancel() thaws the process before killing
    // it, so no special case is needed here.
    const bool batchRunning = (m_batchPage && m_batchPage->isRunning())
                              || (m_postBatchPage && m_postBatchPage->isRunning());
    if (m_runner->isRunning() || batchRunning) {
        const bool quit = TrajectaUi::confirm(
            this, batchRunning ? tr("Batch running") : tr("Analysis running"),
            batchRunning
                ? tr("A batch is still running.\n\n"
                     "Closing Trajecta Studio stops it. The rows already finished "
                     "keep their results; the row in progress is lost. Close the "
                     "application anyway?")
                : tr("An analysis is still running.\n\n"
                     "Closing Trajecta Studio stops it, and everything computed so "
                     "far is lost. Close the application anyway?"));
        if (!quit) {
            event->ignore();
            return;
        }
        if (m_batchPage && m_batchPage->isRunning())
            m_batchPage->cancelForShutdown();
        if (m_postBatchPage && m_postBatchPage->isRunning())
            m_postBatchPage->cancelForShutdown();
        if (m_runner->isRunning())
            m_runner->cancel();
        // The finished() signal that normally does this arrives through the
        // event loop, and the application is about to stop running one. So the
        // session is stamped here instead: this shutdown was asked for, and
        // the next start-up should say "interrupted", not "unexpected".
        markSessionDeliberate();
    }
    // Before the settings are written, not after: saveSettings() records the
    // mode and the page as they are at this instant, and during the tour those
    // belong to the tour, not to the user. Closing the window on the batch
    // screen would otherwise leave Batch as the mode that comes back the next
    // time Trajecta starts.
    //
    // Restored by hand rather than by closing the tour and letting it happen:
    // closeTour() only emits tourFinished when its fade-out animation ends,
    // which needs turns of an event loop this application is about to stop
    // running. The call also takes the tour's example layers back out of the
    // Viewer, which is why it comes before anything else here.
    if (m_tour && m_tour->isActive()) {
        restoreAfterWalkthrough();
        m_tour->closeTour();
    }
    saveSettings();
    event->accept();
}

// "Custom…" is the only entry whose data is 0, so it is also the test for
// whether the spin box has anything to say.
void MainWindow::refreshNeighboursCustom()
{
    if (!m_neighboursCombo || !m_neighboursCustom)
        return;
    m_neighboursCustom->setVisible(m_neighboursCombo->currentData().toInt() == 0);
}

// The chip only changes appearance if the style is re-applied to it: Qt caches
// the polished result and a property change alone does not invalidate it.
void MainWindow::setCmpState(const QString &text, const QString &state)
{
    if (!m_cmpChip)
        return;
    m_cmpChip->setText(text);
    m_cmpChip->setProperty("state", state);
    m_cmpChip->style()->unpolish(m_cmpChip);
    m_cmpChip->style()->polish(m_cmpChip);
}

// Runs the geometric comparison, shows the report in the summary card and the
// steps in the log. Also writes the report next to the known route, because a
// number read once and lost is not evidence: the file is what ends up in a
// publication's supplementary data.
void MainWindow::runRouteComparison()
{
    const QString computed = m_cmpComputedPicker->path().trimmed();
    const QString known = m_cmpKnownPicker->path().trimmed();
    if (computed.isEmpty() || known.isEmpty()) {
        setCmpState(tr("IDLE"), QStringLiteral("idle"));
        m_cmpPhase->setText(
            tr("Choose both a computed route layer and a known route layer."));
        m_cmpSummaryCard->setVisible(false);
        return;
    }

    // The comparison does not use the engine — it reads the two layers here —
    // but it holds this thread while it works, and a computation that is
    // already running is entitled to the machine.
    if (refuseIfEngineBusy())
        return;

    // A fresh transcript each time: a second answer appended under the first is
    // how two comparisons get read as one.
    m_cmpConsole->clearAll();
    setCmpState(tr("RUNNING"), QStringLiteral("running"));
    m_cmpPhase->setText(tr("Comparing…"));
    m_cmpSummaryCard->setVisible(false);
    m_cmpButton->setEnabled(false);
    // Set by hand rather than through refreshRunTicker(): this run has no
    // subprocess and no signals, so there is nothing for that function to see.
    if (m_ticker) {
        RunTicker::State s;
        s.active = true;
        s.kind = tr("Route comparison");
        s.remaining = tr("Reading both layers");
        m_ticker->setState(s);
    }
    QCoreApplication::processEvents();

    // The comparison reads two vector layers, and this page can be the first
    // one a user opens. Without this the library is only ever loaded by the
    // Viewer, and a comparison run before visiting it reports GDAL missing on a
    // machine that has it.
    if (!ensureGdalLoaded()) {
        // RouteCompare says GDAL is missing; only this has the reason.
        m_cmpConsole->appendChunk(
            tr("GDAL could not be loaded: %1\n").arg(GdalApi::instance().loadError()));
    }

    // The comparison is synchronous and runs on this thread, so without a turn
    // of the event loop per line the log would arrive in one burst at the end.
    // User input stays excluded: the only clickable thing left is the button
    // that started this, and re-entering here would be a second comparison
    // writing into the first one's log.
    ConsoleView *const console = m_cmpConsole;
    const RouteCompare::Result res = RouteCompare::compare(
        computed, known, m_cmpToleranceSpin->value(),
        [console](const QString &line) {
            console->appendChunk(line + QLatin1Char('\n'));
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        });

    m_cmpButton->setEnabled(true);
    refreshRunTicker();          // empties it: nothing is running any more
    m_cmpSummaryCard->setVisible(true);
    // A proportional font would throw the report's aligned columns out; the run
    // panels wrap their summaries the same way.
    const auto asSummary = [](const QString &text) {
        return QStringLiteral("<pre style='white-space:pre-wrap; font-family:Consolas, "
                              "monospace; margin:0;'>%1</pre>")
            .arg(text.toHtmlEscaped());
    };
    m_cmpResult->setText(asSummary(res.report()));

    if (!res.ok) {
        setCmpState(tr("FAILED"), QStringLiteral("failed"));
        m_cmpPhase->setText(tr("The comparison did not run."));
        m_cmpSummaryTitle->setText(tr("✖ Comparison failed"));
        console->appendMarker(tr("— Comparison failed."), QColor(0xff, 0x6b, 0x6b));
        return;
    }

    setCmpState(tr("COMPLETED"), QStringLiteral("success"));
    m_cmpPhase->setText(tr("Comparison completed."));
    m_cmpSummaryTitle->setText(tr("✓ Comparison completed"));

    const QFileInfo info(known);
    const QString outPath =
        info.absoluteDir().filePath(info.completeBaseName() + QStringLiteral("_comparison.txt"));
    QFile f(outPath);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&f);
        out << tr("Trajecta route comparison\n")
            << QDateTime::currentDateTime().toString(Qt::ISODate) << "\n\n"
            << tr("Computed: ") << QDir::toNativeSeparators(computed) << "\n"
            << tr("Known:    ") << QDir::toNativeSeparators(known) << "\n\n"
            << res.report();
        f.close();
        m_cmpResult->setText(asSummary(res.report() + tr("\nWritten to %1")
                                           .arg(QDir::toNativeSeparators(outPath))));
        console->appendMarker(tr("— Report written to %1")
                                  .arg(QDir::toNativeSeparators(outPath)),
                              QColor(0x5f, 0xd0, 0x68));
    } else {
        // Silently losing the file would leave the numbers on screen only.
        console->appendMarker(tr("— Could not write the report next to the known "
                                 "route (%1): %2")
                                  .arg(QDir::toNativeSeparators(outPath), f.errorString()),
                              QColor(0xff, 0xd1, 0x66));
    }
}

void MainWindow::setCohState(const QString &text, const QString &state)
{
    if (!m_cohChip)
        return;
    m_cohChip->setText(text);
    m_cohChip->setProperty("state", state);
    m_cohChip->style()->unpolish(m_cohChip);
    m_cohChip->style()->polish(m_cohChip);
}

// Scores the sites against the corridors and shows the report. Written files go
// to the folder the user chose; what appears here is the same text, so the
// screen and the supplementary data of a paper cannot disagree.
void MainWindow::runCoherence()
{
    if (m_cohRunning)
        return;                      // the button is disabled, but belt and braces
    const QString raster = m_cohRasterPicker->path().trimmed();
    const QString points = m_cohPointsPicker->path().trimmed();
    if (raster.isEmpty() || points.isEmpty()) {
        setCohState(tr("IDLE"), QStringLiteral("idle"));
        m_cohPhase->setText(tr("Choose both a FETE surface and a point layer."));
        m_cohSummaryCard->setVisible(false);
        return;
    }
    if (refuseIfEngineBusy())
        return;

    Coherence::Params p;
    p.rasterPath = raster;
    p.pointsPath = points;
    p.radiusMetres = m_cohRadiusSpin->value();
    switch (m_cohThresholdCombo->currentData().toInt()) {
    case 1: p.thresholdMode = Coherence::ThresholdMode::Otsu; break;
    case 2: p.thresholdMode = Coherence::ThresholdMode::Absolute; break;
    default: p.thresholdMode = Coherence::ThresholdMode::TopPercent; break;
    }
    p.thresholdValue = m_cohThresholdSpin->value();
    p.nullModel = m_cohNullCheck->isChecked();
    p.nullMode = m_cohNullModeCombo->currentData().toInt() == 1
                     ? Coherence::NullMode::Uniform
                     : Coherence::NullMode::RandomShift;
    p.nullReplicates = m_cohRepsSpin->value();
    p.sensitivity = m_cohSensCheck->isChecked();
    if (p.sensitivity) {
        const QStringList parts = m_cohSensEdit->text().split(
            QRegularExpression(QStringLiteral("[,;\\s]+")), Qt::SkipEmptyParts);
        for (const QString &s : parts) {
            bool okNum = false;
            const double r = s.toDouble(&okNum);
            if (okNum && r > 0.0)
                p.sensitivityRadii << r;
        }
    }
    // Left empty the tool falls back to its own ladder, so a cleared box is a
    // request for the default rather than for no table at all.
    {
        const QStringList parts = m_cohEcdfEdit->text().split(
            QRegularExpression(QStringLiteral("[,;\\s]+")), Qt::SkipEmptyParts);
        for (const QString &s : parts) {
            bool okNum = false;
            const double d = s.toDouble(&okNum);
            if (okNum && d >= 0.0)
                p.ecdfDistances << d;
        }
    }
    p.edgeGuard = m_cohEdgeCheck->isChecked();
    p.writeVector = true;
    p.vectorAsGeoPackage = m_cohVectorCombo->currentData().toInt() == 0;
    p.writeDistanceRaster = m_cohRasterCheck->isChecked();
    p.writeHistogramScript = m_cohRScriptCheck->isChecked();
    p.outputPrefix = m_cohPrefixEdit->text().trimmed();
    p.outputDir = m_cohOutPicker->path().trimmed();
    // Somewhere to write is not optional — the table is the result — so an empty
    // folder falls back to the raster's own, which is where a user who did not
    // think about it will look first anyway.
    if (p.outputDir.isEmpty())
        p.outputDir = QFileInfo(raster).absolutePath();

    m_cohRunning = true;
    m_cohConsole->clearAll();
    setCohState(tr("RUNNING"), QStringLiteral("running"));
    m_cohPhase->setText(tr("Scoring…"));
    m_cohSummaryCard->setVisible(false);
    m_cohButton->setEnabled(false);
    if (m_ticker) {
        RunTicker::State s;
        s.active = true;
        s.kind = tr("Site coherence");
        s.remaining = tr("Reading the surface");
        m_ticker->setState(s);
    }
    QCoreApplication::processEvents();

    if (!ensureGdalLoaded()) {
        m_cohConsole->appendChunk(
            tr("GDAL could not be loaded: %1\n").arg(GdalApi::instance().loadError()));
    }

    // Synchronous, like the comparison, and pumped the same way: the module
    // reports each stage and each block of replicates, and every one of those
    // lines is a chance for the window to repaint. User input stays excluded so
    // a second run cannot be started into the first one's log.
    ConsoleView *const console = m_cohConsole;
    const Coherence::Result res = Coherence::run(p, [console](const QString &line) {
        console->appendChunk(line + QLatin1Char('\n'));
        QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    });

    m_cohRunning = false;
    m_cohButton->setEnabled(true);
    refreshRunTicker();
    m_cohSummaryCard->setVisible(true);
    const auto asSummary = [](const QString &text) {
        return QStringLiteral("<pre style='white-space:pre-wrap; font-family:Consolas, "
                              "monospace; margin:0;'>%1</pre>")
            .arg(text.toHtmlEscaped());
    };
    m_cohResult->setText(asSummary(res.report()));

    if (!res.ok) {
        setCohState(tr("FAILED"), QStringLiteral("failed"));
        m_cohPhase->setText(tr("The sites were not scored."));
        m_cohSummaryTitle->setText(tr("✖ Scoring failed"));
        console->appendMarker(tr("— Scoring failed."), QColor(0xff, 0x6b, 0x6b));
        return;
    }

    setCohState(tr("COMPLETED"), QStringLiteral("success"));
    m_cohPhase->setText(tr("%1 site(s) scored.").arg(res.pointsUsed));
    m_cohSummaryTitle->setText(tr("✓ Sites scored"));
    console->appendMarker(tr("— Done."), QColor(0x5f, 0xd0, 0x68));

    // Straight into the Viewer, as a pair: the distance raster underneath and
    // the scored sites on top. Seeing the two together is how the numbers become
    // an argument, and loading them by hand afterwards is four dialogs.
    if (m_viewer) {
        if (!res.rasterPath.isEmpty()) {
            m_viewer->registerRaster(tr("Distance to corridor"), res.rasterPath, true);
        } else {
            m_viewer->registerRaster(QFileInfo(raster).completeBaseName(), raster, true);
        }
        if (!res.vectorPath.isEmpty())
            m_viewer->registerVectorOverlay(tr("Scored sites"), res.vectorPath);
    }
}

void MainWindow::pickFeatureForTest(int pointIndex)
{
    if (!m_viewer)
        return;
    showPage(2);
    QCoreApplication::processEvents();
    m_viewer->clickFeatureForTest(pointIndex);
}

void MainWindow::pickColourForTest(int overlayIndex)
{
    if (!m_viewer)
        return;
    showPage(2);
    QCoreApplication::processEvents();
    m_viewer->pickColourForTest(overlayIndex);
}

void MainWindow::setOverlaySizeForTest(int overlayIndex, int percent)
{
    if (!m_viewer)
        return;
    showPage(2);
    QCoreApplication::processEvents();
    m_viewer->setOverlaySizeForTest(overlayIndex, percent);
}

void MainWindow::triggerCoherence(const QString &raster, const QString &points,
                                  double radiusMetres)
{
    if (!m_cohRasterPicker)
        return;
    selectPostMode(QStringLiteral("coherence"));
    m_cohRasterPicker->setPath(raster);
    m_cohPointsPicker->setPath(points);
    if (radiusMetres > 0.0)
        m_cohRadiusSpin->setValue(radiusMetres);
    refreshCoherenceUi();
    runCoherence();
}

void MainWindow::triggerRouteComparison(const QString &computed, const QString &known,
                                        double tolerance)
{
    if (!m_cmpComputedPicker)
        return;
    selectPostMode(QStringLiteral("compare"));
    m_cmpComputedPicker->setPath(computed);
    m_cmpKnownPicker->setPath(known);
    if (tolerance > 0.0)
        m_cmpToleranceSpin->setValue(tolerance);
    runRouteComparison();
}

// The walkthrough's content. This is the only place that knows both the tour
// and Trajecta: the overlay is handed widgets and strings and iterates them.
// Rebuilt on every start so the QPointers are fresh — batch chunks come and go.
//
// The wording is condensed from the "?" help beside each field, deliberately:
// two descriptions of the same parameter, written separately, drift apart.
void MainWindow::buildWalkthrough()
{
    if (!m_tour) {
        m_tour = new TourOverlay(this);
        connect(m_tour, &TourOverlay::tourFinished,
                this, &MainWindow::restoreAfterWalkthrough);
        connect(m_tour, &TourOverlay::closeRequested,
                this, &MainWindow::confirmCloseWalkthrough);
    }

    QVector<TourStep> steps;

    // Every screen has to be able to stand on its own. The tour is walked
    // backwards as often as forwards, and a screen that relied on the page the
    // one before it happened to leave behind lights nothing at all when it is
    // arrived at from the other side — which is exactly what "Running it" did,
    // sitting on the Processing page pointing at a button on the Post-
    // processing one. So each block of screens ends by handing its own
    // navigation to every screen in it that did not set its own.
    int blockStart = 0;
    auto closeBlock = [&steps, &blockStart](const std::function<void()> &enter) {
        for (int i = blockStart; i < steps.size(); ++i) {
            if (!steps[i].onEnter)
                steps[i].onEnter = enter;
        }
        blockStart = steps.size();
    };

    {   // 1 — the top bar
        TourStep s;
        for (QPushButton *b : std::as_const(m_navButtons))
            s.targets.append(b);
        s.title = tr("The five sections");
        s.text = tr(
            "Trajecta is organised into five pages. Use this bar to move "
            "between them. <b>Processing</b> is where an analysis is set up and "
            "run. <b>Post-processing</b> works on results that already exist. "
            "<b>Viewer</b> lets you visualise any raster or vector layer "
            "directly in Trajecta Studio. <b>Guide</b> is the manual you have "
            "just come from and contains information about Trajecta and its "
            "use. <b>About</b> carries the version and the contacts."
            "<br><br>"
            "Move through this tour with <b>Continue</b> and <b>Back</b>, or "
            "with the <b>←</b> and <b>→</b> keys. The count between the two "
            "buttons says where you are.");
        steps.append(s);
    }

    {   // 2 — the gear, with a picture of its menu where the real one would drop
        TourStep s;
        s.targets = { m_gearButton };
        QVector<QRect> groups;
        QRect advancedRect;
        s.inset = renderMenuPicture(m_gearButton ? m_gearButton->menu() : nullptr, &groups,
                                    m_advancedSettingsAction, &advancedRect);
        s.title = tr("Appearance and automatic saving");
        s.text = tr(
            "Everything behind the gear is about how Trajecta looks, how it "
            "protects a long run from data loss, and a handful of settings used "
            "rarely enough to stay out of the way otherwise. <b>Auto-save</b> "
            "deserves the most attention: with it on, a FETE writes its state to "
            "disk at a fixed interval, so a crash, a power cut or a deliberate "
            "shutdown costs you that interval instead of the whole analysis — "
            "Trajecta offers to resume an interrupted run the next time it "
            "starts. Auto-save is on by default, and it is set to be performed "
            "every 30 minutes. Importantly, auto-save applies to FETE runs "
            "only.");
        if (groups.size() == 3) {
            s.annotations = {
                { {}, tr("Four colour themes. Purely cosmetic — no setting "
                         "changes with them."), groups.at(0) },
                { {}, tr("The interface font and its size."), groups.at(1) },
                { {}, tr("Auto-save, how often it writes and where it puts "
                         "the file."), groups.at(2) },
            };
            if (advancedRect.isValid()) {
                s.annotations.append(
                    { {}, tr("Explore Advanced settings for additional "
                             "functionalities and personalisations."),
                      advancedRect });
            }
        }
        steps.append(s);
    }

    {   // 3 — the Processing page itself
        TourStep s;
        // The tab, not the page under it: this screen is about where you are,
        // and the answer is a place in the top bar.
        if (!m_navButtons.isEmpty())
            s.targets = { m_navButtons.first() };
        s.onEnter = [this] { showPage(0); };
        s.title = tr("Processing — where an analysis is set up");
        s.text = tr(
            "The first of the five pages, and the one you will spend the most "
            "time on. It holds one form: what to compute, what to read, how "
            "movement is modelled, what to write and where — and, at the foot of "
            "it, the button that starts the engine and the panel that reports on "
            "it.<br><br>"
            "Every setting on it carries a small <b>?</b> beside it. <b>Click "
            "one</b> and it opens an explanation of that parameter — what it "
            "does, what it costs and what a sensible value is — which stays on "
            "screen until you click away. Those badges are the real "
            "documentation of this form; this tour is the map of it.<br><br>"
            "The next screens go down that form, card by card.");
        steps.append(s);
    }

    {   // 3b — analysis type, the card
        TourStep s;
        s.lightCard(m_cardAnalysisType);
        s.title = tr("The first choice: one analysis, or a batch");
        s.text = tr(
            "Every page that runs something asks this first, before which tool: "
            "set up and run <b>one</b> analysis here and now, or hand Trajecta a "
            "<b>queue</b> of them and leave it running. The choice decides which "
            "card appears next — the tool card below stays a single form for "
            "Single analysis, and turns into the batch page for Batch "
            "processing — everything else on the page follows from it.");
        s.annotations = {
            { m_modeSingle, tr("The tool card below chooses FETE or LCPA; "
                               "everything else on the page is that one analysis.") },
            { m_modeBatch, tr("A queue of analyses (FETE or LCPA), run one "
                              "after another unattended. Its own screens are a "
                              "little further into this tour.") },
        };
        steps.append(s);
    }

    {   // 4 — analysis mode, then the hardware it runs with, one card
        TourStep s;
        s.lightCard(m_cardMode);
        // Six things on this one card need a caption, and they all fit below
        // it only once the panel itself is not down there competing for the
        // same room — so the panel goes above instead, narrow band or not,
        // and every caption stays on its usual side, below.
        s.calloutWidthCap = 1650;
        s.preferAbove = true;
        s.title = tr("Processing tool");
        s.text = tr(
            "Everything below this card changes with the choice made in it, "
            "and it only appears while Single analysis is selected above. The "
            "lower half is what the run may use — raising the defaults does "
            "not change the result, only how long you wait.");
        s.annotations = {
            { m_modeFete, tr("Every point to every other. Produces a density raster "
                             "representing how many routes crossed each cell.") },
            { m_modeLcpa, tr("One origin to one or more chosen destinations via "
                             "the least-costly path. Produces the routes themselves.") },
            { m_threadsSpin, tr("Cores the engine may keep busy.") },
            { m_ramSpin, tr("Ceiling on memory. The engine works in blocks under it.") },
            { m_verboseCheck, tr("More detail in the engine log.") },
            { m_manifestCheck, tr("A text record of every input, setting and "
                                  "output of the run.") },
        };
        steps.append(s);
    }

    {   // 5 — FETE, the mode this whole block is about
        TourStep s;
        s.targets = { m_modeFete };
        s.onEnter = [this] { showPage(0); selectMode(QStringLiteral("fete")); };
        s.title = tr("The first mode: FETE");
        s.text = tr(
            "<b>FETE stands for From-Everywhere-To-Everywhere.</b> Given a set "
            "of points, FETE computes the cheapest route between <i>every "
            "pair</i> of them and counts, for each cell of the map, how many of "
            "those routes crossed it. What comes out is not a route but a "
            "<b>density</b>: the ground that movement between these places "
            "would have worn.<br><br>"
            "It is the mode for a question about a landscape rather than about a "
            "journey — where the corridors are, which passes carry everything, "
            "which valleys nothing goes through and so on.");
        steps.append(s);
    }

    {   // 6 — input data, the card
        TourStep s;
        s.lightCard(m_cardInput);
        s.onEnter = [this] { selectMode(QStringLiteral("fete")); };
        s.title = tr("Input data");
        s.text = tr(
            "What the analysis reads: an elevation model (e.g. a DEM), the "
            "sample points to connect, and a folder to write into. None of it "
            "is modified — Trajecta only ever writes inside the output "
            "folder.<br><br>"
            "Everything else on this page has a usable default; these do not.");
        s.annotations = {
            { m_demPicker, tr("A georeferenced GeoTIFF. Every cost comes from its slope.") },
            { m_pointsSourceCombo, tr("A layer you already have, or points generated "
                                      "from the DEM.") },
            { m_outputDirPicker, tr("Where the results are written.") },
        };
        steps.append(s);
    }

    {   // 7 — where the points come from
        TourStep s;
        // The whole card again: when the points are imported the generation
        // group is hidden, and a screen lit around a group that is not there
        // reads as a fault. Turning the setting on to show it off would break
        // the promise that the tour changes nothing.
        s.lightCard(m_cardInput);
        s.title = tr("Where the sample points come from");
        s.text = tr(
            "FETE connects points, so it needs a set of them. Either you bring "
            "your own layer, or Trajecta lays them out over the DEM for you — "
            "only on ground a route could actually cross, so NoData stays empty. "
            "The generated layer is written to the output folder <i>before</i> "
            "the analysis starts and read back as its input, so what the run "
            "used is always on disk.<br><br>"
            "This is also the setting that decides how long the analysis takes. "
            "FETE connects every point to every other, so the work grows with "
            "the square of their number: twice the points is four times the run.");
        steps.append(s);
    }

    {   // 8 — generating them, field by field
        TourStep s;
        s.lightCard(m_cardInput);
        s.title = tr("Laying the points out");
        s.text = tr("Only shown while the points are being generated; an "
                    "imported layer needs none of it.");
        s.annotations = {
            { m_genDensityCombo, tr("Spacing between points, or a target number.") },
            { m_genArrangementCombo, tr("A regular grid, or one random point per block.") },
            { m_genSeedSpin, tr("Makes a random layout reproducible.") },
            { m_genEdgeSpin, tr("Keeps points away from the edge of the DEM.") },
            { m_genNameEdit, tr("Name of the layer written to disk.") },
        };
        steps.append(s);
    }

    {   // 9 — cost modifiers, the idea
        TourStep s;
        s.lightCard(m_cardModifiers);
        s.title = tr("Cost modifiers");
        s.text = tr(
            "Slope decides the cost of every move. Modifiers change it "
            "afterwards, cell by cell: a river or a marsh made dearer, a road "
            "or a ford made cheaper. Optional, and off by default — they make "
            "the run longer, which is why the switch says so.");
        steps.append(s);
    }

    {   // 10 — cost modifiers, the fields
        TourStep s;
        s.lightCard(m_cardModifiers);
        s.title = tr("The four ways to alter cost");
        s.text = tr("A multiplier of 1 changes nothing; below 1 is cheaper, above "
                    "1 is dearer.");
        s.annotations = {
            { m_costVectorPicker, tr("Lines carrying a 'cost' attribute.") },
            { m_polylineBufferSpin, tr("How many cells wide those lines are painted.") },
            { m_costRasterPicker, tr("A multiplier raster aligned with the DEM.") },
            { m_barrierCheck, tr("Above this multiplier, ground is impassable.") },
        };
        steps.append(s);
    }

    {   // 11 — algorithm, the idea
        TourStep s;
        s.lightCard(m_cardAlgorithm);
        s.title = tr("The algorithm");
        s.text = tr(
            "How movement itself is modelled: in which directions a step may go, "
            "what a step costs, and what is refused outright. These four settings "
            "decide the result more than anything else on the page — two runs "
            "over the same DEM with different cost functions are two different "
            "claims about the past.");
        steps.append(s);
    }

    {   // 12 — algorithm, the four settings
        TourStep s;
        s.lightCard(m_cardAlgorithm);
        s.title = tr("What each one decides");
        s.text = tr("The cost function also fixes the unit of the result — hours for "
                    "the walking models, kilojoules per kilogram for Herzog.");
        s.annotations = {
            { m_neighboursCombo, tr("How many directions a step may take. More is "
                                    "smoother and slower.") },
            { m_costFunctionCombo, tr("Which published model turns slope into cost.") },
            { m_slopeCapCheck, tr("Beyond these slopes a cell is simply refused.") },
            { m_smoothingSpin, tr("Widens the route before drawing it.") },
        };
        steps.append(s);
    }

    {   // 14 — output files, the idea
        TourStep s;
        s.lightCard(m_cardOutputs);
        s.title = tr("What comes out");
        s.text = tr(
            "Every raster the run writes, and what to call it. They all land in "
            "the output folder chosen earlier; nothing is written anywhere else. "
            "Leave a name empty and that file is not produced at all.");
        steps.append(s);
    }

    {   // 15 — output files, one by one
        TourStep s;
        s.lightCard(m_cardOutputs);
        s.title = tr("The files, one by one");
        s.text = tr("The density raster is the FETE result proper; the others are "
                    "the working surfaces it was built from.");
        s.annotations = {
            { m_slopeNameEdit, tr("Slope derived from the DEM.") },
            { m_costNameEdit, tr("Cost of crossing each cell, before modifiers.") },
            { m_totalNameEdit, tr("The same after modifiers — what the search used.") },
            { m_densityNameEdit, tr("How many routes crossed each cell: the result.") },
        };
        steps.append(s);
    }

    {   // 16 — the button
        TourStep s;
        s.targets = { m_runButton };
        s.title = tr("Starting the analysis");
        s.text = tr(
            "Trajecta checks the form first and says what is missing rather than "
            "failing halfway. The engine then runs as a separate program: the "
            "interface stays responsive, and you can watch it work in the panel "
            "that appears below.");
        steps.append(s);
    }

    {   // 17 — the run panel
        TourStep s;
        s.lightCard(m_runPanel);
        s.title = tr("While it runs");
        s.text = tr(
            "A long analysis can take days, so this panel is built to be left "
            "alone and glanced at. Pausing releases the processor without losing "
            "the work — the memory stays held, so the machine may sleep but must "
            "not be shut down.");
        s.annotations = {
            { m_runUi.chip, tr("Idle, running, paused, completed or failed.") },
            { m_runUi.progress, tr("How far along, to a tenth of a percent.") },
            { m_runUi.logHandle, tr("Opens the engine's own output.") },
            { m_runUi.pauseButton, tr("Freezes the computation; the run survives.") },
            { m_runUi.cancelButton, tr("Stops it for good. Red, and it asks first.") },
        };
        steps.append(s);
    }

    {   // 18 — surviving an interruption
        TourStep s;
        s.lightCard(m_runPanel);
        s.title = tr("If the run is interrupted");
        s.text = tr(
            "<b>Auto-save</b> is on by default and writes the progress of a FETE "
            "to disk every 30 minutes, so Trajecta can offer to resume it the "
            "next time it starts. The two buttons on the left of this row make "
            "that state something you can handle deliberately rather than only "
            "after an accident: keep a copy of an analysis in progress, and pick "
            "a saved one back up whenever you like — including on another "
            "machine.");
        s.annotations = {
            { m_ckptSaveButton, tr("Copies the state of the running analysis "
                                   "wherever you want it. The run carries on.") },
            { m_ckptLoadButton, tr("Continues a saved analysis from where it "
                                   "stopped.") },
        };
        steps.append(s);
    }

    // Everything so far is the Processing page in FETE mode, including the two
    // opening screens about the top bar and the gear.
    closeBlock([this] { showPage(0); selectMode(QStringLiteral("fete")); });

    {   // 19 — LCPA
        TourStep s;
        s.targets = { m_modeLcpa };
        s.onEnter = [this] { showPage(0); selectMode(QStringLiteral("lcpa")); };
        s.title = tr("The second mode: LCPA");
        s.text = tr(
            "Everything you have just seen — the DEM, the modifiers, the "
            "algorithm, the hardware — works exactly the same way here. What "
            "changes is the question: instead of connecting every point to every "
            "other, LCPA goes from one origin to the destinations you name, and "
            "gives you those routes rather than a density.");
        steps.append(s);
    }

    {   // 20 — LCPA inputs
        TourStep s;
        s.lightCard(m_cardInput);
        s.title = tr("Origin and destinations");
        s.text = tr("The two fields FETE does not have.");
        s.annotations = {
            { m_originPicker, tr("Exactly one point: where the journey starts.") },
            { m_destinationsPicker, tr("One or more points to reach.") },
        };
        steps.append(s);
    }

    {   // 21 — LCPA outputs and the corridor
        TourStep s;
        s.lightCard(m_cardOutputs);
        s.title = tr("What LCPA writes");
        s.text = tr("The routes themselves, in both of the forms they are "
                    "useful in.");
        s.annotations = {
            { m_pathRasterNameEdit, tr("The routes as a raster, one value per "
                                       "crossed cell.") },
            { m_pathLinesNameEdit, tr("The same routes as lines, each carrying "
                                      "its cost and its length.") },
        };
        steps.append(s);
    }

    {   // 22 — the corridor
        TourStep s;
        s.lightCard(m_cardOutputs);
        s.title = tr("The ground around the route");
        s.text = tr(
            "The optimal route is a line on a map, and a line invites more "
            "confidence than it deserves: change the cost function slightly and "
            "it moves. The cost corridor answers the honest question instead — "
            "which ground is nearly as good? Every cell within the chosen "
            "percentage of the best total cost is kept.<br><br>"
            "A wide corridor over open country and a narrow one through a pass "
            "say something a single line cannot: how forced the route was.");
        s.annotations = {
            { m_corridorCheck, tr("Also map the near-optimal ground.") },
            { m_corridorWidthSpin, tr("How much worse than the best still counts.") },
        };
        steps.append(s);
    }

    closeBlock([this] { showPage(0); selectMode(QStringLiteral("lcpa")); });

    {   // 23 — Batch, the mode
        TourStep s;
        s.targets = { m_modeBatch };
        s.onEnter = [this] { showPage(0); selectMode(QStringLiteral("batch")); };
        s.settleMs = 120;
        s.title = tr("The second choice, revisited: Batch processing");
        s.text = tr(
            "Single analysis runs one FETE or LCPA. This one runs a queue of "
            "them, unattended, and that changes what the interface is for: "
            "instead of filling a form and pressing a button, you build a table "
            "of analyses and leave.<br><br>"
            "It is the mode for the work that actually takes days — the same DEM "
            "at four cost functions, or twenty DEMs with the same settings — and "
            "for making that work reproducible, because the whole queue is saved "
            "to a file and can be run again exactly as it was. The page is read "
            "from the top down, and so are the next few screens.");
        steps.append(s);
    }

    // The rest of the batch block is written by the page itself — every widget
    // it points at is private to it — and only the navigation that has to
    // happen first is added here. It goes on every screen of the block and not
    // just the first: the user can walk backwards into the middle of it, and
    // --tour-step can be pointed straight at one.
    if (m_batchPage) {
        QVector<TourStep> batchSteps = m_batchPage->walkthroughSteps();
        for (TourStep &s : batchSteps) {
            s.onEnter = [this] {
                showPage(0);
                selectMode(QStringLiteral("batch"));
            };
            // The batch page is long and its cards are tall; a chunk unfolds
            // with an animation when the mode is switched to.
            s.settleMs = qMax(s.settleMs, 120);
        }
        steps += batchSteps;
    }

    closeBlock([this] { showPage(0); selectMode(QStringLiteral("batch")); });

    {   // 28 — Post-processing, the page
        TourStep s;
        if (m_navButtons.size() > 1)
            s.targets = { m_navButtons.at(1) };
        s.onEnter = [this] { showPage(1); };
        s.title = tr("Post-processing — working on a finished result");
        s.text = tr(
            "The second page does not start an analysis. It works on results you "
            "already have — which is where most of the thinking happens, because "
            "a raw density raster is evidence about the model, and what the two "
            "tools here produce is evidence about the past.<br><br>"
            "One of them makes a result readable; the other makes it testable.");
        steps.append(s);
    }

    {   // 28b — analysis type, the card
        TourStep s;
        s.lightCard(m_postCardAnalysisType);
        s.title = tr("The first choice here too: one analysis, or a batch");
        s.text = tr(
            "Same question as on the Processing page, asked before which tool: "
            "one analysis here and now, or a queue of them left running. The "
            "tool card below stays a single form for Single analysis, and "
            "turns into the post-processing batch page for Batch processing.");
        s.annotations = {
            { m_postModeSingle, tr("The tool card below chooses NNI, Compare or "
                                   "Coherence; everything else on the page is "
                                   "that one analysis.") },
            { m_postModeBatch, tr("A queue of NNI, comparison or coherence runs. "
                                  "Its own screens are further into this tour.") },
        };
        steps.append(s);
    }

    {   // 29 — the three tools, then NNI's hardware
        TourStep s;
        s.lightCard(m_cardPostTool);
        // Same reasoning as the Processing page's tool card: five captions
        // need the room below the card, and the panel only leaves enough of
        // it if it is not resting there itself — see "4 — analysis mode".
        s.calloutWidthCap = 1650;
        s.preferAbove = true;
        s.title = tr("The three tools");
        s.text = tr("The card chooses which one the rest of the page belongs to, "
                    "exactly as the mode card does on the Processing page, and "
                    "only appears while Single analysis is selected above. NNI "
                    "is the one real engine run of the three — Compare and "
                    "Coherence run inside the interface and finish in a moment "
                    "— so it is the one that shows the hardware row beneath "
                    "the cards.");
        s.annotations = {
            { m_postModeNni, tr("Turns a sparse density into a continuous surface.") },
            { m_postModeCompare, tr("Measures a computed route against a real one.") },
            { m_postModeCoherence, tr("Scores how well the sites sit on the corridors.") },
            { m_postThreadsSpin, tr("Cores the engine may keep busy — NNI only.") },
            { m_postRamSpin, tr("Ceiling on memory — NNI only.") },
        };
        steps.append(s);
    }

    // The tool card is on the page whichever tool is chosen, so these two ask
    // only for the page.
    closeBlock([this] { showPage(1); });

    {   // 30 — NNI, the tool
        TourStep s;
        s.targets = { m_postModeNni };
        s.onEnter = [this] { showPage(1); selectPostMode(QStringLiteral("nni")); };
        s.title = tr("The first tool: Natural Neighbour Interpolation");
        s.text = tr(
            "A FETE density is spiky: corridors of crossed cells with empty "
            "ground between them. Interpolating turns it into a continuous "
            "surface that keeps the sampled values exactly and passes smoothly "
            "between corridors — easier to read, and easier to compare with "
            "other surfaces.<br><br>"
            "Sibson's method weights each neighbour by the area it gives up to "
            "the new point, so the result is smooth everywhere except at the "
            "samples themselves, and never invents a value outside the range of "
            "the data.");
        steps.append(s);
    }

    {   // 31 — NNI, what it reads and writes
        TourStep s;
        s.lightCard(m_postNniCard);
        s.title = tr("What it reads, and where it puts the answer");
        s.text = tr("Filled in for you after a FETE run on this machine.");
        s.annotations = {
            { m_interpInputPicker, tr("The density raster to interpolate.") },
            { m_interpOutputDirPicker, tr("Where the interpolated raster goes.") },
            { m_interpNameEdit, tr("Its name.") },
        };
        steps.append(s);
    }

    {   // 32 — NNI, the parameters
        TourStep s;
        s.lightCard(m_postNniCard);
        s.title = tr("How much to generalise");
        s.text = tr(
            "Spacing is the one that matters. At 1 every qualifying cell is a "
            "sample and the result barely differs from the input; larger values "
            "keep only the broad structure — and can thin a narrow corridor to "
            "nothing, which is what preserving the peaks guards against.");
        s.annotations = {
            { m_interpThresholdSpin, tr("Cells at or above this become samples.") },
            { m_interpSpacingSpin, tr("Take a sample every N cells.") },
            { m_interpPeaksCheck, tr("Also keep each block's maximum. Available "
                                     "when the spacing is above 1.") },
            { m_interpRadiusSpin, tr("How far the interpolation reaches into empty ground.") },
        };
        steps.append(s);
    }

    {   // 23 — running the interpolation
        TourStep s;
        s.targets = { m_runInterpButton };
        s.title = tr("Running it");
        s.text = tr(
            "The same panel as an analysis, with the same state, progress and "
            "engine log underneath — it is the same engine doing the work.");
        steps.append(s);
    }

    closeBlock([this] { showPage(1); selectPostMode(QStringLiteral("nni")); });

    {   // 34 — comparison, the tool
        TourStep s;
        s.targets = { m_postModeCompare };
        s.onEnter = [this] { showPage(1); selectPostMode(QStringLiteral("compare")); };
        s.title = tr("The second tool: comparing with a route that is known");
        s.text = tr(
            "This is the step that turns a least-cost path from an illustration "
            "into a claim that can be wrong. Without it a model can only ever "
            "agree with itself.<br><br>"
            "A Roman road, a drove road, a surveyed track: anything whose course "
            "is known independently of the model can be used to ask how well the "
            "model recovers it — and a disagreement is as informative as a "
            "match, because it points at the thing the cost surface does not "
            "know about.");
        steps.append(s);
    }

    {   // 35 — comparison, the two layers
        TourStep s;
        s.lightCard(m_postCompareCard);
        s.title = tr("The two routes, and how close counts");
        s.text = tr("Both layers must be in the same projected system — degrees "
                    "are refused rather than quietly measured as if they were "
                    "metres.");
        s.annotations = {
            { m_cmpComputedPicker, tr("What the model produced.") },
            { m_cmpKnownPicker, tr("The real route: a surveyed road, a historic track.") },
            { m_cmpToleranceSpin, tr("How close counts as close.") },
        };
        steps.append(s);
    }

    {   // 25 — comparison, reading the answer
        TourStep s;
        s.targets = { m_cmpButton };
        s.title = tr("Reading the result");
        s.text = tr(
            "Distances are reported <b>both ways</b>, and the two are not the "
            "same question: how far the computed route strays from the real one, "
            "and how much of the real one the model recovered. A short route "
            "lying on top of a long one scores perfectly one way and badly the "
            "other.<br><br>"
            "The <b>median</b> and the <b>90th percentile</b> describe the usual "
            "agreement; the <b>Hausdorff distance</b> is the single worst "
            "disagreement anywhere. The report is also written next to the known "
            "route as a text file, because a number read once and lost is not "
            "evidence.");
        steps.append(s);
    }

    closeBlock([this] { showPage(1); selectPostMode(QStringLiteral("compare")); });

    {   // 36a — coherence, the tool
        TourStep s;
        s.targets = { m_postModeCoherence };
        s.onEnter = [this] { showPage(1); selectPostMode(QStringLiteral("coherence")); };
        s.title = tr("The third tool: do the sites sit on the corridors?");
        s.text = tr(
            "This is the question the FETE surface was computed for. The surface "
            "says where movement concentrates; a layer of sites says where people "
            "stayed. This measures how far each site is from the nearest "
            "corridor, how busy the ground around it is, and — the part that "
            "matters — how much of that could have happened by "
            "chance.<br><br>"
            "Because the answer is expressed as a comparison against chance, two "
            "periods or two regions can be set side by side even when their "
            "surfaces were computed from different numbers of points.");
        steps.append(s);
    }

    {   // 36b — coherence, the parameters
        TourStep s;
        s.lightCard(m_postCoherenceCard);
        s.title = tr("What to measure, and what to measure it against");
        s.text = tr(
            "The radius is restated in <b>cells</b> beside the box, because that "
            "is what decides whether the measurement means anything.<br><br>"
            "Every setting here has a <b>?</b> beside it, and the null model's is "
            "worth reading: without it a score of 64 cannot be told from the "
            "score random points would have got.");
        s.annotations = {
            { m_cohRasterPicker, tr("The FETE surface, raw or interpolated.") },
            { m_cohPointsPicker, tr("The sites, in the same projected CRS.") },
            { m_cohRadiusSpin, tr("How far around each site to look.") },
            { m_cohThresholdCombo, tr("What counts as a corridor — the top 1% by default.") },
            { m_cohNullCheck, tr("The comparison against chance. Leave it on.") },
        };
        steps.append(s);
    }

    {   // 36c — coherence, reading the answer
        TourStep s;
        s.targets = { m_cohButton };
        s.title = tr("Reading the result");
        s.text = tr(
            "The report answers four questions, from the most general down. "
            "<b>How many sites are near a corridor at all?</b> — a table of "
            "shares within fixed distances. <b>How far are they?</b> — the "
            "median, the deciles and a histogram, none of which depend on the "
            "radius you chose. <b>How much corridor is around them?</b> — the "
            "<b>proximity index</b>, the share of the neighbourhood that is "
            "corridor, and <b>enrichment</b>, that share against the whole "
            "surface's, where <b>1.00 is exactly chance</b>. <b>How busy is "
            "that ground?</b> — the <b>intensity index</b>, where <b>50 is the "
            "average location</b>.<br><br>"
            "Quote the <b>means</b>: those are the figures the 1.00 and the 50 "
            "are statements about. The medians beside them are usually much "
            "lower, and that is normal — corridors are thin, so most ground has "
            "none within reach.<br><br>"
            "For the sample as a whole there is also the <b>ratio</b>: 0.5 means "
            "the sites are half as far from a corridor as chance would put "
            "them.<br><br>"
            "The scored sites and the distance raster open in the Viewer as soon "
            "as the run finishes, with the sites coloured by their score — and "
            "clicking one opens its whole row in a panel, your own columns "
            "included.");
        steps.append(s);
    }

    closeBlock([this] { showPage(1); selectPostMode(QStringLiteral("coherence")); });

    {   // 36d — Batch, revisited
        TourStep s;
        s.targets = { m_postModeBatch };
        s.onEnter = [this] { showPage(1); selectPostMode(QStringLiteral("batch")); };
        s.settleMs = 120;
        s.title = tr("Batch processing, for the three post-processing tools");
        s.text = tr(
            "The same idea as Processing's own Batch processing: queue several "
            "runs and leave them unattended.<br><br>"
            "A chunk here is simpler than a Processing chunk, because it does "
            "not need to be anything else: one chunk is already one whole "
            "analysis — one interpolation, one comparison or one coherence "
            "score — with the same fields as the single-tool tab above this "
            "page, plus the choice of loading its result into the Viewer once "
            "it finishes.<br><br>"
            "One tool runs for the whole batch, chosen once at the top, exactly "
            "like FETE or LCPA is for a Processing batch.");
        steps.append(s);
    }

    // The rest of the block is written by the page itself, the same way the
    // Processing batch's screens are.
    if (m_postBatchPage) {
        QVector<TourStep> postBatchSteps = m_postBatchPage->walkthroughSteps();
        for (TourStep &s : postBatchSteps) {
            s.onEnter = [this] {
                showPage(1);
                selectPostMode(QStringLiteral("batch"));
            };
            s.settleMs = qMax(s.settleMs, 120);
        }
        steps += postBatchSteps;
    }

    closeBlock([this] { showPage(1); selectPostMode(QStringLiteral("batch")); });

    {   // 37 — the Viewer, the page
        TourStep s;
        if (m_navButtons.size() > 2)
            s.targets = { m_navButtons.at(2) };
        s.onEnter = [this] {
            showPage(2);
            // The example layers go on here rather than on the first screen of
            // the block: this is where the page comes into view, and an empty
            // canvas behind a screen that says "here is the map" reads badly.
            if (m_viewer && m_viewer->loadTourSamples())
                m_viewerSamplesLoaded = true;
        };
        s.settleMs = 150;
        s.title = tr("Viewer — looking at what came out");
        s.text = tr(
            "Every result Trajecta writes is a map, and a map has to be looked "
            "at before it can be believed. The third page draws any raster or "
            "vector layer — its own outputs, which appear here on their own "
            "after a run, or anything else you open.<br><br>"
            "It is a reading room, not an editor: nothing here changes a file on "
            "disk. For this part of the tour a small elevation model and a set "
            "of sample points have been loaded as an example, and both are taken "
            "away again at the end.");
        steps.append(s);
    }

    // The Viewer block, written by the page for the same reason as the batch
    // one. Every screen of it also makes sure the example layers are on the
    // canvas: explaining a map viewer against an empty canvas explains nothing,
    // and the block can be walked into backwards or jumped straight to.
    if (m_viewer) {
        QVector<TourStep> viewerSteps = m_viewer->walkthroughSteps();
        for (TourStep &s : viewerSteps) {
            s.onEnter = [this] {
                showPage(2);
                // Idempotent: it registers the same two paths again, which the
                // Viewer treats as a refresh of layers it already has.
                if (m_viewer->loadTourSamples())
                    m_viewerSamplesLoaded = true;
                // This block is the one immediately before the status-bar
                // screens, so it is also where a reader walking *backwards* out
                // of them arrives: the demonstration ticker is taken down here.
                setTourTicker(false);
            };
            // The first draw of a raster goes through GDAL and a full rebuild
            // of the display buffer.
            s.settleMs = qMax(s.settleMs, 150);
        }
        steps += viewerSteps;
    }

    closeBlock([this] {
        showPage(2);
        if (m_viewer && m_viewer->loadTourSamples())
            m_viewerSamplesLoaded = true;
    });

    {   // 42 — the status bar
        TourStep s;
        s.targets = { m_statusBar };
        s.padding = 0;
        // Nothing is running during a tour, and the ticker is invisible when
        // nothing is. It is put on screen for these two screens exactly as the
        // Viewer's example layers are put on the canvas: an explanation of
        // something that is not there explains nothing.
        s.onEnter = [this] { setTourTicker(true); };
        s.title = tr("The bar along the bottom");
        s.text = tr(
            "It answers one question, and it is always on screen: has Trajecta "
            "got everything it needs to work? Green on both means you can run an "
            "analysis. Anything else is worth reading before filling in a form, "
            "because the run would fail at the end of it.<br><br>"
            "The middle of the bar answers a second question, but only while "
            "there is one to answer. <b>As soon as an analysis starts, a small "
            "progress strip appears there</b> and stays until the run ends — "
            "paused runs included, in paler colours. It is the only report on a "
            "run that is visible from every page: whatever you are looking at, "
            "you can see that the machine is busy, how far it has got and what "
            "kind of analysis it is working on.");
        steps.append(s);
    }

    {   // 43 — the status bar, item by item
        TourStep s;
        s.targets = { m_statusBar };
        s.padding = 0;
        s.onEnter = [this] { setTourTicker(true); };
        s.title = tr("The two indicators, the strip, and the two remedies");
        s.text = tr("What each one says depends on the machine; what each one "
                    "means does not.");
        s.annotations = {
            { m_engineStatus, tr("The computing engine. Without it nothing runs.") },
            { m_gdalStatus, tr("The library that reads and writes geospatial files.") },
            { m_ticker, tr("The run in progress, shown only while there is one. "
                           "Click it for the chunk, the hardware and the time "
                           "left.") },
            { m_locateEngineButton, tr("Point Trajecta at the engine yourself.") },
            { m_locateGdalButton, tr("Point it at a GDAL installation yourself.") },
        };
        steps.append(s);
    }

    {   // 44 — Last: how to see all this again
        TourStep s;   // no targets: nothing to light, the callout sits centred
        // The demonstration ends here: from this screen on, the ticker tells
        // the truth again, which on a machine with nothing running is silence.
        s.onEnter = [this] { setTourTicker(false); };
        s.title = tr("The end");
        s.text = tr(
            // Centred on its own, against the justified paragraph underneath:
            // the closing line is a salute, not part of the explanation. The
            // margin is zeroed because the block would otherwise add its own
            // spacing on top of the blank line that follows it.
            "<div align=\"center\" style=\"margin:0\">"
            "<b>Congratulations, that's the end of the tour!</b></div><br>"
            "You can run it again whenever you like: open the <b>Guide</b> "
            "section and click the <b>tutorial</b> link near the top of the "
            "page. Nothing you had set up has been changed — Trajecta is "
            "exactly where you left it.");
        steps.append(s);
    }

    // The bar along the bottom, and the closing screen: back on the page the
    // tour started on, which is also where it will leave the user.
    closeBlock([this] { showPage(0); });

    m_tour->setSteps(steps);
}

void MainWindow::startWalkthrough(int index)
{
    // Photograph where the user was, before a single thing moves. A tour taken
    // from the Guide with LCPA set up must give back the Guide with LCPA set
    // up — not the Processing page in whatever mode the last screen happened
    // to leave behind.
    m_tourReturnPage = m_pages ? m_pages->currentIndex() : 0;
    m_tourReturnMode = (m_modeBatch && m_modeBatch->isChecked())
                           ? QStringLiteral("batch")
                       : (m_modeLcpa && m_modeLcpa->isChecked())
                           ? QStringLiteral("lcpa")
                           : QStringLiteral("fete");
    // Restored via selectPostMode(), which already treats an unrecognised
    // name as "nni" — the same fallback this had before "coherence" and
    // "batch" existed, kept here rather than fixed silently as a separate
    // change: with only two names it never had a wrong answer to give.
    m_tourReturnPostMode = (m_postModeCompare && m_postModeCompare->isChecked())
                               ? QStringLiteral("compare")
                           : (m_postModeCoherence && m_postModeCoherence->isChecked())
                               ? QStringLiteral("coherence")
                           : (m_postModeBatch && m_postModeBatch->isChecked())
                               ? QStringLiteral("batch")
                               : QStringLiteral("nni");
    // The log canvases are folded away for the duration, and put back as they
    // were at the end. An open one is a full screen of empty transcript: the
    // run panel then grows past the bottom of the window, and the captions on
    // the screens that describe it point at buttons that have been pushed out
    // of sight. Folded, the whole panel fits and every leader lands on
    // something the reader can see — including the handle itself, which is
    // what the caption about the log is about in the first place.
    m_tourReturnLogsOpen.clear();
    for (QToolButton *h : {m_runUi.logHandle, m_postUi.logHandle, m_cmpLogHandle}) {
        m_tourReturnLogsOpen.append(h && h->isChecked());
        if (h)
            h->setChecked(false);
    }
    // The batch chunks go the other way: unfolded for the duration, and folded
    // back exactly as they were found. Same reasoning, opposite direction — a
    // folded chunk is a header with nothing under it, so the screen about the
    // rows had no rows to point at and the one about the chunk lit a strip.
    if (m_batchPage)
        m_tourReturnChunksFolded = m_batchPage->unfoldChunks();
    if (m_postBatchPage)
        m_tourReturnPostChunksFolded = m_postBatchPage->unfoldChunks();

    // The tour opens on the Processing page with FETE selected, whichever way
    // it was started: the first screens describe that page.
    showPage(0);
    selectMode(QStringLiteral("fete"));
    buildWalkthrough();
    m_tour->startAt(index);
}

// The two doors a person comes through — the Guide's link and the offer at the
// first start — both promise a maximised window, and this is where the promise
// is kept. Deliberately not inside startWalkthrough(): the hidden --tour switch
// goes through that one too, and it is used together with --size to photograph
// a screen at a chosen width, which maximising would silently override.
//
// The window is left maximised at the end. That is not an oversight against the
// "the tour changes nothing" rule: the two dialogs say it will happen, so it is
// something the user agreed to rather than something done behind their back —
// which is exactly why they say it.
void MainWindow::startWalkthroughMaximised()
{
    if (!isMaximized())
        showMaximized();
    // The tour measures widgets, and the resize the line above asks for has not
    // happened yet. It would right itself anyway — the overlay recomputes
    // everything on the host's resize — but the first screen would be placed
    // twice, and the first placement would be visible.
    QTimer::singleShot(0, this, [this] { startWalkthrough(0); });
}

void MainWindow::closeWalkthrough()
{
    if (m_tour)
        m_tour->closeTour();
}

void MainWindow::confirmCloseWalkthrough()
{
    if (!m_tour || !m_tour->isActive())
        return;
    // "Keep going" wears the fill: leaving is the answer with a cost, and the
    // one the ✕ may well have been pressed for by accident.
    if (!TrajectaUi::confirm(
            this, tr("End the walkthrough"),
            tr("Stop the guided tour here?\n\nYou can start it again from the "
               "beginning whenever you like."),
            tr("Stop the tour"), tr("Keep going"), 0,
            TrajectaUi::Fill::Reject))
        return;

    m_tour->closeTour();
    // After the tour is on its way out, not before: a notice about where to
    // find something again, printed over the thing itself, says nothing.
    TrajectaUi::notify(
        this, tr("It stays available"),
        tr("The walkthrough can be started whenever you like: open the "
           "<b>Guide</b> page and click the <b>tutorial</b> link near the top."));
}

void MainWindow::restoreAfterWalkthrough()
{
    // The example layers go before anything else: they are the only thing the
    // tour added that outlives it, and they must not survive being abandoned
    // halfway any more than being watched to the end.
    if (m_viewerSamplesLoaded && m_viewer) {
        m_viewer->unloadTourSamples();
        m_viewerSamplesLoaded = false;
    }
    // Whatever the tour folded away goes back exactly as it was found. The
    // list is empty when the tour never started, which is why it is checked by
    // size rather than assumed to have three entries.
    const QVector<QToolButton *> handles = {m_runUi.logHandle, m_postUi.logHandle,
                                            m_cmpLogHandle};
    for (int i = 0; i < handles.size() && i < m_tourReturnLogsOpen.size(); ++i) {
        if (handles.at(i))
            handles.at(i)->setChecked(m_tourReturnLogsOpen.at(i));
    }
    m_tourReturnLogsOpen.clear();

    if (m_batchPage)
        m_batchPage->restoreChunkFolds(m_tourReturnChunksFolded);
    m_tourReturnChunksFolded.clear();

    if (m_postBatchPage)
        m_postBatchPage->restoreChunkFolds(m_tourReturnPostChunksFolded);
    m_tourReturnPostChunksFolded.clear();

    selectMode(m_tourReturnMode);
    selectPostMode(m_tourReturnPostMode);
    showPage(m_tourReturnPage);
}

void MainWindow::openAllLogs()
{
    for (QToolButton *h : {m_runUi.logHandle, m_postUi.logHandle, m_cmpLogHandle}) {
        if (h)
            h->setChecked(true);
    }
    if (m_batchPage)
        m_batchPage->openLogs();
    if (m_postBatchPage)
        m_postBatchPage->openLogs();
}

void MainWindow::selectPostMode(const QString &name)
{
    if (!m_postModeNni || !m_postModeCompare)
        return;
    const QString wanted = name.trimmed().toLower();
    if (wanted == QLatin1String("batch") || wanted == QLatin1String("postbatch")) {
        if (m_postModeBatch)
            m_postModeBatch->setChecked(true);
        updatePostModeUi();
        return;
    }
    // Analysis type and tool are two different button groups now (see
    // buildPostPage()'s "Analysis type" card), so picking a tool no longer
    // implies Single the way it did when all four shared one group.
    if (m_postModeSingle)
        m_postModeSingle->setChecked(true);
    if (wanted == QLatin1String("compare"))
        m_postModeCompare->setChecked(true);
    else if (wanted == QLatin1String("coherence") && m_postModeCoherence)
        m_postModeCoherence->setChecked(true);
    else
        m_postModeNni->setChecked(true);
    updatePostModeUi();
}

// Shows one post-processing tool at a time, each with its own start button and
// its own result panel. The two panels are never on screen together: the run
// panel reports on an engine run the comparison never starts, and offering a
// Pause button for nothing is worse than showing nothing. Batch replaces the
// tool card and every single-tool form wholesale, the same relationship
// Batch processing has with FETE/LCPA on the Processing page.
void MainWindow::updatePostModeUi()
{
    if (!m_postModeNni || !m_postNniCard || !m_postCompareCard)
        return;
    const bool batch = m_postModeBatch && m_postModeBatch->isChecked();
    if (m_cardPostTool)
        m_cardPostTool->setVisible(!batch);
    for (QWidget *w : std::as_const(m_postSingleRunCards))
        w->setVisible(!batch);
    if (m_postBatchPage)
        m_postBatchPage->setVisible(batch);
    if (batch)
        return;

    const bool nni = m_postModeNni->isChecked();
    const bool compare = m_postModeCompare && m_postModeCompare->isChecked();
    const bool coherence = m_postModeCoherence && m_postModeCoherence->isChecked();
    m_postNniCard->setVisible(nni);
    if (m_postHardwareBox)
        m_postHardwareBox->setVisible(nni);
    m_postRunRow->setVisible(nni);
    if (m_postRunPanel)
        m_postRunPanel->setVisible(nni);

    m_postCompareCard->setVisible(compare);
    if (m_cmpRunRow)
        m_cmpRunRow->setVisible(compare);
    if (m_cmpPanel)
        m_cmpPanel->setVisible(compare);

    if (m_postCoherenceCard)
        m_postCoherenceCard->setVisible(coherence);
    if (m_cohRunRow)
        m_cohRunRow->setVisible(coherence);
    if (m_cohPanel)
        m_cohPanel->setVisible(coherence);
}

// The radius in cells, and which fields the chosen options actually use. Both
// exist for the same reason: a parameter whose effect you cannot see is a
// parameter you set once and never understand.
void MainWindow::refreshCoherenceUi()
{
    if (!m_cohThresholdCombo || !m_cohRadiusSpin)
        return;
    const int mode = m_cohThresholdCombo->currentData().toInt();
    m_cohThresholdSpin->setEnabled(mode != 1);         // Otsu decides for itself
    m_cohThresholdSpin->setSuffix(mode == 0 ? tr(" %") : QString());
    // A per-cent wants two decimals, a raw path count wants three; and a value
    // left over from the other mode has to be brought back into range, or the
    // box shows 100 when the user typed 91574.
    m_cohThresholdSpin->setDecimals(mode == 0 ? 2 : 3);
    m_cohThresholdSpin->setMaximum(mode == 0 ? 100.0 : 1.0e9);
    if (mode == 0 && m_cohThresholdSpin->value() > 100.0)
        m_cohThresholdSpin->setValue(1.0);

    const bool null = m_cohNullCheck && m_cohNullCheck->isChecked();
    if (m_cohNullModeCombo)
        m_cohNullModeCombo->setEnabled(null);
    if (m_cohRepsSpin)
        m_cohRepsSpin->setEnabled(null);
    if (m_cohSensEdit && m_cohSensCheck)
        m_cohSensEdit->setEnabled(m_cohSensCheck->isChecked());

    if (!m_cohCellNote)
        return;
    // The cell size comes from the raster that is actually selected, so the
    // note is silent until there is one.
    const QString path = m_cohRasterPicker ? m_cohRasterPicker->path() : QString();
    double cell = 0.0;
    // The library is loaded lazily by whichever page needs it first, and this
    // page can be the first. Without asking here, the note stays "pick a
    // surface" even with one picked, which reads as if the file were wrong.
    // Loading is idempotent and costs nothing once it has happened.
    if (!path.isEmpty() && !GdalApi::instance().isLoaded())
        ensureGdalLoaded();
    if (!path.isEmpty() && GdalApi::instance().isLoaded()) {
        GdalApi &api = GdalApi::instance();
        if (GDALDatasetH ds = api.OpenEx(path.toUtf8().constData(),
                                         GdalApi::OF_Raster, nullptr, nullptr, nullptr)) {
            double gt[6];
            if (api.GetGeoTransform(ds, gt) == 0)
                cell = std::fabs(gt[1]);
            api.Close(ds);
        }
    }
    if (cell > 0.0) {
        const double cells = m_cohRadiusSpin->value() / cell;
        m_cohCellNote->setText(tr("— %1 cells of %2 m")
                                   .arg(cells, 0, 'f', 1)
                                   .arg(cell, 0, 'f', 2));
    } else {
        m_cohCellNote->setText(tr("— pick a surface to see this in cells"));
    }
}

void MainWindow::refreshCostUnitsNote()
{
    if (!m_costFunctionCombo || !m_costUnitsNote)
        return;
    const bool energy = m_costFunctionCombo->currentData().toInt() == 4;
    m_costUnitsNote->setText(
        energy ? tr("— costs come out in kJ per kg of walker, not in hours: this is an "
                    "energy model and its cost surfaces cannot be compared with the others")
               : tr("— costs come out in hours of walking"));
}

int MainWindow::selectedNeighbours() const
{
    if (!m_neighboursCombo)
        return 16;
    const int preset = m_neighboursCombo->currentData().toInt();
    if (preset > 0)
        return preset;
    return TrajectaUi::snapNeighbourCount(m_neighboursCustom ? m_neighboursCustom->value() : 16);
}

bool MainWindow::confirmOverwritingSavedProcess(const QString &dir)
{
    const Checkpoint::Session session = Checkpoint::readSession(dir);
    if (!session.valid)
        return true;
    // A session file with no engine state behind it is worth nothing: the run
    // it belonged to never reached its first save.
    const Checkpoint::Info info = Checkpoint::latest(dir);
    if (!info.found && !session.batch)
        return true;
    return TrajectaUi::confirm(
        this, tr("An unfinished analysis is saved"),
        tr("%1 was interrupted, and its progress is still saved.\n\n"
           "Trajecta keeps one unfinished analysis at a time, so starting this "
           "run deletes it. To keep it instead, cancel here, restart Trajecta "
           "Studio and choose Resume — or \"Save process...\" to put it aside.\n\n"
           "Start the new run anyway?")
            .arg(session.label.isEmpty() ? tr("An earlier analysis") : session.label),
        tr("Start anyway"), tr("Cancel"));
}

void MainWindow::markSessionDeliberate()
{
    // A single run records the folder it is saving into; a batch keeps its own
    // copy, and both take it from Checkpoint::activeDir() when they start.
    const QString dir = m_lastRunCheckpointDir.isEmpty() ? Checkpoint::activeDir()
                                                         : m_lastRunCheckpointDir;
    if (dir.isEmpty())
        return;
    Checkpoint::Session session = Checkpoint::readSession(dir);
    if (!session.valid || session.deliberate)
        return;
    session.deliberate = true;
    Checkpoint::writeSession(dir, session);
}

// ---------------------------------------------------------------------------
// Settings persistence
// ---------------------------------------------------------------------------

void MainWindow::loadSettings()
{
    QSettings s;

    // 1 FETE, 2 LCPA, 3 batch. Without the third value, leaving the app in
    // batch mode brought it back up in LCPA.
    switch (s.value(QStringLiteral("form/mode"), 1).toInt()) {
    case 3:  m_modeBatch->setChecked(true); break;
    case 2:  m_modeLcpa->setChecked(true); break;
    default: m_modeFete->setChecked(true); break;
    }
    if (m_batchPage) {
        // A batch that was never run comes back: it is typed by hand, row by
        // row, and losing it on exit would be the most expensive thing the
        // window could throw away. Save/Load to a file stays, for keeping and
        // sharing a batch on purpose.
        //
        // A batch that *was* run does not. Its rows have been computed, and
        // handing them back as a page ready to run again invites exactly one
        // mistake: pressing Run and recomputing the lot over its own output.
        // There is one road back, and it is the recovery prompt a moment from
        // now — resumeFromCheckpoint() rebuilds the job from the session file
        // with the finished rows already marked done, which is the only form
        // in which a run batch is worth restoring. Decline that, or let the
        // batch have finished properly, and the page is what it is on a fresh
        // install: a single empty chunk.
        //
        // The flag is cleared here, with the job it refers to, so only this
        // one start-up pays for the last run.
        // The version is as much a part of the test as the flag, and for a
        // reason worth spelling out: a job stored by a build that did not yet
        // record whether it had been run tells us nothing about itself. It may
        // be a batch typed and never started, or one that ran for four days —
        // and there is no way to tell them apart after the fact. Discarded, on
        // the ground that offering a finished batch as ready to run again is
        // the more expensive mistake of the two.
        //
        // Without this every user upgrading from an earlier build would get
        // their last batch back exactly once, which is precisely the behaviour
        // the flag was added to stop, and would look like the fix not working.
        const bool processed = s.value(QStringLiteral("batch/processed"), false).toBool();
        const bool known = s.value(QStringLiteral("batch/jobVersion"), 0).toInt()
                           >= kBatchJobVersion;
        if (processed || !known) {
            s.remove(QStringLiteral("batch/processed"));
            s.remove(QStringLiteral("batch/job"));
            s.remove(QStringLiteral("batch/jobVersion"));
        } else {
            m_batchPage->restoreState(s.value(QStringLiteral("batch/job")).toString());
        }
    }
    if (m_postBatchPage) {
        // Same reasoning as the Processing batch just above, in its own
        // namespace: a post-processing batch that was never run comes back
        // untouched, one that was is discarded rather than offered as ready to
        // run again over its own output.
        const bool processed =
            s.value(QStringLiteral("postbatch/processed"), false).toBool();
        const bool known =
            s.value(QStringLiteral("postbatch/jobVersion"), 0).toInt()
            >= kPostBatchJobVersion;
        if (processed || !known) {
            s.remove(QStringLiteral("postbatch/processed"));
            s.remove(QStringLiteral("postbatch/job"));
            s.remove(QStringLiteral("postbatch/jobVersion"));
        } else {
            m_postBatchPage->restoreState(
                s.value(QStringLiteral("postbatch/job")).toString());
        }
    }

    m_demPicker->setPath(s.value(QStringLiteral("form/dem")).toString());
    m_pointsSourceCombo->setCurrentIndex(
        s.value(QStringLiteral("form/pointsSource"), 0).toInt() == 1 ? 1 : 0);
    m_pointsPicker->setPath(s.value(QStringLiteral("form/points")).toString());
    m_genDensityCombo->setCurrentIndex(
        s.value(QStringLiteral("form/genByTarget"), true).toBool() ? 1 : 0);
    m_genSpacingSpin->setValue(s.value(QStringLiteral("form/genSpacing"), 10).toInt());
    m_genTargetSpin->setValue(s.value(QStringLiteral("form/genTarget"), 5000).toInt());
    m_genArrangementCombo->setCurrentIndex(
        s.value(QStringLiteral("form/genRandom"), false).toBool() ? 1 : 0);
    m_genSeedSpin->setValue(s.value(QStringLiteral("form/genSeed"), 1).toInt());
    m_genEdgeSpin->setValue(s.value(QStringLiteral("form/genEdgeBuffer"), 0).toInt());
    m_originPicker->setPath(s.value(QStringLiteral("form/origin")).toString());
    m_destinationsPicker->setPath(s.value(QStringLiteral("form/destinations")).toString());
    m_outputDirPicker->setPath(s.value(QStringLiteral("form/outputDir")).toString());

    // Deliberately not restored from the last session: cost modifiers are
    // opt-in every time, even if the paths below stay filled in.
    m_modifiersGroup->setChecked(false);
    m_costVectorPicker->setPath(s.value(QStringLiteral("form/costVector")).toString());
    m_polylineBufferSpin->setValue(s.value(QStringLiteral("form/polylineBuffer"), 2).toInt());
    m_costRasterPicker->setPath(s.value(QStringLiteral("form/costRaster")).toString());
    m_barrierCheck->setChecked(s.value(QStringLiteral("form/barrierOn"), true).toBool());
    m_barrierSpin->setValue(s.value(QStringLiteral("form/barrierValue"), 1000.0).toDouble());

    // A stored value that is not one of the presets came from the custom box,
    // so it goes back there rather than being silently reset to 16.
    const int neighbours = s.value(QStringLiteral("form/neighbours"), 16).toInt();
    const int nIdx = m_neighboursCombo->findData(neighbours);
    if (nIdx >= 0) {
        m_neighboursCombo->setCurrentIndex(nIdx);
    } else {
        m_neighboursCustom->setValue(TrajectaUi::snapNeighbourCount(neighbours));
        m_neighboursCombo->setCurrentIndex(m_neighboursCombo->findData(0));
    }
    refreshNeighboursCustom();
    const int cf = s.value(QStringLiteral("form/costFunction"), 1).toInt();
    const int cfIdx = m_costFunctionCombo->findData(cf);
    m_costFunctionCombo->setCurrentIndex(cfIdx >= 0 ? cfIdx : 0);
    refreshCostUnitsNote();
    m_slopeCapCheck->setChecked(s.value(QStringLiteral("form/slopeCutoff"), false).toBool());
    m_slopeCapUp->setValue(s.value(QStringLiteral("form/maxSlopeUp"), 30).toInt());
    m_slopeCapDown->setValue(s.value(QStringLiteral("form/maxSlopeDown"), 30).toInt());
    m_slopeCapUp->setEnabled(m_slopeCapCheck->isChecked());
    m_slopeCapDown->setEnabled(m_slopeCapCheck->isChecked());
    m_smoothingSpin->setValue(s.value(QStringLiteral("form/smoothing"), 0).toInt());

    if (s.contains(QStringLiteral("form/threads")))
        m_threadsSpin->setValue(s.value(QStringLiteral("form/threads")).toInt());
    if (s.contains(QStringLiteral("form/ram"))) {
        const int stored = s.value(QStringLiteral("form/ram")).toInt();
        // Two stored values were never chosen by anyone: 60 percent of the
        // installed memory, which is what versions up to 1.0.0 put there, and
        // the flat 4096 MB that a 1.0.1 build recommended for a day before the
        // per-thread arithmetic showed it was too low. Both are what the box
        // happened to hold when the app last closed, so both give way to the
        // current recommendation. A figure somebody actually typed is restored
        // untouched.
        const qint64 totalRam = SystemInfo::totalRamMb();
        const int sixtyPercent = int(qMax<qint64>(1024, (totalRam * 60) / 100));
        const bool wasNeverChosen = (stored == sixtyPercent || stored == 4096);
        m_ramSpin->setValue(
            wasNeverChosen
                ? int(qMin<qint64>(SystemInfo::kRecommendedRamMb, totalRam))
                : stored);
    }
    m_verboseCheck->setChecked(s.value(QStringLiteral("form/verbose"), false).toBool());
    m_manifestCheck->setChecked(s.value(QStringLiteral("form/manifest"), true).toBool());

    auto loadName = [&s](QLineEdit *edit, const char *key) {
        // An empty stored value is a deliberate "do not save this output", so
        // it has to come back empty. Only a key that was never written keeps
        // the built-in default the field was constructed with.
        const QString k = QString::fromLatin1(key);
        if (s.contains(k))
            edit->setText(s.value(k).toString().trimmed());
    };
    loadName(m_slopeNameEdit, "form/slopeName");
    loadName(m_costNameEdit, "form/costName");
    loadName(m_additionalNameEdit, "form/additionalName");
    loadName(m_totalNameEdit, "form/totalName");
    loadName(m_densityNameEdit, "form/densityName");
    loadName(m_pathRasterNameEdit, "form/pathRasterName");
    loadName(m_pathLinesNameEdit, "form/pathLinesName");
    loadName(m_corridorNameEdit, "form/corridorName");
    m_corridorCheck->setChecked(
        s.value(QStringLiteral("form/costCorridor"), false).toBool());
    m_corridorWidthSpin->setValue(
        s.value(QStringLiteral("form/corridorWidth"), 10.0).toDouble());
    m_corridorWidthSpin->setEnabled(m_corridorCheck->isChecked());
    m_corridorNameLabel->setEnabled(m_corridorCheck->isChecked());
    m_corridorNameEdit->setEnabled(m_corridorCheck->isChecked());
    loadName(m_genNameEdit, "form/genLayerName");

    m_interpInputPicker->setPath(s.value(QStringLiteral("post/input")).toString());
    m_interpOutputDirPicker->setPath(s.value(QStringLiteral("post/outputDir")).toString());
    m_interpThresholdSpin->setValue(s.value(QStringLiteral("post/threshold"), 1.0).toDouble());
    m_interpSpacingSpin->setValue(s.value(QStringLiteral("post/spacing"), 4).toInt());
    m_interpRadiusSpin->setValue(s.value(QStringLiteral("post/radius"), 0).toInt());
    loadName(m_interpNameEdit, "post/name");
    if (s.contains(QStringLiteral("post/threads")))
        m_postThreadsSpin->setValue(s.value(QStringLiteral("post/threads")).toInt());
    if (s.contains(QStringLiteral("post/ram")))
        m_postRamSpin->setValue(s.value(QStringLiteral("post/ram")).toInt());
    m_postManifestCheck->setChecked(s.value(QStringLiteral("post/manifest"), true).toBool());

    const QByteArray geometry = s.value(QStringLiteral("window/geometry")).toByteArray();
    if (!geometry.isEmpty())
        restoreGeometry(geometry);
}

void MainWindow::saveSettings() const
{
    QSettings s;

    s.setValue(QStringLiteral("form/mode"),
               m_modeBatch->isChecked() ? 3 : (m_modeFete->isChecked() ? 1 : 2));
    if (m_batchPage) {
        s.setValue(QStringLiteral("batch/job"), m_batchPage->saveState());
        // Written with it, never separately: the stamp says "this job comes from
        // a build that also keeps batch/processed honest", and a job without one
        // is not trusted at load time.
        s.setValue(QStringLiteral("batch/jobVersion"), kBatchJobVersion);
    }
    if (m_postBatchPage) {
        s.setValue(QStringLiteral("postbatch/job"), m_postBatchPage->saveState());
        s.setValue(QStringLiteral("postbatch/jobVersion"), kPostBatchJobVersion);
    }
    s.setValue(QStringLiteral("form/dem"), m_demPicker->path());
    s.setValue(QStringLiteral("form/pointsSource"), m_pointsSourceCombo->currentIndex());
    s.setValue(QStringLiteral("form/points"), m_pointsPicker->path());
    s.setValue(QStringLiteral("form/genByTarget"), m_genDensityCombo->currentIndex() == 1);
    s.setValue(QStringLiteral("form/genSpacing"), m_genSpacingSpin->value());
    s.setValue(QStringLiteral("form/genTarget"), m_genTargetSpin->value());
    s.setValue(QStringLiteral("form/genRandom"), m_genArrangementCombo->currentIndex() == 1);
    s.setValue(QStringLiteral("form/genSeed"), m_genSeedSpin->value());
    s.setValue(QStringLiteral("form/genEdgeBuffer"), m_genEdgeSpin->value());
    s.setValue(QStringLiteral("form/genLayerName"), m_genNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/origin"), m_originPicker->path());
    s.setValue(QStringLiteral("form/destinations"), m_destinationsPicker->path());
    s.setValue(QStringLiteral("form/outputDir"), m_outputDirPicker->path());

    s.setValue(QStringLiteral("form/costVector"), m_costVectorPicker->path());
    s.setValue(QStringLiteral("form/polylineBuffer"), m_polylineBufferSpin->value());
    s.setValue(QStringLiteral("form/costRaster"), m_costRasterPicker->path());
    s.setValue(QStringLiteral("form/barrierOn"), m_barrierCheck->isChecked());
    s.setValue(QStringLiteral("form/barrierValue"), m_barrierSpin->value());

    s.setValue(QStringLiteral("form/neighbours"), selectedNeighbours());
    s.setValue(QStringLiteral("form/costFunction"), m_costFunctionCombo->currentData().toInt());
    s.setValue(QStringLiteral("form/slopeCutoff"), m_slopeCapCheck->isChecked());
    s.setValue(QStringLiteral("form/maxSlopeUp"), m_slopeCapUp->value());
    s.setValue(QStringLiteral("form/maxSlopeDown"), m_slopeCapDown->value());
    s.setValue(QStringLiteral("form/smoothing"), m_smoothingSpin->value());
    s.setValue(QStringLiteral("form/threads"), m_threadsSpin->value());
    s.setValue(QStringLiteral("form/ram"), m_ramSpin->value());
    s.setValue(QStringLiteral("form/verbose"), m_verboseCheck->isChecked());
    s.setValue(QStringLiteral("form/manifest"), m_manifestCheck->isChecked());

    s.setValue(QStringLiteral("form/slopeName"), m_slopeNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/costName"), m_costNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/additionalName"), m_additionalNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/totalName"), m_totalNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/densityName"), m_densityNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/pathRasterName"), m_pathRasterNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/pathLinesName"), m_pathLinesNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/corridorName"), m_corridorNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/costCorridor"), m_corridorCheck->isChecked());
    s.setValue(QStringLiteral("form/corridorWidth"), m_corridorWidthSpin->value());

    s.setValue(QStringLiteral("post/input"), m_interpInputPicker->path());
    s.setValue(QStringLiteral("post/outputDir"), m_interpOutputDirPicker->path());
    s.setValue(QStringLiteral("post/threshold"), m_interpThresholdSpin->value());
    s.setValue(QStringLiteral("post/spacing"), m_interpSpacingSpin->value());
    s.setValue(QStringLiteral("post/radius"), m_interpRadiusSpin->value());
    s.setValue(QStringLiteral("post/name"), m_interpNameEdit->text().trimmed());
    s.setValue(QStringLiteral("post/threads"), m_postThreadsSpin->value());
    s.setValue(QStringLiteral("post/ram"), m_postRamSpin->value());
    s.setValue(QStringLiteral("post/manifest"), m_postManifestCheck->isChecked());

    s.setValue(QStringLiteral("window/geometry"), saveGeometry());
}
