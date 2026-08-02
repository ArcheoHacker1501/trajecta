#include "mainwindow.h"

#include "confirmdialog.h"
#include "consoleview.h"
#include "gdalapi.h"
#include "pathpicker.h"
#include "smoothcombobox.h"
#include "systeminfo.h"
#include "thememanager.h"
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
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QPainterPath>
#include <QPalette>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QScreen>
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

// Prevents accidental value changes when the user scrolls the form with the
// mouse wheel over a combo box or spin box that does not have focus.
class WheelGuard : public QObject
{
public:
    using QObject::QObject;

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        if (event->type() == QEvent::Wheel) {
            auto *w = qobject_cast<QWidget *>(watched);
            if (w && !w->hasFocus()) {
                event->ignore();
                return true;  // let the scroll area handle it instead
            }
        }
        return QObject::eventFilter(watched, event);
    }
};

void guardWheel(QWidget *w)
{
    w->setFocusPolicy(Qt::StrongFocus);
    w->installEventFilter(new WheelGuard(w));
}

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

// Minimise / maximise / restore / close, drawn for the same reason as the gear:
// they have to take the active palette's colour. Stroked, not filled, so they
// stay legible at 10 px on both light and dark themes.
enum class WindowGlyph { Minimise, Maximise, Restore, Close };

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

// Small "?" badge whose tooltip carries a short description of the setting.
QLabel *makeHelpDot(const QString &help, QWidget *parent)
{
    auto *dot = new QLabel(QStringLiteral("?"), parent);
    dot->setObjectName(QStringLiteral("HelpDot"));
    dot->setAlignment(Qt::AlignCenter);
    dot->setFixedSize(16, 16);
    dot->setToolTip(help);
    dot->setCursor(Qt::WhatsThisCursor);
    return dot;
}

// Field label followed by a "?" help badge.
QWidget *makeFieldLabel(const QString &text, const QString &help, QWidget *parent)
{
    auto *box = new QWidget(parent);
    auto *layout = new QHBoxLayout(box);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    auto *label = new QLabel(text, box);
    label->setObjectName(QStringLiteral("FieldLabel"));
    layout->addWidget(label);
    layout->addWidget(makeHelpDot(help, box));
    layout->addStretch(1);
    return box;
}

// ---------------------------------------------------------------------------
// Large memory pages
//
// Three independent gates decide whether the engine actually gets 2 MB pages:
//   1. the Windows privilege, granted once per account and picked up only by a
//      NEW logon token, which is why signing out is unavoidable;
//   2. this checkbox;
//   3. the allocation itself, which can still fail on fragmented memory.
// The UI's job is to say which gate the user is standing at.
// ---------------------------------------------------------------------------

enum class LargePageState { NotGranted, GrantedNeedsRelogin, Ready };

// Is SeLockMemoryPrivilege present in *this* process's token? Presence, not
// enabled state: a privilege the account holds is in the token even while off.
bool tokenHasLockMemory()
{
#ifdef Q_OS_WIN
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token))
        return false;
    DWORD needed = 0;
    GetTokenInformation(token, TokenPrivileges, nullptr, 0, &needed);
    QByteArray buf(int(needed), 0);
    bool found = false;
    if (GetTokenInformation(token, TokenPrivileges, buf.data(), needed, &needed)) {
        auto *tp = reinterpret_cast<TOKEN_PRIVILEGES *>(buf.data());
        LUID want{};
        if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &want)) {
            for (DWORD i = 0; i < tp->PrivilegeCount; ++i) {
                if (tp->Privileges[i].Luid.LowPart == want.LowPart
                    && tp->Privileges[i].Luid.HighPart == want.HighPart) {
                    found = true;
                    break;
                }
            }
        }
    }
    CloseHandle(token);
    return found;
#else
    return false;
#endif
}

LargePageState largePageState()
{
#ifdef Q_OS_WIN
    if (tokenHasLockMemory())
        return LargePageState::Ready;
    // The grant is recorded so the app can tell "never asked" apart from
    // "granted, but this session's token predates it".
    if (QSettings().value(QStringLiteral("engine/largePagesGranted"), false).toBool())
        return LargePageState::GrantedNeedsRelogin;
    return LargePageState::NotGranted;
#else
    return LargePageState::NotGranted;
#endif
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
        "<b>How to enable it</b><br>"
        "1. Press <i>Set up…</i> and approve the Windows prompt. This grants "
        "the “Lock pages in memory” privilege to your account, once.<br>"
        "2. <b>Sign out of Windows and back in.</b> This step cannot be "
        "skipped: the privilege list is fixed when you log on, so a new "
        "session is needed before it exists. Restarting Trajecta is not "
        "enough, and neither is running it as administrator.<br>"
        "3. Tick the box. From then on Trajecta runs as a normal program — no "
        "administrator rights are needed for analyses.<br><br>"
        "<b>If it still does not engage</b><br>"
        "A 2 MB page needs 2 MB of physically contiguous memory. On a computer "
        "that has been running for a long time, memory is fragmented and the "
        "request can be refused. Restarting the PC before a large run is the "
        "usual remedy. Trajecta always falls back to normal pages on its own, "
        "and reports what actually happened at the end of the run.");
}

} // namespace

void MainWindow::refreshLargePagesStatus()
{
    if (!m_largePagesCheck)
        return;
    const LargePageState state = largePageState();
    const bool ready = (state == LargePageState::Ready);

    m_largePagesCheck->setEnabled(ready);
    if (!ready)
        m_largePagesCheck->setChecked(false);
    m_largePagesSetup->setVisible(state == LargePageState::NotGranted);

    switch (state) {
    case LargePageState::NotGranted:
        m_largePagesStatus->setText(
            tr("— not available: Windows privilege not granted"));
        break;
    case LargePageState::GrantedNeedsRelogin:
        m_largePagesStatus->setText(
            tr("— granted: sign out of Windows and back in to finish"));
        break;
    case LargePageState::Ready:
        m_largePagesStatus->setText(tr("— available"));
        break;
    }
}

// Grants SeLockMemoryPrivilege to the current account. Needs elevation, so it
// is done by a one-shot elevated helper rather than by elevating Trajecta:
// running the whole application as administrator would break drag & drop from
// Explorer, hide mapped drives, and leave output files with the wrong owner.
void MainWindow::setUpLargePages()
{
#ifdef Q_OS_WIN
    if (!TrajectaUi::confirm(
            this, tr("Enable large memory pages"),
            tr("Windows will ask for administrator approval.\n\n"
               "The \"Lock pages in memory\" privilege will be granted to your "
               "account. You then have to sign out of Windows and back in "
               "before it takes effect.\n\nContinue?")))
        return;

    // ntrights-style grant via PowerShell + secedit: export the current policy,
    // add this account to SeLockMemoryPrivilege, import it back.
    const QString user = qEnvironmentVariable("USERNAME");
    const QString script = QStringLiteral(
        "$ErrorActionPreference='Stop';"
        "$tmp=[IO.Path]::GetTempPath();"
        "$inf=Join-Path $tmp 'tj_sec.inf'; $db=Join-Path $tmp 'tj_sec.sdb';"
        "secedit /export /areas USER_RIGHTS /cfg $inf | Out-Null;"
        "$c=Get-Content $inf;"
        "$sid=(New-Object Security.Principal.NTAccount('%1')).Translate("
        "[Security.Principal.SecurityIdentifier]).Value;"
        "if($c -match 'SeLockMemoryPrivilege'){"
        "$c=$c -replace '(SeLockMemoryPrivilege\\s*=\\s*)(.*)', \"`$1`$2,*$sid\"}"
        "else{$c+=\"SeLockMemoryPrivilege = *$sid\"}"
        "$c|Set-Content $inf -Encoding Unicode;"
        "secedit /configure /db $db /cfg $inf /areas USER_RIGHTS | Out-Null;"
        "Remove-Item $inf,$db -Force -ErrorAction SilentlyContinue")
        .arg(user);

    // ShellExecute with "runas" is what raises the UAC prompt.
    const QString args = QStringLiteral("-NoProfile -ExecutionPolicy Bypass -Command \"%1\"")
                             .arg(QString(script).replace('"', "\\\""));
    const auto res = reinterpret_cast<qintptr>(ShellExecuteW(
        nullptr, L"runas", L"powershell.exe",
        reinterpret_cast<const wchar_t *>(args.utf16()), nullptr, SW_HIDE));

    if (res <= 32) {   // <=32 is the documented failure range, incl. user cancel
        QMessageBox::warning(
            this, tr("Large memory pages"),
            tr("The privilege could not be granted (the request was declined or "
               "failed).\n\nYou can also do it by hand: Local Security Policy → "
               "Local Policies → User Rights Assignment → \"Lock pages in "
               "memory\" → add your account."));
        return;
    }

    QSettings().setValue(QStringLiteral("engine/largePagesGranted"), true);
    refreshLargePagesStatus();
    QMessageBox::information(
        this, tr("Large memory pages"),
        tr("The privilege has been granted to your account.\n\n"
           "Sign out of Windows and back in to finish. Restarting Trajecta is "
           "not enough — the privilege list is fixed when you log on."));
#endif
}

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
    m_pages->addWidget(buildSetupPage());
    m_pages->addWidget(buildRunPage());
    m_pages->addWidget(buildPostPage());
    m_viewer = new ViewerPage(this);
    m_pages->addWidget(m_viewer);
    m_pages->addWidget(buildGuidePage());
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
    switchPage(0);
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

    const QStringList navNames = {tr("Analysis Setup"), tr("Processing"),
                                  tr("Post-processing"), tr("Viewer"),
                                  tr("Guide"), tr("About")};
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
    refreshWindowButtons();

    // Everything painted outside the stylesheet has to be told.
    updateEnvironmentStatus();
    if (m_viewer)
        m_viewer->applyTheme();
    if (m_runUi.console)
        m_runUi.console->applyTheme();
    if (m_postUi.console)
        m_postUi.console->applyTheme();
    // Only ever a GuideBrowser; the type is file-local so it cannot be named
    // in the header, and it carries no Q_OBJECT for qobject_cast to key on.
    if (m_guideBrowser)
        static_cast<GuideBrowser *>(m_guideBrowser)->relayout(false, true);
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

// ---------------------------------------------------------------------------
// Bottom status bar (environment indicators + locate actions)
// ---------------------------------------------------------------------------

QWidget *MainWindow::buildStatusBar()
{
    auto *bar = new QFrame(this);
    bar->setObjectName(QStringLiteral("StatusBar"));
    bar->setFixedHeight(36);

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

    layout->addStretch(1);

    auto *locateEngineBtn = new QPushButton(tr("Locate engine..."), bar);
    locateEngineBtn->setObjectName(QStringLiteral("LinkButton"));
    locateEngineBtn->setCursor(Qt::PointingHandCursor);
    connect(locateEngineBtn, &QPushButton::clicked, this, &MainWindow::locateEngine);
    layout->addWidget(locateEngineBtn);

    auto *locateGdalBtn = new QPushButton(tr("Locate GDAL folder..."), bar);
    locateGdalBtn->setObjectName(QStringLiteral("LinkButton"));
    locateGdalBtn->setCursor(Qt::PointingHandCursor);
    connect(locateGdalBtn, &QPushButton::clicked, this, &MainWindow::locateGdal);
    layout->addWidget(locateGdalBtn);

    return bar;
}

void MainWindow::switchPage(int index)
{
    m_pages->setCurrentIndex(index);
    if (index >= 0 && index < m_navButtons.size())
        m_navButtons.at(index)->setChecked(true);
}

// ---------------------------------------------------------------------------
// Cards helper
// ---------------------------------------------------------------------------

QWidget *MainWindow::makeCard(const QString &title, const QString &subtitle, QWidget *content)
{
    auto *card = new QFrame(this);
    card->setObjectName(QStringLiteral("Card"));

    auto *layout = new QVBoxLayout(card);
    layout->setContentsMargins(18, 14, 18, 16);
    layout->setSpacing(6);

    auto *titleLabel = new QLabel(title, card);
    titleLabel->setObjectName(QStringLiteral("CardTitle"));
    layout->addWidget(titleLabel);

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

    // ----- Mode selector -----
    {
        auto *content = new QWidget(inner);
        auto *row = new QHBoxLayout(content);
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
               "Computes optimal routes from one origin to one or more destinations"),
            content);
        m_modeLcpa->setToolTip(
            tr("Computes the optimal routes from a single origin point to one or "
               "more destinations: paths raster and polyline shapefile."));
        for (QPushButton *b : {m_modeFete, m_modeLcpa}) {
            b->setObjectName(QStringLiteral("ModeCard"));
            b->setCheckable(true);
            b->setCursor(Qt::PointingHandCursor);
            b->setMinimumHeight(72);
            // Share the row equally even if one caption is longer.
            b->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
            row->addWidget(b, 1);
        }
        auto *modeGroup = new QButtonGroup(content);
        modeGroup->setExclusive(true);
        modeGroup->addButton(m_modeFete);
        modeGroup->addButton(m_modeLcpa);
        m_modeFete->setChecked(true);
        connect(modeGroup, &QButtonGroup::buttonClicked, this,
                [this](QAbstractButton *) { updateModeUi(); });

        layout->addWidget(makeCard(tr("Analysis mode"),
                                   tr("Choose what Trajecta should compute."), content));
    }

    // ----- Input data -----
    {
        auto *content = new QWidget(inner);
        auto *grid = new QGridLayout(content);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);

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
               tr("Digital Elevation Model in GeoTIFF format. Must be "
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
                      "analysis, so the exact input stays on disk."),
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
                             "cell, which is only realistic on very small rasters."));

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
                                     tr("Vector file with one or more points: the targets the "
                                        "optimal routes are computed to."),
                                     m_destinationsPicker);

        m_outputDirPicker = new PathPicker(PathPicker::Kind::Directory,
                                           tr("Select the output folder"), QString(), content);
        m_outputDirPicker->setPlaceholder(tr("Folder where all results will be written"));
        addRow(tr("Output folder"),
               tr("Folder where every result file (rasters and shapefiles) "
                  "will be written."),
               m_outputDirPicker);

        layout->addWidget(makeCard(
            tr("Input data"),
            tr("The DEM and every vector file must share the same coordinate "
               "reference system, and all points must fall inside the DEM extent."),
            content));
    }

    // ----- Cost modifiers -----
    {
        m_modifiersGroup = new QGroupBox(tr("Use cost modifiers in this analysis"), inner);
        m_modifiersGroup->setCheckable(true);
        m_modifiersGroup->setChecked(false);

        auto *grid = new QGridLayout(m_modifiersGroup);
        // Generous top margin: the checkable title is drawn inside the group
        // box area and must not overlap the first row.
        grid->setContentsMargins(8, 44, 8, 10);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);

        int r = 0;
        // The description lives only on the "?" badge, not on the input itself.
        auto addRow = [&](const QString &label, QWidget *w, const QString &tip) {
            QWidget *l = makeFieldLabel(label, tip, m_modifiersGroup);
            grid->addWidget(l, r, 0);
            grid->addWidget(w, r, 1);
            ++r;
        };

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
                  "2 cells per side is safe for 16-connectivity."));

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

        layout->addWidget(makeCard(
            tr("Cost modifiers (optional)"),
            tr("Increase traversal costs over specific features such as rivers, "
               "restricted areas or difficult terrain."),
            groupHolder));
    }

    // ----- Algorithm -----
    {
        auto *content = new QWidget(inner);
        auto *grid = new QGridLayout(content);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);

        int r = 0;
        // The description lives only on the "?" badge, not on the input itself.
        auto addRow = [&](const QString &label, QWidget *w, const QString &tip) {
            QWidget *l = makeFieldLabel(label, tip, content);
            grid->addWidget(l, r, 0);
            grid->addWidget(w, r, 1);
            ++r;
        };

        m_neighboursCombo = new SmoothComboBox(content);
        m_neighboursCombo->addItem(tr("8 — basic (3×3 grid)"), 8);
        m_neighboursCombo->addItem(tr("16 — knight moves (recommended)"), 16);
        m_neighboursCombo->addItem(tr("24 — extended"), 24);
        m_neighboursCombo->addItem(tr("32 — more extended"), 32);
        m_neighboursCombo->addItem(tr("64 — full extended"), 64);
        m_neighboursCombo->setCurrentIndex(1);
        addRow(tr("Neighbours"), m_neighboursCombo,
               tr("Connectivity used when building the cumulative cost surface. "
                  "More neighbours = smoother, straighter paths but slower runs."));

        m_costFunctionCombo = new SmoothComboBox(content);
        m_costFunctionCombo->addItem(tr("Modified Tobler's Hiking Function (White 2015) — recommended"), 1);
        m_costFunctionCombo->addItem(tr("Márquez-Pérez et al. (2017)"), 2);
        m_costFunctionCombo->addItem(tr("Irmischer & Clarke (2017)"), 3);
        addRow(tr("Cost function"), m_costFunctionCombo,
               tr("Anisotropic walking-cost model applied to the terrain slope. "
                  "The slope unit (degrees or percent) is set automatically."));

        m_smoothingSpin = new QSpinBox(content);
        m_smoothingSpin->setRange(0, 10);
        m_smoothingSpin->setValue(0);
        m_smoothingSpin->setSuffix(tr(" cell(s) per side"));
        addRow(tr("Path smoothing buffer"), m_smoothingSpin,
               tr("Buffer applied around computed paths when accumulating the "
                  "result. 0 keeps the raw single-cell paths."));

        layout->addWidget(makeCard(tr("Algorithm"), QString(), content));
    }

    // ----- Hardware resources -----
    {
        auto *content = new QWidget(inner);
        auto *grid = new QGridLayout(content);
        grid->setContentsMargins(0, 0, 0, 0);
        grid->setHorizontalSpacing(12);
        grid->setVerticalSpacing(16);
        grid->setColumnStretch(1, 1);

        const int maxThreads = qMax(1, QThread::idealThreadCount());
        const int recommendedThreads = qMax(1, maxThreads - 4);
        const qint64 totalRam = SystemInfo::totalRamMb();
        const int recommendedRam = int(qMax<qint64>(1024, (totalRam * 60) / 100));

        int r = 0;
        auto addRow = [&](const QString &label, const QString &help, QWidget *w,
                          const QString &hint) {
            QWidget *l = makeFieldLabel(label, help, content);
            grid->addWidget(l, r, 0);
            auto *rowWidget = new QWidget(content);
            auto *rowLayout = new QHBoxLayout(rowWidget);
            rowLayout->setContentsMargins(0, 0, 0, 0);
            rowLayout->setSpacing(10);
            rowLayout->addWidget(w);
            auto *hintLabel = new QLabel(hint, content);
            hintLabel->setObjectName(QStringLiteral("HintLabel"));
            rowLayout->addWidget(hintLabel, 1);
            grid->addWidget(rowWidget, r, 1);
            ++r;
        };

        m_threadsSpin = new QSpinBox(content);
        m_threadsSpin->setRange(1, maxThreads);
        m_threadsSpin->setValue(recommendedThreads);
        addRow(tr("CPU threads"),
               tr("Number of parallel CPU threads used for the computation. "
                  "Keeping a few cores free preserves system responsiveness."),
               m_threadsSpin,
               tr("%1 available — %2 recommended (keeps 4 cores for the system)")
                   .arg(maxThreads)
                   .arg(recommendedThreads));

        m_ramSpin = new QSpinBox(content);
        m_ramSpin->setRange(512, int(totalRam));
        m_ramSpin->setSingleStep(512);
        m_ramSpin->setSuffix(QStringLiteral(" MB"));
        m_ramSpin->setValue(recommendedRam);
        addRow(tr("Maximum RAM"),
               tr("Memory ceiling used for raster processing. About 60 percent "
                  "of the installed RAM is a safe value."),
               m_ramSpin,
               tr("%1 MB installed — about 60 percent is recommended").arg(totalRam));

        // ----- Large memory pages -----
        // Belongs here rather than in the appearance menu: it is an execution
        // parameter of the engine, like the two spin boxes above it.
        m_largePagesCheck =
            new QCheckBox(tr("Use large memory pages (advanced)"), content);
        auto *lpRow = new QWidget(content);
        auto *lpLayout = new QHBoxLayout(lpRow);
        lpLayout->setContentsMargins(0, 0, 0, 0);
        lpLayout->setSpacing(6);
        lpLayout->addWidget(m_largePagesCheck);
        lpLayout->addWidget(makeHelpDot(largePagesHelpText(), lpRow));

        m_largePagesStatus = new QLabel(lpRow);
        m_largePagesStatus->setObjectName(QStringLiteral("HintLabel"));
        lpLayout->addWidget(m_largePagesStatus);

        m_largePagesSetup = new QPushButton(tr("Set up…"), lpRow);
        m_largePagesSetup->setCursor(Qt::PointingHandCursor);
        connect(m_largePagesSetup, &QPushButton::clicked,
                this, &MainWindow::setUpLargePages);
        lpLayout->addWidget(m_largePagesSetup);
        lpLayout->addStretch(1);
        grid->addWidget(lpRow, r, 1);
        ++r;
        refreshLargePagesStatus();

        m_verboseCheck = new QCheckBox(
            tr("Detailed debug output (verbose console log)"), content);
        auto *verboseRow = new QWidget(content);
        auto *verboseLayout = new QHBoxLayout(verboseRow);
        verboseLayout->setContentsMargins(0, 0, 0, 0);
        verboseLayout->setSpacing(6);
        verboseLayout->addWidget(m_verboseCheck);
        verboseLayout->addWidget(makeHelpDot(
            tr("Prints detailed diagnostic messages in the console log. "
               "Useful for troubleshooting and bug reports."),
            verboseRow));
        verboseLayout->addStretch(1);
        grid->addWidget(verboseRow, r, 1);

        layout->addWidget(makeCard(tr("Hardware resources"), QString(), content));
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

        int r = 0, c = 0;
        auto addName = [&](const QString &label, const QString &defaultName,
                           const QString &tip) {
            QWidget *l = makeFieldLabel(label, tip, content);
            auto *e = new QLineEdit(defaultName, content);
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
                    tr("Main LCPA result: polyline shapefile of the optimal routes."));
        std::tie(m_densityNameLabel, m_densityNameEdit) =
            addName(tr("Density raster"), QStringLiteral("FETE_density"),
                    tr("Main FETE result: accumulated path-usage density."));

        layout->addWidget(makeCard(
            tr("Output files"),
            tr("Names only, without extension — everything is written inside the "
               "output folder."),
            content));
    }

    // ----- Run bar -----
    {
        // A bare layout, not a QWidget wrapper: a wrapper inherits the opaque
        // base background and shows up as a slab behind the button on any
        // theme whose pages are transparent.
        auto *row = new QHBoxLayout;
        row->setContentsMargins(0, 4, 0, 0);
        row->addStretch(1);

        m_runButton = new QPushButton(tr("▶  Run analysis"), inner);
        m_runButton->setObjectName(QStringLiteral("RunButton"));
        m_runButton->setCursor(Qt::PointingHandCursor);
        m_runButton->setMinimumSize(220, 46);
        connect(m_runButton, &QPushButton::clicked, this, &MainWindow::startRun);
        row->addWidget(m_runButton);

        layout->addLayout(row);
    }

    layout->addStretch(1);
    scroll->setWidget(inner);
    pageLayout->addWidget(scroll);

    // Scrolling the form must never silently change values.
    for (QWidget *w : std::initializer_list<QWidget *>{
             m_polylineBufferSpin, m_barrierSpin, m_neighboursCombo,
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

    return page;
}

void MainWindow::updateModeUi()
{
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
        if (!api.isLoaded()) {
            const GdalEnvironment env = detectGdalEnvironment();
            QStringList dirs;
            const QString engine = engineExePath();
            if (!engine.isEmpty())
                dirs << QFileInfo(engine).absolutePath();
            if (!env.binDir.isEmpty())
                dirs << env.binDir;
            dirs.removeDuplicates();
            api.load(dirs, env.projData, env.gdalData);
        }
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

QWidget *MainWindow::buildRunPanel(RunUi &ui, QWidget *parent,
                                   const QString &idlePhrase, QWidget *leadingButton)
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
    ui.chip->setFixedHeight(26);
    ui.chip->setMinimumWidth(110);
    statusRow->addWidget(ui.chip);

    ui.phase = new QLabel(idlePhrase, panel);
    ui.phase->setObjectName(QStringLiteral("PhaseLabel"));
    statusRow->addWidget(ui.phase, 1);

    ui.elapsed = new QLabel(QStringLiteral("0:00:00"), panel);
    ui.elapsed->setObjectName(QStringLiteral("ElapsedLabel"));
    statusRow->addWidget(ui.elapsed);

    layout->addLayout(statusRow);

    ui.progress = new QProgressBar(panel);
    ui.progress->setRange(0, 1000);
    ui.progress->setValue(0);
    ui.progress->setTextVisible(true);
    ui.progress->setFormat(QStringLiteral("%p%"));
    ui.progress->setFixedHeight(20);
    layout->addWidget(ui.progress);

    ui.console = new ConsoleView(panel);
    layout->addWidget(ui.console, 1);

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

    ui.openFolderButton = new QPushButton(tr("Open output folder"), panel);
    ui.openFolderButton->setCursor(Qt::PointingHandCursor);
    ui.openFolderButton->setEnabled(false);
    connect(ui.openFolderButton, &QPushButton::clicked, this, &MainWindow::openOutputFolder);
    buttonRow->addWidget(ui.openFolderButton);

    ui.pauseButton = new QPushButton(tr("Pause"), panel);
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

QWidget *MainWindow::buildRunPage()
{
    auto *page = new QWidget(this);
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(28, 24, 28, 24);

    auto *backButton = new QPushButton(tr("‹ Back to setup"), page);
    backButton->setCursor(Qt::PointingHandCursor);
    connect(backButton, &QPushButton::clicked, this, [this] { switchPage(0); });

    layout->addWidget(buildRunPanel(
        m_runUi, page,
        tr("Configure the analysis and press “Run analysis”."), backButton));
    return page;
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
                  "the surface, keeping only the broad corridor structure."),
               m_interpSpacingSpin);

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

        layout->addWidget(makeCard(
            tr("Natural Neighbour Interpolation (NNI)"),
            tr("Turns the sparse FETE density raster into a smooth, continuous "
               "surface with discrete Sibson interpolation: each cell receives "
               "the weighted average of the sample cells whose influence area "
               "it would claim. The result keeps the sample values exactly and "
               "transitions smoothly between corridors."),
            content));
    }

    // Run bar
    {
        auto *row = new QHBoxLayout;
        row->setContentsMargins(0, 4, 0, 0);
        row->addStretch(1);

        m_runInterpButton = new QPushButton(tr("▶  Run interpolation"), inner);
        m_runInterpButton->setObjectName(QStringLiteral("RunButton"));
        m_runInterpButton->setCursor(Qt::PointingHandCursor);
        m_runInterpButton->setMinimumSize(220, 46);
        connect(m_runInterpButton, &QPushButton::clicked, this, &MainWindow::startInterpRun);
        row->addWidget(m_runInterpButton);

        layout->addLayout(row);
    }

    // Live run panel: now in the same scrolling column as the parameters,
    // instead of a separately-scrolled block pinned below.
    layout->addSpacing(10);
    layout->addWidget(buildRunPanel(
        m_postUi, inner,
        tr("Set the parameters and press “Run interpolation”.")));
    m_postUi.console->setMinimumHeight(320);

    layout->addStretch(1);
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
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(28, 24, 28, 24);

    auto *browser = new GuideBrowser(page);
    browser->setObjectName(QStringLiteral("GuideBrowser"));
    browser->setOpenExternalLinks(true);
    // Not tr(): a translation would have to reproduce the %HALF%/%FULL%
    // placeholders verbatim or silently break the figure layout.
    browser->setTemplate(QStringLiteral(R"HTML(
<style>
 h2 { color:%H2%; } h3 { color:%H3%; }
 a { color:%LINK%; }
 td, th { padding: 3px 10px; }
</style>
<h1>Overview</h1>
<p>Trajecta is a least-cost analysis software specifically developed for users with only a basic computer science background. <b>Be patient, this software is currently under development and can contain bugs or errors</b>. Please, contact me for bug reporting, problems during the installation, improvements or additional features you would like to see developed and included in future releases (see About for contacs).</p>

<h2>Core Functions of Trajecta</h2>
<p>Currently, Trajecta provides two complementary workflows for movement modeling (FETE and LCPA, see below). Both modes use anisotropic cost functions (e.g. Modified Tobler's Hiking Function) and support cost surface modifiers (e.g. waterbodies).</p>

<h3>FETE — From Everywhere To Everywhere</h3>
<p>As analysis model, FETE was originally conceptualized by White &amp; Barber (2012). The FETE algorithm implemented by Trajecta allows to calculate a high number of least-cost paths connecting every point to every other point of a regular or randomly scattered point grid. To compute LCPs, DEM or other elevation based data can be used to calculate slope which can then be transformed using cost functions (e.g. Tobler's Hiking Function). Additional costs can be added as for waterbodies or terrain indexes. 

Based on all the LCPs calculated, overlap analysis is then computed and a density raster is created, with the same resolution as the input DEM. Different color gradients can be used to display most probable paths among all calculated LCPs. </p>

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

<h3>LCPA — Least-Cost Path Analysis</h3>
<p>For an introduction to Least-Cost Path Analysis (LCPA), see White (2015). Trajecta allows high-speed computation of Least-Cost Paths (LCPs) between a single origin and one or more destinations. To compute LCPs, DEM or other elevation based data can be used to calculate slope which can then be transformed using cost functions (e.g. Modified Tobler's Hiking Function). Additional costs can be added as for waterbodies or terrain indexes. </p>

<table border="0" cellspacing="10" width="100%">
<tr><td align="center"><img src="guide:lcpa" width="%FULL%"></td></tr>
<tr><td align="center"><i>Least-Cost Paths from single origin to multiple destinations calculated using
Trajecta and SRTM 30m DEM.</i></td></tr>
</table>

<h2>Post-processing: NNI — Natural Neighbour Interpolation</h2>
<p>The <b>Post-processing</b> page turns a FETE density raster into a smooth,
continuous surface using <b>discrete Sibson (natural neighbour) interpolation (Park et al. 2006)</b>.
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

<h2>Sample points: imported or generated</h2>
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

<h2>Input requirements</h2>
<p>Trajecta allows for different types of inputs and file formats:</p>
<table border="0" cellspacing="0">
<tr><th align="left">Input</th><th align="left">Requirements</th></tr>
<tr><td><b>DEM</b></td><td>GeoTIFF (.tif/.tiff), georeferenced, with a CRS.</td></tr>
<tr><td><b>Points</b></td><td>.shp, .geojson/.json, .kml, .gml/.xml or .csv
    (coordinate columns named x/y, lon/lat or easting/northing).
    In FETE mode they can be generated from the DEM instead.</td></tr>
<tr><td><b>Vector modifiers</b></td><td>Polylines with a float <b>cost</b> field
    holding the multiplier; for .csv the geometry must be in a WKT column.</td></tr>
<tr><td><b>Raster modifiers</b></td><td>GeoTIFF with the same dimensions as the DEM;
    cell values are multipliers (1.0 = unchanged, 2.0 = double cost).</td></tr>
</table>

<h2>Cost modifiers &amp; barriers</h2>
<p>Modifiers let you make specific features (water bodies, restricted areas, specific types of terrain) more
expensive to cross. The <b>polyline buffer</b> widens rasterized lines so paths cannot
slip diagonally across them. The <b>barrier threshold</b> turns extreme multipliers
(e.g. 999999) into hard barriers: cells at or above the threshold are excluded from
movement, which also keeps the computation fast.</p>

<h2>Algorithm parameters</h2>
<p><b>Neighbours</b> — connectivity of the search grid (8, 16, 24, 32, 64). Higher
values allow finer path angles at the price of speed. A connectivity radius of 16 (Knigth's Move) is the usual choice.</p>
<p><b>Cost function</b> — the anisotropic hiking model applied to slope. Currently, the following cost function have been implemented in Trajecta:</p>
<ul>
<li>Modified Tobler hiking function (White 2015);</li>
<li>M&aacute;rquez-P&eacute;rez et al. (2017);</li>
<li>Irmischer &amp; Clarke (2017).</li>
</ul>
<p><b>Path smoothing buffer</b> — buffer in cells applied around each computed path
when accumulating results.This function allows to calculate wider paths, thus maximising the possibility that actual, historical path(s) were located inside the modeled high-traversability areas.</p>

<h2>Outputs</h2>
<p><b>Both modes:</b> slope raster, base cost surface, and (with modifiers) the
additional and total cost surfaces.<br/>
<b>FETE:</b> the path-density raster, plus the sample points shapefile when the
points were generated from the DEM.<br/>
<b>LCPA:</b> the paths raster and the paths polyline shapefile.</p>

<h2>GDAL requirement</h2>
<p>The Trajecta engine relies on the <b>GDAL</b> geospatial libraries, installed
through <a href="https://trac.osgeo.org/osgeo4w">OSGeo4W</a>. Trajecta Studio finds a
standard OSGeo4W installation (C:\OSGeo4W or C:\OSGeo4W64) automatically — no manual
PATH configuration is needed. If GDAL lives elsewhere, use <b>Locate GDAL folder</b>
in the sidebar and select your <code>OSGeo4W\bin</code> directory. The status is shown
at the bottom of the sidebar.</p>

<h2>References</h2>
<p>Irmischer, I. J., &amp; Clarke, K. C. (2017). Measuring and modeling the
speed of human navigation. <i>Cartography and Geographic Information Science</i>,
45(2), 177&ndash;186.
<a href="https://doi.org/10.1080/15230406.2017.1292150">doi:10.1080/15230406.2017.1292150</a></p>
<p>M&aacute;rquez-P&eacute;rez, J., Vallejo-Villalta, I., &amp;
&Aacute;lvarez-Francoso, J. I. (2017). Estimated travel time for walking trails
in natural areas. <i>Geografisk Tidsskrift&ndash;Danish Journal of Geography</i>,
117(1), 53&ndash;62.
<a href="https://doi.org/10.1080/00167223.2017.1316212">doi:10.1080/00167223.2017.1316212</a></p>
<p>Park, S. W., Linsen, L., Kreylos, O., Owens, J. D., &amp; Hamann, B. (2006).
Discrete Sibson interpolation. <i>IEEE Transactions on Visualization and
Computer Graphics</i>, 12(2), 243&ndash;253.
<a href="https://doi.org/10.1109/TVCG.2006.27">doi:10.1109/TVCG.2006.27</a></p>
<p>White, D. A. (2015). The Basics of Least Cost Analysis for Archaeological
Applications. <i>Advances in Archaeological Practice</i>, 3(4), 407&ndash;414.
<a href="https://doi.org/10.7183/2326-3768.3.4.407">doi:10.7183/2326-3768.3.4.407</a></p>
<p>White, D. A., &amp; Barber, S. B. (2012). Geospatial modeling of pedestrian
transportation networks: A case study from precolumbian Oaxaca, Mexico.
<i>Journal of Archaeological Science</i>, 39(8), 2684&ndash;2696.
<a href="https://doi.org/10.1016/j.jas.2012.04.017">doi:10.1016/j.jas.2012.04.017</a></p>
)HTML"));

    // Figures, in the order they appear above. The second argument marks the
    // ones that span the full text column; the rest share a row with a sibling
    // and are laid out at half the width.
    browser->addFigure(QStringLiteral("grid"),
                       QStringLiteral(":/assets/guide/Grid_FETE.jpg"), false);
    browser->addFigure(QStringLiteral("unfiltered"),
                       QStringLiteral(":/assets/guide/unfiltered_FETE.jpg"), false);
    browser->addFigure(QStringLiteral("filtered"),
                       QStringLiteral(":/assets/guide/filtered_FETE.jpg"), true);
    browser->addFigure(QStringLiteral("lcpa"),
                       QStringLiteral(":/assets/guide/LCPA.jpg"), true);
    browser->addFigure(QStringLiteral("density"),
                       QStringLiteral(":/assets/guide/FETE_density.jpg"), false);
    browser->addFigure(QStringLiteral("nni"),
                       QStringLiteral(":/assets/guide/FETE_density_NNI.jpg"), false);

    layout->addWidget(browser);
    m_guideBrowser = browser;
    return page;
}

QWidget *MainWindow::buildAboutPage()
{
    auto *page = new QWidget(this);
    auto *pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(28, 24, 28, 24);

    // One panel for the whole page, like the Guide, rather than each label
    // drawing its own background: on a theme with a picture behind it, loose
    // labels read as a stack of unrelated strips.
    auto *card = new QFrame(page);
    card->setObjectName(QStringLiteral("Card"));
    pageLayout->addWidget(card);

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

    auto *body = new QLabel(card);
    body->setObjectName(QStringLiteral("AboutBody"));
    body->setAlignment(Qt::AlignHCenter);
    body->setWordWrap(true);
    body->setOpenExternalLinks(true);
    body->setTextFormat(Qt::RichText);
    {
        QPalette pal = body->palette();
        pal.setColor(QPalette::Link, QColor(0x7e, 0xa8, 0xa0));
        body->setPalette(pal);
    }
    // No inline colours: a hard-coded grey is invisible on a light theme and
    // is the one thing the palette mapping cannot reach. Everything here
    // inherits AboutBody, so it follows the theme like the rest of the text.
    body->setText(tr(
        "<p><b>Trajecta</b> is a software by <b>Stefano Aprà</b></p>"
        "<p><a href=\"%1\">%1</a></p>"
        "<p>License: GPL-3.0 • Powered by Qt and GDAL</p>"
        "<p>If you use Trajecta in your research, please cite:<br/>"
        "<i>Stefano Aprà, Ph.D. candidate — (<a href=\"https://isaw.nyu.edu/people/students/stefano-apra\">Institute for the Study of the Ancient World at New York University</a>)</i></p>")
                      .arg(QString::fromLatin1(kProjectUrl)));
    layout->addWidget(body);

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
    const QStringList pathDirs = pathValue.split(QLatin1Char(';'), Qt::SkipEmptyParts);
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

void MainWindow::updateEnvironmentStatus()
{
    const QString engine = engineExePath();
    if (engine.isEmpty()) {
        m_engineStatus->setText(tr("⚠ Engine not found"));
        m_engineStatus->setStyleSheet(QStringLiteral("color:%1;").arg(ThemeManager::mapped("#cf7f7f").name()));
        m_engineStatus->setToolTip(tr("trajecta.exe was not found next to the "
                                      "interface. Use \"Locate engine\" below."));
    } else {
        m_engineStatus->setText(tr("✓ Engine ready"));
        m_engineStatus->setStyleSheet(QStringLiteral("color:%1;").arg(ThemeManager::mapped("#7fb08a").name()));
        m_engineStatus->setToolTip(QDir::toNativeSeparators(engine));
    }

    const GdalEnvironment gdal = detectGdalEnvironment();
    if (!gdal.found) {
        m_gdalStatus->setText(tr("⚠ GDAL not found"));
        m_gdalStatus->setStyleSheet(QStringLiteral("color:%1;").arg(ThemeManager::mapped("#cf7f7f").name()));
        m_gdalStatus->setToolTip(tr("Install GDAL through OSGeo4W (see the Guide "
                                    "page) or use \"Locate GDAL folder\" below."));
    } else if (gdal.binDir.isEmpty()) {
        m_gdalStatus->setText(tr("✓ GDAL ready"));
        m_gdalStatus->setStyleSheet(QStringLiteral("color:%1;").arg(ThemeManager::mapped("#7fb08a").name()));
        m_gdalStatus->setToolTip(tr("GDAL libraries are already reachable (bundled "
                                    "with the engine or on the system PATH)."));
    } else {
        m_gdalStatus->setText(tr("✓ GDAL detected"));
        m_gdalStatus->setStyleSheet(QStringLiteral("color:%1;").arg(ThemeManager::mapped("#7fb08a").name()));
        m_gdalStatus->setToolTip(tr("Using %1 (added to PATH automatically for the engine).")
                                     .arg(QDir::toNativeSeparators(gdal.binDir)));
    }

    configureViewerGdal();
}

void MainWindow::configureViewerGdal()
{
    if (!m_viewer)
        return;

    // Every folder that may hold gdal*.dll, most specific first. The viewer
    // loads the GDAL C API dynamically from the first one that works.
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
            .split(QLatin1Char(';'), Qt::SkipEmptyParts);
    for (const QString &dir : pathDirs) {
        if (dirHasGdal(dir)) {
            dirs << dir;
            break;
        }
    }
    dirs.removeDuplicates();

    m_viewer->configureGdal(dirs, gdal.projData, gdal.gdalData);
}

void MainWindow::viewerLoadFile(const QString &path)
{
    switchPage(3);
    static const QStringList kVectorSuffixes = {
        QStringLiteral("shp"), QStringLiteral("geojson"), QStringLiteral("json"),
        QStringLiteral("kml"), QStringLiteral("gml")};
    if (kVectorSuffixes.contains(QFileInfo(path).suffix().toLower())) {
        m_viewer->registerVectorOverlay(QFileInfo(path).completeBaseName(), path);
        return;
    }
    m_viewer->openRasterFile(path);
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

    QStringList names = {m_slopeNameEdit->text(), m_costNameEdit->text()};
    if (m_modifiersGroup->isChecked() && !m_costVectorPicker->path().isEmpty())
        names << m_additionalNameEdit->text() << m_totalNameEdit->text();
    if (fete)
        names << m_densityNameEdit->text();
    else
        names << m_pathRasterNameEdit->text() << m_pathLinesNameEdit->text();
    for (const QString &name : names) {
        if (!isValidFileName(name))
            return tr("Output file names cannot be empty or contain "
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
    p.maxThreads = m_threadsSpin->value();
    p.maxRamMb = m_ramSpin->value();
    // Only ever true when the privilege is actually in this session's token:
    // the box is disabled and cleared otherwise (refreshLargePagesStatus).
    p.largePages = m_largePagesCheck && m_largePagesCheck->isChecked();

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

    p.neighbours = m_neighboursCombo->currentData().toInt();
    p.costFunction = m_costFunctionCombo->currentData().toInt();
    p.smoothingBufferRadius = m_smoothingSpin->value();

    p.slopeName = m_slopeNameEdit->text().trimmed();
    p.costName = m_costNameEdit->text().trimmed();
    p.additionalCostName = m_additionalNameEdit->text().trimmed();
    p.totalCostName = m_totalNameEdit->text().trimmed();
    p.densityName = m_densityNameEdit->text().trimmed();
    p.pathRasterName = m_pathRasterNameEdit->text().trimmed();
    p.pathLinesName = m_pathLinesNameEdit->text().trimmed();

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
    if (m_runner->isRunning())
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
    if (m_runner->isRunning()) {
        QMessageBox::information(this, tr("Analysis running"),
                                 tr("An analysis is already running. Wait for it "
                                    "to finish or cancel it first."));
        switchPage(m_activeUi == &m_postUi ? 2 : 1);
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
    if (m_runner->isRunning()) {
        QMessageBox::information(this, tr("Analysis running"),
                                 tr("An analysis is already running. Wait for it "
                                    "to finish or cancel it first."));
        switchPage(m_activeUi == &m_postUi ? 2 : 1);
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
    params.interpMaxRadius = m_interpRadiusSpin->value();
    params.interpOutputName = m_interpNameEdit->text().trimmed();

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

void MainWindow::beginRun(const TrajectaRunner::Parameters &params)
{
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

    switchPage(interp ? 2 : 1);
    m_runner->start(params);
}

void MainWindow::onRunFinished(TrajectaRunner::Outcome outcome, const QString &report)
{
    RunUi &ui = *m_activeUi;

    m_elapsedTimer->stop();
    ui.elapsed->setText(formatElapsed(m_elapsed.elapsed() - m_pausedMs));
    ui.pauseButton->setEnabled(false);
    ui.pauseButton->setText(tr("Pause"));
    ui.cancelButton->setEnabled(false);
    m_runButton->setEnabled(true);
    m_runInterpButton->setEnabled(true);
    m_genPointsButton->setEnabled(true);

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
        switchPage(3);
}

void MainWindow::onPauseStateChanged(bool paused)
{
    RunUi &ui = *m_activeUi;

    if (paused) {
        m_pauseClock.start();
        m_elapsedTimer->stop();
        ui.pauseButton->setText(tr("▶ Resume"));
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
        if (auto *browser = m_pages->currentWidget()->findChild<QTextBrowser *>())
            area = browser;
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

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (m_runner->isRunning()) {
        const auto answer = QMessageBox::question(
            this, tr("Analysis running"),
            tr("An analysis is still running. Stop it and quit?"));
        if (answer != QMessageBox::Yes) {
            event->ignore();
            return;
        }
        m_runner->cancel();
    }
    saveSettings();
    event->accept();
}

// ---------------------------------------------------------------------------
// Settings persistence
// ---------------------------------------------------------------------------

void MainWindow::loadSettings()
{
    QSettings s;

    if (s.value(QStringLiteral("form/mode"), 1).toInt() == 2)
        m_modeLcpa->setChecked(true);
    else
        m_modeFete->setChecked(true);

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

    const int neighbours = s.value(QStringLiteral("form/neighbours"), 16).toInt();
    const int nIdx = m_neighboursCombo->findData(neighbours);
    m_neighboursCombo->setCurrentIndex(nIdx >= 0 ? nIdx : 1);
    const int cf = s.value(QStringLiteral("form/costFunction"), 1).toInt();
    const int cfIdx = m_costFunctionCombo->findData(cf);
    m_costFunctionCombo->setCurrentIndex(cfIdx >= 0 ? cfIdx : 0);
    m_smoothingSpin->setValue(s.value(QStringLiteral("form/smoothing"), 0).toInt());

    if (s.contains(QStringLiteral("form/threads")))
        m_threadsSpin->setValue(s.value(QStringLiteral("form/threads")).toInt());
    if (s.contains(QStringLiteral("form/ram")))
        m_ramSpin->setValue(s.value(QStringLiteral("form/ram")).toInt());
    m_verboseCheck->setChecked(s.value(QStringLiteral("form/verbose"), false).toBool());

    auto loadName = [&s](QLineEdit *edit, const char *key) {
        const QString v = s.value(QString::fromLatin1(key)).toString().trimmed();
        if (!v.isEmpty())
            edit->setText(v);
    };
    loadName(m_slopeNameEdit, "form/slopeName");
    loadName(m_costNameEdit, "form/costName");
    loadName(m_additionalNameEdit, "form/additionalName");
    loadName(m_totalNameEdit, "form/totalName");
    loadName(m_densityNameEdit, "form/densityName");
    loadName(m_pathRasterNameEdit, "form/pathRasterName");
    loadName(m_pathLinesNameEdit, "form/pathLinesName");
    loadName(m_genNameEdit, "form/genLayerName");

    m_interpInputPicker->setPath(s.value(QStringLiteral("post/input")).toString());
    m_interpOutputDirPicker->setPath(s.value(QStringLiteral("post/outputDir")).toString());
    m_interpThresholdSpin->setValue(s.value(QStringLiteral("post/threshold"), 1.0).toDouble());
    m_interpSpacingSpin->setValue(s.value(QStringLiteral("post/spacing"), 4).toInt());
    m_interpRadiusSpin->setValue(s.value(QStringLiteral("post/radius"), 0).toInt());
    loadName(m_interpNameEdit, "post/name");

    const QByteArray geometry = s.value(QStringLiteral("window/geometry")).toByteArray();
    if (!geometry.isEmpty())
        restoreGeometry(geometry);
}

void MainWindow::saveSettings() const
{
    QSettings s;

    s.setValue(QStringLiteral("form/mode"), m_modeFete->isChecked() ? 1 : 2);
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

    s.setValue(QStringLiteral("form/neighbours"), m_neighboursCombo->currentData().toInt());
    s.setValue(QStringLiteral("form/costFunction"), m_costFunctionCombo->currentData().toInt());
    s.setValue(QStringLiteral("form/smoothing"), m_smoothingSpin->value());
    s.setValue(QStringLiteral("form/threads"), m_threadsSpin->value());
    s.setValue(QStringLiteral("form/ram"), m_ramSpin->value());
    s.setValue(QStringLiteral("form/verbose"), m_verboseCheck->isChecked());

    s.setValue(QStringLiteral("form/slopeName"), m_slopeNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/costName"), m_costNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/additionalName"), m_additionalNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/totalName"), m_totalNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/densityName"), m_densityNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/pathRasterName"), m_pathRasterNameEdit->text().trimmed());
    s.setValue(QStringLiteral("form/pathLinesName"), m_pathLinesNameEdit->text().trimmed());

    s.setValue(QStringLiteral("post/input"), m_interpInputPicker->path());
    s.setValue(QStringLiteral("post/outputDir"), m_interpOutputDirPicker->path());
    s.setValue(QStringLiteral("post/threshold"), m_interpThresholdSpin->value());
    s.setValue(QStringLiteral("post/spacing"), m_interpSpacingSpin->value());
    s.setValue(QStringLiteral("post/radius"), m_interpRadiusSpin->value());
    s.setValue(QStringLiteral("post/name"), m_interpNameEdit->text().trimmed());

    s.setValue(QStringLiteral("window/geometry"), saveGeometry());
}
