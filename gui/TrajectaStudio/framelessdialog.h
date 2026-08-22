#pragma once

#include <QDialog>

class QLabel;
class QVBoxLayout;

namespace TrajectaUi {

// A QDialog with none of the window manager's own chrome — no title bar, no
// minimise, no maximise/restore, no system menu — replaced with a title row
// this class draws itself: a name and a single "X", styled like the rest of
// Trajecta rather than like Windows. Every dialog in the application that is
// not a native file/folder picker is meant to build on this one in place of
// a bare QDialog — see confirmdialog.cpp, advancedsettingsdialog.cpp and
// reportform.cpp for the three places that already do. A native picker stays
// a native picker: redrawing it would mean replacing it entirely, at the
// cost of the OS-familiar navigation (recent places and the rest) it offers
// for free.
//
// A dialog needs far less of this than the main window did to get its own
// custom title bar (see the Q_OS_WIN block in MainWindow's constructor and
// nativeEvent()): that one keeps the native WS_OVERLAPPEDWINDOW frame and
// answers WM_NCCALCSIZE by hand, purely so Windows keeps animating minimise
// and restore — which nothing here needs, because nothing here has a
// minimise button. Qt::FramelessWindowHint alone is enough to take the rest
// of the frame away; showEvent() below restores the drop shadow with the
// same one-line DWM call the main window already makes for the same reason.
//
// The window's own outline — border and rounded corners — is not left to
// the window manager. Windows only rounds a borderless top-level window
// according to its own fixed presets, none of which is the 12 px radius
// #ModeCard and every other "chosen surface with a border" in the app is
// built on (see theme.qss), and on Windows 10 it does not round one at all.
// Qt::WA_TranslucentBackground turns the window itself into a normal
// per-pixel-alpha surface instead, and paintEvent() below fills and strokes
// exactly the rounded shape the rest of Trajecta already uses, antialiased,
// with the four corners left fully transparent — so a dialog looks the same
// whichever OS or Windows version draws it. A caller whose own content is
// opaque and reaches flush to a corner has to curve away with it there or
// paint a square edge over the curve — see AdvancedSettingsSidebar's own
// border-bottom-left-radius in theme.qss, set to kCornerRadius, for the one
// caller today whose content does.
class FramelessDialog : public QDialog
{
    Q_OBJECT

public:
    explicit FramelessDialog(QWidget *parent, const QString &title);

    // Height of the title row this class adds. A caller that used to
    // setFixedSize() a bare QDialog is now sizing the title bar and the
    // content together, where a native title bar used to sit outside the
    // client area and cost the content nothing — so that content needs this
    // much added back to keep the same room it had before.
    static constexpr int kTitleBarHeight = 42;

    // Same number as #ModeCard's border-radius in theme.qss — one shape
    // language for every bordered surface in the app, popup windows
    // included. A public constant rather than a value repeated in
    // paintEvent() and in whichever caller's content needs to keep clear of
    // the curve at a corner.
    static constexpr qreal kCornerRadius = 12.0;

    // Where a caller's own content goes — everything below the title row.
    // Margins and spacing are the caller's to set, exactly as they would be
    // on a QVBoxLayout put directly on a bare QDialog.
    QVBoxLayout *contentLayout() const { return m_content; }

protected:
    void paintEvent(QPaintEvent *event) override;
#ifdef Q_OS_WIN
    void showEvent(QShowEvent *event) override;
#endif

private:
    QLabel *m_titleLabel = nullptr;
    QVBoxLayout *m_content = nullptr;
};

} // namespace TrajectaUi
