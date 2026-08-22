#pragma once

#include <QColor>
#include <QIcon>
#include <QProgressBar>
#include <QString>
#include <QWidget>

#include <functional>

class QFrame;
class QGroupBox;
class QLabel;
class QObject;
class QPropertyAnimation;
class QPushButton;
class QTimer;
class QWidget;

// A progress bar with a highlight travelling along the filled part, the way
// Windows' own bars behave. It is the same bar in every other respect: the
// groove, the fill and the text are still drawn by the stylesheet, so it keeps
// whatever colours the current theme gives it, and the sweep is painted over
// the fill afterwards.
//
// The timer only runs while the bar is both visible and actually in progress:
// an empty or finished bar has nothing to say, and a repaint every 33 ms of a
// widget nobody is looking at is a waste of an analysis's CPU.
class ActivityBar : public QProgressBar
{
    Q_OBJECT

public:
    explicit ActivityBar(QWidget *parent = nullptr);

    // One decimal place, so a long analysis visibly moves: at whole percent a
    // FETE over a large DEM can sit on the same number for many minutes and
    // read as stuck. Only for the default "%p%" format — a bar given its own
    // wording (the batch's "3 of 40 rows") keeps it.
    QString text() const override;

protected:
    void paintEvent(QPaintEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;
    void changeEvent(QEvent *event) override;

private:
    void updateAnimation();

    QTimer *m_timer = nullptr;
    // 0 → 1 over one pass of the highlight, then round again.
    double m_phase = 0.0;
};

// The strip in the middle of the status bar, and the drawer it opens.
//
// Everything else that reports on a run — the chip, the bar, the phase line —
// lives inside the panel of the page that started it. Leave that page and there
// is nothing on screen to say the machine is busy, which on an analysis that
// takes days is the wrong answer to the only question the user has. This says
// it from the bottom of every page, and stays for as long as the run does,
// paused included.
//
// It knows nothing about engines or batches: MainWindow assembles a State and
// hands it over, which is what lets one widget serve FETE, LCPA, a batch, the
// interpolator and the route comparison.
class RunTicker : public QWidget
{
    Q_OBJECT

public:
    explicit RunTicker(QWidget *parent = nullptr);

    struct State {
        bool active = false;
        bool paused = false;
        // "FETE", "LCPA", "Batch — FETE", "NNI", "Route comparison". Shown to
        // the right of the bar, because "something is running" is only half an
        // answer when three different things can be running.
        QString kind;
        // Negative until the engine reports for the first time: the bar then
        // sweeps instead of filling, exactly as the big one does.
        double percent = -1.0;
        // The three lines of the drawer. Empty ones are left out, which is how
        // a single-run drawer ends up shorter than a batch's.
        QString chunks;
        QString hardware;
        QString remaining;
    };

    void setState(const State &s);

    // Opens the drawer without a click, for the hidden --ticker-drawer switch:
    // a screenshot run has no pointer to press it with.
    void openDrawerForTest() { setDrawerOpen(true); }

protected:
    // The bar is not in the status bar's layout — see reposition().
    bool eventFilter(QObject *watched, QEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    void reposition();
    void buildDrawer();
    void setDrawerOpen(bool open);

    QLabel *m_chip = nullptr;
    QProgressBar *m_bar = nullptr;
    QLabel *m_kind = nullptr;

    QFrame *m_drawer = nullptr;
    QLabel *m_drawerChunks = nullptr;
    QLabel *m_drawerHardware = nullptr;
    QLabel *m_drawerRemaining = nullptr;
    QPropertyAnimation *m_drawerAnim = nullptr;
    bool m_drawerOpen = false;

    State m_state;
};

// Small building blocks shared by the setup form and the batch page, so a
// setting explained on one page is presented the same way on the other.
namespace TrajectaUi {

// How tall a log canvas is once unfolded — the single-run panels and the
// post-processing tools share this one value, so a log is the same size
// wherever it is read.
//
// Down from six screens of transcript, then from three: enough that the log is
// read instead of scrolled a line at a time, while leaving the panel it belongs
// to visible above it rather than pushed off the top.
constexpr int kLogCanvasHeight = 600;

// The batch page's log is shorter, by a fifth. It is the one page whose panel
// above the log is a table that grows with the job: at the full height the
// queue was pushed off the top exactly when there was most of it to watch.
constexpr int kBatchLogCanvasHeight = 480;

// Stops the mouse wheel from ever changing this control's value; the wheel
// scrolls the page instead, as if the control were not under the pointer. The
// value is changed by clicking, by typing, or with the arrow keys.
//
// For spin boxes, which have no subclass of their own. Every combo box in the
// application is a SmoothComboBox, which refuses the wheel itself.
void guardWheel(QWidget *w);

// Small "?" badge carrying a short description of the setting.
//
// It opens on a click, not on a hover. These texts are paragraphs — for several
// parameters they are the only explanation there is — and a tooltip has to be
// aimed at with the pointer and then read before it times out. A click is
// deliberate, and what it opens stays open until it is dismissed.
QLabel *makeHelpDot(const QString &help, QWidget *parent);

// Field label followed by a "?" help badge.
QWidget *makeFieldLabel(const QString &text, const QString &help, QWidget *parent);

// A caption with its own "?" badge, for a checkbox or any widget that already
// carries its label: the badge is placed to the right of `w`.
QWidget *withHelpDot(QWidget *w, const QString &help);

// A caption beside the title of a checkable group box, with its own "?".
//
// It sits on the title's own line, starting just after it: the note is about
// the switch, and at the far end of a wide card it was a long way from the
// thing it qualifies. Where the title ends is measured from the style, since
// the title is drawn by it, in a font that is a user setting.
//
// A checkable QGroupBox greys out everything inside it while it is unticked —
// which is exactly when a warning about what ticking it costs has to be read
// and its badge clicked. The row therefore puts itself back on every toggle.
QWidget *makeGroupNote(QGroupBox *group, const QString &note, const QString &help);

// The pause mark — two filled bars — on a Pause button, or off it again.
//
// Drawn rather than typed, and sized from the ▶ that the Run buttons carry, so
// the two marks weigh the same whichever of the interface fonts is in use and
// whatever the fallback face for ▶ turns out to be. It repaints itself when the
// theme or the font changes, so a caller sets it once and forgets it.
//
// `on == false` takes it off, which is what a Pause button needs when it becomes
// "▶ Resume": the mark has to leave with the word it belongs to.
void setPauseMark(QPushButton *button, bool on);

// Shown behind the "?" next to the large-pages checkbox, on both the setup form
// and the batch page: the same setting deserves the same explanation.
QString largePagesHelpText();

// Behind the "?" next to the run-manifest checkbox, on the setup form and on
// the batch page.
QString manifestHelpText();

// Appended to every explanation of the RAM ceiling — the setup form's and the
// batch page's. One sentence, in one place, because the two texts are otherwise
// written separately and a warning that appears on only one of them is worse
// than none: the user who needs it is the one whose DEM is large, and that DEM
// is just as likely to be the input of a batch.
QString ramHeadroomNote();

// "Time left: about 2 h 10 min", from how long the job has been going and how
// far through it is. Shared by the single-run ticker and the batch's, which
// measure different things — percent of one analysis, rows of a queue — and
// still have to phrase the answer the same way.
//
// Deliberately coarse. An estimate to the second on a job measured in hours is
// a promise nobody can keep, and reads as one.
QString timeLeftText(qint64 workedMs, double percent);

// Behind the "?" next to the neighbours selector, on the setup form and on
// every batch chunk. Covers which totals are admissible and what raising the
// number costs.
QString neighboursHelpText();

// Every neighbourhood size the engine accepts, smallest first. The list is a
// property of the square grid, not a setting: see src/neighbourhood.h.
QList<int> admissibleNeighbourCounts();

// The largest admissible count not above `wanted`, so the form can show the
// number the engine will really use rather than the one that was typed.
int snapNeighbourCount(int wanted);

// Behind the "?" next to the cost-function selector, on the setup form and on
// every batch chunk. Gives the formula each entry actually evaluates, the
// published variant it corresponds to, and the units of the result.
QString costFunctionHelpText();

// Behind the "?" next to the slope cut-off, on the setup form and on every
// batch chunk. Explains that the limit applies to a move, not to a cell.
QString slopeCutoffHelpText();

// Behind the "?" next to the NNI peak-preservation checkbox.
QString preservePeaksHelpText();

// Behind the "?" next to the cost-corridor switch, in the LCPA output card
// and on every batch chunk.
QString costCorridorHelpText();

// Behind the "?" next to the known-route comparison in post-processing.
QString routeCompareHelpText();

// Behind the "?" next to the cost-modifiers switch, on the setup form and on
// every batch chunk. Explains what the option does and why it costs time.
QString costModifiersHelpText();

// The always-visible half of that warning, in a few words.
QString costModifiersNoteText();

// Behind the "?" icons of the site-corridor coherence tool. One per parameter,
// and each says the same three things: what it does, what changes if you move
// it, and what it is set to by default and why.
QString coherenceSurfaceHelpText();
QString coherenceSitesHelpText();
QString coherenceRadiusHelpText();
QString coherenceThresholdHelpText();
QString coherenceNullHelpText();
QString coherenceEcdfHelpText();
QString coherenceSensitivityHelpText();
QString coherenceEdgeHelpText();
QString coherenceHistogramScriptHelpText();
QString coherenceOutputHelpText();

// Opens a hue/saturation wheel with a value bar and a feature-size slider,
// under `globalPos`, over `anchor`'s window — the way every drawing program
// has drawn "pick a colour" for thirty years, plus the one other thing a
// vector layer's own appearance needs.
//
// Not QColorDialog: that is a modal box with an OK button, four ways to type a
// colour and a row of custom swatches — a whole sitting, for a choice that is
// worth about a second. This follows the pointer live, so the map recolours
// and resizes as the hand moves, and it is dismissed by clicking away or by
// its own close button. There is no OK, because there is nothing to confirm:
// what is on screen is the answer. `onColour` and `onSizePercent` are called
// continuously as the pointer moves, never only once at the end.
//
// `initialSizePercent` is where the slider starts — 100 is a layer's ordinary
// size — and the widget itself lives entirely in uiwidgets.cpp: it is a plain
// child of the window, the same reason and the same pattern as HelpPopup
// there, which is what lets its QSlider pick up the application stylesheet
// like any other.
void pickColour(QWidget *anchor, const QPoint &globalPos, const QColor &initialColour,
                int initialSizePercent,
                const std::function<void(const QColor &)> &onColour,
                const std::function<void(int)> &onSizePercent);

// Minimise / maximise / restore / close, drawn rather than typed: they have
// to take the active palette's colour, which a font glyph cannot do reliably
// across fallback fonts. Stroked, not filled, so they stay legible at 10 px
// on both light and dark themes. Shared by the main window's own title bar
// and by FramelessDialog's close button, so the "X" on a dialog and the one
// that closes the whole application are the same mark.
enum class WindowGlyph { Minimise, Maximise, Restore, Close };

QIcon makeWindowIcon(WindowGlyph glyph, const QColor &color, int size);

} // namespace TrajectaUi
