#include "advancedsettingsdialog.h"

#include "largepages.h"
#include "uiwidgets.h"

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QScrollArea>
#include <QStackedWidget>
#include <QVBoxLayout>

namespace {
// Wide enough for LargePagesOption's own row — tick, status line and "Set
// up…" button side by side — to fit its longest status without wrapping or
// running under the button: "granted, but withheld from ordinary programs:
// start Trajecta as administrator" is the one that used to lose its tail.
constexpr int kDialogWidth = 1100;
constexpr int kDialogHeight = 440;
}

AdvancedSettingsDialog::AdvancedSettingsDialog(QWidget *parent)
    : FramelessDialog(parent, tr("Advanced settings"))
{
    // Same styling idiom as ConfirmDialog: an id of its own rather than that
    // one, because this dialog is a different shape and size — see theme.qss.
    setObjectName(QStringLiteral("AdvancedSettingsDialog"));
    setFixedSize(kDialogWidth, TrajectaUi::FramelessDialog::kTitleBarHeight + kDialogHeight);

    // A bare layout, not a QWidget wrapper: an unnamed QWidget here would be
    // caught by the "QMainWindow, QWidget" rule at the top of theme.qss and
    // paint its own opaque #14171c square behind the sidebar and the page
    // stack — the same failure FramelessDialog's own contentLayout() avoids
    // the same way, and for the same reason (see framelessdialog.cpp).
    auto *bodyLayout = new QHBoxLayout;
    // Flush on every side, including the bottom: the sidebar is meant to
    // read as one frame with the dialog itself, the same "frame the pages
    // sit inside" its own theme.qss comment describes, all the way down to
    // the window's own edge. What keeps this from squaring off the dialog's
    // rounded bottom-left corner is not a margin here but
    // #AdvancedSettingsSidebar's own border-bottom-left-radius in
    // theme.qss, set to the same kCornerRadius so the sidebar's opaque fill
    // curves away with the window instead of stopping short of it or
    // cutting across it. The page stack on the right needs no matching
    // radius: it and the page inside it are both transparent (see
    // theme.qss), so FramelessDialog's own rounded fill already shows
    // through correctly there without any help from this layout.
    bodyLayout->setContentsMargins(0, 0, 0, 0);
    bodyLayout->setSpacing(0);

    // The sidebar. A QListWidget rather than a row of buttons: the list it
    // will hold does not have to stay short, and a list scrolls on its own
    // once it no longer fits, which a row of stacked buttons would not.
    m_sidebar = new QListWidget(this);
    m_sidebar->setObjectName(QStringLiteral("AdvancedSettingsSidebar"));
    m_sidebar->setFixedWidth(190);
    m_sidebar->setFocusPolicy(Qt::NoFocus);
    m_sidebar->setFrameShape(QFrame::NoFrame);
    bodyLayout->addWidget(m_sidebar);

    m_pages = new QStackedWidget(this);
    m_pages->setObjectName(QStringLiteral("AdvancedSettingsPages"));
    bodyLayout->addWidget(m_pages, 1);
    contentLayout()->addLayout(bodyLayout, 1);

    connect(m_sidebar, &QListWidget::currentRowChanged,
            m_pages, &QStackedWidget::setCurrentIndex);

    // --- Large memory pages: the one entry today ---
    {
        auto *page = new QWidget;
        page->setObjectName(QStringLiteral("AdvancedSettingsPage"));
        auto *layout = new QVBoxLayout(page);
        layout->setContentsMargins(24, 20, 24, 20);
        layout->setSpacing(10);

        auto *heading = new QLabel(tr("Large memory pages (experimental)"), page);
        heading->setObjectName(QStringLiteral("AdvancedSettingsPageTitle"));
        layout->addWidget(heading);

        auto *intro = new QLabel(
            tr("Lets the analysis engine address memory in 2 MB pages instead "
               "of the usual 4 KB, which on a large DEM typically makes the "
               "propagation phase 15–30% faster. It cannot change a result — "
               "only how long it takes to arrive. Applies to every run, single "
               "or batch, on either page."),
            page);
        intro->setObjectName(QStringLiteral("AdvancedSettingsPageIntro"));
        intro->setWordWrap(true);
        layout->addWidget(intro);

        auto *option = new LargePagesOption(page);
        layout->addWidget(option);

        // The help-dot's own popup used to carry this — sized to overlay a
        // whole page, it drew clipped and unreadable inside a 720x440
        // dialog. Printed here instead, as a plain label the scroll area
        // below lets the reader reach the same way they would read any long
        // page: the wheel, not a click.
        auto *explanation = new QLabel(TrajectaUi::largePagesHelpText(), page);
        explanation->setObjectName(QStringLiteral("AdvancedSettingsPageIntro"));
        explanation->setWordWrap(true);
        explanation->setTextFormat(Qt::RichText);
        layout->addWidget(explanation);
        layout->addStretch(1);

        auto *scroll = new QScrollArea(m_pages);
        scroll->setObjectName(QStringLiteral("AdvancedSettingsScroll"));
        scroll->setWidget(page);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

        addPage(tr("Large memory pages"), scroll);
    }

    m_sidebar->setCurrentRow(0);
}

void AdvancedSettingsDialog::addPage(const QString &title, QWidget *page)
{
    m_sidebar->addItem(title);
    m_pages->addWidget(page);
}
