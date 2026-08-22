#pragma once

#include "framelessdialog.h"

class QListWidget;
class QStackedWidget;

// The dialog behind the gear menu's "Advanced settings…" entry: a sidebar
// list on the left, naming each setting, and its own panel on the right — the
// same shape as the Guide's own layout, at dialog scale. Built for one entry
// today (large memory pages, moved out of the four hardware cards it used to
// be duplicated across) but meant to hold more without changing shape: adding
// a setting is a new row in the sidebar and a new page in the stack, not a
// new dialog.
//
// Modal, centred on the window and dressed by the active theme, like every
// other dialog in the application — see confirmdialog.h for why that means a
// plain QDialog rather than QMessageBox, and framelessdialog.h for why this
// one, and not that one, is where the window's own title and close button
// come from.
class AdvancedSettingsDialog : public TrajectaUi::FramelessDialog
{
    Q_OBJECT

public:
    explicit AdvancedSettingsDialog(QWidget *parent = nullptr);

private:
    void addPage(const QString &title, QWidget *page);

    QListWidget *m_sidebar = nullptr;
    QStackedWidget *m_pages = nullptr;
};
