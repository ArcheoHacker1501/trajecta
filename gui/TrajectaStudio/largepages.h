#pragma once

#include <QWidget>

class QCheckBox;
class QLabel;
class QPushButton;

// Large memory pages, as one widget used by every page that offers them.
//
// Three independent gates decide whether the engine actually gets 2 MB pages:
//   1. the Windows privilege, granted once per account and picked up only by a
//      NEW logon token, which is why signing out is unavoidable;
//   2. this checkbox;
//   3. the allocation itself, which can still fail on fragmented memory.
// The interface's job is to say which gate the user is standing at.
//
// It is a widget rather than three loose controls because the setting appears
// in two places — the single-run form and the batch page — and "the same words
// and the same warnings in both" is a requirement, not a coincidence. Sharing
// the widget is the only way to keep it true.

enum class LargePageState {
    NotGranted,           // the account does not hold SeLockMemoryPrivilege
    GrantedNeedsRelogin,  // granted, but this logon token predates the grant
    // Granted, the logon is new enough, and it still is not in the token:
    // the account is an administrator, and UAC hands an ordinary process a
    // *filtered* token with this privilege removed. Signing out again changes
    // nothing — only running elevated does. Told apart from the case above
    // because the advice is completely different.
    NeedsElevation,
    Ready
};

LargePageState largePageState();

// The one saved preference behind the checkbox below, read by every place
// that used to hold its own copy of the widget (the single-run forms, the
// batch page, the post-processing batch page) now that there is only one
// copy of it, in the Advanced settings dialog. Already gated on the
// privilege actually being in this session's token — a caller never has to
// check largePageState() itself before honouring it.
bool largePagesRequested();
void setLargePagesRequested(bool on);

// Grants SeLockMemoryPrivilege to the account this process is running as, and
// then reads the policy back to confirm it is there. Returns false with a
// reason in `error`.
//
// Must run elevated: adding an account right needs POLICY_CREATE_ACCOUNT on the
// local security policy, which an ordinary token does not have. It is called in
// exactly one place — the one-shot elevated copy of Trajecta Studio that
// "Set up…" launches with --grant-large-pages — and is declared here so that
// copy and the button that starts it share one implementation.
//
// This replaces an earlier attempt that drove secedit through PowerShell: it
// exported the security policy to an INF, appended the privilege line to the
// end of the file and imported it back. On the default Windows install, where
// "Lock pages in memory" is assigned to nobody, there is no such line to edit
// and the appended one landed after [Version] — outside [Privilege Rights],
// where secedit ignores it. secedit still exited 0, so the grant was recorded
// as done and the option waited for a sign-out that could never change
// anything. The LSA call below cannot fail quietly: it returns a status, and
// what it wrote is read back before this returns true.
bool grantLockMemoryPrivilege(QString *error);

class LargePagesOption : public QWidget
{
    Q_OBJECT

public:
    explicit LargePagesOption(QWidget *parent = nullptr);

    bool isChecked() const;
    // Ignored when the privilege is not in this session's token: the engine
    // would only fall back to 4 KB pages, and a tick that means nothing is
    // worse than no tick at all.
    void setChecked(bool on);

    // Re-reads the privilege state and updates the label, the checkbox and the
    // visibility of "Set up…". Called after a grant, and by anything that may
    // have changed the state behind our back.
    void refresh();

    // The tick itself, for a caption in the walkthrough to point at. The widget
    // is a whole row — tick, explanation, live status, "Set up…" — so a leader
    // drawn to the middle of it lands on empty space between two of those.
    QWidget *tickAnchor() const;

signals:
    void toggled(bool checked);
    // The privilege was just granted; other instances of this widget have to
    // catch up.
    void privilegeGranted();

private:
    void runSetup();

    QCheckBox *m_check = nullptr;
    QLabel *m_status = nullptr;
    QPushButton *m_setup = nullptr;
};
