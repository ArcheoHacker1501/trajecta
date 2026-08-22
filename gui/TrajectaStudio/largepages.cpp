#include "largepages.h"

#include "confirmdialog.h"
#include "uiwidgets.h"

#include <QApplication>
#include <QByteArray>
#include <QCheckBox>
#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSettings>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// After windows.h: it defines the types ntsecapi.h builds on.
#include <ntsecapi.h>
#include <shellapi.h>
#endif

namespace {

#ifdef Q_OS_WIN
// Is SeLockMemoryPrivilege present in the given token? Presence, not enabled
// state: a privilege the account holds is in the token even while off.
bool tokenHasLockMemory(HANDLE token)
{
    DWORD needed = 0;
    GetTokenInformation(token, TokenPrivileges, nullptr, 0, &needed);
    if (needed == 0)
        return false;
    QByteArray buf(int(needed), 0);
    if (!GetTokenInformation(token, TokenPrivileges, buf.data(), needed, &needed))
        return false;
    auto *tp = reinterpret_cast<TOKEN_PRIVILEGES *>(buf.data());
    LUID want{};
    if (!LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &want))
        return false;
    for (DWORD i = 0; i < tp->PrivilegeCount; ++i) {
        if (tp->Privileges[i].Luid.LowPart == want.LowPart
            && tp->Privileges[i].Luid.HighPart == want.HighPart) {
            return true;
        }
    }
    return false;
}
#endif

// Is it present in *this* process's token — the one the engine will inherit?
bool tokenHasLockMemory()
{
#ifdef Q_OS_WIN
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token))
        return false;
    const bool found = tokenHasLockMemory(token);
    CloseHandle(token);
    return found;
#else
    return false;
#endif
}

#ifdef Q_OS_WIN
// Is this an administrator's UAC-filtered token that *would* hold the
// privilege if the process ran elevated?
//
// This is the case that made the option look broken. On an administrator
// account Windows issues two tokens at logon: the full one, and a filtered one
// with the powerful privileges — SeLockMemoryPrivilege among them — stripped
// out. Every ordinary program, Trajecta included, gets the filtered one, so the
// privilege is genuinely absent no matter how many times the user signs out.
// The engine is launched from Trajecta and inherits the same restriction.
//
// The linked token is the authoritative answer, and asking for it needs no
// privilege of its own. TokenElevationType is the fallback: if the linked token
// cannot be examined, a "limited" elevation type at least says the token is a
// filtered administrator one.
bool elevationWouldHelp()
{
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_DUPLICATE, &token))
        return false;

    bool wouldHelp = false;
    bool answered = false;

    TOKEN_LINKED_TOKEN linked{};
    DWORD needed = 0;
    if (GetTokenInformation(token, TokenLinkedToken, &linked, sizeof(linked), &needed)
        && linked.LinkedToken) {
        wouldHelp = tokenHasLockMemory(linked.LinkedToken);
        answered = true;
        CloseHandle(linked.LinkedToken);
    }

    if (!answered) {
        TOKEN_ELEVATION_TYPE type = TokenElevationTypeDefault;
        if (GetTokenInformation(token, TokenElevationType, &type, sizeof(type), &needed))
            wouldHelp = (type == TokenElevationTypeLimited);
    }

    CloseHandle(token);
    return wouldHelp;
}
#endif

} // namespace

LargePageState largePageState()
{
#ifdef Q_OS_WIN
    if (tokenHasLockMemory())
        return LargePageState::Ready;
    // Asked before the "sign out and back in" advice, because when it is true
    // that advice is simply wrong: the account already holds the privilege and
    // no number of logons will put it in a filtered token.
    if (elevationWouldHelp())
        return LargePageState::NeedsElevation;
    // The grant is recorded so the application can tell "never asked" apart
    // from "granted, but this session's token predates it".
    if (QSettings().value(QStringLiteral("engine/largePagesGranted"), false).toBool())
        return LargePageState::GrantedNeedsRelogin;
    return LargePageState::NotGranted;
#else
    return LargePageState::NotGranted;
#endif
}

bool largePagesRequested()
{
    return QSettings().value(QStringLiteral("engine/largePagesEnabled"), false).toBool()
           && largePageState() == LargePageState::Ready;
}

void setLargePagesRequested(bool on)
{
    QSettings().setValue(QStringLiteral("engine/largePagesEnabled"), on);
}

LargePagesOption::LargePagesOption(QWidget *parent)
    : QWidget(parent)
{
    // Named for the same reason as ChunkHost and the rest: an unnamed QWidget
    // is caught by the "QMainWindow, QWidget" rule at the top of theme.qss
    // and paints its own opaque page background behind the row — visible as
    // a grey band on Light, where that colour is not close to white — rather
    // than staying transparent over whatever it is actually sitting on
    // (Advanced settings' own page today).
    setObjectName(QStringLiteral("LargePagesOption"));

    auto *row = new QHBoxLayout(this);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(6);

    m_check = new QCheckBox(tr("Use large memory pages (advanced)"), this);
    row->addWidget(m_check);
    // No "?" here any more: this widget now lives in exactly one place, the
    // Advanced settings dialog, whose own page prints largePagesHelpText()
    // directly instead of behind a popup — see advancedsettingsdialog.cpp.
    // A help-dot's popup is sized to overlay a page, not a 720x440 dialog,
    // and inside one it drew clipped and unreadable.

    m_status = new QLabel(this);
    m_status->setObjectName(QStringLiteral("HintLabel"));
    row->addWidget(m_status);

    m_setup = new QPushButton(tr("Set up…"), this);
    // Filled like every other button that does something: it is the action
    // that makes the option available, not an aside.
    m_setup->setObjectName(QStringLiteral("PrimaryButton"));
    m_setup->setCursor(Qt::PointingHandCursor);
    connect(m_setup, &QPushButton::clicked, this, &LargePagesOption::runSetup);
    row->addWidget(m_setup);
    row->addStretch(1);

    connect(m_check, &QCheckBox::toggled, this, [this](bool on) {
        setLargePagesRequested(on);
        emit toggled(on);
    });
    refresh();
    // There is only one instance of this widget now (the Advanced settings
    // dialog), so its checkbox is free to just start at the saved preference
    // rather than waiting for a caller to set it.
    setChecked(QSettings().value(QStringLiteral("engine/largePagesEnabled"), false).toBool());
}

bool LargePagesOption::isChecked() const
{
    return m_check && m_check->isChecked();
}

void LargePagesOption::setChecked(bool on)
{
    if (!m_check)
        return;
    m_check->setChecked(on && largePageState() == LargePageState::Ready);
}

QWidget *LargePagesOption::tickAnchor() const
{
    // The tick if there is one, the row itself otherwise: a caller drawing a
    // line to it must always get somewhere sensible, never nullptr.
    return m_check ? static_cast<QWidget *>(m_check)
                   : const_cast<LargePagesOption *>(this);
}

void LargePagesOption::refresh()
{
    if (!m_check)
        return;
    const LargePageState state = largePageState();
    const bool ready = (state == LargePageState::Ready);

    m_check->setEnabled(ready);
    if (!ready)
        m_check->setChecked(false);
    // The status line is the only place the reason is written out, and in the
    // elevation case it is the difference between a fix and a wasted sign-out.
    m_status->setToolTip(ready ? QString() : TrajectaUi::largePagesHelpText());
    // Offered whenever the option is not usable, including after a grant that
    // did not take: a machine where a group policy overrides the local one
    // comes back from the sign-out still without the privilege, and hiding the
    // button there would leave no way to try again.
    m_setup->setVisible(!ready);

    switch (state) {
    case LargePageState::NotGranted:
        m_status->setText(tr("— not available: Windows privilege not granted"));
        break;
    case LargePageState::GrantedNeedsRelogin:
        m_status->setText(tr("— granted: sign out of Windows and back in to finish"));
        break;
    case LargePageState::NeedsElevation:
        m_status->setText(tr("— granted, but withheld from ordinary programs: "
                             "start Trajecta as administrator"));
        break;
    case LargePageState::Ready:
        m_status->setText(tr("— available"));
        break;
    }
}

// --------------------------------------------------------------- the grant
//
// Adding an account right through the LSA, which is what ntrights.exe does and
// what the secedit/INF route was a roundabout imitation of. Three steps, each
// of which can fail loudly: find this account's SID, open the local security
// policy for writing, add the right — and then read the rights back to make
// sure the one we asked for is among them.
bool grantLockMemoryPrivilege(QString *error)
{
#ifdef Q_OS_WIN
    const auto fail = [error](const QString &text) {
        if (error)
            *error = text;
        return false;
    };

    // --- the SID of the account this process runs as ---
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token))
        return fail(QObject::tr("The process token could not be read."));
    DWORD needed = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &needed);
    QByteArray buf(int(needed), 0);
    if (needed == 0 || !GetTokenInformation(token, TokenUser, buf.data(), needed, &needed)) {
        CloseHandle(token);
        return fail(QObject::tr("The account could not be identified."));
    }
    CloseHandle(token);
    PSID sid = reinterpret_cast<TOKEN_USER *>(buf.data())->User.Sid;

    // --- the policy, opened for writing ---
    LSA_OBJECT_ATTRIBUTES attrs{};
    LSA_HANDLE policy = nullptr;
    NTSTATUS status = LsaOpenPolicy(nullptr, &attrs,
                                    POLICY_CREATE_ACCOUNT | POLICY_LOOKUP_NAMES
                                        | POLICY_VIEW_LOCAL_INFORMATION,
                                    &policy);
    if (status != 0) {
        return fail(QObject::tr("The local security policy could not be opened "
                                "(error %1). Administrator rights are needed.")
                        .arg(LsaNtStatusToWinError(status)));
    }

    wchar_t rightName[] = L"SeLockMemoryPrivilege";
    LSA_UNICODE_STRING right{};
    right.Buffer = rightName;
    right.Length = USHORT(wcslen(rightName) * sizeof(wchar_t));
    right.MaximumLength = USHORT(right.Length + sizeof(wchar_t));

    status = LsaAddAccountRights(policy, sid, &right, 1);
    if (status != 0) {
        const ULONG win32 = LsaNtStatusToWinError(status);
        LsaClose(policy);
        return fail(QObject::tr("The privilege could not be added (error %1).")
                        .arg(win32));
    }

    // --- and read it back, because "it returned success" is not the same
    //     claim as "the account has it" ---
    PLSA_UNICODE_STRING granted = nullptr;
    ULONG count = 0;
    bool found = false;
    if (LsaEnumerateAccountRights(policy, sid, &granted, &count) == 0) {
        for (ULONG i = 0; i < count && !found; ++i) {
            const QString name = QString::fromWCharArray(
                granted[i].Buffer, granted[i].Length / sizeof(wchar_t));
            found = (name.compare(QLatin1String("SeLockMemoryPrivilege"),
                                  Qt::CaseInsensitive) == 0);
        }
        LsaFreeMemory(granted);
    }
    LsaClose(policy);

    if (!found) {
        return fail(QObject::tr("The privilege was accepted but is not in the "
                                "policy afterwards, which usually means a domain "
                                "or group policy is overriding the local one."));
    }
    return true;
#else
    if (error)
        *error = QObject::tr("Large pages are a Windows feature.");
    return false;
#endif
}

// Grants SeLockMemoryPrivilege to the current account. Needs elevation, so it
// is done by a one-shot elevated helper rather than by elevating Trajecta:
// running the whole application as administrator would break drag & drop from
// Explorer, hide mapped drives, and leave output files with the wrong owner.
void LargePagesOption::runSetup()
{
#ifdef Q_OS_WIN
    // Two coloured buttons rather than two outlined ones: this is a question
    // with a real answer on both sides — it raises a UAC prompt and changes a
    // security policy — and a pair of grey buttons made a consequential choice
    // look like a passive notice.
    if (!TrajectaUi::confirm(
            this, tr("Enable large memory pages"),
            tr("Windows will ask for administrator approval.\n\n"
               "The \"Lock pages in memory\" privilege will be granted to your "
               "account. You then have to sign out of Windows and back in "
               "before it takes effect.\n\nContinue?"),
            // Taller than the standard box: five lines and a question do not
            // fit in it, and the question is the line that was being cut off.
            QString(), QString(), 70, TrajectaUi::Fill::AcceptDanger))
        return;

    // A one-shot elevated copy of Trajecta Studio itself, run with a switch
    // that does nothing but grant the right and exit. Not the whole application
    // elevated — that would break drag & drop from Explorer, hide mapped drives
    // and leave output files owned by the wrong account — and not PowerShell
    // either: what this needs is one API call, and going through a script is
    // what cost us a failure nobody could see.
    SHELLEXECUTEINFOW sei{};
    sei.cbSize = sizeof(sei);
    sei.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC;
    sei.lpVerb = L"runas";          // this is what raises the UAC prompt
    const QString exe =
        QDir::toNativeSeparators(QCoreApplication::applicationFilePath());
    const QString params = QStringLiteral("--grant-large-pages");
    sei.lpFile = reinterpret_cast<const wchar_t *>(exe.utf16());
    sei.lpParameters = reinterpret_cast<const wchar_t *>(params.utf16());
    sei.nShow = SW_HIDE;

    const QString manualHint =
        tr("\n\nYou can also do it by hand: Local Security Policy → Local "
           "Policies → User Rights Assignment → \"Lock pages in memory\" → add "
           "your account.");

    if (!ShellExecuteExW(&sei) || !sei.hProcess) {
        TrajectaUi::notify(
            this, tr("Large memory pages"),
            tr("The privilege could not be granted: the administrator prompt "
               "was declined, or could not be shown.")
                + manualHint,
            QString(), 80);
        return;
    }

    // The wait is mostly the user reading the UAC prompt, which can sit there
    // for a while. So it is pumped in short slices rather than made in one
    // blocking call: a straight WaitForSingleObject would stop the window
    // repainting and Windows would grey it out as "not responding" while its
    // own prompt is on screen. User input stays excluded — nothing here should
    // be clickable meanwhile.
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QElapsedTimer clock;
    clock.start();
    DWORD waited = WAIT_TIMEOUT;
    while (clock.elapsed() < 120000) {
        waited = WaitForSingleObject(sei.hProcess, 50);
        if (waited != WAIT_TIMEOUT)
            break;
        QApplication::processEvents(QEventLoop::ExcludeUserInputEvents, 20);
    }
    DWORD exitCode = 1;
    if (waited == WAIT_OBJECT_0)
        GetExitCodeProcess(sei.hProcess, &exitCode);
    CloseHandle(sei.hProcess);
    QApplication::restoreOverrideCursor();

    if (waited != WAIT_OBJECT_0) {
        TrajectaUi::notify(
            this, tr("Large memory pages"),
            tr("The grant is taking longer than expected. Restart Trajecta in a "
               "moment to see whether it succeeded.")
                + manualHint,
            QString(), 80);
        return;
    }
    if (exitCode != 0) {
        // The elevated copy reads the policy back before returning 0, so a
        // non-zero code here means the right is genuinely not there — never
        // "we could not tell", which is what the old route reported.
        TrajectaUi::notify(
            this, tr("Large memory pages"),
            tr("The privilege was not granted. Either the administrator prompt "
               "was refused, or a domain or group policy is overriding the "
               "local security policy.")
                + manualHint,
            QString(), 100);
        return;
    }

    QSettings().setValue(QStringLiteral("engine/largePagesGranted"), true);
    refresh();
    emit privilegeGranted();
    TrajectaUi::notify(
        this, tr("Large memory pages"),
        tr("The privilege has been granted to your account, and read back to "
           "check that it is really there.\n\n"
           "Sign out of Windows and back in to finish: restarting Trajecta is "
           "not enough, because the list of privileges is fixed when you log "
           "on.\n\n"
           "On an administrator account there is one more step. Windows hands "
           "ordinary programs a filtered token with this privilege removed, so "
           "after signing back in Trajecta has to be started with \"Run as "
           "administrator\" for large pages to be available. The status line "
           "beside this option always says which of the two you are at."),
        QString(), 200);
#endif
}
