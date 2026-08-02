#include "pathpicker.h"

#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QMimeData>
#include <QToolButton>

PathPicker::PathPicker(Kind kind, const QString &dialogTitle,
                       const QString &filter, QWidget *parent)
    : QWidget(parent)
    , m_kind(kind)
    , m_dialogTitle(dialogTitle)
    , m_filter(filter)
{
    setAcceptDrops(true);

    m_edit = new QLineEdit(this);
    m_edit->setClearButtonEnabled(true);

    m_browse = new QToolButton(this);
    m_browse->setText(tr("..."));
    m_browse->setCursor(Qt::PointingHandCursor);

    m_indicator = new QLabel(this);
    m_indicator->setFixedWidth(14);
    m_indicator->setAlignment(Qt::AlignCenter);

    auto *layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    layout->addWidget(m_indicator);
    layout->addWidget(m_edit, 1);
    layout->addWidget(m_browse);

    connect(m_browse, &QToolButton::clicked, this, &PathPicker::browse);
    connect(m_edit, &QLineEdit::textChanged, this, [this](const QString &text) {
        updateIndicator();
        emit pathChanged(text);
    });

    updateIndicator();
}

QString PathPicker::path() const
{
    return m_edit->text().trimmed();
}

void PathPicker::setPath(const QString &path)
{
    m_edit->setText(QDir::toNativeSeparators(path));
}

void PathPicker::setPlaceholder(const QString &text)
{
    m_edit->setPlaceholderText(text);
}

void PathPicker::setOptional(bool optional)
{
    m_optional = optional;
    updateIndicator();
}

bool PathPicker::isSatisfied() const
{
    const QString p = path();
    if (p.isEmpty())
        return m_optional;
    const QFileInfo info(p);
    return m_kind == Kind::Directory ? info.isDir() : info.isFile();
}

void PathPicker::browse()
{
    QString start = path();
    if (start.isEmpty())
        start = QDir::homePath();

    QString chosen;
    if (m_kind == Kind::Directory)
        chosen = QFileDialog::getExistingDirectory(this, m_dialogTitle, start);
    else
        chosen = QFileDialog::getOpenFileName(this, m_dialogTitle, start, m_filter);

    if (!chosen.isEmpty())
        setPath(chosen);
}

void PathPicker::updateIndicator()
{
    const QString p = path();
    if (p.isEmpty()) {
        m_indicator->setText(QStringLiteral("●"));
        m_indicator->setStyleSheet(QStringLiteral("color: #3a4656;"));
        m_indicator->setToolTip(m_optional ? tr("Optional — leave empty to skip")
                                           : tr("Required"));
        return;
    }
    const QFileInfo info(p);
    const bool ok = m_kind == Kind::Directory ? info.isDir() : info.isFile();
    if (ok) {
        m_indicator->setText(QStringLiteral("●"));
        m_indicator->setStyleSheet(QStringLiteral("color: #3fa34d;"));
        m_indicator->setToolTip(tr("Found"));
    } else {
        m_indicator->setText(QStringLiteral("●"));
        m_indicator->setStyleSheet(QStringLiteral("color: #e05555;"));
        m_indicator->setToolTip(m_kind == Kind::Directory
                                    ? tr("This folder does not exist (it will be offered for creation)")
                                    : tr("File not found"));
    }
}

void PathPicker::dragEnterEvent(QDragEnterEvent *event)
{
    if (event->mimeData()->hasUrls())
        event->acceptProposedAction();
}

void PathPicker::dropEvent(QDropEvent *event)
{
    const QList<QUrl> urls = event->mimeData()->urls();
    if (!urls.isEmpty() && urls.first().isLocalFile())
        setPath(urls.first().toLocalFile());
}
