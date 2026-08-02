#pragma once

#include <QLineEdit>
#include <QWidget>

class QLabel;
class QToolButton;

// One-line path selector: text field + browse button + validity indicator.
// Accepts files dragged from Explorer.
class PathPicker : public QWidget
{
    Q_OBJECT

public:
    enum class Kind { ExistingFile, Directory };

    PathPicker(Kind kind, const QString &dialogTitle,
               const QString &filter = QString(), QWidget *parent = nullptr);

    QString path() const;
    void setPath(const QString &path);
    void setPlaceholder(const QString &text);
    void setOptional(bool optional);
    bool isOptional() const { return m_optional; }

    // True when the field points to something usable: an existing file or
    // directory of the right kind, or an empty field if the path is optional.
    bool isSatisfied() const;

signals:
    void pathChanged(const QString &path);

protected:
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;

private:
    void browse();
    void updateIndicator();

    Kind m_kind;
    QString m_dialogTitle;
    QString m_filter;
    bool m_optional = false;
    QLineEdit *m_edit = nullptr;
    QToolButton *m_browse = nullptr;
    QLabel *m_indicator = nullptr;
};
