#pragma once

#include <QPlainTextEdit>
#include <QTextCharFormat>

// Read-only log view that understands the subset of ANSI behaviour trajecta
// uses: SGR color/bold codes, carriage-return line rewriting (progress bars)
// and the CSI "K" clear-line sequence.
class ConsoleView : public QPlainTextEdit
{
    Q_OBJECT

public:
    explicit ConsoleView(QWidget *parent = nullptr);

    void appendChunk(const QString &raw);                          // engine output
    void appendMarker(const QString &text, const QColor &color);   // GUI-generated note
    void clearAll();

    // Re-reads the palette after a theme change. The ANSI colours the engine
    // emits are chosen for a dark terminal; on a light palette they are
    // darkened rather than replaced, so red stays red and the transcript keeps
    // meaning the same thing.
    void applyTheme();
    static QColor forTheme(const QColor &ansi);

private:
    void insertRun(const QString &text);
    void applySgrCodes(const QList<int> &codes);
    bool atBottom() const;
    void scrollToBottom();

    QTextCharFormat m_fmt;
    QTextCharFormat m_defaultFmt;
    bool m_pendingCr = false;
};
