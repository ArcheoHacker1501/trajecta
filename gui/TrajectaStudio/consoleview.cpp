#include "consoleview.h"

#include "thememanager.h"

#include <QFontDatabase>
#include <QScrollBar>
#include <QTextBlock>

ConsoleView::ConsoleView(QWidget *parent)
    : QPlainTextEdit(parent)
{
    setObjectName(QStringLiteral("Console"));
    setReadOnly(true);
    setWordWrapMode(QTextOption::WrapAnywhere);
    setMaximumBlockCount(8000);  // keep memory bounded on long runs

    QFont mono(QStringLiteral("Cascadia Mono"));
    mono.setStyleHint(QFont::Monospace);
    if (!QFontDatabase::families().contains(QStringLiteral("Cascadia Mono")))
        mono.setFamily(QStringLiteral("Consolas"));
    mono.setPointSize(11);
    setFont(mono);

    m_defaultFmt.setForeground(forTheme(QColor(0xd7, 0xde, 0xe8)));
    m_fmt = m_defaultFmt;
}

// The engine's ANSI palette assumes a dark terminal. On a light theme the
// same hues are darkened until they carry on paper, which keeps the colour
// coding (red = error, amber = warning) intact instead of remapping it.
QColor ConsoleView::forTheme(const QColor &ansi)
{
    if (!ThemeManager::isLight())
        return ansi;
    QColor c = ansi.toHsl();
    c.setHsl(c.hslHue(), qMin(255, int(c.hslSaturation() * 1.15)),
             qMax(40, int(c.lightness() * 0.42)));
    return c;
}

void ConsoleView::applyTheme()
{
    // Only the palette-dependent defaults; text already in the document keeps
    // the colours it was written with.
    m_defaultFmt.setForeground(forTheme(QColor(0xd7, 0xde, 0xe8)));
    m_fmt = m_defaultFmt;
}

bool ConsoleView::atBottom() const
{
    const QScrollBar *sb = verticalScrollBar();
    return sb->value() >= sb->maximum() - 4;
}

void ConsoleView::scrollToBottom()
{
    verticalScrollBar()->setValue(verticalScrollBar()->maximum());
}

void ConsoleView::clearAll()
{
    clear();
    m_fmt = m_defaultFmt;
    m_pendingCr = false;
}

void ConsoleView::applySgrCodes(const QList<int> &codes)
{
    for (int code : codes) {
        switch (code) {
        case 0:
            m_fmt = m_defaultFmt;
            break;
        case 1:
            m_fmt.setFontWeight(QFont::Bold);
            break;
        case 22:
            m_fmt.setFontWeight(QFont::Normal);
            break;
        case 31: m_fmt.setForeground(forTheme(QColor(0xff, 0x6b, 0x6b))); break;  // red
        case 32: m_fmt.setForeground(forTheme(QColor(0x5f, 0xd0, 0x68))); break;  // green
        case 33: m_fmt.setForeground(forTheme(QColor(0xff, 0xd1, 0x66))); break;  // yellow
        case 34: m_fmt.setForeground(forTheme(QColor(0x6e, 0xa8, 0xfe))); break;  // blue
        case 35: m_fmt.setForeground(forTheme(QColor(0xd9, 0x8b, 0xf9))); break;  // magenta
        case 36: m_fmt.setForeground(forTheme(QColor(0x66, 0xd9, 0xef))); break;  // cyan
        case 37: m_fmt.setForeground(forTheme(QColor(0xd7, 0xde, 0xe8))); break;  // white
        case 90: m_fmt.setForeground(forTheme(QColor(0x8a, 0x97, 0xa5))); break;  // bright black
        case 91: m_fmt.setForeground(forTheme(QColor(0xff, 0x8a, 0x8a))); break;
        case 92: m_fmt.setForeground(forTheme(QColor(0x7c, 0xe0, 0x83))); break;
        case 93: m_fmt.setForeground(forTheme(QColor(0xff, 0xe0, 0x8a))); break;
        case 96: m_fmt.setForeground(forTheme(QColor(0x8f, 0xe3, 0xf2))); break;
        default:
            break;  // unsupported code: ignore
        }
    }
}

void ConsoleView::insertRun(const QString &text)
{
    if (text.isEmpty())
        return;

    QTextCursor cursor(document());
    cursor.movePosition(QTextCursor::End);

    if (m_pendingCr) {
        // Carriage return without newline: rewrite the current line
        // (this is how trajecta animates its progress bar).
        cursor.movePosition(QTextCursor::StartOfBlock, QTextCursor::KeepAnchor);
        cursor.removeSelectedText();
        m_pendingCr = false;
    }

    cursor.insertText(text, m_fmt);
}

void ConsoleView::appendChunk(const QString &raw)
{
    const bool keepScrolled = atBottom();

    QString run;
    run.reserve(raw.size());

    int i = 0;
    const int n = raw.size();
    while (i < n) {
        const QChar ch = raw.at(i);

        if (ch == QLatin1Char('\x1b')) {
            insertRun(run);
            run.clear();
            // Parse a CSI sequence: ESC '[' params final-letter
            int j = i + 1;
            if (j < n && raw.at(j) == QLatin1Char('[')) {
                ++j;
                QString params;
                while (j < n && (raw.at(j).isDigit() || raw.at(j) == QLatin1Char(';')
                                 || raw.at(j) == QLatin1Char('?'))) {
                    params += raw.at(j);
                    ++j;
                }
                if (j < n) {
                    const QChar final = raw.at(j);
                    if (final == QLatin1Char('m')) {
                        QList<int> codes;
                        if (params.isEmpty()) {
                            codes << 0;
                        } else {
                            const QStringList parts = params.split(QLatin1Char(';'));
                            for (const QString &part : parts)
                                codes << part.toInt();
                        }
                        applySgrCodes(codes);
                    }
                    // "K" (erase line) is implied by our carriage-return
                    // handling; every other sequence is ignored.
                    i = j + 1;
                    continue;
                }
            }
            ++i;  // lone ESC: drop it
            continue;
        }

        if (ch == QLatin1Char('\r')) {
            insertRun(run);
            run.clear();
            m_pendingCr = true;
            ++i;
            continue;
        }

        if (ch == QLatin1Char('\n')) {
            insertRun(run);
            run.clear();
            QTextCursor cursor(document());
            cursor.movePosition(QTextCursor::End);
            cursor.insertBlock();
            m_pendingCr = false;
            ++i;
            continue;
        }

        run += ch;
        ++i;
    }
    insertRun(run);

    if (keepScrolled)
        scrollToBottom();
}

void ConsoleView::appendMarker(const QString &text, const QColor &color)
{
    const bool keepScrolled = atBottom();

    QTextCursor cursor(document());
    cursor.movePosition(QTextCursor::End);
    if (!document()->isEmpty() && !cursor.block().text().isEmpty())
        cursor.insertBlock();

    QTextCharFormat fmt = m_defaultFmt;
    fmt.setForeground(forTheme(color));
    fmt.setFontWeight(QFont::Bold);
    cursor.insertText(text, fmt);
    cursor.insertBlock();
    m_pendingCr = false;

    if (keepScrolled)
        scrollToBottom();
}
