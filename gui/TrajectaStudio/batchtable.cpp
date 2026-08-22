#include "batchtable.h"

#include "smoothcombobox.h"
#include "thememanager.h"
#include "uiwidgets.h"

#include <QApplication>
#include <QBrush>
#include <QComboBox>
#include <QDir>
#include <QEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QLineEdit>
#include <QMenu>
#include <QPainter>
#include <QSpinBox>
#include <QToolButton>

#include <algorithm>
#include <functional>

namespace {

// Status colours already exist in every palette (they are the ones the status
// bar uses), so the batch table needs no new entry in ThemeManager's map.
QColor okColor()      { return ThemeManager::mapped("#7fb08a"); }
QColor badColor()     { return ThemeManager::mapped("#cf7f7f"); }
QColor mutedColor()   { return ThemeManager::mapped("#5c646e"); }
QColor accentColor()  { return ThemeManager::mapped("#7ea8a0"); }

// A line edit with a "..." button, used for every path cell.
class PathCellEditor : public QWidget
{
public:
    PathCellEditor(bool directory, const QString &filter, QWidget *parent)
        : QWidget(parent)
        , m_directory(directory)
        , m_filter(filter)
    {
        auto *layout = new QHBoxLayout(this);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(2);
        m_edit = new QLineEdit(this);
        m_edit->setFrame(false);
        layout->addWidget(m_edit, 1);
        auto *browse = new QToolButton(this);
        browse->setText(QStringLiteral("..."));
        browse->setCursor(Qt::PointingHandCursor);
        browse->setFocusPolicy(Qt::NoFocus);
        layout->addWidget(browse);
        setFocusProxy(m_edit);
        setAutoFillBackground(true);

        connect(browse, &QToolButton::clicked, this, [this] {
            const QString start = m_edit->text().isEmpty()
                                      ? QString()
                                      : QFileInfo(m_edit->text()).absolutePath();
            const QString picked =
                m_directory
                    ? QFileDialog::getExistingDirectory(this, tr("Choose a folder"), start)
                    : QFileDialog::getOpenFileName(this, tr("Choose a file"), start, m_filter);
            if (!picked.isEmpty())
                m_edit->setText(QDir::toNativeSeparators(picked));
            m_edit->setFocus();
        });
    }

    QString path() const { return m_edit->text(); }
    void setPath(const QString &p) { m_edit->setText(p); }

private:
    bool m_directory;
    QString m_filter;
    QLineEdit *m_edit = nullptr;
};

} // namespace

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

BatchTableModel::BatchTableModel(TrajectaRunner::Mode mode, QObject *parent)
    : QAbstractTableModel(parent)
    , m_mode(mode)
{
    rebuildColumns();
}

void BatchTableModel::rebuildColumns()
{
    m_columns.clear();
    auto add = [this](ColumnId id, Kind kind, const QString &title,
                      const QString &filter = QString(),
                      const QStringList &items = QStringList(),
                      int lo = 0, int hi = 1000000) {
        m_columns.append(Column{id, kind, title, filter, items, lo, hi});
    };

    const QString rasterFilter = tr("Rasters (*.tif *.tiff);;All files (*)");
    const QString vectorFilter = tr("Vectors (*.shp *.gpkg *.geojson *.csv);;All files (*)");

    add(ColStatus, Kind::Status, tr("Status"));
    add(ColDem, Kind::FilePath, tr("DEM"), rasterFilter);

    if (m_mode == TrajectaRunner::Mode::Fete) {
        add(ColPointsSource, Kind::Choice, tr("Points source"), QString(),
            {tr("Import from a file"), tr("Generate from the DEM")});
        add(ColPointsFile, Kind::FilePath, tr("Points file"), vectorFilter);
        add(ColDensityMode, Kind::Choice, tr("Density given as"), QString(),
            {tr("Point spacing"), tr("Target number of points")});
        add(ColDensityValue, Kind::Integer, tr("Spacing / target"), QString(),
            QStringList(), 1, 100000000);
        add(ColArrangement, Kind::Choice, tr("Arrangement"), QString(),
            {tr("Regular grid"), tr("Stratified random")});
        add(ColSeed, Kind::Integer, tr("Seed"), QString(), QStringList(), 0, 1000000000);
        add(ColEdgeBuffer, Kind::Integer, tr("Edge buffer"), QString(),
            QStringList(), 0, 100000);
        add(ColLayerName, Kind::Text, tr("Layer name"));
    } else {
        add(ColOrigin, Kind::FilePath, tr("Origin"), vectorFilter);
        add(ColDestinations, Kind::FilePath, tr("Destination(s)"), vectorFilter);
    }

    add(ColOutputDir, Kind::DirPath, tr("Output folder"));
    add(ColOutputName, Kind::Text,
        m_mode == TrajectaRunner::Mode::Fete ? tr("Density raster name")
                                             : tr("Paths raster name"));
    if (m_mode == TrajectaRunner::Mode::Lcpa)
        add(ColPathLinesName, Kind::Text, tr("Paths shapefile name"));
}

void BatchTableModel::setMode(TrajectaRunner::Mode mode)
{
    if (m_mode == mode)
        return;
    beginResetModel();
    m_mode = mode;
    rebuildColumns();
    endResetModel();
    emit rowsChanged();
}

int BatchTableModel::rowCount(const QModelIndex &parent) const
{
    return parent.isValid() ? 0 : m_rows.size();
}

int BatchTableModel::columnCount(const QModelIndex &parent) const
{
    return parent.isValid() ? 0 : m_columns.size();
}

int BatchTableModel::columnIndex(ColumnId id) const
{
    for (int i = 0; i < m_columns.size(); ++i)
        if (m_columns.at(i).id == id)
            return i;
    return -1;
}

bool BatchTableModel::isCellInactive(int row, int section) const
{
    if (row < 0 || row >= m_rows.size() || section < 0 || section >= m_columns.size())
        return false;
    const Batch::Row &r = m_rows.at(row);
    switch (m_columns.at(section).id) {
    case ColPointsFile:
        return r.generatePoints;
    case ColDensityMode:
    case ColDensityValue:
    case ColArrangement:
    case ColEdgeBuffer:
    case ColLayerName:
        return !r.generatePoints;
    case ColSeed:
        // Only a stratified random arrangement has anything to seed.
        return !r.generatePoints || !r.genRandom;
    default:
        return false;
    }
}

QVariant BatchTableModel::cellValue(const Batch::Row &row, ColumnId id) const
{
    switch (id) {
    case ColDem:           return row.demPath;
    case ColPointsSource:  return row.generatePoints ? 1 : 0;
    case ColPointsFile:    return row.pointsPath;
    case ColDensityMode:   return row.genByTargetCount ? 1 : 0;
    case ColDensityValue:  return row.genByTargetCount ? row.genTargetCount : row.genSpacing;
    case ColArrangement:   return row.genRandom ? 1 : 0;
    case ColSeed:          return row.genSeed;
    case ColEdgeBuffer:    return row.genEdgeBuffer;
    case ColLayerName:     return row.genLayerName;
    case ColOrigin:        return row.originPath;
    case ColDestinations:  return row.destinationsPath;
    case ColOutputDir:     return row.outputDir;
    case ColOutputName:    return row.outputName;
    case ColPathLinesName: return row.pathLinesName;
    case ColStatus:        break;
    }
    return QVariant();
}

bool BatchTableModel::applyValue(Batch::Row &row, ColumnId id, const QVariant &value)
{
    switch (id) {
    case ColDem:           row.demPath = value.toString(); return true;
    case ColPointsSource:  row.generatePoints = value.toInt() == 1; return true;
    case ColPointsFile:    row.pointsPath = value.toString(); return true;
    case ColDensityMode:   row.genByTargetCount = value.toInt() == 1; return true;
    case ColDensityValue:
        // One column drives two fields; which one depends on the mode of that
        // same row, so the number the user sees is always the one in use.
        if (row.genByTargetCount)
            row.genTargetCount = value.toInt();
        else
            row.genSpacing = value.toInt();
        return true;
    case ColArrangement:   row.genRandom = value.toInt() == 1; return true;
    case ColSeed:          row.genSeed = value.toInt(); return true;
    case ColEdgeBuffer:    row.genEdgeBuffer = value.toInt(); return true;
    case ColLayerName:     row.genLayerName = value.toString(); return true;
    case ColOrigin:        row.originPath = value.toString(); return true;
    case ColDestinations:  row.destinationsPath = value.toString(); return true;
    case ColOutputDir:     row.outputDir = value.toString(); return true;
    case ColOutputName:    row.outputName = value.toString(); return true;
    case ColPathLinesName: row.pathLinesName = value.toString(); return true;
    case ColStatus:        break;
    }
    return false;
}

QVariant BatchTableModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid() || index.row() >= m_rows.size())
        return QVariant();
    const Batch::Row &row = m_rows.at(index.row());
    const Column &col = m_columns.at(index.column());

    if (col.kind == Kind::Status) {
        if (role == Qt::ToolTipRole)
            return status(index.row()).message;
        return QVariant();
    }

    switch (role) {
    case Qt::DisplayRole: {
        const QVariant v = cellValue(row, col.id);
        if (col.kind == Kind::Choice) {
            const int i = v.toInt();
            return i >= 0 && i < col.items.size() ? col.items.at(i) : QString();
        }
        if (col.kind == Kind::FilePath || col.kind == Kind::DirPath)
            return QDir::toNativeSeparators(v.toString());
        return v;
    }
    case Qt::EditRole:
        return cellValue(row, col.id);
    case Qt::ForegroundRole:
        if (isCellInactive(index.row(), index.column()))
            return QBrush(mutedColor());
        return QVariant();
    case Qt::ToolTipRole: {
        if (isCellInactive(index.row(), index.column()))
            return tr("Not used by this row.");
        const QVariant v = cellValue(row, col.id);
        if (col.kind == Kind::FilePath || col.kind == Kind::DirPath)
            return v.toString();
        return QVariant();
    }
    case Qt::TextAlignmentRole:
        if (col.kind == Kind::Integer)
            return int(Qt::AlignRight | Qt::AlignVCenter);
        return int(Qt::AlignLeft | Qt::AlignVCenter);
    default:
        break;
    }
    return QVariant();
}

bool BatchTableModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    if (!index.isValid() || index.row() >= m_rows.size())
        return false;
    const Column &col = m_columns.at(index.column());
    if (col.kind == Kind::Status)
        return false;

    if (role != Qt::EditRole)
        return false;

    if (!applyValue(m_rows[index.row()], col.id, value))
        return false;

    emit dataChanged(index, index);
    // Switching the points source or the density mode changes which cells are
    // active and what the value column means, so the whole row is repainted.
    emit dataChanged(this->index(index.row(), 0),
                     this->index(index.row(), m_columns.size() - 1));
    emit rowsChanged();
    return true;
}

Qt::ItemFlags BatchTableModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return Qt::NoItemFlags;
    const Column &col = m_columns.at(index.column());
    Qt::ItemFlags f = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
    if (col.kind == Kind::Status)
        return f;
    if (isCellInactive(index.row(), index.column()))
        return Qt::ItemIsSelectable;  // visible, dim, and not editable
    return f | Qt::ItemIsEditable;
}

QVariant BatchTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole) {
        if (orientation == Qt::Horizontal && section < m_columns.size())
            return m_columns.at(section).title;
        if (orientation == Qt::Vertical)
            return section + 1;
    }
    if (role == Qt::ToolTipRole && orientation == Qt::Horizontal)
        return tr("Right-click for copy down and sequential fill.");
    return QVariant();
}

void BatchTableModel::setRows(const QList<Batch::Row> &rows)
{
    beginResetModel();
    m_rows = rows;
    m_status.clear();
    m_status.resize(rows.size());
    endResetModel();
    emit rowsChanged();
}

void BatchTableModel::addRow(const Batch::Row &row)
{
    addRows({row});
}

void BatchTableModel::addRows(const QList<Batch::Row> &rows)
{
    if (rows.isEmpty())
        return;
    beginInsertRows(QModelIndex(), m_rows.size(), m_rows.size() + rows.size() - 1);
    m_rows.append(rows);
    for (int i = 0; i < rows.size(); ++i)
        m_status.append(Status{});
    endInsertRows();
    emit rowsChanged();
}

void BatchTableModel::removeRowsAt(const QList<int> &rows)
{
    QList<int> sorted = rows;
    std::sort(sorted.begin(), sorted.end(), std::greater<int>());
    for (int r : sorted) {
        if (r < 0 || r >= m_rows.size())
            continue;
        beginRemoveRows(QModelIndex(), r, r);
        m_rows.removeAt(r);
        m_status.removeAt(r);
        endRemoveRows();
    }
    emit rowsChanged();
}

void BatchTableModel::duplicateRow(int row)
{
    if (row < 0 || row >= m_rows.size())
        return;
    beginInsertRows(QModelIndex(), row + 1, row + 1);
    m_rows.insert(row + 1, m_rows.at(row));
    m_status.insert(row + 1, Status{});
    endInsertRows();
    emit rowsChanged();
}

void BatchTableModel::moveRowTo(int from, int to)
{
    if (from < 0 || from >= m_rows.size() || to < 0 || to >= m_rows.size() || from == to)
        return;
    beginResetModel();
    m_rows.move(from, to);
    m_status.move(from, to);
    endResetModel();
    emit rowsChanged();
}

void BatchTableModel::setStatus(int row, const Status &status)
{
    if (row < 0 || row >= m_status.size())
        return;
    m_status[row] = status;
    const QModelIndex i = index(row, 0);
    emit dataChanged(i, i);
}

void BatchTableModel::clearStatuses()
{
    for (Status &s : m_status)
        s = Status{};
    if (!m_status.isEmpty())
        emit dataChanged(index(0, 0), index(m_status.size() - 1, 0));
}

BatchTableModel::Status BatchTableModel::status(int row) const
{
    return row >= 0 && row < m_status.size() ? m_status.at(row) : Status{};
}

void BatchTableModel::fillColumnFrom(int section, int sourceRow)
{
    if (section < 0 || section >= m_columns.size())
        return;
    if (sourceRow < 0 || sourceRow >= m_rows.size())
        return;
    const ColumnId id = m_columns.at(section).id;
    const QVariant v = cellValue(m_rows.at(sourceRow), id);
    for (int r = 0; r < m_rows.size(); ++r) {
        if (r == sourceRow)
            continue;
        applyValue(m_rows[r], id, v);
    }
    emit dataChanged(index(0, 0), index(m_rows.size() - 1, m_columns.size() - 1));
    emit rowsChanged();
}

void BatchTableModel::fillColumnSequential(int section, const QString &base, int start)
{
    if (section < 0 || section >= m_columns.size())
        return;
    const ColumnId id = m_columns.at(section).id;
    for (int r = 0; r < m_rows.size(); ++r)
        applyValue(m_rows[r], id, base + QString::number(start + r));
    emit dataChanged(index(0, 0), index(m_rows.size() - 1, m_columns.size() - 1));
    emit rowsChanged();
}

// ---------------------------------------------------------------------------
// Delegate
// ---------------------------------------------------------------------------

BatchItemDelegate::BatchItemDelegate(QObject *parent)
    : QStyledItemDelegate(parent)
{
}

QWidget *BatchItemDelegate::createEditor(QWidget *parent,
                                         const QStyleOptionViewItem &option,
                                         const QModelIndex &index) const
{
    const auto *model = qobject_cast<const BatchTableModel *>(index.model());
    if (!model)
        return QStyledItemDelegate::createEditor(parent, option, index);
    const BatchTableModel::Column &col = model->column(index.column());

    switch (col.kind) {
    case BatchTableModel::Kind::FilePath:
        return new PathCellEditor(false, col.filter, parent);
    case BatchTableModel::Kind::DirPath:
        return new PathCellEditor(true, QString(), parent);
    case BatchTableModel::Kind::Choice: {
        // The same combo as everywhere else in the application, animated arrow
        // included: a drop-down in a table should not look like a different
        // control from the one three centimetres above it.
        auto *combo = new SmoothComboBox(parent);
        combo->addItems(col.items);
        return combo;
    }
    case BatchTableModel::Kind::Integer: {
        auto *spin = new QSpinBox(parent);
        // Especially here: the wheel is how a long batch is scrolled, and the
        // editor sits under the pointer while it is open.
        TrajectaUi::guardWheel(spin);
        spin->setRange(col.min, col.max);
        spin->setFrame(false);
        return spin;
    }
    default:
        return QStyledItemDelegate::createEditor(parent, option, index);
    }
}

void BatchItemDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    // dynamic_cast, not qobject_cast: PathCellEditor is local to this file and
    // carries no Q_OBJECT, which qobject_cast requires.
    if (auto *path = dynamic_cast<PathCellEditor *>(editor)) {
        path->setPath(index.data(Qt::EditRole).toString());
        return;
    }
    if (auto *combo = qobject_cast<QComboBox *>(editor)) {
        combo->setCurrentIndex(index.data(Qt::EditRole).toInt());
        return;
    }
    if (auto *spin = qobject_cast<QSpinBox *>(editor)) {
        spin->setValue(index.data(Qt::EditRole).toInt());
        return;
    }
    QStyledItemDelegate::setEditorData(editor, index);
}

void BatchItemDelegate::setModelData(QWidget *editor, QAbstractItemModel *model,
                                     const QModelIndex &index) const
{
    // dynamic_cast, not qobject_cast: PathCellEditor is local to this file and
    // carries no Q_OBJECT, which qobject_cast requires.
    if (auto *path = dynamic_cast<PathCellEditor *>(editor)) {
        model->setData(index, path->path(), Qt::EditRole);
        return;
    }
    if (auto *combo = qobject_cast<QComboBox *>(editor)) {
        model->setData(index, combo->currentIndex(), Qt::EditRole);
        return;
    }
    if (auto *spin = qobject_cast<QSpinBox *>(editor)) {
        spin->interpretText();
        model->setData(index, spin->value(), Qt::EditRole);
        return;
    }
    QStyledItemDelegate::setModelData(editor, model, index);
}

bool BatchItemDelegate::eventFilter(QObject *object, QEvent *event)
{
    if (event->type() == QEvent::FocusOut) {
        // The file dialog opened from the cell steals the focus; letting the
        // base class close the editor here would delete it from under the
        // handler that opened the dialog.
        if (QApplication::activeModalWidget())
            return false;
        auto *w = qobject_cast<QWidget *>(object);
        QWidget *focus = QApplication::focusWidget();
        if (w && focus && w->isAncestorOf(focus))
            return false;
    }
    return QStyledItemDelegate::eventFilter(object, event);
}

void BatchItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option,
                              const QModelIndex &index) const
{
    const auto *model = qobject_cast<const BatchTableModel *>(index.model());
    if (!model || model->column(index.column()).kind != BatchTableModel::Kind::Status) {
        QStyledItemDelegate::paint(painter, option, index);
        return;
    }

    // Draw the cell's background and selection, but never its text: the status
    // is a glyph or a bar. Passing an invalid index to the base class instead
    // would hit its own Q_ASSERT and leaves the text of the style option
    // untouched, which is how the layer names ended up here.
    QStyleOptionViewItem opt = option;
    initStyleOption(&opt, index);
    opt.text.clear();
    opt.features &= ~QStyleOptionViewItem::HasDisplay;
    const QWidget *w = opt.widget;
    (w ? w->style() : QApplication::style())
        ->drawControl(QStyle::CE_ItemViewItem, &opt, painter, w);

    const BatchTableModel::Status st = model->status(index.row());
    QRect r = option.rect.adjusted(6, 0, -6, 0);
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, true);

    using RowState = BatchController::RowState;
    switch (st.state) {
    case RowState::Running: {
        // A slim bar, so a long batch reads as a column of progress at a glance.
        const int h = 6;
        QRect bar(r.left(), r.center().y() - h / 2, r.width(), h);
        painter->setPen(Qt::NoPen);
        painter->setBrush(mutedColor().darker(130));
        painter->drawRoundedRect(bar, 3, 3);
        QRect fill = bar;
        fill.setWidth(int(bar.width() * qBound(0.0, st.percent, 100.0) / 100.0));
        painter->setBrush(accentColor());
        painter->drawRoundedRect(fill, 3, 3);
        break;
    }
    case RowState::Done: {
        painter->setPen(QPen(okColor(), 2));
        const QPoint c = r.center();
        painter->drawLine(c.x() - 6, c.y(), c.x() - 2, c.y() + 4);
        painter->drawLine(c.x() - 2, c.y() + 4, c.x() + 6, c.y() - 5);
        break;
    }
    case RowState::Failed:
    case RowState::Invalid: {
        painter->setPen(QPen(badColor(), 2));
        const QPoint c = r.center();
        painter->drawLine(c.x() - 5, c.y() - 5, c.x() + 5, c.y() + 5);
        painter->drawLine(c.x() + 5, c.y() - 5, c.x() - 5, c.y() + 5);
        break;
    }
    case RowState::Cancelled: {
        painter->setPen(QPen(mutedColor(), 2));
        const QPoint c = r.center();
        painter->drawEllipse(c, 5, 5);
        painter->drawLine(c.x() - 4, c.y() + 4, c.x() + 4, c.y() - 4);
        break;
    }
    case RowState::Pending:
        painter->setPen(QPen(mutedColor(), 2));
        painter->drawLine(r.center().x() - 5, r.center().y(),
                          r.center().x() + 5, r.center().y());
        break;
    }
    painter->restore();
}

QSize BatchItemDelegate::sizeHint(const QStyleOptionViewItem &option,
                                  const QModelIndex &index) const
{
    QSize s = QStyledItemDelegate::sizeHint(option, index);
    s.setHeight(qMax(s.height(), 30));
    return s;
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

BatchTableView::BatchTableView(QWidget *parent)
    : QTableView(parent)
{
    // Named so theme.qss can reach it with "QFrame#Card QTableView#BatchTable":
    // the sheet makes every widget inside a card transparent, and only a more
    // specific selector than that can give the table a background of its own.
    setObjectName(QStringLiteral("BatchTable"));
    setItemDelegate(new BatchItemDelegate(this));
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setSelectionMode(QAbstractItemView::ExtendedSelection);
    setAlternatingRowColors(false);
    setWordWrap(false);
    setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::SelectedClicked
                    | QAbstractItemView::EditKeyPressed | QAbstractItemView::AnyKeyPressed);
    setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    verticalHeader()->setDefaultSectionSize(30);
    horizontalHeader()->setSectionsMovable(false);
    // Interactive, not stretched: the widths start wide enough to read (see
    // applyDefaultColumnWidths) and the user can still drag one wider for a
    // very long path.
    horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    horizontalHeader()->setStretchLastSection(true);
    horizontalHeader()->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(horizontalHeader(), &QWidget::customContextMenuRequested,
            this, &BatchTableView::showHeaderMenu);
}

void BatchTableView::setBatchModel(BatchTableModel *model)
{
    m_model = model;
    setModel(model);
    // The columns are rebuilt when the mode changes, and the widths have to be
    // recomputed for the new headings.
    connect(model, &BatchTableModel::modelReset,
            this, &BatchTableView::applyDefaultColumnWidths);
    applyDefaultColumnWidths();
}

void BatchTableView::applyDefaultColumnWidths()
{
    if (!m_model)
        return;

    // Every generation column stays visible whatever the rows contain: they go
    // dim and read-only for a row that imports its points (isCellInactive), but
    // a column that disappears when the last row switches source is disorienting
    // and makes the table jump about while it is being filled in.
    const QFontMetrics headerFm(horizontalHeader()->font());
    const QFontMetrics cellFm(font());
    // Sort indicator, cell padding and the frame; measured against the style's
    // own margins rather than guessed per column.
    const int headerPad = 34;
    const int cellPad = 26;

    horizontalHeader()->resizeSection(0, 64);
    for (int c = 1; c < m_model->columnCount(); ++c) {
        const BatchTableModel::Column &col = m_model->column(c);
        int w = headerFm.horizontalAdvance(col.title) + headerPad;

        switch (col.kind) {
        case BatchTableModel::Kind::Choice:
            // The whole point: "Generate from the DEM" has to be readable
            // without dragging the column wider first.
            for (const QString &item : col.items)
                w = qMax(w, cellFm.horizontalAdvance(item) + cellPad + 24);  // + arrow
            break;
        case BatchTableModel::Kind::FilePath:
        case BatchTableModel::Kind::DirPath:
            // A path never fits whole; this is enough for the file name and the
            // tail of its folder, and the tooltip carries the rest.
            w = qMax(w, 240);
            break;
        case BatchTableModel::Kind::Integer:
            w = qMax(w, 110);
            break;
        default:
            w = qMax(w, 150);
            break;
        }
        horizontalHeader()->resizeSection(c, w);
    }
}

void BatchTableView::showHeaderMenu(const QPoint &pos)
{
    if (!m_model || m_model->rowCount() == 0)
        return;
    const int section = horizontalHeader()->logicalIndexAt(pos);
    if (section < 0)
        return;
    const BatchTableModel::Column &col = m_model->column(section);
    if (col.kind == BatchTableModel::Kind::Status)
        return;

    QMenu menu(this);
    const int current = currentIndex().isValid() ? currentIndex().row() : 0;
    QAction *copyDown = menu.addAction(
        tr("Copy row %1 down the whole column").arg(current + 1));
    QAction *sequential = nullptr;
    if (col.kind == BatchTableModel::Kind::Text)
        sequential = menu.addAction(tr("Fill with a numbered sequence..."));

    QAction *chosen = menu.exec(horizontalHeader()->mapToGlobal(pos));
    if (!chosen)
        return;
    if (chosen == copyDown) {
        m_model->fillColumnFrom(section, current);
    } else if (chosen == sequential) {
        bool ok = false;
        const QString base = QInputDialog::getText(
            this, tr("Numbered sequence"),
            tr("Prefix — the row number is appended, so \"output_\" gives "
               "output_1, output_2, ..."),
            QLineEdit::Normal, QStringLiteral("output_"), &ok);
        if (ok && !base.isEmpty())
            m_model->fillColumnSequential(section, base, 1);
    }
}
