#pragma once

#include <QAbstractTableModel>
#include <QStyledItemDelegate>
#include <QTableView>

#include "batchcontroller.h"
#include "batchmodel.h"

// The rows of one chunk, as a table.
//
// Columns are described once, in a list built from the analysis mode, and the
// model, the delegate and the header menu all read that list — so adding a
// field is one entry, not four switch statements.
//
// A cell that does not apply to its row (the points file of a row that
// generates its points, the seed of a regular grid) is not hidden: it goes
// read-only and dim, the same treatment the single-run form gives the cost
// modifier fields when they are switched off. Hiding it per row is impossible
// anyway — a column exists for the whole table, not for one row.
class BatchTableModel : public QAbstractTableModel
{
    Q_OBJECT

public:
    enum ColumnId {
        ColStatus,
        ColDem,
        ColPointsSource,
        ColPointsFile,
        ColDensityMode,
        ColDensityValue,
        ColArrangement,
        ColSeed,
        ColEdgeBuffer,
        ColLayerName,
        ColOrigin,
        ColDestinations,
        ColOutputDir,
        ColOutputName,
        ColPathLinesName,
    };

    enum class Kind { Status, Text, FilePath, DirPath, Choice, Integer };

    struct Column {
        ColumnId id = ColStatus;
        Kind kind = Kind::Text;
        QString title;
        QString filter;     // FilePath only
        QStringList items;  // Choice only
        int min = 0;        // Integer only
        int max = 1000000;
    };

    // Live state of a row while the batch runs, painted in the status column.
    struct Status {
        BatchController::RowState state = BatchController::RowState::Pending;
        double percent = 0.0;
        QString message;
    };

    explicit BatchTableModel(TrajectaRunner::Mode mode, QObject *parent = nullptr);

    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role) const override;
    bool setData(const QModelIndex &index, const QVariant &value, int role) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

    void setMode(TrajectaRunner::Mode mode);
    TrajectaRunner::Mode mode() const { return m_mode; }

    const QList<Batch::Row> &rows() const { return m_rows; }
    void setRows(const QList<Batch::Row> &rows);
    void addRow(const Batch::Row &row);
    void addRows(const QList<Batch::Row> &rows);
    // Named apart from QAbstractItemModel::removeRows/moveRow on purpose:
    // taking the same names would hide the base virtuals.
    void removeRowsAt(const QList<int> &rows);
    void duplicateRow(int row);
    void moveRowTo(int from, int to);

    const Column &column(int section) const { return m_columns.at(section); }
    int columnIndex(ColumnId id) const;
    // A cell that does not apply to its row.
    bool isCellInactive(int row, int section) const;

    void setStatus(int row, const Status &status);
    void clearStatuses();
    Status status(int row) const;

    // Copy one cell down the whole column, and the sequential fill that makes
    // 30 output names bearable to type.
    void fillColumnFrom(int section, int sourceRow);
    void fillColumnSequential(int section, const QString &base, int start);

signals:
    void rowsChanged();  // anything that may alter column visibility or validity

private:
    void rebuildColumns();
    QVariant cellValue(const Batch::Row &row, ColumnId id) const;
    bool applyValue(Batch::Row &row, ColumnId id, const QVariant &value);

    TrajectaRunner::Mode m_mode;
    QList<Column> m_columns;
    QList<Batch::Row> m_rows;
    QList<Status> m_status;
};

// Editors for the cell kinds: a line edit with a "..." button for paths, a
// combo for choices, a spin box for numbers.
class BatchItemDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    explicit BatchItemDelegate(QObject *parent = nullptr);

    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                          const QModelIndex &index) const override;
    void setEditorData(QWidget *editor, const QModelIndex &index) const override;
    void setModelData(QWidget *editor, QAbstractItemModel *model,
                      const QModelIndex &index) const override;
    void paint(QPainter *painter, const QStyleOptionViewItem &option,
               const QModelIndex &index) const override;
    QSize sizeHint(const QStyleOptionViewItem &option,
                   const QModelIndex &index) const override;
    // The browse button opens a modal dialog, which takes the focus away from
    // the editor. Without this the default filter would close (and delete) the
    // editor while its own button handler is still running.
    bool eventFilter(QObject *object, QEvent *event) override;
};

// A QTableView that knows about the batch: sizes its columns so every heading
// and every drop-down entry reads in full, and offers copy-down / sequential
// fill from the header.
class BatchTableView : public QTableView
{
    Q_OBJECT

public:
    explicit BatchTableView(QWidget *parent = nullptr);

    void setBatchModel(BatchTableModel *model);
    // Column widths taken from the text they have to show, so nothing has to be
    // dragged wider by hand before it can be read.
    void applyDefaultColumnWidths();

private:
    void showHeaderMenu(const QPoint &pos);

    BatchTableModel *m_model = nullptr;
};
