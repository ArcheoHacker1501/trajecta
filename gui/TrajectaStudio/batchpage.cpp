#include "batchpage.h"

#include "checkpointstore.h"
#include "confirmdialog.h"
#include "largepages.h"
#include "neighbourhood.h"   // kMin/kMax for the custom neighbours box
#include "consoleview.h"
#include "smoothcombobox.h"
#include "thememanager.h"
#include "uiwidgets.h"

#include <QCheckBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QEasingCurve>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPropertyAnimation>
#include <QPushButton>
#include <QSpinBox>
#include <QStyle>
#include <QThread>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>

#include "systeminfo.h"

#include <algorithm>

namespace {

// Same shape as MainWindow::makeCard, and the same object names, so the cards
// on this page are styled by theme.qss exactly like every other card.
QFrame *makeCard(const QString &title, const QString &subtitle, QWidget *content,
                 QWidget *parent)
{
    auto *card = new QFrame(parent);
    card->setObjectName(QStringLiteral("Card"));
    auto *layout = new QVBoxLayout(card);
    layout->setContentsMargins(18, 14, 18, 16);
    layout->setSpacing(6);

    // A chunk card draws its own header row (title plus its buttons), so it
    // passes an empty title and gets no label of its own.
    if (!title.isEmpty()) {
        auto *titleLabel = new QLabel(title, card);
        titleLabel->setObjectName(QStringLiteral("CardTitle"));
        layout->addWidget(titleLabel);
    }

    if (!subtitle.isEmpty()) {
        auto *sub = new QLabel(subtitle, card);
        sub->setObjectName(QStringLiteral("CardSubtitle"));
        sub->setWordWrap(true);
        layout->addWidget(sub);
    }
    layout->addSpacing(4);
    layout->addWidget(content);
    return card;
}

QPushButton *smallButton(const QString &text, const QString &tip, QWidget *parent)
{
    auto *b = new QPushButton(text, parent);
    b->setToolTip(tip);
    b->setCursor(Qt::PointingHandCursor);
    // A minimum, not a fixed height: theme.qss gives buttons their own padding
    // and a fixed value clips the label on the larger UI fonts. 38 is what the
    // stylesheet's own padding produces on an ordinary button, so these come
    // out the same thickness as every other button instead of a thin strip.
    b->setMinimumHeight(38);
    return b;
}

QLineEdit *pathRow(QGridLayout *grid, int row, const QString &label,
                   const QString &help, const QString &filter, QWidget *parent,
                   bool directory = false, const QString &placeholder = QString())
{
    grid->addWidget(TrajectaUi::makeFieldLabel(label, help, parent), row, 0);
    auto *edit = new QLineEdit(parent);
    if (!placeholder.isEmpty())
        edit->setPlaceholderText(placeholder);
    grid->addWidget(edit, row, 1);
    auto *browse = new QPushButton(QStringLiteral("..."), parent);
    // Minimum, not fixed: the stylesheet's 16 px side padding eats a fixed 38
    // and leaves a single dot showing.
    browse->setMinimumWidth(48);
    browse->setMaximumWidth(60);
    browse->setCursor(Qt::PointingHandCursor);
    grid->addWidget(browse, row, 2);
    QObject::connect(browse, &QPushButton::clicked, edit, [edit, filter, directory, parent] {
        const QString start = edit->text().isEmpty()
                                  ? QString()
                                  : QFileInfo(edit->text()).absolutePath();
        const QString picked =
            directory ? QFileDialog::getExistingDirectory(parent, QObject::tr("Choose a folder"), start)
                      : QFileDialog::getOpenFileName(parent, QObject::tr("Choose a file"), start, filter);
        if (!picked.isEmpty())
            edit->setText(QDir::toNativeSeparators(picked));
    });
    return edit;
}

} // namespace

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

BatchChunkWidget::BatchChunkWidget(TrajectaRunner::Mode mode, QWidget *parent)
    : QWidget(parent)
    , m_mode(mode)
{
    auto *outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(10);

    auto *content = new QWidget(this);
    auto *layout = new QVBoxLayout(content);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);

    // --- header: title + chunk-level buttons ---
    auto *header = new QWidget(content);
    auto *headerRow = new QHBoxLayout(header);
    headerRow->setContentsMargins(0, 0, 0, 0);
    headerRow->setSpacing(6);
    m_title = new QLabel(tr("Chunk"), header);
    m_title->setObjectName(QStringLiteral("CardTitle"));
    headerRow->addWidget(m_title);
    headerRow->addStretch(1);
    auto *up = smallButton(QStringLiteral("↑"), tr("Move this chunk up"), header);
    auto *down = smallButton(QStringLiteral("↓"), tr("Move this chunk down"), header);
    auto *dup = smallButton(tr("Duplicate"),
                            tr("Copy this chunk with all its rows — the quickest way to "
                               "run the same inputs through a different algorithm."),
                            header);
    auto *del = smallButton(QStringLiteral("−"), tr("Delete this chunk"), header);
    // Moving and copying are ordinary actions and get the filled treatment;
    // only the one that destroys work is set apart.
    for (QPushButton *b : {up, down, dup})
        b->setObjectName(QStringLiteral("PrimaryButton"));
    del->setObjectName(QStringLiteral("DangerButton"));
    headerRow->addWidget(TrajectaUi::makeHelpDot(
        tr("<b>What these buttons do</b><br><br>"
           "<b>↑ and ↓</b> move this chunk up or down the page, and that order "
           "<i>is</i> the processing order: the batch runs the chunks from the "
           "top of the page to the bottom, and every row of a chunk before "
           "moving on to the next one. Put the analyses you most want finished "
           "first at the top.<br><br>"
           "<b>Duplicate</b> makes a copy of this chunk with all its rows, "
           "placed immediately below the original. It is the quickest way to "
           "run the same inputs through a different algorithm or a different "
           "set of cost modifiers: duplicate, then change the one thing.<br><br>"
           "<b>−</b> deletes this chunk and every row in it. You are asked to "
           "confirm when it still has rows, and a batch always keeps at least "
           "one chunk."),
        header));
    for (QPushButton *b : {up, down, dup, del})
        headerRow->addWidget(b);
    connect(up, &QPushButton::clicked, this, &BatchChunkWidget::moveUpRequested);
    connect(down, &QPushButton::clicked, this, &BatchChunkWidget::moveDownRequested);
    connect(dup, &QPushButton::clicked, this, &BatchChunkWidget::duplicateRequested);
    connect(del, &QPushButton::clicked, this, &BatchChunkWidget::removeRequested);
    layout->addWidget(header);

    // Everything below the header folds away; the header itself always stays,
    // so a collapsed chunk can still be moved, duplicated or deleted.
    m_body = new QWidget(content);
    auto *bodyLayout = new QVBoxLayout(m_body);
    bodyLayout->setContentsMargins(0, 0, 0, 0);
    bodyLayout->setSpacing(14);

    // --- algorithm ---
    auto *algo = new QWidget(m_body);
    auto *algoRow = new QHBoxLayout(algo);
    algoRow->setContentsMargins(0, 0, 0, 0);
    algoRow->setSpacing(10);
    algoRow->addWidget(TrajectaUi::makeFieldLabel(
        tr("Neighbours"), TrajectaUi::neighboursHelpText(), algo));
    m_neighbours = new SmoothComboBox(algo);
    // The number alone, as on the setup form.
    for (int n : {8, 16, 24, 32, 64})
        m_neighbours->addItem(QString::number(n), n);
    m_neighbours->addItem(tr("Custom…"), 0);
    m_neighbours->setCurrentIndex(1);
    algoRow->addWidget(m_neighbours, 1);

    // Same arrangement as the setup form: the box only appears on "Custom…".
    m_neighboursCustom = new QSpinBox(algo);
    m_neighboursCustom->setRange(neighbourhood::kMin, neighbourhood::kMax);
    m_neighboursCustom->setValue(48);
    m_neighboursCustom->setSuffix(tr(" directions"));
    m_neighboursCustom->setVisible(false);
    TrajectaUi::guardWheel(m_neighboursCustom);
    algoRow->addWidget(m_neighboursCustom);
    // The other algorithm controls are read on demand by toChunk() rather than
    // signalling, so these follow suit; only the visibility and the snapping
    // need doing here.
    connect(m_neighbours, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] {
        m_neighboursCustom->setVisible(m_neighbours->currentData().toInt() == 0);
    });
    connect(m_neighboursCustom, &QSpinBox::editingFinished, this, [this] {
        m_neighboursCustom->setValue(
            TrajectaUi::snapNeighbourCount(m_neighboursCustom->value()));
    });
    algoRow->addWidget(TrajectaUi::makeFieldLabel(
        tr("Cost function"),
        tr("The walking-speed model applied to the slope. All three are "
           "published hiking functions; they differ in how sharply they "
           "penalise steep ground."),
        algo));
    m_costFunction = new SmoothComboBox(algo);
    m_costFunction->addItem(tr("Modified Tobler's Hiking Function (White 2015)"), 1);
    m_costFunction->addItem(tr("Márquez-Pérez et al. (2017)"), 2);
    m_costFunction->addItem(tr("Irmischer & Clarke (2017) — on-path, male"), 3);
    m_costFunction->addItem(tr("Herzog (2013) — energy, kJ/kg"), 4);
    m_costFunction->addItem(tr("Campbell et al. (2019) — 5th percentile"), 5);
    m_costFunction->addItem(tr("Campbell et al. (2019) — 50th percentile"), 6);
    algoRow->addWidget(m_costFunction, 2);
    algoRow->addWidget(TrajectaUi::makeFieldLabel(
        tr("Smoothing"),
        tr("Widens each computed path by this many cells before it is "
           "accumulated, which turns a one-pixel line into a corridor. 0 leaves "
           "the paths exactly as computed."),
        algo));
    m_smoothing = new QSpinBox(algo);
    TrajectaUi::guardWheel(m_smoothing);
    m_smoothing->setRange(0, 10);
    m_smoothing->setSuffix(tr(" cell(s)"));
    algoRow->addWidget(m_smoothing);
    bodyLayout->addWidget(algo);

    // --- slope cut-off, one row of its own so the algorithm row stays readable ---
    auto *capRow = new QWidget(m_body);
    auto *capLayout = new QHBoxLayout(capRow);
    capLayout->setContentsMargins(0, 0, 0, 0);
    capLayout->setSpacing(8);
    m_slopeCap = new QCheckBox(tr("Refuse moves steeper than"), capRow);
    capLayout->addWidget(m_slopeCap);
    m_slopeCapUp = new QSpinBox(capRow);
    m_slopeCapUp->setRange(1, 89);
    m_slopeCapUp->setValue(30);
    m_slopeCapUp->setSuffix(tr("\u00b0 uphill"));
    m_slopeCapUp->setEnabled(false);
    TrajectaUi::guardWheel(m_slopeCapUp);
    capLayout->addWidget(m_slopeCapUp);
    m_slopeCapDown = new QSpinBox(capRow);
    m_slopeCapDown->setRange(1, 89);
    m_slopeCapDown->setValue(30);
    m_slopeCapDown->setSuffix(tr("\u00b0 downhill"));
    m_slopeCapDown->setEnabled(false);
    TrajectaUi::guardWheel(m_slopeCapDown);
    capLayout->addWidget(m_slopeCapDown);
    capLayout->addWidget(TrajectaUi::makeHelpDot(TrajectaUi::slopeCutoffHelpText(), capRow));
    capLayout->addStretch(1);
    connect(m_slopeCap, &QCheckBox::toggled, this, [this](bool on) {
        m_slopeCapUp->setEnabled(on);
        m_slopeCapDown->setEnabled(on);
    });
    bodyLayout->addWidget(capRow);

    // --- cost corridor, LCPA only ---
    m_corridorRow = new QWidget(m_body);
    auto *corrLayout = new QHBoxLayout(m_corridorRow);
    corrLayout->setContentsMargins(0, 0, 0, 0);
    corrLayout->setSpacing(8);
    m_corridorCheck = new QCheckBox(tr("Also compute the cost corridor"), m_corridorRow);
    corrLayout->addWidget(m_corridorCheck);
    m_corridorWidth = new QDoubleSpinBox(m_corridorRow);
    m_corridorWidth->setRange(1.0, 500.0);
    m_corridorWidth->setDecimals(1);
    m_corridorWidth->setValue(10.0);
    m_corridorWidth->setSuffix(tr("% above the optimum"));
    m_corridorWidth->setEnabled(false);
    TrajectaUi::guardWheel(m_corridorWidth);
    corrLayout->addWidget(m_corridorWidth);
    corrLayout->addWidget(
        TrajectaUi::makeHelpDot(TrajectaUi::costCorridorHelpText(), m_corridorRow));
    corrLayout->addStretch(1);
    connect(m_corridorCheck, &QCheckBox::toggled, this, [this](bool on) {
        m_corridorWidth->setEnabled(on);
    });
    bodyLayout->addWidget(m_corridorRow);

    // --- cost modifiers ---
    m_modifiers = new QGroupBox(tr("Cost modifiers for every row of this chunk"), m_body);
    m_modifiers->setCheckable(true);
    m_modifiers->setChecked(false);
    // See the setup form: the title band is row 0, and the note shares the line
    // with the title only if the row starts where the title is drawn.
    m_modifiers->setStyleSheet(QStringLiteral("QGroupBox { margin-top: 0px; }"));
    auto *modGrid = new QGridLayout(m_modifiers);
    // The same arrangement as the single-run form: row 0 is the title band, and
    // the note shares it with the title. The bottom margin keeps the last row
    // clear of the table underneath.
    modGrid->setContentsMargins(8, 0, 8, 16);
    modGrid->setHorizontalSpacing(12);
    modGrid->setVerticalSpacing(16);
    modGrid->setColumnStretch(1, 1);
    // The same warning the setup form carries, in the same words: a chunk's
    // modifiers cost exactly what a single run's do, once per row.
    {
        QWidget *note = TrajectaUi::makeGroupNote(m_modifiers,
                                                  TrajectaUi::costModifiersNoteText(),
                                                  TrajectaUi::costModifiersHelpText());
        note->setMinimumHeight(m_modifiers->fontMetrics().height() + 8);
        // Beside the switch it qualifies, not at the far end of the line; the
        // note indents itself past the title (TitleFollower in uiwidgets.cpp).
        modGrid->addWidget(note, 0, 0, 1, 3, Qt::AlignLeft | Qt::AlignVCenter);
    }
    m_costVector = pathRow(modGrid, 1, tr("Vector modifiers"),
                           tr("A vector layer whose features carry a cost multiplier: "
                              "below 1 makes the ground cheaper to cross (a path, a "
                              "ford), above 1 makes it dearer (a marsh, a river). "
                              "Applies to every row of this chunk."),
                           tr("Vectors (*.shp *.gpkg *.geojson *.csv);;All files (*)"),
                           m_modifiers, false,
                           tr("Polylines with a 'cost' attribute — leave empty to skip"));
    modGrid->addWidget(TrajectaUi::makeFieldLabel(
                           tr("Polyline buffer"),
                           tr("How many cells on each side of a modifier line are "
                              "given its multiplier. A road drawn as a single line "
                              "is one pixel wide otherwise."),
                           m_modifiers),
                       2, 0);
    m_polylineBuffer = new QSpinBox(m_modifiers);
    TrajectaUi::guardWheel(m_polylineBuffer);
    m_polylineBuffer->setRange(0, 100);
    m_polylineBuffer->setValue(2);
    m_polylineBuffer->setSuffix(tr(" cell(s) per side"));
    modGrid->addWidget(m_polylineBuffer, 2, 1);
    m_costRaster = pathRow(modGrid, 3, tr("Raster modifiers"),
                           tr("A raster of cost multipliers, one value per cell, on "
                              "the same grid as the DEM. Combined with the vector "
                              "modifiers when both are given."),
                           tr("Rasters (*.tif *.tiff);;All files (*)"), m_modifiers, false,
                           tr("Multiplier raster aligned with the DEM — leave empty to skip"));
    // Checkbox and threshold on one row, side by side: the value only means
    // anything as the tail of that sentence, and across the full width of the
    // card it read as an unrelated field.
    auto *barrierRow = new QWidget(m_modifiers);
    auto *barrierLayout = new QHBoxLayout(barrierRow);
    barrierLayout->setContentsMargins(0, 0, 0, 0);
    barrierLayout->setSpacing(8);
    m_barrier = new QCheckBox(tr("Treat extreme multipliers as impassable barriers"),
                              barrierRow);
    m_barrier->setChecked(true);
    barrierLayout->addWidget(m_barrier);
    m_barrierValue = new QDoubleSpinBox(barrierRow);
    TrajectaUi::guardWheel(m_barrierValue);
    m_barrierValue->setRange(1.0, 1e9);
    m_barrierValue->setDecimals(1);
    m_barrierValue->setValue(1000.0);
    m_barrierValue->setMaximumWidth(140);
    barrierLayout->addWidget(m_barrierValue);
    barrierLayout->addWidget(TrajectaUi::makeHelpDot(
        tr("A multiplier at or above this value makes the cell impassable "
           "instead of merely expensive — a path will go round it however long "
           "the detour. Unticking the box turns every multiplier back into an "
           "ordinary cost, however large."),
        barrierRow));
    barrierLayout->addStretch(1);
    modGrid->addWidget(barrierRow, 4, 0, 1, 3);
    connect(m_barrier, &QCheckBox::toggled, m_barrierValue, &QWidget::setEnabled);
    bodyLayout->addWidget(m_modifiers);

    // --- what the rows of this chunk write and show ---
    // Outside the cost-modifiers group on purpose: these apply whether or not
    // modifiers are used, and a checkable group would grey them out with it.
    m_keepExtra = new QCheckBox(
        tr("Also save the slope and cost surfaces for every row of this chunk"),
        m_body);
    bodyLayout->addWidget(TrajectaUi::withHelpDot(
        m_keepExtra,
        tr("Off by default. A batch is normally run for its main result, and "
           "the intermediate rasters — slope, cost surface and, with modifiers, "
           "the additional and total cost surfaces — take as much disk space "
           "again. The numbers in the main result are identical either way: the "
           "engine simply does not write the files.")));

    auto *viewerRow = new QWidget(m_body);
    auto *viewerLayout = new QHBoxLayout(viewerRow);
    viewerLayout->setContentsMargins(0, 0, 0, 0);
    viewerLayout->setSpacing(18);
    m_loadRasters = new QCheckBox(tr("Load the result rasters into the Viewer"), viewerRow);
    m_loadVectors = new QCheckBox(tr("Load the result vectors into the Viewer"), viewerRow);
    viewerLayout->addWidget(TrajectaUi::withHelpDot(
        m_loadRasters,
        tr("Off by default in a batch: thirty rows would leave thirty rasters "
           "stacked in the Viewer, each holding its file open.")));
    m_loadVectorsRow = TrajectaUi::withHelpDot(
        m_loadVectors,
        tr("Off by default in a batch, for the same reason as the rasters: the "
           "path layers of every row would pile up."));
    viewerLayout->addWidget(m_loadVectorsRow);
    viewerLayout->addStretch(1);
    bodyLayout->addWidget(viewerRow);

    // --- rows ---
    m_model = new BatchTableModel(mode, this);
    m_table = new BatchTableView(m_body);
    m_table->setBatchModel(m_model);
    m_table->setMinimumHeight(160);
    bodyLayout->addWidget(m_table, 1);

    auto *rowButtons = new QWidget(m_body);
    auto *rowRow = new QHBoxLayout(rowButtons);
    rowRow->setContentsMargins(0, 0, 0, 0);
    rowRow->setSpacing(6);
    auto *addRow = smallButton(QStringLiteral("+"), tr("Add an empty row"), rowButtons);
    auto *delRow = smallButton(QStringLiteral("−"), tr("Delete the selected rows"),
                               rowButtons);
    auto *dupRow = smallButton(tr("Duplicate row"), tr("Copy the selected row"), rowButtons);
    auto *fromFiles = smallButton(tr("Add rows from files..."),
                                  tr("Pick several DEMs at once: one row per file, with "
                                     "the output name taken from the file name."),
                                  rowButtons);
    for (QPushButton *b : {addRow, dupRow, fromFiles})
        b->setObjectName(QStringLiteral("PrimaryButton"));
    delRow->setObjectName(QStringLiteral("DangerButton"));
    rowRow->addWidget(addRow);
    rowRow->addWidget(delRow);
    rowRow->addWidget(dupRow);
    rowRow->addWidget(fromFiles);
    rowRow->addWidget(TrajectaUi::makeHelpDot(
        tr("<b>Building the list of rows</b><br><br>"
           "One row is one analysis. The rows of a chunk run from top to "
           "bottom, and they all share the algorithm and the cost modifiers set "
           "above them.<br><br>"
           "<b>+</b> adds an empty row at the end. It inherits the output "
           "folder, the DEM and the point-generation settings of the row above "
           "it, because in a batch those are nearly always the same and "
           "retyping them is the tedious part.<br><br>"
           "<b>−</b> deletes every selected row. Click a row number to select "
           "it, or drag to select several.<br><br>"
           "<b>Duplicate row</b> copies the current row and inserts the copy "
           "just below it. Useful for running the same DEM with a different "
           "number of sample points, or writing to a second output name.<br><br>"
           "<b>Add rows from files…</b> asks for several DEMs at once and "
           "creates one row per file, taking the output name from the file "
           "name. It asks for the output folder once, and only when no row has "
           "one yet. This is the fast way to set up a batch of twenty.<br><br>"
           "A last trick that is easy to miss: <b>right-click a column heading</b> "
           "to copy the current row's value down the whole column, or to fill a "
           "text column with a numbered sequence."),
        rowButtons));
    rowRow->addStretch(1);
    bodyLayout->addWidget(rowButtons);
    m_rowButtons = {addRow, delRow, dupRow, fromFiles, up, down, dup, del};

    layout->addWidget(m_body);

    connect(addRow, &QPushButton::clicked, this, [this] {
        Batch::Row row;
        // A new row inherits the folder of the last one: in a batch they are
        // nearly always the same, and retyping it every time is the single
        // most tedious part of building one.
        if (!m_model->rows().isEmpty()) {
            const Batch::Row &last = m_model->rows().constLast();
            row.outputDir = last.outputDir;
            row.demPath = last.demPath;
            row.generatePoints = last.generatePoints;
            row.genByTargetCount = last.genByTargetCount;
            row.genSpacing = last.genSpacing;
            row.genTargetCount = last.genTargetCount;
            row.genRandom = last.genRandom;
            row.genEdgeBuffer = last.genEdgeBuffer;
        }
        m_model->addRow(row);
        emit changed();
    });
    connect(delRow, &QPushButton::clicked, this, [this] {
        QList<int> rows;
        const auto selected = m_table->selectionModel()->selectedRows();
        for (const QModelIndex &i : selected)
            rows << i.row();
        if (rows.isEmpty())
            return;
        m_model->removeRowsAt(rows);
        emit changed();
    });
    connect(dupRow, &QPushButton::clicked, this, [this] {
        const QModelIndex i = m_table->currentIndex();
        if (i.isValid()) {
            m_model->duplicateRow(i.row());
            emit changed();
        }
    });
    connect(fromFiles, &QPushButton::clicked, this, [this] { addRowsFromFiles(); });
    connect(m_model, &BatchTableModel::rowsChanged, this, &BatchChunkWidget::changed);

    // --- collapse handle, bottom right ---
    auto *handleRow = new QWidget(content);
    auto *handleLayout = new QHBoxLayout(handleRow);
    handleLayout->setContentsMargins(0, 0, 0, 0);
    handleLayout->addStretch(1);
    m_collapseButton = new QToolButton(handleRow);
    m_collapseButton->setObjectName(QStringLiteral("CollapseHandle"));
    m_collapseButton->setText(QStringLiteral("▲"));
    m_collapseButton->setToolTip(tr("Fold this chunk away"));
    m_collapseButton->setCursor(Qt::PointingHandCursor);
    m_collapseButton->setAutoRaise(true);
    handleLayout->addWidget(m_collapseButton);
    layout->addWidget(handleRow);
    connect(m_collapseButton, &QToolButton::clicked, this,
            [this] { setCollapsed(!m_collapsed); });

    // Height, not visibility: animating maximumHeight makes the body grow
    // downwards when it opens and shrink upwards when it closes, which is what
    // the eye expects from a fold.
    m_collapseAnim = new QPropertyAnimation(m_body, "maximumHeight", this);
    m_collapseAnim->setDuration(180);
    m_collapseAnim->setEasingCurve(QEasingCurve::InOutCubic);
    connect(m_collapseAnim, &QPropertyAnimation::finished, this, [this] {
        // Released once open, so the body can still grow when rows are added.
        if (!m_collapsed)
            m_body->setMaximumHeight(QWIDGETSIZE_MAX);
        else
            m_body->setVisible(false);
    });

    outer->addWidget(makeCard(QString(), QString(), content, this));
    // The card draws its own title, so the generated one is dropped and the
    // header row above takes its place.
    setIndex(1);
    applyModeVisibility();
}

void BatchChunkWidget::setCollapsed(bool collapsed, bool animate)
{
    if (m_collapsed == collapsed)
        return;
    m_collapsed = collapsed;
    m_collapseButton->setText(collapsed ? QStringLiteral("▼") : QStringLiteral("▲"));
    m_collapseButton->setToolTip(collapsed ? tr("Unfold this chunk")
                                           : tr("Fold this chunk away"));

    m_collapseAnim->stop();
    if (!animate) {
        m_body->setVisible(!collapsed);
        m_body->setMaximumHeight(collapsed ? 0 : QWIDGETSIZE_MAX);
        return;
    }

    // sizeHint rather than height(): a chunk collapsed before it was ever shown
    // has no height yet, and would open to nothing.
    const int full = qMax(m_body->sizeHint().height(), m_body->height());
    m_body->setVisible(true);
    if (collapsed) {
        m_collapseAnim->setStartValue(full);
        m_collapseAnim->setEndValue(0);
    } else {
        m_body->setMaximumHeight(0);
        m_collapseAnim->setStartValue(0);
        m_collapseAnim->setEndValue(full);
    }
    m_collapseAnim->start();
}

void BatchChunkWidget::addRowsFromFiles()
{
    const QStringList files = QFileDialog::getOpenFileNames(
        this, tr("Choose the DEMs — one row will be created for each"), QString(),
        tr("Rasters (*.tif *.tiff);;All files (*)"));
    if (files.isEmpty())
        return;

    // The folder is asked once: every row of a batch normally writes under the
    // same root, and the per-row subfolder keeps them apart.
    QString outputDir;
    if (!m_model->rows().isEmpty())
        outputDir = m_model->rows().constLast().outputDir;
    if (outputDir.isEmpty()) {
        outputDir = QFileDialog::getExistingDirectory(
            this, tr("Where should these rows write their results?"),
            QFileInfo(files.first()).absolutePath());
        if (outputDir.isEmpty())
            return;
    }

    QList<Batch::Row> rows;
    for (const QString &f : files) {
        Batch::Row row;
        row.demPath = QDir::toNativeSeparators(f);
        row.outputDir = QDir::toNativeSeparators(outputDir);
        // The DEM's own name is the one thing that already tells the rows
        // apart, so it becomes the output name.
        row.outputName = QFileInfo(f).completeBaseName();
        row.genLayerName = row.outputName + QStringLiteral("_points");
        rows << row;
    }
    m_model->addRows(rows);
    emit changed();
}

void BatchChunkWidget::setMode(TrajectaRunner::Mode mode)
{
    m_mode = mode;
    m_model->setMode(mode);
    applyModeVisibility();
}

// Called from the constructor too, and that is the point: a chunk is built
// already knowing its mode, but until this ran there as well the LCPA-only
// rows were laid out visible and only corrected when the user happened to
// switch modes — so a batch that started in FETE and stayed there showed them
// the whole time.
void BatchChunkWidget::applyModeVisibility()
{
    const bool lcpa = m_mode == TrajectaRunner::Mode::Lcpa;
    // The corridor is an LCPA answer: a FETE run has no single origin and
    // destination for a detour to be measured against.
    if (m_corridorRow)
        m_corridorRow->setVisible(lcpa);
    // Same for the result vectors. What a FETE row produces is a density
    // raster; the one vector file it can write is the sample-point layer it
    // was asked to generate, which is an input it made for itself rather than
    // a result to stack in the Viewer.
    //
    // Unticked as well as hidden: a chunk switched over from LCPA would
    // otherwise carry the setting on where nothing on the page admits to it.
    if (m_loadVectorsRow) {
        m_loadVectorsRow->setVisible(lcpa);
        if (!lcpa)
            m_loadVectors->setChecked(false);
    }
}

void BatchChunkWidget::setIndex(int index)
{
    m_title->setText(tr("Chunk %1").arg(index));
}

Batch::Chunk BatchChunkWidget::chunk() const
{
    Batch::Chunk c;
    const int preset = m_neighbours->currentData().toInt();
    c.neighbours = preset > 0
        ? preset
        : TrajectaUi::snapNeighbourCount(m_neighboursCustom->value());
    c.costFunction = m_costFunction->currentData().toInt();
    c.costCorridor = m_corridorCheck->isChecked();
    c.corridorWidthPercent = m_corridorWidth->value();
    c.slopeCutoffEnabled = m_slopeCap->isChecked();
    c.maxSlopeUpDeg = m_slopeCapUp->value();
    c.maxSlopeDownDeg = m_slopeCapDown->value();
    c.smoothingBufferRadius = m_smoothing->value();
    c.useCostModifiers = m_modifiers->isChecked();
    c.costVectorPath = m_costVector->text().trimmed();
    c.polylineBufferRadius = m_polylineBuffer->value();
    c.costRasterPath = m_costRaster->text().trimmed();
    c.barrierEnabled = m_barrier->isChecked();
    c.barrierThreshold = m_barrierValue->value();
    c.keepExtraRasters = m_keepExtra->isChecked();
    c.loadRastersInViewer = m_loadRasters->isChecked();
    // Never reported in FETE, where the box is not on the page at all: a batch
    // saved from here should not carry a setting it does not offer.
    c.loadVectorsInViewer =
        m_mode == TrajectaRunner::Mode::Lcpa && m_loadVectors->isChecked();
    c.collapsed = m_collapsed;
    c.rows = m_model->rows();
    return c;
}

void BatchChunkWidget::setChunk(const Batch::Chunk &c)
{
    // A saved value outside the presets belongs in the custom box.
    const int n = m_neighbours->findData(c.neighbours);
    if (n >= 0) {
        m_neighbours->setCurrentIndex(n);
    } else {
        m_neighboursCustom->setValue(TrajectaUi::snapNeighbourCount(c.neighbours));
        m_neighbours->setCurrentIndex(m_neighbours->findData(0));
    }
    m_neighboursCustom->setVisible(m_neighbours->currentData().toInt() == 0);
    const int f = m_costFunction->findData(c.costFunction);
    m_costFunction->setCurrentIndex(f >= 0 ? f : 0);
    m_corridorCheck->setChecked(c.costCorridor);
    m_corridorWidth->setValue(c.corridorWidthPercent);
    m_corridorWidth->setEnabled(c.costCorridor);
    m_slopeCap->setChecked(c.slopeCutoffEnabled);
    m_slopeCapUp->setValue(c.maxSlopeUpDeg);
    m_slopeCapDown->setValue(c.maxSlopeDownDeg);
    m_slopeCapUp->setEnabled(c.slopeCutoffEnabled);
    m_slopeCapDown->setEnabled(c.slopeCutoffEnabled);
    m_smoothing->setValue(c.smoothingBufferRadius);
    m_modifiers->setChecked(c.useCostModifiers);
    m_costVector->setText(c.costVectorPath);
    m_polylineBuffer->setValue(c.polylineBufferRadius);
    m_costRaster->setText(c.costRasterPath);
    m_barrier->setChecked(c.barrierEnabled);
    m_barrierValue->setValue(c.barrierThreshold);
    m_keepExtra->setChecked(c.keepExtraRasters);
    m_loadRasters->setChecked(c.loadRastersInViewer);
    m_loadVectors->setChecked(c.loadVectorsInViewer);
    m_model->setRows(c.rows);
    // After the boxes are filled, not before: a chunk restored into a FETE
    // batch from a file that was saved as LCPA would otherwise arrive with the
    // vector box ticked and hidden.
    applyModeVisibility();
    // No animation here: this runs while the chunk is being built, and folding
    // it open over 180 ms as the page appears looks like a glitch.
    setCollapsed(c.collapsed, false);
}

void BatchChunkWidget::setEditingEnabled(bool enabled)
{
    m_neighbours->setEnabled(enabled);
    m_neighboursCustom->setEnabled(enabled);
    m_slopeCap->setEnabled(enabled);
    m_corridorCheck->setEnabled(enabled);
    m_corridorWidth->setEnabled(enabled && m_corridorCheck->isChecked());
    m_slopeCapUp->setEnabled(enabled && m_slopeCap->isChecked());
    m_slopeCapDown->setEnabled(enabled && m_slopeCap->isChecked());
    m_costFunction->setEnabled(enabled);
    m_smoothing->setEnabled(enabled);
    m_modifiers->setEnabled(enabled);
    for (QWidget *w : m_rowButtons)
        w->setEnabled(enabled);
    // The table stays readable and scrollable while the batch runs — it is
    // where the progress is shown — but nothing in it can be edited.
    m_table->setEditTriggers(enabled ? (QAbstractItemView::DoubleClicked
                                        | QAbstractItemView::SelectedClicked
                                        | QAbstractItemView::EditKeyPressed
                                        | QAbstractItemView::AnyKeyPressed)
                                     : QAbstractItemView::NoEditTriggers);
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

BatchPage::BatchPage(QWidget *parent)
    : QWidget(parent)
    , m_controller(new BatchController(this))
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(14);

    // ----- mode + hardware -----
    {
        auto *content = new QWidget(this);
        auto *box = new QVBoxLayout(content);
        box->setContentsMargins(0, 0, 0, 0);
        box->setSpacing(12);

        auto *modeRow = new QWidget(content);
        auto *modeLayout = new QHBoxLayout(modeRow);
        modeLayout->setContentsMargins(0, 0, 0, 0);
        modeLayout->setSpacing(12);
        // Same wording as the single-analysis cards above, on the page this
        // batch card sits under — copied verbatim rather than left at the bare
        // "FETE"/"LCPA" the row used to show, which named the mode but not
        // what it does. The full text is what makes a "?" unnecessary here.
        m_modeFete = new QPushButton(
            tr("FETE — From Everywhere To Everywhere\n"
               "Models general mobility across the landscape"),
            modeRow);
        m_modeFete->setToolTip(
            tr("Computes least-cost paths between every pair of sample points and "
               "accumulates them into a path-density raster: natural movement "
               "corridors and accessibility patterns."));
        m_modeLcpa = new QPushButton(
            tr("LCPA — Least-Cost Path Analysis\n"
               "Computes optimal routes from origin to destination(s)"),
            modeRow);
        m_modeLcpa->setToolTip(
            tr("Computes the optimal routes from a single origin point to one or "
               "more destinations: paths raster and polyline shapefile."));
        // The same two colours the mode cards use on the page above: choosing
        // FETE here is the same decision, made in a smaller place.
        m_modeFete->setProperty("mode", QStringLiteral("fete"));
        m_modeLcpa->setProperty("mode", QStringLiteral("lcpa"));
        for (QPushButton *b : {m_modeFete, m_modeLcpa}) {
            b->setObjectName(QStringLiteral("ModeCard"));
            b->setCheckable(true);
            b->setCursor(Qt::PointingHandCursor);
            b->setMinimumHeight(72);
            modeLayout->addWidget(b);
        }
        m_modeFete->setChecked(true);
        auto switchMode = [this](bool) {
            const TrajectaRunner::Mode m = mode();
            for (BatchChunkWidget *c : m_chunks)
                c->setMode(m);
        };
        connect(m_modeFete, &QPushButton::clicked, this, [this, switchMode](bool) {
            m_modeFete->setChecked(true);
            m_modeLcpa->setChecked(false);
            switchMode(true);
        });
        connect(m_modeLcpa, &QPushButton::clicked, this, [this, switchMode](bool) {
            m_modeLcpa->setChecked(true);
            m_modeFete->setChecked(false);
            switchMode(true);
        });
        box->addWidget(modeRow);

        auto *hw = new QWidget(content);
        auto *hwRow = new QHBoxLayout(hw);
        hwRow->setContentsMargins(0, 0, 0, 0);
        hwRow->setSpacing(10);
        hwRow->addWidget(TrajectaUi::makeFieldLabel(
            tr("CPU threads"),
            tr("How many source points the engine works on at once. The rows run "
               "one after another, so this is the whole machine's budget. Leaving "
               "a few cores free keeps the rest of the system usable."),
            hw));
        m_threads = new QSpinBox(hw);
        TrajectaUi::guardWheel(m_threads);
        m_threads->setRange(1, 1024);
        m_threads->setValue(qMax(1, QThread::idealThreadCount() - 4));
        hwRow->addWidget(m_threads);
        hwRow->addWidget(TrajectaUi::makeFieldLabel(
            tr("Maximum RAM"),
            tr("The ceiling the engine keeps to. The analysis needs far less "
               "than most machines have: at least %1 MB of RAM is recommended, "
               "and raising the ceiling further does not make the computation "
               "any faster. A batch never runs two engines at once, so this is "
               "not divided between rows.")
                    .arg(SystemInfo::kRecommendedRamMb)
                + TrajectaUi::ramHeadroomNote(),
            hw));
        m_ram = new QSpinBox(hw);
        TrajectaUi::guardWheel(m_ram);
        m_ram->setRange(256, 1024 * 1024);
        m_ram->setSuffix(tr(" MB"));
        m_ram->setValue(int(qMin<qint64>(SystemInfo::kRecommendedRamMb,
                                         SystemInfo::totalRamMb())));
        hwRow->addWidget(m_ram);
        auto *ramHint = new QLabel(
            tr("at least %1 MB of RAM is recommended")
                .arg(SystemInfo::kRecommendedRamMb),
            hw);
        ramHint->setObjectName(QStringLiteral("HintLabel"));
        hwRow->addWidget(ramHint);
        hwRow->addStretch(1);
        box->addWidget(hw);

        auto *optionsRow = new QWidget(content);
        auto *optionsLayout = new QHBoxLayout(optionsRow);
        optionsLayout->setContentsMargins(0, 0, 0, 0);
        optionsLayout->setSpacing(24);

        m_folderPerRow = new QCheckBox(
            tr("Create a separate folder for each row, named after its output name"),
            optionsRow);
        m_folderPerRow->setChecked(true);
        optionsLayout->addWidget(TrajectaUi::withHelpDot(
            m_folderPerRow,
            tr("On by default. Without it every row writes into the same folder, "
               "and two rows sharing an output name — or any two rows at all, "
               "once the slope and cost surfaces are saved, since those have "
               "fixed names — overwrite each other.")));

        // Same option and the same words as the single-run form. On by default
        // here and not there: a batch is left running unattended, so the
        // detailed transcript is usually the only account of what happened.
        m_verbose = new QCheckBox(
            tr("Detailed debug output (verbose console log)"), optionsRow);
        m_verbose->setChecked(true);
        optionsLayout->addWidget(TrajectaUi::withHelpDot(
            m_verbose,
            tr("Prints detailed diagnostic messages in the console log. "
               "Useful for troubleshooting and bug reports. On by default in a "
               "batch: nobody is watching while it runs, so the log is what is "
               "left to look at afterwards.")));

        // One manifest per row, written into that row's own output folder.
        m_manifest = new QCheckBox(
            tr("Write a run manifest next to each row's results"), optionsRow);
        m_manifest->setChecked(true);
        optionsLayout->addWidget(
            TrajectaUi::withHelpDot(m_manifest, TrajectaUi::manifestHelpText()));
        optionsLayout->addStretch(1);
        box->addWidget(optionsRow);

        m_settingsCard = makeCard(
            tr("Tool selection"),
            tr("Choose the analysis tool to use."),
            content, this);
        layout->addWidget(m_settingsCard);
    }

    // ----- chunks -----
    m_chunkHost = new QWidget(this);
    // Named for the same reason as AddChunkRow just below, and it matters more
    // here: an unnamed QWidget is caught by the "QMainWindow, QWidget" rule at
    // the top of theme.qss, and this one spans every chunk at once. Its opaque
    // rectangle turned the whole stack into one slab.
    m_chunkHost->setObjectName(QStringLiteral("ChunkHost"));
    m_chunkLayout = new QVBoxLayout(m_chunkHost);
    m_chunkLayout->setContentsMargins(0, 0, 0, 0);
    m_chunkLayout->setSpacing(14);
    layout->addWidget(m_chunkHost);

    {
        auto *addChunkRow = new QWidget(this);
        // Named so the stylesheet can keep it transparent: a full-width opaque
        // band between two cards is a stripe, not a background.
        addChunkRow->setObjectName(QStringLiteral("AddChunkRow"));
        auto *r = new QHBoxLayout(addChunkRow);
        r->setContentsMargins(0, 0, 0, 0);
        auto *addChunkButton = smallButton(tr("+  Add chunk"),
                                           tr("A new group of rows with its own algorithm "
                                              "and cost modifiers."),
                                           addChunkRow);
        addChunkButton->setObjectName(QStringLiteral("PrimaryButton"));
        // Centred, between two stretches: it adds a chunk *below* the last one,
        // so it reads as the seam between the chunk above and the next, rather
        // than as a control belonging to the left edge of the page.
        r->addStretch(1);
        r->addWidget(addChunkButton);
        r->addStretch(1);
        connect(addChunkButton, &QPushButton::clicked, this,
                [this] { addChunk(Batch::Chunk()); });
        layout->addWidget(addChunkRow);
    }

    // ----- run -----
    {
        auto *content = new QWidget(this);
        auto *box = new QVBoxLayout(content);
        box->setContentsMargins(0, 0, 0, 0);
        box->setSpacing(10);

        auto *buttons = new QWidget(content);
        auto *br = new QHBoxLayout(buttons);
        br->setContentsMargins(0, 0, 0, 0);
        br->setSpacing(8);
        m_runButton = new QPushButton(tr("▶  Run batch"), buttons);
        m_runButton->setObjectName(QStringLiteral("PrimaryButton"));
        m_runButton->setCursor(Qt::PointingHandCursor);
        m_pauseButton = smallButton(tr("Pause"), tr("Freeze the running row."), buttons);
        m_pauseButton->setObjectName(QStringLiteral("PrimaryButton"));
        // The two bars, to answer the ▶ on Run batch beside it.
        TrajectaUi::setPauseMark(m_pauseButton, true);
        m_skipButton = smallButton(tr("Skip row"),
                                   tr("Abandon the row in progress and go to the next."),
                                   buttons);
        // Skipping throws away the work done on one row, stopping throws away
        // the rest of the batch: both are the destructive colour.
        m_skipButton->setObjectName(QStringLiteral("DangerButton"));
        m_stopButton = smallButton(tr("Stop batch"), tr("Abandon the whole batch."), buttons);
        m_stopButton->setObjectName(QStringLiteral("DangerButton"));
        br->addWidget(m_runButton);
        br->addWidget(m_pauseButton);
        br->addWidget(m_skipButton);
        br->addWidget(m_stopButton);
        br->addWidget(TrajectaUi::makeHelpDot(
            tr("<b>The four run controls</b><br><br>"
               "<b>Run batch</b> starts at the first row of the first chunk and "
               "works down the queue, one analysis at a time, until it runs out "
               "of rows. Nothing else needs to happen: the page can be left, the "
               "window minimised, the machine unattended.<br><br>"
               "<b>Pause</b> freezes the row being computed. The engine stops "
               "using the processor but keeps everything it has in memory, so "
               "resuming costs nothing and loses nothing — this is the button "
               "for \"I need the machine for an hour\", not for stopping.<br><br>"
               "<b>Skip row</b> abandons the row in progress and moves to the "
               "next one. The work done on that row is lost and it is marked "
               "cancelled; every other row is untouched. Use it when one input "
               "turns out to be wrong and the rest of the queue is still "
               "wanted.<br><br>"
               "<b>Stop batch</b> abandons the row in progress and the whole "
               "queue with it. The rows already finished keep their results, and "
               "the batch can be picked up later from its checkpoint — see the "
               "two buttons at the end of this row."),
            buttons));
        br->addStretch(1);

        // The saved state of the row in progress, worded exactly as on the
        // single-run panel and, like it, buttons rather than links: they do
        // something. They sit at the far end of the row, with the stretch
        // between them and the run controls, because they are not about this
        // batch in this window — they are about a folder on disk that outlives
        // it. MainWindow owns the checkpoint folder and the resume path, so
        // the page only asks.
        auto *saveCkpt = new QPushButton(tr("Save a copy of the checkpoint..."), buttons);
        saveCkpt->setObjectName(QStringLiteral("PrimaryButton"));
        saveCkpt->setCursor(Qt::PointingHandCursor);
        saveCkpt->setToolTip(tr(
            "Writes a copy of the state of the row being computed to a folder "
            "of your choosing. The batch is not affected.\n\n"
            "There is something to copy only once auto-save has written its "
            "first checkpoint."));
        connect(saveCkpt, &QPushButton::clicked,
                this, &BatchPage::exportCheckpointRequested);
        br->addWidget(TrajectaUi::makeHelpDot(
            tr("<b>The two checkpoint buttons</b><br><br>"
               "A checkpoint is the state of a single analysis, written to disk "
               "while it runs, from which the engine can carry on instead of "
               "starting again. On a batch measured in days that is the "
               "difference between a power cut costing an hour and costing the "
               "week.<br><br>"
               "<b>Save a copy of the checkpoint…</b> writes the state of the "
               "row being computed to a folder you choose. The batch carries on "
               "regardless — this only copies. There is something to copy once "
               "auto-save has written its first checkpoint, so not in the first "
               "minutes of a row.<br><br>"
               "<b>Resume from a checkpoint file…</b> picks an interrupted batch "
               "up again: the rows that had finished stay finished, and the row "
               "it stopped on carries on from where it was rather than from its "
               "first source point."),
            buttons));
        br->addWidget(saveCkpt);

        auto *loadCkpt = new QPushButton(tr("Resume from a checkpoint file..."), buttons);
        loadCkpt->setObjectName(QStringLiteral("PrimaryButton"));
        loadCkpt->setCursor(Qt::PointingHandCursor);
        loadCkpt->setToolTip(tr(
            "Picks up an interrupted batch from a checkpoint saved earlier: the "
            "rows already finished stay finished, and the one it stopped on "
            "carries on from where it was."));
        connect(loadCkpt, &QPushButton::clicked,
                this, &BatchPage::importCheckpointRequested);
        br->addWidget(loadCkpt);

        box->addWidget(buttons);

        // The same chip as the setup page, in the same place — above the bar,
        // not below it: the two pages report the same three states and should
        // say so the same way.
        auto *statusRow = new QWidget(content);
        auto *sr = new QHBoxLayout(statusRow);
        sr->setContentsMargins(0, 0, 0, 0);
        sr->setSpacing(12);
        m_chip = new QLabel(tr("IDLE"), statusRow);
        m_chip->setObjectName(QStringLiteral("StateChip"));
        m_chip->setProperty("state", QStringLiteral("idle"));
        m_chip->setAlignment(Qt::AlignCenter);
        m_chip->setMinimumHeight(26);
        m_chip->setMinimumWidth(110);
        sr->addWidget(m_chip);
        m_status = new QLabel(tr("Waiting to start"), statusRow);
        m_status->setObjectName(QStringLiteral("PhaseLabel"));
        sr->addWidget(m_status, 1);
        box->addWidget(statusRow);

        m_progress = new ActivityBar(content);
        m_progress->setRange(0, 100);
        m_progress->setValue(0);
        box->addWidget(m_progress);

        m_summary = new QLabel(QString(), content);
        m_summary->setWordWrap(true);
        m_summary->setVisible(false);
        box->addWidget(m_summary);

        // Folded away by default, exactly as on the single-run panel: a batch
        // left running unattended is watched by its progress bar, and the
        // engine's own output only matters when something needs explaining.
        auto *logHandle = new QToolButton(content);
        m_logHandle = logHandle;
        logHandle->setObjectName(QStringLiteral("LogHandle"));
        logHandle->setCursor(Qt::PointingHandCursor);
        logHandle->setAutoRaise(true);
        logHandle->setCheckable(true);
        logHandle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        logHandle->setArrowType(Qt::RightArrow);
        logHandle->setText(tr("Engine log"));

        auto *logRow = new QHBoxLayout;
        logRow->setSpacing(14);
        logRow->addWidget(logHandle);
        logRow->addStretch(1);
        box->addLayout(logRow);

        m_console = new ConsoleView(content);
        // A fifth shorter than the other log canvases: this is the one page
        // where what sits above the log is the queue itself, and it has to stay
        // in view while the transcript runs.
        m_console->setMinimumHeight(TrajectaUi::kBatchLogCanvasHeight);
        m_console->setVisible(false);
        box->addWidget(m_console);

        ConsoleView *const console = m_console;
        connect(logHandle, &QToolButton::toggled, this, [logHandle, console](bool open) {
            logHandle->setArrowType(open ? Qt::DownArrow : Qt::RightArrow);
            console->setVisible(open);
        });

        // Saving and loading a batch belongs at the end of the page, next to
        // the thing you do with it, and reads as an action rather than as a
        // setting — hence the same filled treatment as Run batch.
        auto *fileRow = new QWidget(content);
        auto *fileLayout = new QHBoxLayout(fileRow);
        fileLayout->setContentsMargins(0, 0, 0, 0);
        fileLayout->setSpacing(8);
        auto *save = smallButton(tr("Save batch..."),
                                 tr("Store this batch in a file so it can be run again."),
                                 fileRow);
        auto *load = smallButton(tr("Load batch..."), tr("Open a saved batch."), fileRow);
        for (QPushButton *b : {save, load})
            b->setObjectName(QStringLiteral("PrimaryButton"));
        fileLayout->addStretch(1);
        fileLayout->addWidget(TrajectaUi::makeHelpDot(
            tr("<b>Saving and loading the batch</b><br><br>"
               "These two are about the <i>queue</i>, not about its results: the "
               "list of rows, the chunks they are grouped into and every setting "
               "on them, in a single <b>.trjbatch</b> file.<br><br>"
               "<b>Save batch…</b> writes that file. It is what makes a piece of "
               "work reproducible — the file says exactly which analyses were "
               "run, and can be published beside the results or sent to someone "
               "else, who will get the same queue on their own machine.<br><br>"
               "<b>Load batch…</b> reads one back, replacing what is currently on "
               "the page. Note that this is not needed to keep your work between "
               "sessions: an unfinished batch is remembered on its own. It is "
               "for keeping a queue deliberately, and for moving it."),
            fileRow));
        fileLayout->addWidget(load);
        fileLayout->addWidget(save);
        connect(save, &QPushButton::clicked, this, &BatchPage::saveToFile);
        connect(load, &QPushButton::clicked, this, &BatchPage::loadFromFile);
        box->addWidget(fileRow);

        m_runCard = makeCard(tr("Run"), QString(), content, this);
        layout->addWidget(m_runCard);

        connect(m_runButton, &QPushButton::clicked, this, &BatchPage::startBatch);
        connect(m_pauseButton, &QPushButton::clicked, this, [this] {
            if (m_controller->isPaused())
                m_controller->resume();
            else
                m_controller->pause();
        });
        connect(m_skipButton, &QPushButton::clicked, this, [this] {
            if (TrajectaUi::confirm(this, tr("Skip row"),
                                    tr("Abandon the row in progress and continue with "
                                       "the next one?")))
                m_controller->skipCurrentRow();
        });
        connect(m_stopButton, &QPushButton::clicked, this, [this] {
            if (TrajectaUi::confirm(this, tr("Stop batch"),
                                    tr("Abandon the whole batch? The rows already "
                                       "finished keep their results.")))
                m_controller->stopBatch();
        });
    }

    layout->addStretch(1);

    // ----- controller wiring -----
    connect(m_controller, &BatchController::consoleOutput,
            m_console, &ConsoleView::appendChunk);
    connect(m_controller, &BatchController::consoleErrorLine, this, [this](const QString &l) {
        m_console->appendMarker(l, ThemeManager::mapped("#cf7f7f"));
    });
    connect(m_controller, &BatchController::statusChanged, m_status, &QLabel::setText);
    connect(m_controller, &BatchController::pauseStateChanged, this, [this](bool paused) {
        // Paused, this button offers play, so it wears the ▶ and not the bars —
        // the same swap the single-run panel makes. The mark and the word have
        // to agree, or the button says two things at once.
        m_pauseButton->setText(paused ? tr("▶ Resume") : tr("Pause"));
        TrajectaUi::setPauseMark(m_pauseButton, !paused);
        setChipState(paused ? QStringLiteral("paused") : QStringLiteral("running"),
                     paused ? tr("PAUSED") : tr("RUNNING"));
        emit tickerChanged();
    });
    connect(m_controller, &BatchController::rowStarted, this, [this](int i) {
        const Batch::Index at = m_controller->results().at(i).where;
        if (at.chunk < m_chunks.size()) {
            m_chunks.at(at.chunk)->model()->setStatus(
                at.row, {BatchController::RowState::Running, 0.0, QString()});
        }
        m_status->setText(tr("Row %1 of %2 — chunk %3, row %4")
                              .arg(i + 1).arg(m_controller->total())
                              .arg(at.chunk + 1).arg(at.row + 1));
        // Rewritten at every row, so a crash halfway through a long batch is
        // recovered at the right place and not at the first row.
        writeSessionForRow(i);
        m_rowPercent = 0.0;
        emit tickerChanged();
    });
    connect(m_controller, &BatchController::rowProgress, this, [this](int i, double pct) {
        const Batch::Index at = m_controller->results().at(i).where;
        if (at.chunk < m_chunks.size()) {
            m_chunks.at(at.chunk)->model()->setStatus(
                at.row, {BatchController::RowState::Running, pct, QString()});
        }
        m_rowPercent = pct;
        emit tickerChanged();
    });
    connect(m_controller, &BatchController::rowFinished, this,
            [this](int i, BatchController::RowState state, const QString &message) {
        const Batch::Index at = m_controller->results().at(i).where;
        if (at.chunk < m_chunks.size())
            m_chunks.at(at.chunk)->model()->setStatus(at.row, {state, 100.0, message});
        if (state == BatchController::RowState::Done)
            publishRowLayers(at);
        if (state != BatchController::RowState::Done) {
            m_console->appendMarker(
                tr("[batch] chunk %1 row %2: %3")
                    .arg(at.chunk + 1).arg(at.row + 1).arg(message.trimmed()),
                ThemeManager::mapped("#cf7f7f"));
        }
    });
    connect(m_controller, &BatchController::batchProgress, this, [this](int done, int total) {
        m_progress->setRange(0, qMax(1, total));
        m_progress->setValue(done);
        m_progress->setFormat(tr("%1 of %2 rows").arg(done).arg(total));
        m_rowsDone = done;
        emit tickerChanged();
    });
    connect(m_controller, &BatchController::batchFinished, this, [this](const QString &report) {
        // Green only when every row that was launched actually succeeded; a
        // batch that ends with failures should not look like a clean finish.
        const auto &results = m_controller->results();
        const bool clean = std::all_of(results.cbegin(), results.cend(),
                                       [](const BatchController::RowResult &r) {
                                           return r.state == BatchController::RowState::Done;
                                       });
        setChipState(clean ? QStringLiteral("success") : QStringLiteral("failed"),
                     clean ? tr("FINISHED") : tr("WITH ERRORS"));
        m_status->setText(tr("Batch finished."));
        m_summary->setText(report);
        m_summary->setVisible(true);
        // A batch stopped while paused would otherwise leave the button
        // offering to resume something that is no longer there.
        m_pauseButton->setText(tr("Pause"));
        TrajectaUi::setPauseMark(m_pauseButton, true);
        m_console->appendMarker(QStringLiteral("\n") + report,
                                ThemeManager::mapped("#7fb08a"));
        setEditingEnabled(true);
        updateRunButtons();
        // A batch the user stopped — including by closing the window — keeps
        // its state, so the next start can offer to pick it up at the row it
        // was on. A batch that simply ran out of rows has nothing to recover.
        if (!m_env.checkpointDir.isEmpty()) {
            if (m_controller->wasStopped()) {
                Checkpoint::Session session =
                    Checkpoint::readSession(m_env.checkpointDir);
                if (session.valid && session.batch) {
                    session.deliberate = true;
                    Checkpoint::writeSession(m_env.checkpointDir, session);
                }
            } else {
                Checkpoint::discard(m_env.checkpointDir);
            }
        }
        emit runningChanged(false);
        emit tickerChanged();
    });

    addChunk(Batch::Chunk());
    updateRunButtons();
}

TrajectaRunner::Mode BatchPage::mode() const
{
    return m_modeLcpa->isChecked() ? TrajectaRunner::Mode::Lcpa
                                   : TrajectaRunner::Mode::Fete;
}

QWidget *BatchChunkWidget::algorithmAnchor() const { return m_neighbours; }
QWidget *BatchChunkWidget::modifiersAnchor() const { return m_modifiers; }
QWidget *BatchChunkWidget::tableAnchor() const { return m_table; }

// The batch block, read the way the page is read: down it. The mode card that
// opens the block is added by MainWindow, because the button belongs to the
// page above; from there this is the settings card, then a chunk, then its
// rows, then the run panel — the order in which a batch is actually built.
QVector<TourStep> BatchPage::walkthroughSteps()
{
    QVector<TourStep> steps;
    // The page always carries at least one chunk (addChunk seeds it in the
    // constructor), but the tour is built at start-up and this is a pointer
    // into a widget the user can delete, so it is checked rather than assumed.
    BatchChunkWidget *chunk = m_chunks.isEmpty() ? nullptr : m_chunks.first();

    {   // Everything the whole batch shares
        TourStep s;
        s.lightCard(m_settingsCard);
        s.title = tr("Settings for the whole batch");
        s.text = tr(
            "The top card is the only part of this page that is not about one "
            "analysis: what it holds applies to <b>every chunk and every row</b> "
            "below it, for the whole run.<br><br>"
            "Hardware is here rather than per chunk because the rows run one "
            "after another — two rows never compete for the machine, so there is "
            "nothing to divide. Everything is saved as you type it, and the "
            "batch is still there after Trajecta is closed and reopened.");
        s.annotations = {
            { m_modeFete, tr("Whether the batch runs FETE or LCPA. Every row is "
                             "the same kind of analysis.") },
            { m_threads, tr("Cores the engine may keep busy while a row runs.") },
            { m_ram, tr("Memory ceiling, the same for every row.") },
            { m_folderPerRow, tr("Give each row a folder of its own — otherwise "
                                 "rows overwrite each other's surfaces.") },
        };
        steps.append(s);
    }

    if (chunk) {
        {   // The anatomy of a chunk
            TourStep s;
            s.lightCard(chunk);
            // The one screen that keeps its callout over the lit card: a chunk
            // is taller than the window, so there is no gap to widen into and
            // trying would only produce a panel the width of the screen.
            s.avoidLitArea = false;
            s.title = tr("A chunk, and what is set on it");
            s.text = tr(
                "A batch is made of chunks, and a chunk is a group of analyses "
                "that share an algorithm and a set of cost modifiers. Everything "
                "at the top of the card applies to every row beneath it — so the "
                "natural way to build a batch is one chunk per question: this "
                "chunk is Tobler at 16 neighbours, that one is Herzog at 8.<br><br>"
                "A batch can have as many chunks as you like, and they run from "
                "the top of the page down.");
            s.annotations = {
                { chunk->algorithmAnchor(),
                  tr("The algorithm for this chunk: neighbours, cost function, "
                     "slope cut-off, smoothing.") },
                { chunk->modifiersAnchor(),
                  tr("Cost modifiers for this chunk, exactly as on the "
                     "single-run form.") },
            };
            steps.append(s);
        }

        {   // The rows
            TourStep s;
            s.targets = { chunk->tableAnchor() };
            // The first four only: rowButtons() also carries the chunk's own
            // up/down/duplicate/delete, which live in the header at the top of
            // the card. Lighting the union of all eight lit the entire chunk,
            // which is the screen before this one.
            const QList<QWidget *> buttons = chunk->rowButtons();
            for (int i = 0; i < qMin(4, buttons.size()); ++i)
                s.targets.append(buttons.at(i));
            s.title = tr("The rows: one analysis each");
            s.text = tr(
                "A row is a whole analysis, and the only things it carries are "
                "the ones that change from one to the next: its <b>DEM</b>, its "
                "<b>sample points</b> (or its origin and destinations in LCPA), "
                "its <b>output folder</b> and the <b>name</b> its results are "
                "written under. Everything else it inherits — the algorithm from "
                "its chunk, the hardware from the card at the top.<br><br>"
                "Cells are edited in place, like a spreadsheet, and a new row "
                "inherits the folder, the DEM and the point settings of the row "
                "above it: in a batch those are nearly always the same, and "
                "retyping them is the tedious part.");
            if (buttons.size() >= 4) {
                s.annotations = {
                    { buttons.at(0), tr("Adds an empty row at the end.") },
                    { buttons.at(1), tr("Deletes every selected row.") },
                    { buttons.at(2), tr("Copies the current row just below itself.") },
                    { buttons.at(3), tr("Asks for several DEMs at once: one row "
                                        "per file, named after it.") },
                };
            }
            steps.append(s);
        }
    }

    {   // The run card: starting it, and watching it
        TourStep s;
        s.lightCard(m_runCard);
        s.title = tr("Starting the batch, and watching it");
        s.text = tr(
            "The same panel as a single run, with one addition: a batch has two "
            "ways to stop, and the difference matters. The page also stays "
            "editable throughout — rows further down the queue can be added or "
            "corrected while the ones above them are still computing.");
        s.annotations = {
            { m_runButton, tr("Starts at the first row and works down.") },
            { m_pauseButton, tr("Freezes the running row, keeping its memory.") },
            { m_skipButton, tr("Abandons this row only; the batch carries on.") },
            { m_stopButton, tr("Abandons the whole queue.") },
            { m_progress, tr("Progress through the whole queue.") },
            { m_logHandle, tr("The engine's own output, for the row running now.") },
        };
        steps.append(s);
    }

    return steps;
}

void BatchPage::addChunk(const Batch::Chunk &chunk)
{
    auto *w = new BatchChunkWidget(mode(), m_chunkHost);
    // Every chunk on the page carries at least one row: a chunk with none
    // cannot be processed, so an empty table is never a state worth showing.
    // The rule lives here because this is the single door a chunk comes
    // through — added, duplicated, restored at start-up or loaded from a file.
    if (chunk.rows.isEmpty()) {
        Batch::Chunk seeded = chunk;
        seeded.rows.append(Batch::Row{});
        w->setChunk(seeded);
    } else {
        w->setChunk(chunk);
    }
    m_chunkLayout->addWidget(w);
    m_chunks.append(w);

    connect(w, &BatchChunkWidget::removeRequested, this, [this, w] { removeChunk(w); });
    connect(w, &BatchChunkWidget::duplicateRequested, this, [this, w] {
        const int i = m_chunks.indexOf(w);
        addChunk(w->chunk());
        // The copy lands right after the original rather than at the end,
        // where it would be far from what it was copied from.
        if (i >= 0 && m_chunks.size() >= 2) {
            BatchChunkWidget *copy = m_chunks.takeLast();
            m_chunks.insert(i + 1, copy);
            m_chunkLayout->removeWidget(copy);
            m_chunkLayout->insertWidget(i + 1, copy);
            renumberChunks();
        }
    });
    connect(w, &BatchChunkWidget::moveUpRequested, this, [this, w] {
        const int i = m_chunks.indexOf(w);
        if (i > 0) {
            m_chunks.move(i, i - 1);
            m_chunkLayout->removeWidget(w);
            m_chunkLayout->insertWidget(i - 1, w);
            renumberChunks();
        }
    });
    connect(w, &BatchChunkWidget::moveDownRequested, this, [this, w] {
        const int i = m_chunks.indexOf(w);
        if (i >= 0 && i + 1 < m_chunks.size()) {
            m_chunks.move(i, i + 1);
            m_chunkLayout->removeWidget(w);
            m_chunkLayout->insertWidget(i + 1, w);
            renumberChunks();
        }
    });
    renumberChunks();
}

void BatchPage::removeChunk(BatchChunkWidget *w)
{
    if (m_chunks.size() <= 1) {
        QMessageBox::information(this, tr("Batch processing"),
                                 tr("A batch needs at least one chunk."));
        return;
    }
    if (!w->model()->rows().isEmpty()
        && !TrajectaUi::confirm(this, tr("Delete chunk"),
                                tr("Delete this chunk and its %1 row(s)?")
                                    .arg(w->model()->rows().size())))
        return;
    m_chunks.removeOne(w);
    m_chunkLayout->removeWidget(w);
    w->deleteLater();
    renumberChunks();
}

void BatchPage::renumberChunks()
{
    for (int i = 0; i < m_chunks.size(); ++i)
        m_chunks.at(i)->setIndex(i + 1);
}

Batch::Job BatchPage::buildJob() const
{
    Batch::Job job;
    job.mode = mode();
    job.maxThreads = m_threads->value();
    job.maxRamMb = m_ram->value();
    // The global switch behind Advanced settings, not a control of this
    // page's own any more — see largepages.h.
    job.largePages = largePagesRequested();
    job.folderPerRow = m_folderPerRow->isChecked();
    job.verbose = m_verbose->isChecked();
    job.writeManifest = m_manifest->isChecked();
    for (BatchChunkWidget *w : m_chunks)
        job.chunks.append(w->chunk());
    return job;
}

void BatchPage::applyJob(const Batch::Job &job)
{
    m_modeFete->setChecked(job.mode == TrajectaRunner::Mode::Fete);
    m_modeLcpa->setChecked(job.mode == TrajectaRunner::Mode::Lcpa);
    m_threads->setValue(job.maxThreads);
    m_ram->setValue(job.maxRamMb);
    // job.largePages is not restored into any control of this page any more:
    // it is read fresh from the global setting when the batch actually runs.
    m_folderPerRow->setChecked(job.folderPerRow);
    m_verbose->setChecked(job.verbose);
    m_manifest->setChecked(job.writeManifest);

    for (BatchChunkWidget *w : m_chunks) {
        m_chunkLayout->removeWidget(w);
        w->deleteLater();
    }
    m_chunks.clear();
    if (job.chunks.isEmpty())
        addChunk(Batch::Chunk());
    else
        for (const Batch::Chunk &c : job.chunks)
            addChunk(c);
}

void BatchPage::setEnvironment(const TrajectaRunner::Parameters &env)
{
    // The checkpoint settings are decided when the batch starts and live in
    // m_env from then on; re-detecting the engine mid-run (a theme change does
    // it) must not wipe them, or the session file would stop being updated
    // halfway through and the recovery would resume at the wrong row.
    const bool cpEnabled = m_env.checkpointEnabled;
    const double cpMinutes = m_env.checkpointMinutes;
    const QString cpDir = m_env.checkpointDir;

    m_env = env;

    if (m_controller->isRunning()) {
        m_env.checkpointEnabled = cpEnabled;
        m_env.checkpointMinutes = cpMinutes;
        m_env.checkpointDir = cpDir;
    }
}

bool BatchPage::isRunning() const
{
    return m_controller->isRunning();
}

void BatchPage::cancelForShutdown()
{
    if (m_controller->isRunning())
        m_controller->stopBatch();
}

void BatchPage::applyTheme()
{
    if (m_console)
        m_console->applyTheme();
}

void BatchPage::openLogs()
{
    if (m_logHandle)
        m_logHandle->setChecked(true);
}

RunTicker::State BatchPage::tickerState() const
{
    RunTicker::State s;
    if (!m_controller->isRunning())
        return s;                      // inactive: the ticker hides itself

    s.active = true;
    s.paused = m_controller->isPaused();

    const int total = qMax(1, m_controller->total());
    // Rows finished, plus how far the running one has got. A bar that only
    // moved when a row ended would sit still for hours on a queue of long
    // analyses, which is the opposite of what this is for.
    const double done = double(m_rowsDone) + qBound(0.0, m_rowPercent, 100.0) / 100.0;
    s.percent = qBound(0.0, 100.0 * done / double(total), 100.0);

    const Batch::Job &job = m_controller->job();
    const int chunkTotal = qMax(1, job.chunks.size());
    int chunkNow = 1;
    const int queueIndex = m_controller->currentQueueIndex();
    if (queueIndex >= 0 && queueIndex < m_controller->results().size())
        chunkNow = m_controller->results().at(queueIndex).where.chunk + 1;
    const int rowNow = qBound(1, queueIndex + 1, total);

    // The chunk count travels with the name: for a batch, "how far through" is
    // asked in chunks long before it is asked in percent.
    s.kind = tr("Batch — %1 · chunk %2/%3")
                 .arg(job.mode == TrajectaRunner::Mode::Lcpa ? tr("LCPA") : tr("FETE"))
                 .arg(chunkNow)
                 .arg(chunkTotal);
    s.chunks = tr("Chunk %1 of %2 · row %3 of %4")
                   .arg(chunkNow).arg(chunkTotal).arg(rowNow).arg(total);
    s.hardware = tr("Hardware: %1 threads · %2 MB")
                     .arg(job.maxThreads).arg(job.maxRamMb);
    s.remaining = TrajectaUi::timeLeftText(
        m_batchClock.isValid() ? m_batchClock.elapsed() : 0, s.percent);
    return s;
}

QVector<bool> BatchPage::unfoldChunks()
{
    QVector<bool> folded;
    folded.reserve(m_chunks.size());
    for (BatchChunkWidget *w : m_chunks) {
        folded.append(w->isCollapsed());
        // Without the animation: the tour measures what it points at as soon as
        // the page is shown, and a card still growing would be measured halfway
        // — which is the whole reason a folded chunk is a problem for it.
        w->setCollapsed(false, false);
    }
    return folded;
}

void BatchPage::restoreChunkFolds(const QVector<bool> &folded)
{
    // By index, and only as far as both lists reach: the list is empty when no
    // tour was ever started, and the page stays editable while one is running.
    for (int i = 0; i < m_chunks.size() && i < folded.size(); ++i)
        m_chunks.at(i)->setCollapsed(folded.at(i), false);
}

void BatchPage::setEditingEnabled(bool enabled)
{
    m_modeFete->setEnabled(enabled);
    m_modeLcpa->setEnabled(enabled);
    m_threads->setEnabled(enabled);
    m_ram->setEnabled(enabled);
    m_folderPerRow->setEnabled(enabled);
    m_verbose->setEnabled(enabled);
    m_manifest->setEnabled(enabled);
    for (BatchChunkWidget *w : m_chunks)
        w->setEditingEnabled(enabled);
}

void BatchPage::updateRunButtons()
{
    const bool running = m_controller->isRunning();
    // m_startAllowed is the single-run panel holding the engine. Only the one
    // button that would start a second engine is blocked by it: the rest of the
    // page stays live, so a batch can be built while an analysis runs.
    m_runButton->setEnabled(!running && m_startAllowed);
    m_pauseButton->setEnabled(running);
    m_skipButton->setEnabled(running);
    m_stopButton->setEnabled(running);
}

// One engine at a time, enforced on the button rather than on the page. The
// page used to be disabled outright, which greyed every field and left the
// buttons as outlines — and made it impossible to prepare a batch while a
// single run was going, which is exactly when there is time to prepare one.
void BatchPage::setStartAllowed(bool allowed)
{
    if (m_startAllowed == allowed)
        return;
    m_startAllowed = allowed;
    m_runButton->setToolTip(allowed
        ? QString()
        : tr("An analysis is already running. It has to finish before a batch "
             "can start — everything else on this page can be set up now."));
    updateRunButtons();
}

void BatchPage::startBatch()
{
    if (m_controller->isRunning())
        return;
    if (m_env.exePath.isEmpty()) {
        QMessageBox::warning(this, tr("Engine not found"),
                             tr("Trajecta Studio cannot find trajecta.exe. Use "
                                "\"Locate engine...\" in the status bar first."));
        return;
    }

    const Batch::Job job = buildJob();
    for (BatchChunkWidget *w : m_chunks)
        w->model()->clearStatuses();
    m_console->clearAll();
    m_summary->setVisible(false);

    // The rows that cannot possibly work are reported up front, in one list,
    // instead of one failure at a time over the next few hours.
    const QList<Batch::Issue> issues = Batch::validate(job);
    if (!issues.isEmpty()) {
        QStringList lines;
        for (const Batch::Issue &i : issues) {
            lines << (i.where.isValid()
                          ? tr("Chunk %1 row %2: %3").arg(i.where.chunk + 1)
                                .arg(i.where.row + 1).arg(i.message)
                          : i.message);
        }
        const bool fatal = std::any_of(issues.cbegin(), issues.cend(),
                                       [](const Batch::Issue &i) { return i.where.row < 0; });
        if (fatal) {
            QMessageBox::warning(this, tr("The batch cannot start"),
                                 lines.join(QLatin1Char('\n')));
            return;
        }
        if (!TrajectaUi::confirm(
                this, tr("Some rows will be skipped"),
                tr("%1 row(s) will not run because of the problems listed in the log. "
                   "Start the batch anyway?").arg(lines.size()))) {
            return;
        }
        for (const QString &l : lines)
            m_console->appendMarker(l, ThemeManager::mapped("#cf7f7f"));
    }

    // Two different things, both governed by the gear menu's autosave setting.
    //
    // The engine can only checkpoint the FETE propagation phase, so only a FETE
    // batch gets the environment variables that turn that on. The *session*
    // file is written in either mode: it is what makes a batch interrupted at
    // row 40 of 50 carry on from row 40 instead of from the first one, and that
    // is worth exactly as much to a batch of LCPA rows.
    const Checkpoint::Settings cp = Checkpoint::settings();
    const QString cpDir = Checkpoint::activeDir();
    const bool wantSession = cp.enabled && !cpDir.isEmpty();
    const bool wantCheckpoints =
        wantSession && job.mode == TrajectaRunner::Mode::Fete;

    // Only one unfinished analysis is kept at a time, and this batch is about
    // to take that place: the engine clears the folder as its first row starts.
    // A batch being resumed is exempt — the state in there is its own.
    if (wantSession && m_resumeQueueIndex < 0) {
        const Checkpoint::Session saved = Checkpoint::readSession(cpDir);
        const Checkpoint::Info info = Checkpoint::latest(cpDir);
        if (saved.valid && (info.found || saved.batch)
            && !TrajectaUi::confirm(
                   this, tr("An unfinished analysis is saved"),
                   tr("%1 was interrupted, and its progress is still saved.\n\n"
                      "Trajecta keeps one unfinished analysis at a time, so "
                      "running this batch deletes it. To keep it instead, cancel "
                      "here, restart Trajecta Studio and choose Resume.\n\n"
                      "Run the batch anyway?")
                       .arg(saved.label.isEmpty() ? tr("An earlier analysis")
                                                  : saved.label),
                   tr("Run anyway"), tr("Cancel"))) {
            return;
        }
    }

    m_env.checkpointEnabled = wantCheckpoints;
    m_env.checkpointMinutes = cp.minutes;
    m_env.checkpointDir = wantSession ? cpDir : QString();
    // Deliberately not clearing the session when autosave is off: a marker in
    // that folder belongs to some other analysis the user is keeping, and this
    // batch — which saves nothing — has no business deleting it.

    BatchController::Resume resume;
    resume.queueIndex = m_resumeQueueIndex;
    resume.checkpointPath = m_resumeCheckpoint;
    // One shot: a batch stopped and started again by hand is a fresh batch.
    m_resumeQueueIndex = -1;
    m_resumeCheckpoint.clear();

    QString error;
    if (!m_controller->start(job, m_env, &error, resume)) {
        QMessageBox::warning(this, tr("The batch cannot start"), error);
        return;
    }
    setEditingEnabled(false);
    updateRunButtons();
    setChipState(QStringLiteral("running"), tr("RUNNING"));
    m_status->setText(tr("Starting..."));
    // The clock the status bar's estimate is made from. Started here rather
    // than on the first row, so the minute the engine spends reading a large
    // DEM is counted as part of the batch, which is what it is.
    m_batchClock.start();
    m_rowsDone = 0;
    m_rowPercent = 0.0;
    emit runningChanged(true);
    emit tickerChanged();
}

void BatchPage::publishRowLayers(const Batch::Index &at)
{
    const Batch::Job &job = m_controller->job();
    if (at.chunk < 0 || at.chunk >= job.chunks.size())
        return;
    const Batch::Chunk &chunk = job.chunks.at(at.chunk);
    if (!chunk.loadRastersInViewer && !chunk.loadVectorsInViewer)
        return;
    if (at.row < 0 || at.row >= chunk.rows.size())
        return;
    const Batch::Row &row = chunk.rows.at(at.row);
    const QDir dir(Batch::outputDirFor(job, row));

    QStringList rasters, vectors;
    auto add = [&](QStringList &into, const QString &name, const QString &ext) {
        const QString trimmed = name.trimmed();
        if (trimmed.isEmpty())
            return;
        const QString path = dir.filePath(trimmed + ext);
        // Only what actually reached the disk: a name can be set for an output
        // the engine ended up skipping.
        if (QFileInfo::exists(path))
            into << QDir::toNativeSeparators(path);
    };

    if (chunk.loadRastersInViewer) {
        add(rasters, row.outputName, QStringLiteral(".tif"));
        if (chunk.keepExtraRasters) {
            add(rasters, QStringLiteral("slope"), QStringLiteral(".tif"));
            add(rasters, QStringLiteral("cost_surface"), QStringLiteral(".tif"));
            add(rasters, QStringLiteral("cost_surface_additional"), QStringLiteral(".tif"));
            add(rasters, QStringLiteral("cost_surface_total"), QStringLiteral(".tif"));
        }
    }
    if (chunk.loadVectorsInViewer) {
        if (job.mode == TrajectaRunner::Mode::Fete) {
            if (row.generatePoints)
                add(vectors, row.genLayerName, QStringLiteral(".shp"));
        } else {
            add(vectors, row.pathLinesName, QStringLiteral(".shp"));
        }
    }
    if (!rasters.isEmpty() || !vectors.isEmpty())
        emit viewerLayersReady(rasters, vectors);
}

void BatchPage::writeSessionForRow(int queueIndex)
{
    // Keyed on the folder, not on checkpointEnabled: an LCPA batch never has
    // engine state to save, but the record of which row it had reached is
    // exactly as useful. startBatch() leaves the folder empty when autosave
    // is off, which is what turns all of this off.
    if (m_env.checkpointDir.isEmpty())
        return;
    Checkpoint::Session session;
    session.batch = true;
    session.job = Batch::toJson(m_controller->job());
    session.queueIndex = queueIndex;
    session.params = Checkpoint::toJson(m_env);
    session.label = tr("Batch — row %1 of %2")
                        .arg(queueIndex + 1).arg(m_controller->total());
    Checkpoint::writeSession(m_env.checkpointDir, session);
}

void BatchPage::resumeJob(const QJsonObject &job, int queueIndex,
                          const QString &checkpointPath)
{
    Batch::Job loaded;
    if (!Batch::fromJson(job, &loaded, nullptr)) {
        QMessageBox::warning(this, tr("Batch processing"),
                             tr("The interrupted batch could not be read back."));
        return;
    }
    applyJob(loaded);
    m_resumeQueueIndex = queueIndex;
    m_resumeCheckpoint = checkpointPath;
    // Started from the event loop so the chunks above have been laid out and
    // the user can see what is about to run.
    QTimer::singleShot(0, this, [this] { startBatch(); });
}

void BatchPage::setChipState(const QString &state, const QString &text)
{
    if (!m_chip)
        return;
    m_chip->setText(text);
    m_chip->setProperty("state", state);
    // A property used in a stylesheet selector is only re-evaluated when the
    // widget is re-polished.
    m_chip->style()->unpolish(m_chip);
    m_chip->style()->polish(m_chip);
}

void BatchPage::saveToFile()
{
    const QString path = QFileDialog::getSaveFileName(
        this, tr("Save the batch"), QString(),
        tr("Trajecta batch (*.trjbatch);;JSON (*.json)"));
    if (path.isEmpty())
        return;
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        QMessageBox::warning(this, tr("Save the batch"),
                             tr("Cannot write %1").arg(path));
        return;
    }
    f.write(QJsonDocument(Batch::toJson(buildJob())).toJson(QJsonDocument::Indented));
}

QString BatchPage::saveState() const
{
    return QString::fromUtf8(
        QJsonDocument(Batch::toJson(buildJob())).toJson(QJsonDocument::Compact));
}

void BatchPage::restoreState(const QString &json)
{
    if (json.trimmed().isEmpty())
        return;
    const QJsonDocument doc = QJsonDocument::fromJson(json.toUtf8());
    Batch::Job job;
    // A stored batch that cannot be read is simply ignored: an empty batch page
    // is a much better greeting than an error box at every start-up.
    if (doc.isObject() && Batch::fromJson(doc.object(), &job, nullptr)) {
        // Same reasoning as the setup form: a RAM ceiling left at one of the
        // defaults nobody ever chose — 60 percent of what is installed, or the
        // flat 4096 MB a 1.0.1 build briefly recommended — was not a decision,
        // so the restored page starts from the current recommendation instead.
        // Batch files opened by hand are left exactly as they were saved.
        const qint64 totalRam = SystemInfo::totalRamMb();
        const int sixtyPercent = int(qMax<qint64>(1024, (totalRam * 60) / 100));
        if (job.maxRamMb == sixtyPercent || job.maxRamMb == 4096)
            job.maxRamMb = int(qMin<qint64>(SystemInfo::kRecommendedRamMb, totalRam));
        applyJob(job);
    }
}

bool BatchPage::loadBatchFile(const QString &path, QString *error)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        if (error)
            *error = tr("Cannot read %1").arg(path);
        return false;
    }
    QJsonParseError perr;
    const QJsonDocument doc = QJsonDocument::fromJson(f.readAll(), &perr);
    if (perr.error != QJsonParseError::NoError) {
        if (error)
            *error = perr.errorString();
        return false;
    }
    Batch::Job job;
    if (!Batch::fromJson(doc.object(), &job, error))
        return false;
    applyJob(job);
    return true;
}

void BatchPage::loadFromFile()
{
    const QString path = QFileDialog::getOpenFileName(
        this, tr("Load a batch"), QString(),
        tr("Trajecta batch (*.trjbatch *.json);;All files (*)"));
    if (path.isEmpty())
        return;
    QString error;
    if (!loadBatchFile(path, &error))
        QMessageBox::warning(this, tr("Load a batch"), error);
}
