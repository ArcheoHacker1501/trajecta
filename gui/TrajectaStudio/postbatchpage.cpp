#include "postbatchpage.h"

#include "checkpointstore.h"
#include "confirmdialog.h"
#include "consoleview.h"
#include "largepages.h"
#include "pathpicker.h"
#include "smoothcombobox.h"
#include "thememanager.h"
#include "uiwidgets.h"

#include <QButtonGroup>
#include <QCheckBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QEasingCurve>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
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

const char *kRasterFilter = "GeoTIFF (*.tif *.tiff);;All files (*)";
const char *kVectorFilter =
    "Vector files (*.shp *.geojson *.json *.kml *.gml *.xml *.csv);;All files (*)";

// Identical shape to batchpage.cpp's own makeCard()/smallButton(): same
// object names, so this page is styled by theme.qss exactly like the other
// batch page and every ordinary card.
QFrame *makeCard(const QString &title, const QString &subtitle, QWidget *content,
                 QWidget *parent)
{
    auto *card = new QFrame(parent);
    card->setObjectName(QStringLiteral("Card"));
    auto *layout = new QVBoxLayout(card);
    layout->setContentsMargins(18, 14, 18, 16);
    layout->setSpacing(6);
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
    b->setMinimumHeight(38);
    return b;
}

} // namespace

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

PostBatchChunkWidget::PostBatchChunkWidget(PostBatch::Mode mode, QWidget *parent)
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

    // --- header: title + chunk-level buttons, identical to BatchChunkWidget ---
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
                            tr("Copy this chunk — the quickest way to run the same "
                               "inputs with one setting changed."),
                            header);
    auto *del = smallButton(QStringLiteral("−"), tr("Delete this chunk"), header);
    for (QPushButton *b : {up, down, dup})
        b->setObjectName(QStringLiteral("PrimaryButton"));
    del->setObjectName(QStringLiteral("DangerButton"));
    headerRow->addWidget(TrajectaUi::makeHelpDot(
        tr("<b>What these buttons do</b><br><br>"
           "<b>↑ and ↓</b> move this chunk up or down the page, and that order "
           "<i>is</i> the processing order: the batch runs the chunks from the "
           "top of the page to the bottom.<br><br>"
           "<b>Duplicate</b> makes a copy of this chunk, placed immediately "
           "below the original — the quickest way to run the same inputs "
           "again with one setting changed.<br><br>"
           "<b>−</b> deletes this chunk. A batch always keeps at least one."),
        header));
    for (QPushButton *b : {up, down, dup, del})
        headerRow->addWidget(b);
    connect(up, &QPushButton::clicked, this, &PostBatchChunkWidget::moveUpRequested);
    connect(down, &QPushButton::clicked, this, &PostBatchChunkWidget::moveDownRequested);
    connect(dup, &QPushButton::clicked, this, &PostBatchChunkWidget::duplicateRequested);
    connect(del, &QPushButton::clicked, this, &PostBatchChunkWidget::removeRequested);
    layout->addWidget(header);

    m_body = new QWidget(content);
    auto *bodyLayout = new QVBoxLayout(m_body);
    bodyLayout->setContentsMargins(0, 0, 0, 0);
    bodyLayout->setSpacing(14);

    buildNniFields(m_body, bodyLayout);
    buildCompareFields(m_body, bodyLayout);
    buildCoherenceFields(m_body, bodyLayout);

    // --- load into the Viewer, present in every mode but one ---
    // Absent for Compare: a comparison measures two layers that already exist,
    // it does not write a new spatial one, so there is nothing here for the
    // checkbox to load. See postbatchmodel.h's Chunk::loadInViewer.
    auto *viewerRow = new QWidget(m_body);
    auto *viewerLayout = new QHBoxLayout(viewerRow);
    viewerLayout->setContentsMargins(0, 0, 0, 0);
    m_loadInViewer = new QCheckBox(tr("Load the result into the Viewer"), viewerRow);
    m_loadInViewer->setChecked(true);
    viewerLayout->addWidget(TrajectaUi::withHelpDot(
        m_loadInViewer,
        tr("On by default: unlike a Processing batch, a post-processing batch "
           "is normally a handful of chunks rather than dozens, so looking at "
           "each result as it finishes is usually the point.")));
    viewerLayout->addStretch(1);
    bodyLayout->addWidget(viewerRow);

    layout->addWidget(m_body);

    // --- collapse handle, bottom right — identical to BatchChunkWidget ---
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

    m_collapseAnim = new QPropertyAnimation(m_body, "maximumHeight", this);
    m_collapseAnim->setDuration(180);
    m_collapseAnim->setEasingCurve(QEasingCurve::InOutCubic);
    connect(m_collapseAnim, &QPropertyAnimation::finished, this, [this] {
        if (!m_collapsed)
            m_body->setMaximumHeight(QWIDGETSIZE_MAX);
        else
            m_body->setVisible(false);
    });

    outer->addWidget(makeCard(QString(), QString(), content, this));
    setIndex(1);
    setMode(mode);
}

void PostBatchChunkWidget::buildNniFields(QWidget *body, QVBoxLayout *bodyLayout)
{
    m_nniBox = new QWidget(body);
    auto *grid = new QGridLayout(m_nniBox);
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setHorizontalSpacing(12);
    grid->setVerticalSpacing(14);
    grid->setColumnStretch(1, 1);

    int r = 0;
    auto addRow = [&](const QString &label, const QString &help, QWidget *w) {
        grid->addWidget(TrajectaUi::makeFieldLabel(label, help, m_nniBox), r, 0);
        grid->addWidget(w, r, 1);
        ++r;
    };

    m_nniInputPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                      tr("Select the density raster (GeoTIFF)"),
                                      QString::fromLatin1(kRasterFilter), m_nniBox);
    m_nniInputPicker->setPlaceholder(
        tr("FETE density raster (.tif) — filled automatically after a FETE run"));
    addRow(tr("Density raster"),
          tr("The raster to interpolate, typically a FETE density output."),
          m_nniInputPicker);

    m_nniOutputDirPicker = new PathPicker(PathPicker::Kind::Directory,
                                          tr("Select the output folder"),
                                          QString(), m_nniBox);
    m_nniOutputDirPicker->setPlaceholder(
        tr("Folder where the interpolated raster will be written"));
    addRow(tr("Output folder"),
          tr("Folder where the interpolated raster of this chunk will be written."),
          m_nniOutputDirPicker);

    m_nniThresholdSpin = new QDoubleSpinBox(m_nniBox);
    m_nniThresholdSpin->setRange(0.0, 1e9);
    m_nniThresholdSpin->setDecimals(1);
    m_nniThresholdSpin->setValue(1.0);
    TrajectaUi::guardWheel(m_nniThresholdSpin);
    addRow(tr("Sample threshold"),
          tr("Cells at or above this value become sample points."),
          m_nniThresholdSpin);

    m_nniSpacingSpin = new QSpinBox(m_nniBox);
    m_nniSpacingSpin->setRange(1, 1000);
    m_nniSpacingSpin->setValue(4);
    m_nniSpacingSpin->setSuffix(tr(" cell(s)"));
    TrajectaUi::guardWheel(m_nniSpacingSpin);
    addRow(tr("Sample spacing"), tr("Samples are taken every N cells."),
          m_nniSpacingSpin);

    m_nniPeaksCheck = new QCheckBox(tr("Preserve local peaks"), m_nniBox);
    grid->addWidget(TrajectaUi::withHelpDot(
                        m_nniPeaksCheck, TrajectaUi::preservePeaksHelpText()),
                    r, 1);
    ++r;

    m_nniRadiusSpin = new QSpinBox(m_nniBox);
    m_nniRadiusSpin->setRange(0, 100000);
    m_nniRadiusSpin->setSuffix(tr(" cell(s)"));
    m_nniRadiusSpin->setSpecialValueText(tr("unlimited"));
    TrajectaUi::guardWheel(m_nniRadiusSpin);
    addRow(tr("Max search radius"), tr("0 is unbounded, the classic behaviour."),
          m_nniRadiusSpin);

    m_nniNameEdit = new QLineEdit(QStringLiteral("FETE_density_NNI"), m_nniBox);
    addRow(tr("Output filename"), tr("Without extension."), m_nniNameEdit);

    bodyLayout->addWidget(m_nniBox);
}

void PostBatchChunkWidget::buildCompareFields(QWidget *body, QVBoxLayout *bodyLayout)
{
    m_cmpBox = new QWidget(body);
    auto *grid = new QGridLayout(m_cmpBox);
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setHorizontalSpacing(12);
    grid->setVerticalSpacing(14);
    grid->setColumnStretch(1, 1);

    int r = 0;
    auto addRow = [&](const QString &label, const QString &help, QWidget *w) {
        grid->addWidget(TrajectaUi::makeFieldLabel(label, help, m_cmpBox), r, 0);
        grid->addWidget(w, r, 1);
        ++r;
    };

    m_cmpComputedPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                         tr("Select the computed routes (vector)"),
                                         QString::fromLatin1(kVectorFilter), m_cmpBox);
    m_cmpComputedPicker->setPlaceholder(
        tr("Usually the LCPA paths shapefile produced by a run"));
    addRow(tr("Computed routes"),
          tr("Usually the LCPA paths shapefile from a run."), m_cmpComputedPicker);

    m_cmpKnownPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                      tr("Select the known route (vector)"),
                                      QString::fromLatin1(kVectorFilter), m_cmpBox);
    m_cmpKnownPicker->setPlaceholder(
        tr("The real route: a surveyed road, a historic track, a mapped path"));
    addRow(tr("Known route"), tr("A surveyed road, a historic track, a mapped path."),
          m_cmpKnownPicker);

    m_cmpToleranceSpin = new QDoubleSpinBox(m_cmpBox);
    m_cmpToleranceSpin->setRange(1.0, 100000.0);
    m_cmpToleranceSpin->setDecimals(0);
    m_cmpToleranceSpin->setValue(100.0);
    m_cmpToleranceSpin->setSuffix(tr(" m"));
    TrajectaUi::guardWheel(m_cmpToleranceSpin);
    addRow(tr("Tolerance"), TrajectaUi::routeCompareHelpText(), m_cmpToleranceSpin);

    bodyLayout->addWidget(m_cmpBox);
}

void PostBatchChunkWidget::buildCoherenceFields(QWidget *body, QVBoxLayout *bodyLayout)
{
    m_cohBox = new QWidget(body);
    auto *grid = new QGridLayout(m_cohBox);
    grid->setContentsMargins(0, 0, 0, 0);
    grid->setHorizontalSpacing(12);
    grid->setVerticalSpacing(14);
    grid->setColumnStretch(1, 1);

    int r = 0;
    auto addRow = [&](const QString &label, const QString &help, QWidget *w) {
        grid->addWidget(TrajectaUi::makeFieldLabel(label, help, m_cohBox), r, 0);
        grid->addWidget(w, r, 1);
        ++r;
    };

    m_cohRasterPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                       tr("Select the FETE surface (raster)"),
                                       QString::fromLatin1(kRasterFilter), m_cohBox);
    m_cohRasterPicker->setPlaceholder(
        tr("The FETE density raster, raw or interpolated with NNI"));
    addRow(tr("FETE surface"), TrajectaUi::coherenceSurfaceHelpText(), m_cohRasterPicker);

    m_cohPointsPicker = new PathPicker(PathPicker::Kind::ExistingFile,
                                       tr("Select the sites (point layer)"),
                                       QString::fromLatin1(kVectorFilter), m_cohBox);
    m_cohPointsPicker->setPlaceholder(
        tr("The sites to score, in the same projected CRS as the raster"));
    addRow(tr("Sites"), TrajectaUi::coherenceSitesHelpText(), m_cohPointsPicker);

    m_cohRadiusSpin = new QDoubleSpinBox(m_cohBox);
    m_cohRadiusSpin->setRange(1.0, 1000000.0);
    m_cohRadiusSpin->setDecimals(0);
    m_cohRadiusSpin->setValue(250.0);
    m_cohRadiusSpin->setSuffix(tr(" m"));
    TrajectaUi::guardWheel(m_cohRadiusSpin);
    addRow(tr("Radius"), TrajectaUi::coherenceRadiusHelpText(), m_cohRadiusSpin);

    {
        auto *box = new QWidget(m_cohBox);
        auto *row = new QHBoxLayout(box);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(10);
        m_cohThresholdCombo = new SmoothComboBox(box);
        m_cohThresholdCombo->addItem(tr("Top percentage of the surface"), 0);
        m_cohThresholdCombo->addItem(tr("Automatic (Otsu, on log values)"), 1);
        m_cohThresholdCombo->addItem(tr("Cells at or above a value"), 2);
        row->addWidget(m_cohThresholdCombo, 1);
        m_cohThresholdSpin = new QDoubleSpinBox(box);
        m_cohThresholdSpin->setRange(0.001, 1000000000.0);
        m_cohThresholdSpin->setDecimals(2);
        m_cohThresholdSpin->setValue(1.0);
        m_cohThresholdSpin->setSuffix(tr(" %"));
        TrajectaUi::guardWheel(m_cohThresholdSpin);
        row->addWidget(m_cohThresholdSpin);
        addRow(tr("Corridor"), TrajectaUi::coherenceThresholdHelpText(), box);
        connect(m_cohThresholdCombo, &QComboBox::currentIndexChanged, this,
                [this] { refreshCoherenceEnablement(); });
    }
    {
        auto *box = new QWidget(m_cohBox);
        auto *row = new QHBoxLayout(box);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(10);
        m_cohSensCheck = new QCheckBox(tr("Also report other radii"), box);
        row->addWidget(m_cohSensCheck);
        m_cohSensEdit = new QLineEdit(QStringLiteral("100, 250, 500, 1000"), box);
        row->addWidget(m_cohSensEdit, 1);
        addRow(tr("Sensitivity"), TrajectaUi::coherenceSensitivityHelpText(), box);
        connect(m_cohSensCheck, &QCheckBox::toggled, this,
                [this] { refreshCoherenceEnablement(); });
    }

    m_cohEcdfEdit = new QLineEdit(QStringLiteral("0, 100, 250, 500, 1000, 2500"),
                                  m_cohBox);
    addRow(tr("Distance bands"), TrajectaUi::coherenceEcdfHelpText(), m_cohEcdfEdit);

    {
        // Comes after the distance bands, not before: it tests those same
        // distances against chance, so the choice it depends on has to be
        // on the page above it.
        auto *box = new QWidget(m_cohBox);
        auto *row = new QHBoxLayout(box);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(10);
        m_cohNullCheck = new QCheckBox(tr("Test against random point sets"), box);
        m_cohNullCheck->setChecked(true);
        row->addWidget(m_cohNullCheck);
        m_cohNullModeCombo = new SmoothComboBox(box);
        m_cohNullModeCombo->addItem(tr("the same pattern, moved as a block"), 0);
        m_cohNullModeCombo->addItem(tr("scattered points"), 1);
        row->addWidget(m_cohNullModeCombo, 1);
        m_cohRepsSpin = new QSpinBox(box);
        m_cohRepsSpin->setRange(99, 9999);
        m_cohRepsSpin->setValue(999);
        m_cohRepsSpin->setPrefix(tr("× "));
        TrajectaUi::guardWheel(m_cohRepsSpin);
        row->addWidget(m_cohRepsSpin);
        addRow(tr("Null model"), TrajectaUi::coherenceNullHelpText(), box);
        connect(m_cohNullCheck, &QCheckBox::toggled, this,
                [this] { refreshCoherenceEnablement(); });
    }

    m_cohEdgeCheck = new QCheckBox(
        tr("Flag sites within one radius of the raster's edge"), m_cohBox);
    m_cohEdgeCheck->setChecked(true);
    addRow(tr("Edge guard"), TrajectaUi::coherenceEdgeHelpText(), m_cohEdgeCheck);

    m_cohRScriptCheck = new QCheckBox(
        tr("Write an R script (ggplot2) for the distance histogram"), m_cohBox);
    m_cohRScriptCheck->setChecked(true);
    addRow(tr("Histogram script"), TrajectaUi::coherenceHistogramScriptHelpText(),
           m_cohRScriptCheck);

    m_cohOutPicker = new PathPicker(PathPicker::Kind::Directory,
                                    tr("Select the output folder"), QString(), m_cohBox);
    m_cohOutPicker->setPlaceholder(tr("Folder where the scored sites will be written"));
    addRow(tr("Output folder"), tr("Where the table, the layer and the distance "
                                   "raster of this chunk are written."), m_cohOutPicker);

    {
        auto *box = new QWidget(m_cohBox);
        auto *row = new QHBoxLayout(box);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(10);
        m_cohPrefixEdit = new QLineEdit(QStringLiteral("coherence"), box);
        row->addWidget(m_cohPrefixEdit, 1);
        m_cohVectorCombo = new SmoothComboBox(box);
        m_cohVectorCombo->addItem(tr("GeoPackage"), 0);
        m_cohVectorCombo->addItem(tr("Shapefile"), 1);
        row->addWidget(m_cohVectorCombo);
        m_cohRasterCheck = new QCheckBox(tr("Distance raster"), box);
        m_cohRasterCheck->setChecked(true);
        row->addWidget(m_cohRasterCheck);
        addRow(tr("Outputs"), TrajectaUi::coherenceOutputHelpText(), box);
    }

    bodyLayout->addWidget(m_cohBox);
}

void PostBatchChunkWidget::refreshCoherenceEnablement()
{
    if (!m_cohThresholdCombo)
        return;
    const int mode = m_cohThresholdCombo->currentData().toInt();
    m_cohThresholdSpin->setEnabled(mode != 1);
    m_cohThresholdSpin->setSuffix(mode == 0 ? tr(" %") : QString());
    m_cohThresholdSpin->setDecimals(mode == 0 ? 2 : 3);
    m_cohThresholdSpin->setMaximum(mode == 0 ? 100.0 : 1.0e9);
    if (mode == 0 && m_cohThresholdSpin->value() > 100.0)
        m_cohThresholdSpin->setValue(1.0);
    const bool null = m_cohNullCheck && m_cohNullCheck->isChecked();
    if (m_cohNullModeCombo)
        m_cohNullModeCombo->setEnabled(null);
    if (m_cohRepsSpin)
        m_cohRepsSpin->setEnabled(null);
    if (m_cohSensEdit && m_cohSensCheck)
        m_cohSensEdit->setEnabled(m_cohSensCheck->isChecked());
}

void PostBatchChunkWidget::setCollapsed(bool collapsed, bool animate)
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

void PostBatchChunkWidget::setMode(PostBatch::Mode mode)
{
    m_mode = mode;
    m_nniBox->setVisible(mode == PostBatch::Mode::Nni);
    m_cmpBox->setVisible(mode == PostBatch::Mode::Compare);
    m_cohBox->setVisible(mode == PostBatch::Mode::Coherence);
    // Compare has no spatial result to register with the Viewer — see the
    // constructor's comment above the checkbox.
    m_loadInViewer->setVisible(mode != PostBatch::Mode::Compare);
}

void PostBatchChunkWidget::setIndex(int index)
{
    m_title->setText(tr("Chunk %1").arg(index));
}

PostBatch::Chunk PostBatchChunkWidget::chunk() const
{
    PostBatch::Chunk c;
    c.interpInputRaster = m_nniInputPicker->path();
    c.interpOutputDir = m_nniOutputDirPicker->path();
    c.interpThreshold = m_nniThresholdSpin->value();
    c.interpSampleSpacing = m_nniSpacingSpin->value();
    c.interpPreservePeaks = m_nniPeaksCheck->isChecked() && m_nniSpacingSpin->value() > 1;
    c.interpMaxRadius = m_nniRadiusSpin->value();
    c.interpOutputName = m_nniNameEdit->text().trimmed();

    c.cmpComputedPath = m_cmpComputedPicker->path();
    c.cmpKnownPath = m_cmpKnownPicker->path();
    c.cmpTolerance = m_cmpToleranceSpin->value();

    c.cohRasterPath = m_cohRasterPicker->path();
    c.cohPointsPath = m_cohPointsPicker->path();
    c.cohRadius = m_cohRadiusSpin->value();
    c.cohThresholdMode = m_cohThresholdCombo->currentData().toInt();
    c.cohThresholdValue = m_cohThresholdSpin->value();
    c.cohNullModel = m_cohNullCheck->isChecked();
    c.cohNullMode = m_cohNullModeCombo->currentData().toInt();
    c.cohNullReplicates = m_cohRepsSpin->value();
    c.cohSensitivity = m_cohSensCheck->isChecked();
    c.cohSensitivityRadii = m_cohSensEdit->text();
    c.cohEcdfDistances = m_cohEcdfEdit->text();
    c.cohEdgeGuard = m_cohEdgeCheck->isChecked();
    c.cohWriteHistogramScript = m_cohRScriptCheck->isChecked();
    c.cohOutputDir = m_cohOutPicker->path();
    c.cohPrefix = m_cohPrefixEdit->text().trimmed();
    c.cohVectorAsGeoPackage = m_cohVectorCombo->currentData().toInt() == 0;
    c.cohWriteDistanceRaster = m_cohRasterCheck->isChecked();

    // Never reported for Compare, where the box is hidden: a chunk saved from
    // here should not carry a setting it does not offer.
    c.loadInViewer = m_mode == PostBatch::Mode::Compare ? false
                                                        : m_loadInViewer->isChecked();
    c.collapsed = m_collapsed;
    return c;
}

void PostBatchChunkWidget::setChunk(const PostBatch::Chunk &c)
{
    m_nniInputPicker->setPath(c.interpInputRaster);
    m_nniOutputDirPicker->setPath(c.interpOutputDir);
    m_nniThresholdSpin->setValue(c.interpThreshold);
    m_nniSpacingSpin->setValue(c.interpSampleSpacing);
    m_nniPeaksCheck->setChecked(c.interpPreservePeaks);
    m_nniRadiusSpin->setValue(c.interpMaxRadius);
    m_nniNameEdit->setText(c.interpOutputName);

    m_cmpComputedPicker->setPath(c.cmpComputedPath);
    m_cmpKnownPicker->setPath(c.cmpKnownPath);
    m_cmpToleranceSpin->setValue(c.cmpTolerance);

    m_cohRasterPicker->setPath(c.cohRasterPath);
    m_cohPointsPicker->setPath(c.cohPointsPath);
    m_cohRadiusSpin->setValue(c.cohRadius);
    const int tIdx = m_cohThresholdCombo->findData(c.cohThresholdMode);
    m_cohThresholdCombo->setCurrentIndex(tIdx >= 0 ? tIdx : 0);
    m_cohThresholdSpin->setValue(c.cohThresholdValue);
    m_cohNullCheck->setChecked(c.cohNullModel);
    const int nIdx = m_cohNullModeCombo->findData(c.cohNullMode);
    m_cohNullModeCombo->setCurrentIndex(nIdx >= 0 ? nIdx : 0);
    m_cohRepsSpin->setValue(c.cohNullReplicates);
    m_cohSensCheck->setChecked(c.cohSensitivity);
    m_cohSensEdit->setText(c.cohSensitivityRadii);
    m_cohEcdfEdit->setText(c.cohEcdfDistances);
    m_cohEdgeCheck->setChecked(c.cohEdgeGuard);
    m_cohRScriptCheck->setChecked(c.cohWriteHistogramScript);
    m_cohOutPicker->setPath(c.cohOutputDir);
    m_cohPrefixEdit->setText(c.cohPrefix);
    m_cohVectorCombo->setCurrentIndex(c.cohVectorAsGeoPackage ? 0 : 1);
    m_cohRasterCheck->setChecked(c.cohWriteDistanceRaster);
    refreshCoherenceEnablement();

    m_loadInViewer->setChecked(c.loadInViewer);
    setMode(m_mode);
    setCollapsed(c.collapsed, false);
}

void PostBatchChunkWidget::setEditingEnabled(bool enabled)
{
    for (QWidget *w : {static_cast<QWidget *>(m_nniBox), m_cmpBox, m_cohBox})
        w->setEnabled(enabled);
    m_loadInViewer->setEnabled(enabled);
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

PostBatchPage::PostBatchPage(QWidget *parent)
    : QWidget(parent)
    , m_controller(new PostBatchController(this))
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(14);

    // ----- tool + hardware -----
    {
        auto *content = new QWidget(this);
        auto *box = new QVBoxLayout(content);
        box->setContentsMargins(0, 0, 0, 0);
        box->setSpacing(12);

        auto *modeRow = new QWidget(content);
        auto *modeLayout = new QHBoxLayout(modeRow);
        modeLayout->setContentsMargins(0, 0, 0, 0);
        modeLayout->setSpacing(12);
        // Same wording as the post-processing tool cards above this page —
        // copied verbatim rather than left at the bare name the row used to
        // show, which named the tool but not what it does.
        m_modeNni = new QPushButton(
            tr("NNI — Natural Neighbour Interpolation\n"
               "Turns a sparse density raster into a continuous surface"),
            modeRow);
        m_modeNni->setToolTip(
            tr("Discrete Sibson interpolation over the cells of a density raster."));
        m_modeCompare = new QPushButton(
            tr("Compare with a known route\n"
               "Measures how closely a computed route follows a real one"),
            modeRow);
        m_modeCompare->setToolTip(
            tr("Geometric comparison between two vector layers: distances in both "
               "directions, and how much of each line runs close to the other."));
        m_modeCoherence = new QPushButton(
            tr("Site-corridor coherence\n"
               "Scores how well a set of sites sits on the predicted corridors"),
            modeRow);
        m_modeCoherence->setToolTip(
            tr("Per-site distance to the nearest corridor and intensity of the "
               "surrounding movement, with a test against random point sets."));
        // Same property values as the post-processing tool cards above this
        // page: the same choice, made in a smaller place — see theme.qss.
        m_modeNni->setProperty("mode", QStringLiteral("nni"));
        m_modeCompare->setProperty("mode", QStringLiteral("compare"));
        m_modeCoherence->setProperty("mode", QStringLiteral("coherence"));
        for (QPushButton *b : {m_modeNni, m_modeCompare, m_modeCoherence}) {
            b->setObjectName(QStringLiteral("ModeCard"));
            b->setCheckable(true);
            b->setCursor(Qt::PointingHandCursor);
            b->setMinimumHeight(72);
            modeLayout->addWidget(b);
        }
        auto *group = new QButtonGroup(modeRow);
        group->setExclusive(true);
        group->addButton(m_modeNni);
        group->addButton(m_modeCompare);
        group->addButton(m_modeCoherence);
        m_modeNni->setChecked(true);
        connect(group, &QButtonGroup::buttonClicked, this, [this](QAbstractButton *) {
            const PostBatch::Mode m = mode();
            for (PostBatchChunkWidget *c : m_chunks)
                c->setMode(m);
            updateHardwareVisibility();
        });
        box->addWidget(modeRow);

        // Hardware, NNI only: Compare and Coherence run in the interface and
        // never see a trajecta.exe process, exactly like the single-run forms.
        m_hardwareRow = new QWidget(content);
        auto *hwBox = new QVBoxLayout(m_hardwareRow);
        hwBox->setContentsMargins(0, 0, 0, 0);
        hwBox->setSpacing(12);

        auto *hw = new QWidget(m_hardwareRow);
        auto *hwRow = new QHBoxLayout(hw);
        hwRow->setContentsMargins(0, 0, 0, 0);
        hwRow->setSpacing(10);
        hwRow->addWidget(TrajectaUi::makeFieldLabel(
            tr("CPU threads"),
            tr("How many threads the interpolation may use. Chunks run one "
               "after another, so this is the whole machine's budget."),
            hw));
        m_threads = new QSpinBox(hw);
        TrajectaUi::guardWheel(m_threads);
        m_threads->setRange(1, 1024);
        m_threads->setValue(qMax(1, QThread::idealThreadCount() - 4));
        hwRow->addWidget(m_threads);
        hwRow->addWidget(TrajectaUi::makeFieldLabel(
            tr("Maximum RAM"),
            tr("The ceiling the engine keeps to. At least %1 MB is recommended.")
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
        hwRow->addStretch(1);
        hwBox->addWidget(hw);

        m_manifest = new QCheckBox(
            tr("Write a run manifest next to each chunk's results"), m_hardwareRow);
        m_manifest->setChecked(true);
        hwBox->addWidget(TrajectaUi::withHelpDot(m_manifest, TrajectaUi::manifestHelpText()));

        box->addWidget(m_hardwareRow);

        m_settingsCard = makeCard(
            tr("Tool selection"),
            tr("Choose the analysis tool to use."),
            content, this);
        layout->addWidget(m_settingsCard);
    }

    // ----- chunks -----
    m_chunkHost = new QWidget(this);
    m_chunkHost->setObjectName(QStringLiteral("ChunkHost"));
    m_chunkLayout = new QVBoxLayout(m_chunkHost);
    m_chunkLayout->setContentsMargins(0, 0, 0, 0);
    m_chunkLayout->setSpacing(14);
    layout->addWidget(m_chunkHost);

    {
        auto *addChunkRow = new QWidget(this);
        addChunkRow->setObjectName(QStringLiteral("AddChunkRow"));
        auto *r = new QHBoxLayout(addChunkRow);
        r->setContentsMargins(0, 0, 0, 0);
        auto *addChunkButton = smallButton(tr("+  Add chunk"),
                                           tr("A new analysis, of the tool selected above."),
                                           addChunkRow);
        addChunkButton->setObjectName(QStringLiteral("PrimaryButton"));
        r->addStretch(1);
        r->addWidget(addChunkButton);
        r->addStretch(1);
        connect(addChunkButton, &QPushButton::clicked, this,
                [this] { addChunk(PostBatch::Chunk()); });
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
        m_pauseButton = smallButton(tr("Pause"), tr("Hold before the next chunk (freezes "
                                                    "an NNI chunk immediately)."), buttons);
        m_pauseButton->setObjectName(QStringLiteral("PrimaryButton"));
        TrajectaUi::setPauseMark(m_pauseButton, true);
        m_skipButton = smallButton(tr("Skip row"),
                                   tr("Abandon the chunk in progress and go to the next."),
                                   buttons);
        m_skipButton->setObjectName(QStringLiteral("DangerButton"));
        m_stopButton = smallButton(tr("Stop batch"), tr("Abandon the whole batch."), buttons);
        m_stopButton->setObjectName(QStringLiteral("DangerButton"));
        br->addWidget(m_runButton);
        br->addWidget(m_pauseButton);
        br->addWidget(m_skipButton);
        br->addWidget(m_stopButton);
        br->addWidget(TrajectaUi::makeHelpDot(
            tr("<b>The four run controls</b><br><br>"
               "<b>Run batch</b> starts at the first chunk and works down the "
               "queue until it runs out.<br><br>"
               "<b>Pause</b> freezes an NNI chunk exactly like the single-run "
               "panel — nothing lost, resuming costs nothing. Compare and "
               "Coherence run in the interface with no process to freeze, so "
               "for those two Pause takes effect once the chunk in progress "
               "finishes, which given how fast they are is normally a moment "
               "away.<br><br>"
               "<b>Skip row</b> abandons the chunk in progress (immediately for "
               "NNI, at the same next-chunk boundary as Pause for Compare and "
               "Coherence) and moves on; every other chunk is untouched.<br><br>"
               "<b>Stop batch</b> abandons the chunk in progress and the whole "
               "queue with it. The chunks already finished keep their results."),
            buttons));
        br->addStretch(1);

        auto *saveCkpt = new QPushButton(tr("Save a copy of the checkpoint..."), buttons);
        saveCkpt->setObjectName(QStringLiteral("PrimaryButton"));
        saveCkpt->setCursor(Qt::PointingHandCursor);
        saveCkpt->setToolTip(tr(
            "Writes a copy of which chunks have finished to a folder of your "
            "choosing. The batch is not affected."));
        connect(saveCkpt, &QPushButton::clicked,
                this, &PostBatchPage::exportCheckpointRequested);
        br->addWidget(TrajectaUi::makeHelpDot(
            tr("<b>The two checkpoint buttons</b><br><br>"
               "None of NNI, Compare or Coherence can resume in the middle of "
               "a chunk — unlike a FETE run, none of the three has a state to "
               "pick up mid-computation. What is kept instead is which chunks "
               "have already finished: a batch interrupted at chunk 6 of 10 "
               "picks up at chunk 6, which starts again from its "
               "beginning.<br><br>"
               "<b>Save a copy of the checkpoint…</b> writes that record to a "
               "folder you choose. The batch carries on regardless.<br><br>"
               "<b>Resume from a checkpoint file…</b> picks an interrupted "
               "batch back up: the chunks that had finished stay finished."),
            buttons));
        br->addWidget(saveCkpt);

        auto *loadCkpt = new QPushButton(tr("Resume from a checkpoint file..."), buttons);
        loadCkpt->setObjectName(QStringLiteral("PrimaryButton"));
        loadCkpt->setCursor(Qt::PointingHandCursor);
        loadCkpt->setToolTip(tr(
            "Picks up an interrupted batch from a checkpoint saved earlier."));
        connect(loadCkpt, &QPushButton::clicked,
                this, &PostBatchPage::importCheckpointRequested);
        br->addWidget(loadCkpt);

        box->addWidget(buttons);

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
        m_console->setMinimumHeight(TrajectaUi::kBatchLogCanvasHeight);
        m_console->setVisible(false);
        box->addWidget(m_console);

        ConsoleView *const console = m_console;
        connect(logHandle, &QToolButton::toggled, this, [logHandle, console](bool open) {
            logHandle->setArrowType(open ? Qt::DownArrow : Qt::RightArrow);
            console->setVisible(open);
        });

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
               "These are about the <i>queue</i>, not its results: the tool, "
               "the hardware and the list of chunks, in a single "
               "<b>.trjpbatch</b> file.<br><br>"
               "<b>Save batch…</b> writes that file, so the exact queue can be "
               "kept or handed to someone else.<br><br>"
               "<b>Load batch…</b> reads one back, replacing what is currently "
               "on the page. Not needed to keep your work between sessions — an "
               "unfinished batch is remembered on its own — this is for keeping "
               "a queue deliberately, or moving it."),
            fileRow));
        fileLayout->addWidget(load);
        fileLayout->addWidget(save);
        connect(save, &QPushButton::clicked, this, &PostBatchPage::saveToFile);
        connect(load, &QPushButton::clicked, this, &PostBatchPage::loadFromFile);
        box->addWidget(fileRow);

        m_runCard = makeCard(tr("Run"), QString(), content, this);
        layout->addWidget(m_runCard);

        connect(m_runButton, &QPushButton::clicked, this, &PostBatchPage::startBatch);
        connect(m_pauseButton, &QPushButton::clicked, this, [this] {
            if (m_controller->isPaused())
                m_controller->resume();
            else
                m_controller->pause();
        });
        connect(m_skipButton, &QPushButton::clicked, this, [this] {
            if (TrajectaUi::confirm(this, tr("Skip row"),
                                    tr("Abandon the chunk in progress and continue "
                                       "with the next one?")))
                m_controller->skipCurrentRow();
        });
        connect(m_stopButton, &QPushButton::clicked, this, [this] {
            if (TrajectaUi::confirm(this, tr("Stop batch"),
                                    tr("Abandon the whole batch? The chunks already "
                                       "finished keep their results.")))
                m_controller->stopBatch();
        });
    }

    layout->addStretch(1);

    connect(m_controller, &PostBatchController::consoleOutput,
            m_console, &ConsoleView::appendChunk);
    connect(m_controller, &PostBatchController::consoleErrorLine, this, [this](const QString &l) {
        m_console->appendMarker(l, ThemeManager::mapped("#cf7f7f"));
    });
    connect(m_controller, &PostBatchController::statusChanged, m_status, &QLabel::setText);
    connect(m_controller, &PostBatchController::pauseStateChanged, this, [this](bool paused) {
        m_pauseButton->setText(paused ? tr("▶ Resume") : tr("Pause"));
        TrajectaUi::setPauseMark(m_pauseButton, !paused);
        setChipState(paused ? QStringLiteral("paused") : QStringLiteral("running"),
                     paused ? tr("PAUSED") : tr("RUNNING"));
        emit tickerChanged();
    });
    connect(m_controller, &PostBatchController::rowStarted, this, [this](int i) {
        if (i < m_chunks.size())
            m_status->setText(tr("Chunk %1 of %2").arg(i + 1).arg(m_controller->total()));
        writeSessionForChunk(i);
        m_rowPercent = 0.0;
        emit tickerChanged();
    });
    connect(m_controller, &PostBatchController::rowProgress, this, [this](int, double pct) {
        m_rowPercent = pct;
        emit tickerChanged();
    });
    connect(m_controller, &PostBatchController::chunkLayersReady, this,
            [this](int i) { publishChunkLayers(i); });
    connect(m_controller, &PostBatchController::rowFinished, this,
            [this](int i, PostBatchController::RowState state, const QString &message) {
        if (state != PostBatchController::RowState::Done) {
            m_console->appendMarker(
                tr("[batch] chunk %1: %2").arg(i + 1).arg(message.trimmed()),
                ThemeManager::mapped("#cf7f7f"));
        }
    });
    connect(m_controller, &PostBatchController::batchProgress, this, [this](int done, int total) {
        m_progress->setRange(0, qMax(1, total));
        m_progress->setValue(done);
        m_progress->setFormat(tr("%1 of %2 chunks").arg(done).arg(total));
        m_rowsDone = done;
        emit tickerChanged();
    });
    connect(m_controller, &PostBatchController::batchFinished, this, [this](const QString &report) {
        const auto &results = m_controller->results();
        const bool clean = std::all_of(
            results.cbegin(), results.cend(),
            [](const PostBatchController::RowResult &r) {
                return r.state == PostBatchController::RowState::Done;
            });
        setChipState(clean ? QStringLiteral("success") : QStringLiteral("failed"),
                     clean ? tr("FINISHED") : tr("WITH ERRORS"));
        m_status->setText(tr("Batch finished."));
        m_summary->setText(report);
        m_summary->setVisible(true);
        m_pauseButton->setText(tr("Pause"));
        TrajectaUi::setPauseMark(m_pauseButton, true);
        m_console->appendMarker(QStringLiteral("\n") + report,
                                ThemeManager::mapped("#7fb08a"));
        setEditingEnabled(true);
        updateRunButtons();
        if (!m_env.checkpointDir.isEmpty()) {
            if (m_controller->wasStopped()) {
                Checkpoint::Session session = Checkpoint::readSession(m_env.checkpointDir);
                if (session.valid && session.batch && session.isPostBatch) {
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

    addChunk(PostBatch::Chunk());
    updateHardwareVisibility();
    updateRunButtons();
}

PostBatch::Mode PostBatchPage::mode() const
{
    if (m_modeCompare->isChecked())
        return PostBatch::Mode::Compare;
    if (m_modeCoherence->isChecked())
        return PostBatch::Mode::Coherence;
    return PostBatch::Mode::Nni;
}

void PostBatchPage::updateHardwareVisibility()
{
    if (m_hardwareRow)
        m_hardwareRow->setVisible(mode() == PostBatch::Mode::Nni);
}

void PostBatchPage::addChunk(const PostBatch::Chunk &chunk)
{
    auto *w = new PostBatchChunkWidget(mode(), m_chunkHost);
    w->setChunk(chunk);
    m_chunkLayout->addWidget(w);
    m_chunks.append(w);

    connect(w, &PostBatchChunkWidget::removeRequested, this, [this, w] { removeChunk(w); });
    connect(w, &PostBatchChunkWidget::duplicateRequested, this, [this, w] {
        const int i = m_chunks.indexOf(w);
        addChunk(w->chunk());
        if (i >= 0 && m_chunks.size() >= 2) {
            PostBatchChunkWidget *copy = m_chunks.takeLast();
            m_chunks.insert(i + 1, copy);
            m_chunkLayout->removeWidget(copy);
            m_chunkLayout->insertWidget(i + 1, copy);
            renumberChunks();
        }
    });
    connect(w, &PostBatchChunkWidget::moveUpRequested, this, [this, w] {
        const int i = m_chunks.indexOf(w);
        if (i > 0) {
            m_chunks.move(i, i - 1);
            m_chunkLayout->removeWidget(w);
            m_chunkLayout->insertWidget(i - 1, w);
            renumberChunks();
        }
    });
    connect(w, &PostBatchChunkWidget::moveDownRequested, this, [this, w] {
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

void PostBatchPage::removeChunk(PostBatchChunkWidget *w)
{
    if (m_chunks.size() <= 1) {
        QMessageBox::information(this, tr("Batch processing"),
                                 tr("A batch needs at least one chunk."));
        return;
    }
    if (!TrajectaUi::confirm(this, tr("Delete chunk"), tr("Delete this chunk?")))
        return;
    m_chunks.removeOne(w);
    m_chunkLayout->removeWidget(w);
    w->deleteLater();
    renumberChunks();
}

void PostBatchPage::renumberChunks()
{
    for (int i = 0; i < m_chunks.size(); ++i)
        m_chunks.at(i)->setIndex(i + 1);
}

PostBatch::Job PostBatchPage::buildJob() const
{
    PostBatch::Job job;
    job.mode = mode();
    job.maxThreads = m_threads->value();
    job.maxRamMb = m_ram->value();
    // The global switch behind Advanced settings, not a control of this
    // page's own any more — see largepages.h.
    job.largePages = largePagesRequested();
    job.writeManifest = m_manifest->isChecked();
    for (PostBatchChunkWidget *w : m_chunks)
        job.chunks.append(w->chunk());
    return job;
}

void PostBatchPage::applyJob(const PostBatch::Job &job)
{
    m_modeNni->setChecked(job.mode == PostBatch::Mode::Nni);
    m_modeCompare->setChecked(job.mode == PostBatch::Mode::Compare);
    m_modeCoherence->setChecked(job.mode == PostBatch::Mode::Coherence);
    m_threads->setValue(job.maxThreads);
    m_ram->setValue(job.maxRamMb);
    // job.largePages is not restored into any control of this page any more:
    // it is read fresh from the global setting when the batch actually runs.
    m_manifest->setChecked(job.writeManifest);
    updateHardwareVisibility();

    for (PostBatchChunkWidget *w : m_chunks) {
        m_chunkLayout->removeWidget(w);
        w->deleteLater();
    }
    m_chunks.clear();
    if (job.chunks.isEmpty())
        addChunk(PostBatch::Chunk());
    else
        for (const PostBatch::Chunk &c : job.chunks)
            addChunk(c);
}

void PostBatchPage::setEnvironment(const TrajectaRunner::Parameters &env)
{
    const QString cpDir = m_env.checkpointDir;
    m_env = env;
    if (m_controller->isRunning())
        m_env.checkpointDir = cpDir;
}

bool PostBatchPage::isRunning() const
{
    return m_controller->isRunning();
}

void PostBatchPage::cancelForShutdown()
{
    if (m_controller->isRunning())
        m_controller->stopBatch();
}

void PostBatchPage::applyTheme()
{
    if (m_console)
        m_console->applyTheme();
}

void PostBatchPage::openLogs()
{
    if (m_logHandle)
        m_logHandle->setChecked(true);
}

RunTicker::State PostBatchPage::tickerState() const
{
    RunTicker::State s;
    if (!m_controller->isRunning())
        return s;
    s.active = true;
    s.paused = m_controller->isPaused();

    const int total = qMax(1, m_controller->total());
    const double done = double(m_rowsDone) + qBound(0.0, m_rowPercent, 100.0) / 100.0;
    s.percent = qBound(0.0, 100.0 * done / double(total), 100.0);

    const PostBatch::Job &job = m_controller->job();
    const int chunkNow = qBound(1, m_controller->currentChunkIndex() + 1, total);
    s.kind = tr("Post-processing batch — %1 · chunk %2/%3")
                 .arg(PostBatch::modeLabel(job.mode)).arg(chunkNow).arg(total);
    s.chunks = tr("Chunk %1 of %2").arg(chunkNow).arg(total);
    if (job.mode == PostBatch::Mode::Nni) {
        s.hardware = tr("Hardware: %1 threads · %2 MB")
                         .arg(job.maxThreads).arg(job.maxRamMb);
    }
    s.remaining = TrajectaUi::timeLeftText(
        m_batchClock.isValid() ? m_batchClock.elapsed() : 0, s.percent);
    return s;
}

QVector<bool> PostBatchPage::unfoldChunks()
{
    QVector<bool> folded;
    folded.reserve(m_chunks.size());
    for (PostBatchChunkWidget *w : m_chunks) {
        folded.append(w->isCollapsed());
        w->setCollapsed(false, false);
    }
    return folded;
}

void PostBatchPage::restoreChunkFolds(const QVector<bool> &folded)
{
    for (int i = 0; i < m_chunks.size() && i < folded.size(); ++i)
        m_chunks.at(i)->setCollapsed(folded.at(i), false);
}

void PostBatchPage::setEditingEnabled(bool enabled)
{
    m_modeNni->setEnabled(enabled);
    m_modeCompare->setEnabled(enabled);
    m_modeCoherence->setEnabled(enabled);
    m_threads->setEnabled(enabled);
    m_ram->setEnabled(enabled);
    m_manifest->setEnabled(enabled);
    for (PostBatchChunkWidget *w : m_chunks)
        w->setEditingEnabled(enabled);
}

void PostBatchPage::updateRunButtons()
{
    const bool running = m_controller->isRunning();
    m_runButton->setEnabled(!running && m_startAllowed);
    m_pauseButton->setEnabled(running);
    m_skipButton->setEnabled(running);
    m_stopButton->setEnabled(running);
}

void PostBatchPage::setStartAllowed(bool allowed)
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

void PostBatchPage::startBatch()
{
    if (m_controller->isRunning())
        return;
    const PostBatch::Job job = buildJob();
    if (job.mode == PostBatch::Mode::Nni && m_env.exePath.isEmpty()) {
        QMessageBox::warning(this, tr("Engine not found"),
                             tr("Trajecta Studio cannot find trajecta.exe. Use "
                                "\"Locate engine...\" in the status bar first."));
        return;
    }
    // Compare and Coherence read vector/raster layers directly; the library
    // that does it may not be loaded yet if this is the first thing the user
    // has done with Trajecta this session.
    if (job.mode != PostBatch::Mode::Nni && m_gdalLoader)
        m_gdalLoader();

    m_console->clearAll();
    m_summary->setVisible(false);

    const QList<PostBatch::Issue> issues = PostBatch::validate(job);
    if (!issues.isEmpty()) {
        QStringList lines;
        for (const PostBatch::Issue &i : issues) {
            lines << (i.chunk >= 0 ? tr("Chunk %1: %2").arg(i.chunk + 1).arg(i.message)
                                   : i.message);
        }
        const bool fatal = std::any_of(issues.cbegin(), issues.cend(),
                                       [](const PostBatch::Issue &i) { return i.chunk < 0; });
        if (fatal) {
            QMessageBox::warning(this, tr("The batch cannot start"),
                                 lines.join(QLatin1Char('\n')));
            return;
        }
        if (!TrajectaUi::confirm(
                this, tr("Some chunks will be skipped"),
                tr("%1 chunk(s) will not run because of the problems listed in "
                   "the log. Start the batch anyway?").arg(lines.size()))) {
            return;
        }
        for (const QString &l : lines)
            m_console->appendMarker(l, ThemeManager::mapped("#cf7f7f"));
    }

    // Session-only, like an LCPA Processing batch: none of the three tools has
    // engine state to checkpoint, but which chunk the batch had reached is
    // worth exactly as much to resume from.
    const Checkpoint::Settings cp = Checkpoint::settings();
    const QString cpDir = Checkpoint::activeDir();
    const bool wantSession = cp.enabled && !cpDir.isEmpty();

    if (wantSession && m_resumeChunkIndex < 0) {
        const Checkpoint::Session saved = Checkpoint::readSession(cpDir);
        if (saved.valid
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

    m_env.checkpointDir = wantSession ? cpDir : QString();

    PostBatchController::Resume resume;
    resume.chunkIndex = m_resumeChunkIndex;
    m_resumeChunkIndex = -1;

    QString error;
    if (!m_controller->start(job, m_env, &error, resume)) {
        QMessageBox::warning(this, tr("The batch cannot start"), error);
        return;
    }
    setEditingEnabled(false);
    updateRunButtons();
    setChipState(QStringLiteral("running"), tr("RUNNING"));
    m_status->setText(tr("Starting..."));
    m_batchClock.start();
    m_rowsDone = 0;
    m_rowPercent = 0.0;
    emit runningChanged(true);
    emit tickerChanged();
}

void PostBatchPage::publishChunkLayers(int chunkIndex)
{
    const PostBatch::Job &job = m_controller->job();
    if (chunkIndex < 0 || chunkIndex >= job.chunks.size())
        return;
    const PostBatch::Chunk &chunk = job.chunks.at(chunkIndex);
    if (!chunk.loadInViewer)
        return;

    QStringList rasters, vectors;
    if (job.mode == PostBatch::Mode::Nni) {
        const QString name = chunk.interpOutputName.trimmed();
        if (!name.isEmpty()) {
            const QString path =
                QDir(chunk.interpOutputDir).filePath(name + QStringLiteral(".tif"));
            if (QFileInfo::exists(path))
                rasters << QDir::toNativeSeparators(path);
        }
    } else if (job.mode == PostBatch::Mode::Coherence) {
        const QDir dir(chunk.cohOutputDir.isEmpty()
                           ? QFileInfo(chunk.cohRasterPath).absolutePath()
                           : chunk.cohOutputDir);
        const QString prefix = chunk.cohPrefix.trimmed().isEmpty()
                                   ? QStringLiteral("coherence")
                                   : chunk.cohPrefix.trimmed();
        const QString vectorPath = dir.filePath(
            prefix + (chunk.cohVectorAsGeoPackage ? QStringLiteral(".gpkg")
                                                  : QStringLiteral(".shp")));
        if (QFileInfo::exists(vectorPath))
            vectors << QDir::toNativeSeparators(vectorPath);
        if (chunk.cohWriteDistanceRaster) {
            const QString rasterPath =
                dir.filePath(prefix + QStringLiteral("_distance.tif"));
            if (QFileInfo::exists(rasterPath))
                rasters << QDir::toNativeSeparators(rasterPath);
        }
    }
    // Compare never reaches here: loadInViewer is forced false for it.
    if (!rasters.isEmpty() || !vectors.isEmpty())
        emit viewerLayersReady(rasters, vectors);
}

void PostBatchPage::writeSessionForChunk(int chunkIndex)
{
    if (m_env.checkpointDir.isEmpty())
        return;
    Checkpoint::Session session;
    session.batch = true;
    session.isPostBatch = true;
    session.job = PostBatch::toJson(m_controller->job());
    session.queueIndex = chunkIndex;
    session.params = Checkpoint::toJson(m_env);
    session.label = tr("Post-processing batch — chunk %1 of %2")
                        .arg(chunkIndex + 1).arg(m_controller->total());
    Checkpoint::writeSession(m_env.checkpointDir, session);
}

void PostBatchPage::resumeJob(const QJsonObject &job, int chunkIndex)
{
    PostBatch::Job loaded;
    if (!PostBatch::fromJson(job, &loaded, nullptr)) {
        QMessageBox::warning(this, tr("Batch processing"),
                             tr("The interrupted batch could not be read back."));
        return;
    }
    applyJob(loaded);
    m_resumeChunkIndex = chunkIndex;
    QTimer::singleShot(0, this, [this] { startBatch(); });
}

void PostBatchPage::setChipState(const QString &state, const QString &text)
{
    if (!m_chip)
        return;
    m_chip->setText(text);
    m_chip->setProperty("state", state);
    m_chip->style()->unpolish(m_chip);
    m_chip->style()->polish(m_chip);
}

void PostBatchPage::saveToFile()
{
    const QString path = QFileDialog::getSaveFileName(
        this, tr("Save the batch"), QString(),
        tr("Trajecta post-processing batch (*.trjpbatch);;JSON (*.json)"));
    if (path.isEmpty())
        return;
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        QMessageBox::warning(this, tr("Save the batch"), tr("Cannot write %1").arg(path));
        return;
    }
    f.write(QJsonDocument(PostBatch::toJson(buildJob())).toJson(QJsonDocument::Indented));
}

QString PostBatchPage::saveState() const
{
    return QString::fromUtf8(
        QJsonDocument(PostBatch::toJson(buildJob())).toJson(QJsonDocument::Compact));
}

void PostBatchPage::restoreState(const QString &json)
{
    if (json.trimmed().isEmpty())
        return;
    const QJsonDocument doc = QJsonDocument::fromJson(json.toUtf8());
    PostBatch::Job job;
    if (doc.isObject() && PostBatch::fromJson(doc.object(), &job, nullptr)) {
        const qint64 totalRam = SystemInfo::totalRamMb();
        const int sixtyPercent = int(qMax<qint64>(1024, (totalRam * 60) / 100));
        if (job.maxRamMb == sixtyPercent || job.maxRamMb == 4096)
            job.maxRamMb = int(qMin<qint64>(SystemInfo::kRecommendedRamMb, totalRam));
        applyJob(job);
    }
}

bool PostBatchPage::loadBatchFile(const QString &path, QString *error)
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
    PostBatch::Job job;
    if (!PostBatch::fromJson(doc.object(), &job, error))
        return false;
    applyJob(job);
    return true;
}

void PostBatchPage::loadFromFile()
{
    const QString path = QFileDialog::getOpenFileName(
        this, tr("Load a batch"), QString(),
        tr("Trajecta post-processing batch (*.trjpbatch *.json);;All files (*)"));
    if (path.isEmpty())
        return;
    QString error;
    if (!loadBatchFile(path, &error))
        QMessageBox::warning(this, tr("Load a batch"), error);
}

QVector<TourStep> PostBatchPage::walkthroughSteps()
{
    QVector<TourStep> steps;
    PostBatchChunkWidget *chunk = m_chunks.isEmpty() ? nullptr : m_chunks.first();

    {
        TourStep s;
        s.lightCard(m_settingsCard);
        s.title = tr("Settings for the whole batch");
        s.text = tr(
            "The top card picks which of the three post-processing tools the "
            "whole batch runs, and — while NNI is selected — the hardware it "
            "runs with. Compare and Coherence run in the interface, so they "
            "have nothing here to set.");
        s.annotations = {
            { m_modeNni, tr("Whether the batch runs NNI, Compare or Coherence. "
                            "Every chunk is the same tool.") },
            { m_threads, tr("Cores the engine may keep busy — NNI only.") },
        };
        steps.append(s);
    }

    if (chunk) {
        TourStep s;
        s.lightCard(chunk);
        s.avoidLitArea = true;
        s.calloutWidthCap = 1400;
        s.title = tr("A chunk: one analysis");
        s.text = tr(
            "Simpler than a Processing chunk: one chunk here already is one "
            "analysis, with the same fields as the tabs above. Add as many "
            "as you like.");
        steps.append(s);
    }

    {
        TourStep s;
        s.lightCard(m_runCard);
        s.title = tr("Starting the batch, and watching it");
        s.text = tr(
            "The same panel as a Processing batch. The one difference: none "
            "of these three tools can resume in the middle of a chunk, so "
            "\"Resume\" always restarts the interrupted chunk from its "
            "beginning rather than from where it stopped.");
        s.annotations = {
            { m_runButton, tr("Starts at the first chunk and works down.") },
            { m_pauseButton, tr("Freezes an NNI chunk; holds before the next "
                                "one otherwise.") },
            { m_skipButton, tr("Abandons this chunk only; the batch carries on.") },
            { m_stopButton, tr("Abandons the whole queue.") },
        };
        steps.append(s);
    }

    return steps;
}
