#include "coherence.h"

#include "gdalapi.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QObject>
#include <QTextStream>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace Coherence {

namespace {

constexpr double kNoDataOut = -9999.0;

// The intensity index reads a site against the same quantity measured at cells
// picked at random across the surface, so the reference has to be built by
// walking a disc around each of them: the cost is samples x cells-in-the-disc,
// and a large radius makes the disc grow with the square of the radius. This
// caps the product. 20 000 samples already place a site to a fifth of a
// percentile, far finer than anything read off the number, so the floor is
// what matters more than the ceiling.
constexpr double kIntensityRefBudget = 6.0e7;
constexpr int kIntensityRefWanted = 50000;
constexpr int kIntensityRefFloor = 20000;

double medianOfSorted(const QVector<double> &v)
{
    if (v.isEmpty())
        return 0.0;
    const qsizetype n = v.size();
    return (n % 2) ? v.at(n / 2) : 0.5 * (v.at(n / 2 - 1) + v.at(n / 2));
}

// Linear interpolation between order statistics, the definition most readers
// will assume when they see "97.5th percentile".
double quantileOfSorted(const QVector<double> &v, double q)
{
    if (v.isEmpty())
        return 0.0;
    if (v.size() == 1)
        return v.first();
    const double pos = q * double(v.size() - 1);
    const qsizetype lo = qsizetype(std::floor(pos));
    const qsizetype hi = std::min<qsizetype>(lo + 1, v.size() - 1);
    const double t = pos - double(lo);
    return v.at(lo) * (1.0 - t) + v.at(hi) * t;
}

double medianOf(QVector<double> v)
{
    std::sort(v.begin(), v.end());
    return medianOfSorted(v);
}

// ---------------------------------------------------------------------------
// Percentile ranks
// ---------------------------------------------------------------------------
//
// A histogram, its cumulative sum, and a lookup table from bin to rank. Two
// linear passes instead of a sort: a 4000 x 4000 raster is sixteen million
// values, and sorting them to learn something the histogram already knows is
// wasted time and wasted memory.
//
// Ties get the *mid*-rank — the average rank of the block of equal values —
// which is not a detail here: a path-count surface is mostly exact zeros and
// exact ones, and the convention chosen decides what the bulk of the map
// scores. Mid-ranks are the convention whose average over the whole surface is
// exactly 50, so a location picked at random scores 50 in any dataset, and the
// intensity score arrives with its own null expectation built in.
struct RankLut {
    double minV = 0.0, maxV = 0.0;
    int bins = 0;
    bool integral = false;
    qint64 valid = 0;
    QVector<qint64> count;
    QVector<double> rank;

    int binOf(double v) const
    {
        if (bins <= 1)
            return 0;
        if (integral) {
            const int b = int(std::llround(v - minV));
            return std::clamp(b, 0, bins - 1);
        }
        if (maxV <= minV)
            return 0;
        const double t = (v - minV) / (maxV - minV);
        return std::clamp(int(t * (bins - 1)), 0, bins - 1);
    }

    double rankOf(double v) const { return rank.isEmpty() ? 0.0 : rank.at(binOf(v)); }

    double lowerEdge(int b) const
    {
        if (integral)
            return minV + b;
        if (bins <= 1)
            return minV;
        return minV + (maxV - minV) * double(b) / double(bins - 1);
    }

    // The smallest value whose rank reaches `pct`. Used only to report the
    // threshold as a number the user can recognise; the selection itself is
    // made on ranks, which needs no interpolation.
    double valueAtRank(double pct) const
    {
        for (int b = 0; b < bins; ++b) {
            if (count.at(b) > 0 && rank.at(b) >= pct)
                return lowerEdge(b);
        }
        return maxV;
    }
};

RankLut buildRanks(const QVector<float> &values, const QVector<quint8> &valid)
{
    RankLut lut;
    double lo = std::numeric_limits<double>::max();
    double hi = -std::numeric_limits<double>::max();
    bool allIntegral = true;
    qint64 n = 0;
    for (int i = 0; i < values.size(); ++i) {
        if (!valid.at(i))
            continue;
        const double v = values.at(i);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
        if (allIntegral && v != std::floor(v))
            allIntegral = false;
        ++n;
    }
    if (n == 0)
        return lut;

    lut.minV = lo;
    lut.maxV = hi;
    lut.valid = n;
    // Path counts are integers, and then one bin per value is exact. A surface
    // that has been through NNI is not, and 65536 bins put the quantiles within
    // 0.0015% of the range — far finer than anything downstream can use.
    const double span = hi - lo;
    if (allIntegral && span < 4.0e6) {
        lut.integral = true;
        lut.bins = int(span) + 1;
    } else {
        lut.integral = false;
        lut.bins = 65536;
    }
    lut.count.assign(lut.bins, 0);
    for (int i = 0; i < values.size(); ++i) {
        if (valid.at(i))
            ++lut.count[lut.binOf(values.at(i))];
    }
    lut.rank.assign(lut.bins, 0.0);
    qint64 cum = 0;
    for (int b = 0; b < lut.bins; ++b) {
        lut.rank[b] = 100.0 * (double(cum) + 0.5 * double(lut.count.at(b))) / double(n);
        cum += lut.count.at(b);
    }
    return lut;
}

// ---------------------------------------------------------------------------
// Otsu's threshold, on log values
// ---------------------------------------------------------------------------
//
// The split that maximises the variance *between* the two classes it makes —
// equivalently, the one that minimises the variance within them. On the raw
// counts it is useless here: the histogram is a spike at zero with a tail four
// orders of magnitude long, and the best two-class split is "the spike" against
// "everything else", which is half the map. On log values the multiplicative
// tail becomes an additive one, the distribution opens out into background and
// channelled movement, and the threshold falls in the valley between them.
double otsuOnLog(const QVector<float> &values, const QVector<quint8> &valid)
{
    constexpr int kBins = 2048;
    double lo = std::numeric_limits<double>::max();
    double hi = -std::numeric_limits<double>::max();
    qint64 n = 0;
    for (int i = 0; i < values.size(); ++i) {
        if (!valid.at(i) || values.at(i) <= 0.0f)
            continue;      // the empty half of the map says nothing about the split
        const double x = std::log1p(double(values.at(i)));
        lo = std::min(lo, x);
        hi = std::max(hi, x);
        ++n;
    }
    if (n == 0 || hi <= lo)
        return 0.0;

    QVector<qint64> h(kBins, 0);
    for (int i = 0; i < values.size(); ++i) {
        if (!valid.at(i) || values.at(i) <= 0.0f)
            continue;
        const double x = std::log1p(double(values.at(i)));
        const int b = std::clamp(int((x - lo) / (hi - lo) * (kBins - 1)), 0, kBins - 1);
        ++h[b];
    }

    double best = -1.0;
    int bestBin = 0;
    qint64 w0 = 0;
    double s0 = 0.0, sTotal = 0.0;
    for (int b = 0; b < kBins; ++b) {
        const double centre = lo + (hi - lo) * (double(b) + 0.5) / double(kBins);
        sTotal += double(h.at(b)) * centre;
    }
    for (int b = 0; b < kBins - 1; ++b) {
        const double centre = lo + (hi - lo) * (double(b) + 0.5) / double(kBins);
        w0 += h.at(b);
        s0 += double(h.at(b)) * centre;
        const qint64 w1 = n - w0;
        if (w0 == 0 || w1 == 0)
            continue;
        const double mu0 = s0 / double(w0);
        const double mu1 = (sTotal - s0) / double(w1);
        const double between = double(w0) * double(w1) * (mu0 - mu1) * (mu0 - mu1);
        if (between > best) {
            best = between;
            bestBin = b;
        }
    }
    const double edge = lo + (hi - lo) * double(bestBin + 1) / double(kBins);
    return std::expm1(edge);
}

// ---------------------------------------------------------------------------
// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher)
// ---------------------------------------------------------------------------
//
// Two one-dimensional passes, each a lower envelope of parabolas, give the
// exact squared distance in linear time. Exact matters: an approximation such
// as a chamfer mask is anisotropic, and a site would score differently for
// lying north of a corridor than for lying north-east of it at the same
// distance.
void lowerEnvelope(const double *f, double *d, int n, int *vtx, double *z)
{
    constexpr double kInf = std::numeric_limits<double>::infinity();
    int k = 0;
    vtx[0] = 0;
    z[0] = -kInf;
    z[1] = kInf;
    for (int q = 1; q < n; ++q) {
        double s = ((f[q] + double(q) * q) - (f[vtx[k]] + double(vtx[k]) * vtx[k]))
                   / (2.0 * q - 2.0 * vtx[k]);
        while (k > 0 && s <= z[k]) {
            --k;
            s = ((f[q] + double(q) * q) - (f[vtx[k]] + double(vtx[k]) * vtx[k]))
                / (2.0 * q - 2.0 * vtx[k]);
        }
        ++k;
        vtx[k] = q;
        z[k] = s;
        z[k + 1] = kInf;
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < double(q))
            ++k;
        const double dq = double(q - vtx[k]);
        d[q] = dq * dq + f[vtx[k]];
    }
}

} // namespace

QVector<double> distanceTransformSquared(const QVector<quint8> &mask, int w, int h)
{
    // Not infinity: the envelope arithmetic subtracts these from one another,
    // and inf - inf is not a number. A value larger than any distance this grid
    // can hold behaves the same and stays finite.
    const double big = 4.0 * (double(w) * w + double(h) * h) + 1.0;
    QVector<double> grid(qsizetype(w) * h, big);
    for (qsizetype i = 0; i < grid.size(); ++i) {
        if (i < mask.size() && mask.at(i))
            grid[i] = 0.0;
    }

    const int longest = std::max(w, h);
    QVector<double> f(longest), d(longest), z(longest + 1);
    QVector<int> vtx(longest);

    for (int x = 0; x < w; ++x) {                       // columns
        for (int y = 0; y < h; ++y)
            f[y] = grid.at(qsizetype(y) * w + x);
        lowerEnvelope(f.data(), d.data(), h, vtx.data(), z.data());
        for (int y = 0; y < h; ++y)
            grid[qsizetype(y) * w + x] = d.at(y);
    }
    for (int y = 0; y < h; ++y) {                       // rows
        double *row = grid.data() + qsizetype(y) * w;
        for (int x = 0; x < w; ++x)
            f[x] = row[x];
        lowerEnvelope(f.data(), d.data(), w, vtx.data(), z.data());
        for (int x = 0; x < w; ++x)
            row[x] = d.at(x);
    }
    return grid;
}

namespace {

// ---------------------------------------------------------------------------
// The grid, and what is asked of it
// ---------------------------------------------------------------------------
struct Grid {
    int w = 0, h = 0;
    double gt[6] = {0, 0, 0, 0, 0, 0};
    double cell = 0.0;
    QString wkt;
    QVector<float> values;
    // log(1 + value), computed once for the whole surface rather than inside
    // the disc walk. The intensity index averages these, and the walk visits
    // the same cell for every site and for every reference sample: computing
    // the logarithm there would be the single most expensive thing the tool
    // does, for a result that never changes.
    QVector<float> logv;
    QVector<quint8> valid;
    QVector<double> dist;     // metres to the nearest corridor cell
    RankLut lut;

    qsizetype index(int col, int row) const { return qsizetype(row) * w + col; }
    bool insideGrid(int col, int row) const
    {
        return col >= 0 && row >= 0 && col < w && row < h;
    }
    double xOf(int col) const { return gt[0] + (col + 0.5) * gt[1]; }
    double yOf(int row) const { return gt[3] + (row + 0.5) * gt[5]; }
    int colOf(double x) const { return int(std::floor((x - gt[0]) / gt[1])); }
    int rowOf(double y) const { return int(std::floor((y - gt[3]) / gt[5])); }
};

// The disc, built once per radius instead of once per site.
//
// Every site and every reference sample walks the same shape, and the old code
// recomputed std::hypot for each of them — the reference distribution alone
// visits tens of millions of cells, so the offsets and their weights are
// gathered here and then only multiplied.
struct Kernel {
    QVector<int> dx, dy;
    QVector<double> w;      // triangular, 1 at the centre and 0 at the rim
    qint64 inDisc = 0;      // cells whose centre falls inside, on or off grid
};

Kernel buildKernel(double radius, double cell)
{
    Kernel k;
    const int rc = int(std::ceil(radius / cell));
    for (int dy = -rc; dy <= rc; ++dy) {
        for (int dx = -rc; dx <= rc; ++dx) {
            const double d = std::hypot(double(dx), double(dy)) * cell;
            if (d >= radius)
                continue;
            k.dx << dx;
            k.dy << dy;
            k.w << (1.0 - d / radius);
            ++k.inDisc;
        }
    }
    return k;
}

// One site's readings at one radius. Kept together because they all walk the
// same disc and none is worth a second walk.
struct Scores {
    double distM = 0.0;
    double proxIndex = 0.0;   // % of the valid neighbourhood that is corridor
    double rawInten = 0.0;    // weighted mean of log(1+v); a percentile later
    double coverage = 0.0;
};

Scores scoreAt(const Grid &g, const Kernel &k, int col, int row)
{
    Scores s;
    if (!g.insideGrid(col, row))
        return s;

    s.distM = g.dist.at(g.index(col, row));

    double sumW = 0.0, sumWV = 0.0;
    qint64 withData = 0, corridor = 0;
    for (qsizetype n = 0; n < k.w.size(); ++n) {
        const int cx = col + k.dx.at(n), cy = row + k.dy.at(n);
        // The disc's own size is k.inDisc, counted whether or not the cell
        // exists: a site whose disc hangs over the edge of the raster has
        // genuinely lost that ground, and coverage is where that is reported.
        if (!g.insideGrid(cx, cy))
            continue;
        const qsizetype i = g.index(cx, cy);
        if (!g.valid.at(i))
            continue;
        ++withData;
        // A corridor cell is exactly a cell at distance zero from a corridor,
        // and the transform returns a hard zero there — no tolerance needed,
        // and no second mask to carry around.
        if (g.dist.at(i) == 0.0)
            ++corridor;
        const double w = k.w.at(n);
        sumW += w;
        sumWV += w * double(g.logv.at(i));
    }
    s.coverage = k.inDisc > 0 ? double(withData) / double(k.inDisc) : 0.0;
    // A share, not a count: the same ground at 30 m and at 90 m holds nine
    // times as many cells but the same fraction of corridor, which is what
    // lets two surfaces of different resolution be compared at all.
    s.proxIndex = withData > 0 ? 100.0 * double(corridor) / double(withData) : 0.0;
    s.rawInten = sumW > 0.0 ? sumWV / sumW : 0.0;
    return s;
}

// Where a value sits in an already-sorted reference, 0-100, splitting ties.
// Mid-ranks are what make the reference's own average exactly 50.
double percentileIn(const QVector<double> &sorted, double x)
{
    if (sorted.isEmpty())
        return 50.0;
    const auto lo = std::lower_bound(sorted.cbegin(), sorted.cend(), x);
    const auto hi = std::upper_bound(lo, sorted.cend(), x);
    const double less = double(lo - sorted.cbegin());
    const double equal = double(hi - lo);
    return 100.0 * (less + 0.5 * equal) / double(sorted.size());
}

// The yardstick the intensity index is read against: the same distance-weighted
// mean, measured at cells picked at random across the surface. Sorted, so a
// site's own value can be turned into "busier than N% of this surface".
//
// Sampled rather than computed everywhere. Doing it for every cell is a radial
// convolution over the whole raster — exact, and affordable only with an FFT
// this application does not carry. A sample of tens of thousands places a site
// to a fraction of a percentile, which is finer than the number is ever read.
//
// The samples go through the identical disc walk the sites do, edge handling
// included, so a site whose disc hangs over the rim is compared against
// reference cells that could be in the same position rather than against an
// idealised interior.
QVector<double> intensityReference(const Grid &g, const Kernel &k, quint32 seed,
                                   qint64 validCells)
{
    int wanted = kIntensityRefWanted;
    const double perSample = double(std::max<qint64>(1, k.inDisc));
    wanted = std::min<int>(wanted, int(kIntensityRefBudget / perSample));
    wanted = std::max(wanted, kIntensityRefFloor);
    wanted = int(std::min<qint64>(wanted, std::max<qint64>(1, validCells)));

    // A separate generator from the null model's, seeded from the same number.
    // Sharing one would make the reference depend on how many replicates the
    // null happened to draw, and a run would stop being repeatable for a
    // reason that has nothing to do with it.
    std::mt19937 rng(seed ^ 0x9e3779b9u);
    std::uniform_int_distribution<int> cx(0, std::max(0, g.w - 1));
    std::uniform_int_distribution<int> cy(0, std::max(0, g.h - 1));

    QVector<double> ref;
    ref.reserve(wanted);
    // Rejection sampling, with a ceiling on the attempts: a surface that is
    // mostly NoData would otherwise spin here.
    const qint64 maxTries = qint64(wanted) * 20 + 1000;
    for (qint64 t = 0; t < maxTries && ref.size() < wanted; ++t) {
        const int col = cx(rng), row = cy(rng);
        const qsizetype i = g.index(col, row);
        if (!g.valid.at(i))
            continue;
        ref << scoreAt(g, k, col, row).rawInten;
    }
    std::sort(ref.begin(), ref.end());
    return ref;
}

QString classOf(double distM, double radius, double intensity)
{
    // The cut-offs are stated rather than tuned: half the radius for "close",
    // and five points above the 50 a random location scores for "busy". Both
    // are in the help text, because a label the reader cannot reproduce is
    // worse than no label.
    //
    // "Half the radius" is written directly now. It used to be "proximity >=
    // 50" on a score that was 100*(1 - d/r), which is the same condition —
    // 100(1-d/r) >= 50 exactly when d <= r/2, and both are false beyond the
    // radius — so retiring that score changed the wording here and nothing
    // else about which sites land in which class.
    const bool near = distM <= 0.5 * radius;
    const bool busy = intensity >= 55.0;
    if (near && busy)
        return QStringLiteral("ON_CORRIDOR");
    if (near)
        return QStringLiteral("NEAR_THIN");
    if (busy)
        return QStringLiteral("DIFFUSE");
    return QStringLiteral("OFF");
}

struct Point {
    double x = 0.0, y = 0.0;
    QStringList fields;
};

// The columns this tool writes itself. An input field with one of these names
// would appear twice in the table and be ambiguous in both — a layer whose own
// columns are called "x" and "y" is not unusual — so the input's copy is left
// out, in the table and in the layer alike, and the two therefore carry exactly
// the same set of columns.
const char *const kReservedNames[] = {
    "site_id", "x", "y", "status", "dist_m", "prox_idx", "enrich", "inten_idx",
    "rank_site", "coverage", "near_edge", "class",
};

bool isReserved(const QString &name)
{
    for (const char *r : kReservedNames) {
        if (name.compare(QLatin1String(r), Qt::CaseInsensitive) == 0)
            return true;
    }
    return false;
}

QVector<int> carriedFields(const QStringList &names)
{
    QVector<int> carried;
    for (int i = 0; i < names.size(); ++i) {
        if (!names.at(i).isEmpty() && !isReserved(names.at(i)))
            carried << i;
    }
    return carried;
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------
bool readRaster(const QString &path, Grid &g, QString *error)
{
    GdalApi &api = GdalApi::instance();
    GDALDatasetH ds = api.OpenEx(path.toUtf8().constData(), GdalApi::OF_Raster,
                                 nullptr, nullptr, nullptr);
    if (!ds) {
        *error = QObject::tr("The raster could not be opened:\n%1")
                     .arg(QDir::toNativeSeparators(path));
        return false;
    }
    g.w = api.GetRasterXSize(ds);
    g.h = api.GetRasterYSize(ds);
    if (g.w <= 0 || g.h <= 0) {
        api.Close(ds);
        *error = QObject::tr("The raster has no cells.");
        return false;
    }
    if (api.GetGeoTransform(ds, g.gt) != 0) {
        api.Close(ds);
        *error = QObject::tr("The raster has no geotransform, so its cells have "
                             "no position on the ground.");
        return false;
    }
    if (g.gt[2] != 0.0 || g.gt[4] != 0.0) {
        api.Close(ds);
        *error = QObject::tr("The raster is rotated. Reproject it to a north-up "
                             "grid first.");
        return false;
    }
    const double sx = std::fabs(g.gt[1]), sy = std::fabs(g.gt[5]);
    if (sx <= 0.0 || sy <= 0.0) {
        api.Close(ds);
        *error = QObject::tr("The raster's cell size is not usable.");
        return false;
    }
    // A radius in metres presumes one cell size, not two. A few per cent apart
    // is a rounding difference in the header; more than that is a raster that
    // has to be resampled before any of this means anything.
    if (std::fabs(sx - sy) > 0.02 * std::max(sx, sy)) {
        api.Close(ds);
        *error = QObject::tr("The raster's cells are not square (%1 x %2). "
                             "Resample it to square cells first.")
                     .arg(sx, 0, 'f', 3).arg(sy, 0, 'f', 3);
        return false;
    }
    g.cell = 0.5 * (sx + sy);
    g.wkt = QString::fromUtf8(api.GetProjectionRef(ds));

    GDALRasterBandH band = api.GetRasterBand(ds, 1);
    if (!band) {
        api.Close(ds);
        *error = QObject::tr("The raster has no first band.");
        return false;
    }
    int hasNoData = 0;
    const double noData = api.GetRasterNoDataValue(band, &hasNoData);

    g.values.resize(qsizetype(g.w) * g.h);
    if (api.RasterIO(band, GdalApi::ReadFlag, 0, 0, g.w, g.h, g.values.data(),
                     g.w, g.h, GdalApi::Float32, 0, 0) != 0) {
        api.Close(ds);
        *error = QObject::tr("The raster's values could not be read.");
        return false;
    }
    api.Close(ds);

    g.valid.assign(g.values.size(), 1);
    for (qsizetype i = 0; i < g.values.size(); ++i) {
        const float v = g.values.at(i);
        // NoData stays missing all the way through. It is not zero: zero is a
        // cell that was measured and had no movement, which is a fact about the
        // landscape, and counting the two together moves every rank.
        if (std::isnan(v) || (hasNoData && double(v) == noData))
            g.valid[i] = 0;
    }
    return true;
}

bool readPoints(const QString &path, QVector<Point> &points, QStringList &fieldNames,
                bool *geographic, QString *error)
{
    GdalApi &api = GdalApi::instance();
    GDALDatasetH ds = api.OpenEx(path.toUtf8().constData(), GdalApi::OF_Vector,
                                 nullptr, nullptr, nullptr);
    if (!ds) {
        *error = QObject::tr("The point layer could not be opened:\n%1")
                     .arg(QDir::toNativeSeparators(path));
        return false;
    }
    if (api.DatasetGetLayerCount(ds) < 1) {
        api.Close(ds);
        *error = QObject::tr("The point layer contains no layers.");
        return false;
    }
    OGRLayerH layer = api.DatasetGetLayer(ds, 0);
    if (!layer) {
        api.Close(ds);
        *error = QObject::tr("The point layer could not be read.");
        return false;
    }
    *geographic = false;
    if (api.L_GetSpatialRef && api.OSRIsGeographic) {
        if (OGRSpatialReferenceH srs = api.L_GetSpatialRef(layer))
            *geographic = api.OSRIsGeographic(srs) != 0;
    }

    const bool fields = api.canReadFields();
    api.L_ResetReading(layer);
    while (OGRFeatureH feat = api.L_GetNextFeature(layer)) {
        OGRGeometryH geom = api.F_GetGeometryRef(feat);
        if (geom) {
            const int type = GdalApi::flattenGeomType(api.G_GetGeometryType(geom));
            OGRGeometryH pt = nullptr;
            if (type == GdalApi::WkbPoint) {
                pt = geom;
            } else if (type == GdalApi::WkbMultiPoint
                       && api.G_GetGeometryCount(geom) > 0) {
                // A multipoint with one part is what several editors write for
                // an ordinary point; taking the first part is right there and
                // harmless elsewhere.
                pt = api.G_GetGeometryRef(geom, 0);
            }
            if (pt && api.G_GetPointCount(pt) > 0) {
                Point p;
                double z = 0.0;
                api.G_GetPoint(pt, 0, &p.x, &p.y, &z);
                if (fields) {
                    const int n = api.F_GetFieldCount(feat);
                    for (int i = 0; i < n; ++i) {
                        if (fieldNames.size() < n) {
                            if (OGRFieldDefnH fd = api.F_GetFieldDefnRef(feat, i))
                                fieldNames << QString::fromUtf8(api.Fld_GetNameRef(fd));
                        }
                        p.fields << QString::fromUtf8(api.F_GetFieldAsString(feat, i));
                    }
                }
                points << p;
            }
        }
        api.F_Destroy(feat);
    }
    api.Close(ds);
    if (points.isEmpty()) {
        *error = QObject::tr("The layer holds no point features.");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------
bool writeDistanceRaster(const Grid &g, const QString &path, QString *error)
{
    GdalApi &api = GdalApi::instance();
    if (!api.canWriteRaster()) {
        *error = QObject::tr("this GDAL build cannot create rasters");
        return false;
    }
    GDALDriverH drv = api.GetDriverByName("GTiff");
    if (!drv) {
        *error = QObject::tr("the GeoTIFF driver is not available");
        return false;
    }
    const char *options[] = {"COMPRESS=DEFLATE", "TILED=YES", nullptr};
    GDALDatasetH ds = api.Create(drv, path.toUtf8().constData(), g.w, g.h, 1,
                                 GdalApi::Float32, options);
    if (!ds) {
        *error = QObject::tr("the file could not be created");
        return false;
    }
    double gt[6];
    std::copy(std::begin(g.gt), std::end(g.gt), std::begin(gt));
    api.SetGeoTransform(ds, gt);
    if (!g.wkt.isEmpty())
        api.SetProjection(ds, g.wkt.toUtf8().constData());
    GDALRasterBandH band = api.GetRasterBand(ds, 1);
    api.SetRasterNoDataValue(band, kNoDataOut);

    QVector<float> out(qsizetype(g.w) * g.h);
    for (qsizetype i = 0; i < out.size(); ++i)
        out[i] = g.valid.at(i) ? float(g.dist.at(i)) : float(kNoDataOut);
    const int rc = api.RasterIO(band, GdalApi::WriteFlag, 0, 0, g.w, g.h, out.data(),
                                g.w, g.h, GdalApi::Float32, 0, 0);
    api.Close(ds);
    if (rc != 0) {
        *error = QObject::tr("the values could not be written");
        return false;
    }
    return true;
}

bool writeVector(const Result &res, const Grid &g, const QString &path,
                 bool geoPackage, QString *error)
{
    GdalApi &api = GdalApi::instance();
    if (!api.canWriteVector()) {
        *error = QObject::tr("this GDAL build cannot create vector layers");
        return false;
    }
    GDALDriverH drv = api.GetDriverByName(geoPackage ? "GPKG" : "ESRI Shapefile");
    if (!drv) {
        *error = QObject::tr("the %1 driver is not available")
                     .arg(geoPackage ? QStringLiteral("GeoPackage")
                                     : QStringLiteral("Shapefile"));
        return false;
    }
    QFile::remove(path);
    GDALDatasetH ds = api.Create(drv, path.toUtf8().constData(), 0, 0, 0, 0, nullptr);
    if (!ds) {
        *error = QObject::tr("the file could not be created");
        return false;
    }

    OGRSpatialReferenceH srs = nullptr;
    if (!g.wkt.isEmpty() && api.OSRNewSpatialReference) {
        srs = api.OSRNewSpatialReference(nullptr);
        if (srs && api.OSRSetFromUserInput)
            api.OSRSetFromUserInput(srs, g.wkt.toUtf8().constData());
    }
    const QByteArray layerName = QFileInfo(path).completeBaseName().toUtf8();
    OGRLayerH layer = api.DatasetCreateLayer(ds, layerName.constData(), srs,
                                             GdalApi::WkbPoint, nullptr);
    if (srs && api.OSRDestroySpatialReference)
        api.OSRDestroySpatialReference(srs);
    if (!layer) {
        api.Close(ds);
        *error = QObject::tr("the layer could not be created");
        return false;
    }

    // Every name here is ten characters or fewer, so a shapefile keeps them as
    // written rather than truncating them into each other.
    struct FieldDef { const char *name; int type; };
    static const FieldDef kFields[] = {
        {"site_id", GdalApi::OftInteger}, {"status", GdalApi::OftString},
        {"dist_m", GdalApi::OftReal},     {"prox_idx", GdalApi::OftReal},
        {"enrich", GdalApi::OftReal},     {"inten_idx", GdalApi::OftReal},
        {"rank_site", GdalApi::OftReal},
        {"coverage", GdalApi::OftReal},   {"near_edge", GdalApi::OftInteger},
        {"class", GdalApi::OftString},
    };
    for (const FieldDef &f : kFields) {
        OGRFieldDefnH fd = api.Fld_Create(f.name, f.type);
        if (!fd)
            continue;
        if (f.type == GdalApi::OftString && api.Fld_SetWidth)
            api.Fld_SetWidth(fd, 16);
        if (f.type == GdalApi::OftReal && api.Fld_SetPrecision) {
            api.Fld_SetWidth(fd, 18);
            api.Fld_SetPrecision(fd, 4);
        }
        api.L_CreateField(layer, fd, 1);
        api.Fld_Destroy(fd);
    }
    // The input's own attributes follow, as text.
    const QVector<int> carried = carriedFields(res.fieldNames);
    for (int i : carried) {
        const QString name = res.fieldNames.at(i);
        OGRFieldDefnH fd = api.Fld_Create(name.toUtf8().constData(), GdalApi::OftString);
        if (!fd)
            continue;
        if (api.Fld_SetWidth)
            api.Fld_SetWidth(fd, 64);
        api.L_CreateField(layer, fd, 1);
        api.Fld_Destroy(fd);
    }

    OGRFeatureDefnH defn = api.L_GetLayerDefn(layer);
    int id = 0;
    for (const SiteResult &s : res.sites) {
        ++id;
        OGRFeatureH feat = api.F_Create(defn);
        if (!feat)
            continue;
        // Rounded on the way in, not on the way out: a GeoPackage keeps the
        // full double whatever precision the field declares, and an attribute
        // table showing 158.619371316846 metres claims an accuracy the DEM
        // cannot support. The CSV is written to the same two decimals.
        const auto round2 = [](double v) { return std::round(v * 100.0) / 100.0; };
        int f = 0;
        api.F_SetFieldInteger(feat, f++, id);
        const QByteArray status = (!s.inside ? QByteArrayLiteral("OUTSIDE")
                                             : (!s.hasData ? QByteArrayLiteral("NODATA")
                                                           : QByteArrayLiteral("OK")));
        api.F_SetFieldString(feat, f++, status.constData());
        api.F_SetFieldDouble(feat, f++, round2(s.distM));
        api.F_SetFieldDouble(feat, f++, round2(s.proxIndex));
        api.F_SetFieldDouble(feat, f++, round2(s.enrichment));
        api.F_SetFieldDouble(feat, f++, round2(s.intenIndex));
        api.F_SetFieldDouble(feat, f++, round2(s.rankSite));
        api.F_SetFieldDouble(feat, f++, std::round(s.coverage * 1000.0) / 1000.0);
        api.F_SetFieldInteger(feat, f++, s.nearEdge ? 1 : 0);
        api.F_SetFieldString(feat, f++, s.cls.toUtf8().constData());
        for (int idx : carried) {
            const QString v = idx < s.fields.size() ? s.fields.at(idx) : QString();
            api.F_SetFieldString(feat, f++, v.toUtf8().constData());
        }
        if (OGRGeometryH pt = api.G_CreateGeometry(GdalApi::WkbPoint)) {
            api.G_SetPoint_2D(pt, 0, s.x, s.y);
            api.F_SetGeometryDirectly(feat, pt);
        }
        api.L_CreateFeature(layer, feat);
        api.F_Destroy(feat);
    }
    api.Close(ds);
    return true;
}

bool writeCsv(const Result &res, const QString &path, QString *error)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        *error = QObject::tr("the file could not be opened for writing");
        return false;
    }
    QTextStream out(&file);
    // No parameter block above the header: a comment line turns a clean table
    // into something half the readers of a CSV have to be told about. The
    // parameters are in the summary written beside it.
    out << "site_id,x,y,status,dist_m,prox_idx,enrich,inten_idx,rank_site,"
           "coverage,near_edge,class";
    const QVector<int> carried = carriedFields(res.fieldNames);
    for (int i : carried)
        out << ',' << res.fieldNames.at(i);
    out << '\n';

    int id = 0;
    for (const SiteResult &s : res.sites) {
        ++id;
        const QString status = !s.inside ? QStringLiteral("OUTSIDE")
                                         : (!s.hasData ? QStringLiteral("NODATA")
                                                       : QStringLiteral("OK"));
        out << id << ',' << QString::number(s.x, 'f', 3) << ','
            << QString::number(s.y, 'f', 3) << ',' << status << ','
            << QString::number(s.distM, 'f', 2) << ','
            << QString::number(s.proxIndex, 'f', 2) << ','
            << QString::number(s.enrichment, 'f', 3) << ','
            << QString::number(s.intenIndex, 'f', 2) << ','
            << QString::number(s.rankSite, 'f', 2) << ','
            << QString::number(s.coverage, 'f', 3) << ','
            << (s.nearEdge ? 1 : 0) << ',' << s.cls;
        for (int i : carried) {
            QString v = i < s.fields.size() ? s.fields.at(i) : QString();
            v.replace(QLatin1Char('"'), QLatin1String("\"\""));
            const bool quote = v.contains(QLatin1Char(',')) || v.contains(QLatin1Char('"'))
                               || v.contains(QLatin1Char('\n'));
            out << ',' << (quote ? QLatin1Char('"') + v + QLatin1Char('"') : v);
        }
        out << '\n';
    }
    file.close();
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------
Result run(const Params &params, const LogSink &log)
{
    Result res;
    const auto say = [&log](const QString &line) {
        if (log)
            log(line);
    };

    GdalApi &api = GdalApi::instance();
    if (!api.isLoaded()) {
        res.error = QObject::tr("GDAL is not loaded, so no layer can be read.");
        return res;
    }
    if (params.radiusMetres <= 0.0) {
        res.error = QObject::tr("The radius must be greater than zero.");
        return res;
    }

    QElapsedTimer clock;
    clock.start();

    // --- the surface ---
    Grid g;
    if (!readRaster(params.rasterPath, g, &res.error))
        return res;
    res.rasterWidth = g.w;
    res.rasterHeight = g.h;
    res.cellSize = g.cell;
    res.totalCells = double(g.w) * g.h;

    if (api.OSRNewSpatialReference && api.OSRIsGeographic && !g.wkt.isEmpty()) {
        OGRSpatialReferenceH srs = api.OSRNewSpatialReference(g.wkt.toUtf8().constData());
        if (srs) {
            if (api.OSRIsGeographic(srs)) {
                api.OSRDestroySpatialReference(srs);
                res.error = QObject::tr(
                    "The raster is in degrees. A radius in metres means nothing "
                    "on a geographic coordinate system: reproject it to a "
                    "projected CRS (UTM, for instance) and run this again.");
                return res;
            }
            if (api.OSRGetName)
                res.crsName = QString::fromUtf8(api.OSRGetName(srs));
            api.OSRDestroySpatialReference(srs);
        }
    }
    say(QObject::tr("Surface: %1 x %2 cells of %3 m%4")
            .arg(g.w).arg(g.h).arg(g.cell, 0, 'f', 2)
            .arg(res.crsName.isEmpty() ? QString()
                                       : QStringLiteral(" — ") + res.crsName));

    // --- ranks ---
    g.lut = buildRanks(g.values, g.valid);
    res.validCells = double(g.lut.valid);
    if (g.lut.valid == 0) {
        res.error = QObject::tr("Every cell of the raster is NoData.");
        return res;
    }
    say(QObject::tr("Valid cells: %1 of %2 (%3%). Values ranked by percentile.")
            .arg(g.lut.valid).arg(qint64(res.totalCells))
            .arg(100.0 * res.validCells / res.totalCells, 0, 'f', 1));

    // --- the corridor ---
    double threshold = 0.0;
    QVector<quint8> corridor(g.values.size(), 0);
    if (params.thresholdMode == ThresholdMode::TopPercent) {
        const double q = std::clamp(params.thresholdValue, 0.001, 99.0);
        const double cut = 100.0 - q;
        for (qsizetype i = 0; i < g.values.size(); ++i) {
            if (g.valid.at(i) && g.lut.rankOf(g.values.at(i)) >= cut)
                corridor[i] = 1;
        }
        threshold = g.lut.valueAtRank(cut);
        res.thresholdPercentile = cut;
    } else {
        threshold = (params.thresholdMode == ThresholdMode::Otsu)
                        ? otsuOnLog(g.values, g.valid)
                        : params.thresholdValue;
        for (qsizetype i = 0; i < g.values.size(); ++i) {
            if (g.valid.at(i) && double(g.values.at(i)) >= threshold)
                corridor[i] = 1;
        }
        res.thresholdPercentile = g.lut.rankOf(threshold);
    }
    qint64 corridorCells = 0;
    for (qsizetype i = 0; i < corridor.size(); ++i)
        corridorCells += corridor.at(i);
    res.thresholdValue = threshold;
    res.corridorShare = double(corridorCells) / double(g.lut.valid);
    if (corridorCells == 0) {
        res.error = QObject::tr("The threshold selects no cell at all, so there "
                                "is no corridor to measure distances from.");
        return res;
    }
    if (params.thresholdMode == ThresholdMode::Otsu) {
        if (res.corridorShare > 0.25) {
            res.thresholdNote = QObject::tr(
                "Otsu selected %1% of the surface. That is a lot for a corridor: "
                "this surface does not separate cleanly into movement and "
                "background, and a fixed percentile will be easier to defend.")
                .arg(100.0 * res.corridorShare, 0, 'f', 1);
        } else if (res.corridorShare < 0.001) {
            res.thresholdNote = QObject::tr(
                "Otsu selected %1% of the surface — almost nothing. Check the "
                "result on the map before using it.")
                .arg(100.0 * res.corridorShare, 0, 'f', 3);
        }
    }
    say(QObject::tr("Corridor: value >= %1 (percentile %2), %3% of the valid surface")
            .arg(threshold, 0, 'f', 3)
            .arg(res.thresholdPercentile, 0, 'f', 2)
            .arg(100.0 * res.corridorShare, 0, 'f', 2));
    if (!res.thresholdNote.isEmpty())
        say(res.thresholdNote);

    // --- distances ---
    const QVector<double> sq = distanceTransformSquared(corridor, g.w, g.h);
    g.dist.resize(sq.size());
    for (qsizetype i = 0; i < sq.size(); ++i)
        g.dist[i] = std::sqrt(sq.at(i)) * g.cell;

    // --- the log surface the intensity index averages ---
    // Clamped at zero first: a FETE density and an NNI surface are both
    // non-negative, but this tool will read whatever raster it is handed, and
    // log1p of a negative number is not a number at all — one stray cell would
    // otherwise turn a whole neighbourhood into NaN silently.
    g.logv.resize(g.values.size());
    for (qsizetype i = 0; i < g.values.size(); ++i) {
        g.logv[i] = g.valid.at(i)
                        ? float(std::log1p(std::max(0.0, double(g.values.at(i)))))
                        : 0.0f;
    }
    say(QObject::tr("Distance to the nearest corridor cell computed for every cell "
                    "(%1 ms).").arg(clock.elapsed()));

    // --- the sites ---
    QVector<Point> points;
    bool pointsGeographic = false;
    if (!readPoints(params.pointsPath, points, res.fieldNames, &pointsGeographic,
                    &res.error)) {
        return res;
    }
    if (pointsGeographic) {
        res.error = QObject::tr(
            "The point layer is in degrees while the raster is projected. "
            "Reproject the points to the raster's coordinate system first.");
        return res;
    }
    res.pointsRead = points.size();
    say(QObject::tr("Sites: %1 read.").arg(points.size()));

    const double radius = params.radiusMetres;
    const double edgeBand = radius;

    // Built once and reused by every site and every reference sample below.
    const Kernel mainKernel = buildKernel(radius, g.cell);
    const QVector<double> mainRef =
        intensityReference(g, mainKernel, params.seed, g.lut.valid);
    say(QObject::tr("Intensity yardstick: %1 sample neighbourhoods measured "
                    "across the surface (%2 ms).")
            .arg(mainRef.size()).arg(clock.elapsed()));

    res.sites.reserve(points.size());
    for (const Point &p : points) {
        SiteResult s;
        s.x = p.x;
        s.y = p.y;
        s.fields = p.fields;
        const int col = g.colOf(p.x), row = g.rowOf(p.y);
        if (!g.insideGrid(col, row)) {
            s.inside = false;
            ++res.pointsOutside;
            res.sites << s;
            continue;
        }
        const qsizetype i = g.index(col, row);
        s.hasData = g.valid.at(i) != 0;
        if (!s.hasData)
            ++res.pointsNoData;
        s.rankSite = s.hasData ? g.lut.rankOf(g.values.at(i)) : -1.0;

        const Scores sc = scoreAt(g, mainKernel, col, row);
        s.distM = sc.distM;
        s.proxIndex = sc.proxIndex;
        // Against the surface's own corridor share, so 1 is what a location
        // picked at random gets. The share is the measured one and not the
        // requested percentage: with Otsu or a raw threshold the two differ,
        // and using the measured one is what keeps this honest under all
        // three modes rather than only under the percentile.
        s.enrichment = res.corridorShare > 0.0
                           ? (sc.proxIndex / 100.0) / res.corridorShare
                           : 0.0;
        s.intenIndex = percentileIn(mainRef, sc.rawInten);
        s.coverage = sc.coverage;
        s.cls = classOf(sc.distM, radius, s.intenIndex);
        s.nearEdge = (g.xOf(col) - g.gt[0] < edgeBand)
                     || (g.gt[0] + g.w * g.gt[1] - g.xOf(col) < edgeBand)
                     || (std::fabs(g.yOf(row) - g.gt[3]) < edgeBand)
                     || (std::fabs(g.gt[3] + g.h * g.gt[5] - g.yOf(row)) < edgeBand);
        if (s.nearEdge)
            ++res.pointsNearEdge;
        if (s.coverage < 0.5)
            ++res.lowCoverage;
        res.sites << s;
    }
    // A site on a NoData cell still has a distance — the distance is geometry,
    // not measurement — so it is kept in the sample. One outside the raster has
    // nothing at all and is not.
    QVector<const SiteResult *> used;
    for (const SiteResult &s : res.sites) {
        if (s.inside)
            used << &s;
    }
    res.pointsUsed = used.size();
    if (used.isEmpty()) {
        res.error = QObject::tr(
            "Every one of the %1 sites falls outside the raster. The two layers "
            "are almost certainly in different coordinate systems.")
            .arg(points.size());
        return res;
    }
    if (res.pointsOutside > 0) {
        say(QObject::tr("%1 site(s) fall outside the raster and are excluded.")
                .arg(res.pointsOutside));
    }
    if (res.pointsNoData > 0) {
        say(QObject::tr("%1 site(s) sit on a NoData cell: kept, but their "
                        "neighbourhood is only partly measured.")
                .arg(res.pointsNoData));
    }

    // --- the sample, and the null it is judged against ---
    std::mt19937 rng(params.seed);

    // Cells a random site may land on. Rejection sampling rather than a list:
    // a list of sixteen million indices is sixty-four megabytes to answer a
    // question that two random numbers answer.
    const int guard = params.edgeGuard ? int(std::ceil(radius / g.cell)) : 0;
    auto randomValidCell = [&](int &col, int &row) {
        std::uniform_int_distribution<int> cx(guard, std::max(guard, g.w - 1 - guard));
        std::uniform_int_distribution<int> cy(guard, std::max(guard, g.h - 1 - guard));
        for (int tries = 0; tries < 500; ++tries) {
            col = cx(rng);
            row = cy(rng);
            if (g.insideGrid(col, row) && g.valid.at(g.index(col, row)))
                return true;
        }
        return false;
    };

    // --- the distances, and the null they are judged against ---
    //
    // Once, not once per radius. The distance to the nearest corridor is a
    // property of the site and the surface: asking for it at four radii prints
    // the same number four times, and running the null four times spends the
    // replicates to reach the same answer again.
    {
        DistanceStats &d = res.dist;
        QVector<double> dists;
        dists.reserve(used.size());
        for (const SiteResult *s : used)
            dists << s->distM;
        std::sort(dists.begin(), dists.end());

        d.median = medianOfSorted(dists);
        d.d10 = quantileOfSorted(dists, 0.10);
        d.d25 = quantileOfSorted(dists, 0.25);
        d.d75 = quantileOfSorted(dists, 0.75);
        d.d90 = quantileOfSorted(dists, 0.90);
        d.iqr = d.d75 - d.d25;

        // The "share of sites within X metres" ladder. Rungs finer than one
        // cell are dropped rather than reported: on a 90 m grid every site is
        // either on a corridor or at least 90 m from one, so a 50 m rung would
        // repeat the 0 m rung and invite the reader to believe the raster
        // resolves something it does not.
        QVector<double> ladder = params.ecdfDistances.isEmpty()
                                     ? kDefaultEcdfDistances()
                                     : params.ecdfDistances;
        std::sort(ladder.begin(), ladder.end());
        for (double rung : ladder) {
            if (rung < 0.0)
                continue;
            if (rung > 0.0 && rung < g.cell)
                continue;
            if (!d.ecdfAt.isEmpty() && std::fabs(d.ecdfAt.last() - rung) < 1e-9)
                continue;
            int within = 0;
            for (double x : dists)
                within += (x <= rung) ? 1 : 0;
            d.ecdfAt << rung;
            d.ecdfShare << double(within) / double(dists.size());
        }

        if (params.nullModel) {
            // Whether the pattern can be shifted at all is a property of the
            // pattern, not of the replicate: a layer whose points fill the
            // study area cannot be translated anywhere without pushing some of
            // them out, and trying two hundred times per replicate to discover
            // that again is how a run takes six minutes. Asked once.
            double cx = 0.0, cy = 0.0;
            for (const SiteResult *s : used) {
                cx += s->x;
                cy += s->y;
            }
            cx /= used.size();
            cy /= used.size();

            bool shift = (params.nullMode == NullMode::RandomShift);
            QVector<QPair<double, double>> offsets;
            if (shift) {
                const int wanted = params.nullReplicates;
                for (int attempt = 0;
                     attempt < 40 * wanted && offsets.size() < wanted; ++attempt) {
                    int tc = 0, tr = 0;
                    if (!randomValidCell(tc, tr))
                        break;
                    const double dx = g.xOf(tc) - cx, dy = g.yOf(tr) - cy;
                    bool allIn = true;
                    for (const SiteResult *s : used) {
                        const int c = g.colOf(s->x + dx), r2 = g.rowOf(s->y + dy);
                        if (!g.insideGrid(c, r2) || !g.valid.at(g.index(c, r2))) {
                            allIn = false;
                            break;
                        }
                    }
                    if (allIn)
                        offsets << qMakePair(dx, dy);
                }
                if (offsets.size() < wanted / 2) {
                    shift = false;
                    offsets.clear();
                    say(QObject::tr(
                        "The sites cover too much of the raster to be shifted as "
                        "a block — every translation pushes some of them off the "
                        "surface. Scattered random points are used instead, which "
                        "makes the test slightly optimistic if the sites are "
                        "clustered."));
                }
            }
            const int reps = shift ? offsets.size() : params.nullReplicates;

            // One lookup per site per replicate. The disc is not walked here at
            // all any more, which is what makes the full 999 affordable however
            // large the radius: the null now tests the distance and nothing
            // else, and a distance is read straight out of the transform.
            QVector<double> nullDist;
            nullDist.reserve(reps);
            QVector<int> cols(used.size()), rows(used.size());
            QVector<double> rd(used.size());
            for (int k = 0; k < reps; ++k) {
                bool placed = true;
                if (shift) {
                    const double dx = offsets.at(k).first, dy = offsets.at(k).second;
                    for (int i = 0; i < used.size(); ++i) {
                        cols[i] = g.colOf(used.at(i)->x + dx);
                        rows[i] = g.rowOf(used.at(i)->y + dy);
                    }
                } else {
                    for (int i = 0; i < used.size() && placed; ++i)
                        placed = randomValidCell(cols[i], rows[i]);
                }
                if (!placed)
                    continue;
                for (int i = 0; i < used.size(); ++i)
                    rd[i] = g.dist.at(g.index(cols.at(i), rows.at(i)));
                nullDist << medianOf(rd);
            }

            if (!nullDist.isEmpty()) {
                d.nullDone = true;
                d.nullUsed = nullDist.size();
                d.nullShifted = shift;
                // Hope's correction: the observed value belongs in its own
                // reference set, which is what keeps the p exact and stops it
                // reaching zero. Left-tailed - closer than chance is the claim.
                int le = 0;
                for (double x : nullDist)
                    le += (x <= d.median) ? 1 : 0;
                d.p = double(1 + le) / double(1 + nullDist.size());
                std::sort(nullDist.begin(), nullDist.end());
                d.nullMedian = medianOfSorted(nullDist);
                d.nullLo = quantileOfSorted(nullDist, 0.025);
                d.nullHi = quantileOfSorted(nullDist, 0.975);
                d.ratio = d.nullMedian > 0.0 ? d.median / d.nullMedian : 1.0;
                say(QObject::tr("Null model: %1 random point sets (%2), distance "
                                "only (%3 ms).")
                        .arg(d.nullUsed)
                        .arg(shift ? QObject::tr("the same pattern, moved as a block")
                                   : QObject::tr("scattered points"))
                        .arg(clock.elapsed()));
            }
        }
    }

    // --- what actually varies with the radius ---
    const auto statsFor = [&](double r, const Kernel &k, const QVector<double> &ref) {
        RadiusStats st;
        st.radius = r;
        QVector<double> prox, enr, inten;
        prox.reserve(used.size());
        enr.reserve(used.size());
        inten.reserve(used.size());
        int within = 0;
        for (const SiteResult *s : used) {
            const int col = g.colOf(s->x), row = g.rowOf(s->y);
            const Scores sc = scoreAt(g, k, col, row);
            prox << sc.proxIndex;
            enr << (res.corridorShare > 0.0
                        ? (sc.proxIndex / 100.0) / res.corridorShare
                        : 0.0);
            inten << percentileIn(ref, sc.rawInten);
            within += (sc.distM < r) ? 1 : 0;
        }
        const auto meanOf = [](const QVector<double> &v) {
            if (v.isEmpty())
                return 0.0;
            double t = 0.0;
            for (double x : v)
                t += x;
            return t / double(v.size());
        };
        st.meanProxIdx = meanOf(prox);
        st.meanEnrich = meanOf(enr);
        st.meanIntenIdx = meanOf(inten);
        std::sort(prox.begin(), prox.end());
        std::sort(enr.begin(), enr.end());
        std::sort(inten.begin(), inten.end());
        st.medianProxIdx = medianOfSorted(prox);
        st.medianEnrich = medianOfSorted(enr);
        st.medianIntenIdx = medianOfSorted(inten);
        st.shareWithin = double(within) / double(used.size());
        return st;
    };

    res.main = statsFor(radius, mainKernel, mainRef);
    say(QObject::tr("Scored %1 site(s) at %2 m (%3 ms).")
            .arg(res.pointsUsed).arg(radius, 0, 'f', 0).arg(clock.elapsed()));

    if (params.sensitivity) {
        for (double r : params.sensitivityRadii) {
            if (r <= 0.0 || r < 2.0 * g.cell)
                continue;      // fewer than two cells across is not a radius
            // Each radius needs its own disc and its own yardstick: the
            // intensity index is a percentile of neighbourhoods of that size,
            // and reading a 2 km neighbourhood against 500 m ones would report
            // the radius rather than the site.
            const Kernel k = buildKernel(r, g.cell);
            const QVector<double> ref =
                intensityReference(g, k, params.seed, g.lut.valid);
            res.sensitivity << statsFor(r, k, ref);
        }
        say(QObject::tr("Sensitivity curve: %1 radii (%2 ms).")
                .arg(res.sensitivity.size()).arg(clock.elapsed()));
    }

    // --- outputs ---
    if (!params.outputDir.isEmpty()) {
        QDir dir(params.outputDir);
        if (!dir.exists())
            QDir().mkpath(params.outputDir);
        const QString base = dir.absoluteFilePath(
            params.outputPrefix.isEmpty() ? QStringLiteral("coherence")
                                          : params.outputPrefix);
        QString err;
        const QString csv = base + QStringLiteral("_sites.csv");
        if (writeCsv(res, csv, &err))
            res.csvPath = csv;
        else
            say(QObject::tr("The table could not be written: %1").arg(err));

        if (params.writeVector) {
            const QString vec = base + (params.vectorAsGeoPackage
                                            ? QStringLiteral("_sites.gpkg")
                                            : QStringLiteral("_sites.shp"));
            if (writeVector(res, g, vec, params.vectorAsGeoPackage, &err))
                res.vectorPath = vec;
            else
                say(QObject::tr("The layer could not be written: %1").arg(err));
        }
        if (params.writeDistanceRaster) {
            const QString ras = base + QStringLiteral("_distance.tif");
            if (writeDistanceRaster(g, ras, &err))
                res.rasterPath = ras;
            else
                say(QObject::tr("The distance raster could not be written: %1").arg(err));
        }
        if (params.writeHistogramScript) {
            const QString rs = base + QStringLiteral("_histogram.R");
            if (writeHistogramRScript(res, rs, &err))
                res.rScriptPath = rs;
            else
                say(QObject::tr("The R script could not be written: %1").arg(err));
        }
        res.ok = true;
        const QString summary = base + QStringLiteral("_summary.txt");
        QFile sf(summary);
        if (sf.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream(&sf) << res.report();
            sf.close();
            res.summaryPath = summary;
        }
    }

    res.ok = true;
    return res;
}

// ---------------------------------------------------------------------------
DistHistogram computeDistHistogram(const Result &res)
{
    DistHistogram h;
    QVector<double> ds;
    for (const SiteResult &s : res.sites) {
        if (s.inside)
            ds << s.distM;
    }
    std::sort(ds.begin(), ds.end());
    if (ds.isEmpty())
        return h;

    // Bins run to p90, not to the maximum: one site ten kilometres out would
    // otherwise squeeze every other site into the first bin and the picture
    // would show nothing at all. What is past the top runs into an overflow
    // bin, which is honest and still readable.
    const double top = res.dist.d90 > 0.0 ? res.dist.d90 : ds.last();
    double width = top / 10.0;
    if (width <= 0.0)
        width = 1.0;
    // Rounded to something a reader can hold: 1, 2 or 5 times a power of ten.
    // Bin edges of 91.4 m teach nothing that 100 m does not.
    const double mag = std::pow(10.0, std::floor(std::log10(width)));
    const double norm = width / mag;
    width = (norm <= 1.5 ? 1.0 : norm <= 3.5 ? 2.0 : norm <= 7.5 ? 5.0 : 10.0) * mag;

    h.width = width;
    h.counts = QVector<int>(DistHistogram::kBins + 1, 0);
    for (double x : ds) {
        int b = int(x / width);
        if (b >= DistHistogram::kBins)
            b = DistHistogram::kBins;      // the overflow bin
        h.counts[b] += 1;
    }
    return h;
}

bool writeHistogramRScript(const Result &res, const QString &path, QString *error)
{
    const DistHistogram h = computeDistHistogram(res);
    if (h.counts.isEmpty()) {
        if (error)
            *error = QObject::tr("no sites fell inside the raster");
        return false;
    }

    QStringList labels, counts;
    for (int b = 0; b <= DistHistogram::kBins; ++b) {
        labels << QStringLiteral("\"%1\"").arg(
            b < DistHistogram::kBins
                ? QStringLiteral("%1-%2 m").arg(b * h.width, 0, 'f', 0)
                                           .arg((b + 1) * h.width, 0, 'f', 0)
                : QStringLiteral("over %1 m").arg(DistHistogram::kBins * h.width, 0, 'f', 0));
        counts << QString::number(h.counts.at(b));
    }

    // A literal name, not one inferred from how the script was launched:
    // commandArgs() only carries the script's own path under some ways of
    // running R (Rscript with --file=) and not others (source(), an RStudio
    // "Run"), so deriving it at run time is the more fragile choice here.
    const QString pngName = QFileInfo(path).completeBaseName() + QStringLiteral(".png");

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        if (error)
            *error = f.errorString();
        return false;
    }
    QTextStream out(&f);
    out << "# Trajecta Studio -- site-corridor coherence: distance histogram\n"
        << "# Redraws question 2 of the report (\"how far are they?\") exactly as\n"
        << "# computed there: same bins, same counts. Needs the ggplot2 package\n"
        << "# (install once with install.packages(\"ggplot2\")).\n"
        << "\n"
        << "library(ggplot2)\n"
        << "\n"
        << "label <- c(" << labels.join(QStringLiteral(", ")) << ")\n"
        << "count  <- c(" << counts.join(QStringLiteral(", ")) << ")\n"
        << "\n"
        << "df <- data.frame(label = factor(label, levels = label), count = count)\n"
        << "\n"
        << "p <- ggplot(df, aes(x = label, y = count)) +\n"
        << "  geom_col(fill = \"#7ea8a0\", width = 0.85) +\n"
        << "  labs(\n"
        << "    title = \"Distance to the nearest corridor\",\n"
        << "    subtitle = \"Site-corridor coherence \\u2014 Trajecta Studio\",\n"
        << "    x = \"Distance\",\n"
        << "    y = \"Number of sites\"\n"
        << "  ) +\n"
        << "  theme_minimal(base_size = 13) +\n"
        << "  theme(\n"
        << "    panel.grid.minor = element_blank(),\n"
        << "    panel.grid.major.x = element_blank(),\n"
        << "    axis.text.x = element_text(angle = 40, hjust = 1),\n"
        << "    plot.title = element_text(face = \"bold\"),\n"
        << "    plot.subtitle = element_text(colour = \"grey40\")\n"
        << "  )\n"
        << "\n"
        << "ggsave(\"" << pngName << "\", plot = p, width = 8, height = 5, dpi = 300)\n"
        << "print(p)\n";
    f.close();
    return true;
}

// ---------------------------------------------------------------------------
QString Result::report() const
{
    if (!ok)
        return error;

    QStringList out;
    const auto line = [&out](const QString &s) { out << s; };

    line(QObject::tr("SITE-CORRIDOR COHERENCE"));
    line(QString());
    line(QObject::tr("Surface        %1 x %2 cells of %3 m%4")
             .arg(rasterWidth).arg(rasterHeight).arg(cellSize, 0, 'f', 2)
             .arg(crsName.isEmpty() ? QString() : QStringLiteral(", ") + crsName));
    line(QObject::tr("Valid cells    %1 of %2 (%3%)")
             .arg(qint64(validCells)).arg(qint64(totalCells))
             .arg(100.0 * validCells / std::max(1.0, totalCells), 0, 'f', 1));
    line(QObject::tr("Corridor       value >= %1, percentile %2, %3% of the surface")
             .arg(thresholdValue, 0, 'f', 3).arg(thresholdPercentile, 0, 'f', 2)
             .arg(100.0 * corridorShare, 0, 'f', 2));
    if (!thresholdNote.isEmpty())
        line(QObject::tr("               %1").arg(thresholdNote));
    line(QString());
    line(QObject::tr("Sites          %1 read, %2 scored").arg(pointsRead).arg(pointsUsed));
    if (pointsOutside > 0)
        line(QObject::tr("               %1 outside the raster, excluded").arg(pointsOutside));
    if (pointsNoData > 0)
        line(QObject::tr("               %1 on a NoData cell").arg(pointsNoData));
    if (pointsNearEdge > 0)
        line(QObject::tr("               %1 within one radius of the edge (FETE "
                         "under-counts there)").arg(pointsNearEdge));
    if (lowCoverage > 0)
        line(QObject::tr("               %1 with less than half their disc measured")
                 .arg(lowCoverage));
    line(QString());

    // 1 -- the most general reading there is: did any of them land near one.
    line(QObject::tr("1. HOW MANY SITES ARE NEAR A CORRIDOR AT ALL?"));
    line(QObject::tr("   The share of sites within a given distance of the nearest"));
    line(QObject::tr("   corridor cell. Fixed distances, so two runs can be read"));
    line(QObject::tr("   side by side whatever radius each was given."));
    line(QString());
    for (int i = 0; i < dist.ecdfAt.size(); ++i) {
        const double rung = dist.ecdfAt.at(i);
        const QString label = (rung <= 0.0)
                                  ? QObject::tr("on a corridor cell")
                                  : QObject::tr("within %1 m").arg(rung, 0, 'f', 0);
        line(QStringLiteral("     %1 %2")
                 .arg(label, -26)
                 .arg(QString::number(100.0 * dist.ecdfShare.at(i), 'f', 1)
                          + QStringLiteral("%"), 7));
    }
    line(QString());
    line(QObject::tr("   At the chosen radius of %1 m: %2% have a corridor in range.")
             .arg(main.radius, 0, 'f', 0)
             .arg(100.0 * main.shareWithin, 0, 'f', 1));
    line(QString());

    // 2 -- how far, and how the sample is spread. Radius-independent, which is
    // what makes this the block to quote when comparing two periods.
    line(QObject::tr("2. HOW FAR ARE THEY?"));
    line(QObject::tr("   Metres to the nearest corridor cell. This does not depend"));
    line(QObject::tr("   on the radius at all, so it is the number that compares"));
    line(QObject::tr("   directly between periods and between surfaces."));
    line(QString());
    line(QObject::tr("     median            %1 m").arg(dist.median, 0, 'f', 1));
    line(QObject::tr("     p10 / p25         %1 m / %2 m")
             .arg(dist.d10, 0, 'f', 1).arg(dist.d25, 0, 'f', 1));
    line(QObject::tr("     p75 / p90         %1 m / %2 m")
             .arg(dist.d75, 0, 'f', 1).arg(dist.d90, 0, 'f', 1));
    line(QObject::tr("     IQR               %1 m").arg(dist.iqr, 0, 'f', 1));
    line(QObject::tr("   A wide gap between two neighbouring figures above means the"));
    line(QObject::tr("   sample is in two groups rather than one, and no single"));
    line(QObject::tr("   average describes it. The histogram below shows which."));
    line(QString());

    // The histogram, from the sites themselves. Shared with
    // writeHistogramRScript(), so the ASCII bars here and the ggplot2 figure
    // written next to the .csv are always the same histogram.
    {
        const DistHistogram h = computeDistHistogram(*this);
        if (!h.counts.isEmpty()) {
            int tallest = 0;
            for (int c : h.counts)
                tallest = std::max(tallest, c);
            for (int b = 0; b <= DistHistogram::kBins; ++b) {
                const QString label =
                    (b < DistHistogram::kBins)
                        ? QObject::tr("%1 - %2 m")
                              .arg(b * h.width, 0, 'f', 0).arg((b + 1) * h.width, 0, 'f', 0)
                        : QObject::tr("over %1 m").arg(DistHistogram::kBins * h.width, 0, 'f', 0);
                const int bars = tallest > 0
                                     ? (h.counts.at(b) * 40 + tallest / 2) / tallest : 0;
                line(QStringLiteral("     %1 %2 %3")
                         .arg(label, 20)
                         .arg(QString(bars, QLatin1Char('#')), -40)
                         .arg(h.counts.at(b)));
            }
            line(QString());
        }
    }

    if (dist.nullDone) {
        line(QObject::tr("   Against %1 random point sets (%2):")
                 .arg(dist.nullUsed)
                 .arg(dist.nullShifted
                          ? QObject::tr("the same pattern, moved as a block")
                          : QObject::tr("scattered points")));
        line(QObject::tr("     observed %1 m, expected %2 m (95% of random sets: %3 - %4 m)")
                 .arg(dist.median, 0, 'f', 0).arg(dist.nullMedian, 0, 'f', 0)
                 .arg(dist.nullLo, 0, 'f', 0).arg(dist.nullHi, 0, 'f', 0));
        line(QObject::tr("     the sites are %1x as far from a corridor as chance "
                         "would put them")
                 .arg(dist.ratio, 0, 'f', 2));
        line(dist.p <= 0.05
                 ? QObject::tr("     — more than chance explains (p = %1)")
                       .arg(dist.p, 0, 'f', 3)
                 : QObject::tr("     — which chance explains (p = %1)")
                       .arg(dist.p, 0, 'f', 3));
        line(QString());
    }

    // 3 and 4 -- the two readings that do depend on the radius.
    const auto block = [&](const RadiusStats &s, bool headings) {
        if (headings) {
            line(QObject::tr("3. HOW MUCH CORRIDOR IS AROUND THEM?"));
            line(QObject::tr("   Two sites the same distance from a corridor are not in"));
            line(QObject::tr("   the same place if one has a single thread nearby and the"));
            line(QObject::tr("   other a whole braid of them."));
            line(QString());
        }
        line(QObject::tr("     proximity index   mean %1%, median %2% of the "
                         "neighbourhood is corridor")
                 .arg(s.meanProxIdx, 0, 'f', 2).arg(s.medianProxIdx, 0, 'f', 2));
        line(QObject::tr("     enrichment        mean %1x, median %2x the surface "
                         "average")
                 .arg(s.meanEnrich, 0, 'f', 2).arg(s.medianEnrich, 0, 'f', 2));
        if (headings) {
            line(QObject::tr("   Quote the mean: a point placed at random has, on"));
            line(QObject::tr("   average, exactly the surface's own share of corridor"));
            line(QObject::tr("   around it, so a mean enrichment of 1.00 is chance"));
            line(QObject::tr("   exactly. The median is usually far lower and that is"));
            line(QObject::tr("   normal — corridors are thin and clustered, so most"));
            line(QObject::tr("   ground has none within reach at all."));
            line(QString());
            line(QObject::tr("4. HOW BUSY IS THAT GROUND?"));
            line(QObject::tr("   Not how much corridor, but how heavily travelled it is."));
            line(QString());
        }
        line(QObject::tr("     intensity index   mean %1, median %2 / 100")
                 .arg(s.meanIntenIdx, 0, 'f', 1).arg(s.medianIntenIdx, 0, 'f', 1));
        if (headings) {
            line(QObject::tr("   Again the mean: 50 is the average location on this"));
            line(QObject::tr("   surface, exactly. A median well under 50 means most of"));
            line(QObject::tr("   the surface is quiet ground, not that the sites are."));
        }
    };
    block(main, true);
    line(QString());

    if (!sensitivity.isEmpty()) {
        line(QObject::tr("SENSITIVITY TO THE RADIUS"));
        line(QObject::tr("  (means, the figures that carry the 1.00x and 50 references)"));
        line(QObject::tr("  radius   prox idx   enrichment   intensity   in range"));
        for (const RadiusStats &s : sensitivity) {
            line(QStringLiteral("  %1 %2 %3 %4 %5")
                     .arg(QString::number(s.radius, 'f', 0), 6)
                     .arg(QString::number(s.meanProxIdx, 'f', 2)
                              + QStringLiteral("%"), 10)
                     .arg(QString::number(s.meanEnrich, 'f', 2)
                              + QStringLiteral("x"), 12)
                     .arg(QString::number(s.meanIntenIdx, 'f', 1), 11)
                     .arg(QString::number(100.0 * s.shareWithin, 'f', 0)
                              + QStringLiteral("%"), 10));
        }
        line(QObject::tr("  A relationship that holds at every radius is a "
                         "relationship; one that appears at a single radius is "
                         "usually the radius."));
        line(QString());
    }

    // The glossary. Written into the report itself rather than left to the
    // guide, because the report is what gets saved, mailed and pasted into an
    // appendix, and a number whose definition lives somewhere else is a number
    // the reader has to take on trust.
    line(QObject::tr("HOW THESE NUMBERS ARE CALCULATED"));
    line(QString());
    line(QObject::tr("  corridor        The cells the threshold selected. With the"));
    line(QObject::tr("                  percentage filter this is the top q% of the"));
    line(QObject::tr("                  surface by rank, so it is exactly q% of the"));
    line(QObject::tr("                  valid cells in every dataset — which is what"));
    line(QObject::tr("                  makes two surfaces comparable at all."));
    line(QString());
    line(QObject::tr("  dist_m          Straight-line metres from the site to the"));
    line(QObject::tr("                  nearest corridor cell, from an exact Euclidean"));
    line(QObject::tr("                  distance transform of the whole surface. Not"));
    line(QObject::tr("                  capped at the radius, and quantised to the cell"));
    line(QObject::tr("                  size: a %1 m raster cannot report less than that.")
             .arg(cellSize, 0, 'f', 0));
    line(QString());
    line(QObject::tr("  prox_idx        Of the valid cells within the radius, the"));
    line(QObject::tr("                  percentage that are corridor. A share and not a"));
    line(QObject::tr("                  count, so a 30 m and a 90 m raster give the same"));
    line(QObject::tr("                  answer where a count would differ ninefold."));
    line(QString());
    line(QObject::tr("  enrich          prox_idx divided by the corridor's share of the"));
    line(QObject::tr("                  whole surface (%1%). Averaged over points placed")
             .arg(100.0 * corridorShare, 0, 'f', 2));
    line(QObject::tr("                  at random this is exactly 1.00, so a mean of 5.00"));
    line(QObject::tr("                  means five times as much corridor as average"));
    line(QObject::tr("                  ground. The identity is about the mean and not"));
    line(QObject::tr("                  the median: corridors are thin and clustered, so"));
    line(QObject::tr("                  on most surfaces the majority of locations have"));
    line(QObject::tr("                  none within reach and the median sits at zero for"));
    line(QObject::tr("                  the sites and for chance alike."));
    line(QString());
    line(QObject::tr("  inten_idx       Each cell's value taken as log(1 + value),"));
    line(QObject::tr("                  averaged over the radius with a triangular"));
    line(QObject::tr("                  weight that falls from 1 at the site to 0 at the"));
    line(QObject::tr("                  rim, then read as a percentile of the same"));
    line(QObject::tr("                  quantity measured at sample neighbourhoods across"));
    line(QObject::tr("                  the surface. The logarithm keeps one very busy"));
    line(QObject::tr("                  cell from standing in for its whole"));
    line(QObject::tr("                  neighbourhood; the percentile is what makes"));
    line(QObject::tr("                  surfaces built from different numbers of source"));
    line(QObject::tr("                  points comparable, and means 64 really does mean"));
    line(QObject::tr("                  busier than 64% of this surface. Mid-ranks make"));
    line(QObject::tr("                  the average location score exactly 50 — again a"));
    line(QObject::tr("                  statement about the mean. A median far below 50"));
    line(QObject::tr("                  says the surface is mostly quiet ground, which on"));
    line(QObject::tr("                  a FETE it usually is."));
    line(QString());
    line(QObject::tr("  class           ON_CORRIDOR: within half the radius and busy."));
    line(QObject::tr("                  NEAR_THIN: close, but quiet ground — a single"));
    line(QObject::tr("                  thread. DIFFUSE: busy ground, but no corridor"));
    line(QObject::tr("                  close by. OFF: neither. The cut-offs are half the"));
    line(QObject::tr("                  radius, and 55 on the intensity index."));
    line(QString());
    line(QObject::tr("  coverage        The fraction of the site's disc that had data."));
    line(QObject::tr("                  Below half, read that site with care."));
    line(QString());
    if (dist.nullDone) {
        line(QObject::tr("  the null model  The same median distance, recomputed on point"));
        line(QObject::tr("                  sets that have no relationship with the"));
        line(QObject::tr("                  corridors but share everything else: the same"));
        line(QObject::tr("                  area, the same number of sites, and by default"));
        line(QObject::tr("                  the same internal arrangement, translated as a"));
        line(QObject::tr("                  block so the sample's own clustering is not"));
        line(QObject::tr("                  mistaken for a relationship with the routes."));
        line(QObject::tr("                  Only the distance is tested this way: it has no"));
        line(QObject::tr("                  natural reference point, while enrichment"));
        line(QObject::tr("                  already has one (1.00) and intensity another"));
        line(QObject::tr("                  (50)."));
        line(QString());
    }
    line(QObject::tr("  Comparing runs  Nothing above requires two surfaces to share a"));
    line(QObject::tr("                  resolution or a number of source points — the"));
    line(QObject::tr("                  measures are built not to depend on either. But a"));
    line(QObject::tr("                  surface built from few points is a noisier"));
    line(QObject::tr("                  estimate of the same thing, so small differences"));
    line(QObject::tr("                  between runs should not be pressed. Use the"));
    line(QObject::tr("                  percentage filter for anything comparative."));
    line(QString());

    if (!csvPath.isEmpty() || !vectorPath.isEmpty() || !rasterPath.isEmpty()
        || !rScriptPath.isEmpty()) {
        line(QString());
        line(QObject::tr("WRITTEN"));
        if (!csvPath.isEmpty())
            line(QStringLiteral("  ") + QDir::toNativeSeparators(csvPath));
        if (!vectorPath.isEmpty())
            line(QStringLiteral("  ") + QDir::toNativeSeparators(vectorPath));
        if (!rasterPath.isEmpty())
            line(QStringLiteral("  ") + QDir::toNativeSeparators(rasterPath));
        if (!rScriptPath.isEmpty())
            line(QStringLiteral("  ") + QDir::toNativeSeparators(rScriptPath));
    }
    return out.join(QLatin1Char('\n'));
}

} // namespace Coherence
