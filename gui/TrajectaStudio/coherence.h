#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

#include <functional>

// How well does a set of sites sit on the corridors a FETE surface predicts?
//
// The surface says where movement concentrates; the sites say where people
// stayed. Putting the two together is the question the FETE was computed for,
// and until now it had to be answered by looking at a map. This measures it,
// per site and for the sample as a whole, in a form that can be compared
// between periods and between regions — which is the whole point: the interest
// is rarely "are these sites on the routes" and almost always "are they more on
// them than those others".
//
// Four questions carry the tool, asked from the most general to the most
// specific, and each measure exists to answer exactly one of them:
//
//  1. **Is there anything here at all?** What share of the sites has a corridor
//     within reach. A yes/no reading: if almost none has, the rest is noise.
//
//  2. **How far are they?** `dist_m`, the metres to the nearest corridor cell,
//     summarised by its median, its deciles, a histogram and a table of "share
//     of sites within X metres". This is the only family of numbers that does
//     not depend on the radius the user picked, which is what makes it the one
//     to quote when comparing two periods or two regions.
//
//  3. **How much corridor is around them?** Two sites the same distance from a
//     corridor are not in the same place if one has ten corridor cells within
//     reach and the other two hundred. The *proximity index* is the share of
//     the neighbourhood that is corridor — a share and not a count, so that a
//     30 m and a 90 m raster give the same answer where a count would differ
//     ninefold. Divided by the corridor's share of the whole surface it becomes
//     *enrichment*, whose expected value under chance is exactly 1.
//
//  4. **How busy is that ground?** The *intensity index*: the neighbourhood's
//     values, weighted by distance, then read as a percentile of the same
//     quantity measured across the surface. Taken on a log scale, because a
//     FETE cell holds a count of paths whose distribution is so heavy-tailed
//     that a plain mean would report the single busiest cell nearby rather
//     than the neighbourhood. The final percentile step is what makes it
//     comparable: a surface built from a million source points has counts
//     hundreds of times larger than one built from ten thousand, and that
//     factor cancels. 50 is what a location picked at random scores, in every
//     dataset — and, unlike a mean of ranks, 64 here really does mean "busier
//     than 64% of this surface".
//
// Two rules underneath all four:
//
//  * **A null model, where a null model is needed.** A median distance of
//     118 m means nothing on its own. It is therefore compared against
//     hundreds of point sets that have no relationship with the corridors but
//     share everything else — the same area, the same number of points, and by
//     default the same internal arrangement, translated as a block. Only the
//     distance gets this treatment, and deliberately: a distance has no
//     natural reference point, while enrichment already has an exact one (1)
//     and intensity another (50).
//
//  * **Missing is missing.** A NoData cell is not a cell with no traffic. It
//     is excluded from the ranks, from the weighted mean and from every
//     denominator; what it costs a site is reported as that site's coverage.
//
// Everything is in the raster's own units, so it must be projected — degrees
// would make the radius meaningless, which is refused rather than reported.
namespace Coherence {

// How the corridor cells are told from the rest.
enum class ThresholdMode {
    TopPercent,   // the top q% of the surface, by rank. Comparable across datasets.
    Otsu,         // the split that best separates two classes, on log values.
    Absolute      // a raw value, for someone who knows what theirs mean.
};

// What the random point sets preserve.
enum class NullMode {
    RandomShift,  // the whole pattern, translated as a block: keeps the sites'
                  // own clustering, which uniform points do not.
    Uniform       // independent points anywhere in the valid area.
};

struct Params {
    QString rasterPath;         // the FETE surface, raw or NNI-interpolated
    QString pointsPath;         // any OGR-readable point layer

    double radiusMetres = 250.0;

    ThresholdMode thresholdMode = ThresholdMode::TopPercent;
    double thresholdValue = 1.0;   // top q% (TopPercent) or the raw value (Absolute)

    bool nullModel = true;
    NullMode nullMode = NullMode::RandomShift;
    int nullReplicates = 999;      // 999 makes the smallest reportable p 0.001

    bool sensitivity = false;
    QVector<double> sensitivityRadii;

    // The rungs of the "share of sites within X metres" table, in metres.
    // Fixed distances rather than fractions of the radius, precisely so that
    // two runs can be laid side by side row for row whatever radius each was
    // given. Empty falls back to kDefaultEcdfDistances below; rungs finer than
    // one cell are dropped, since a raster cannot resolve them.
    QVector<double> ecdfDistances;

    bool edgeGuard = true;         // flag sites within one radius of the raster edge

    QString outputDir;
    QString outputPrefix = QStringLiteral("coherence");
    bool writeVector = true;
    bool vectorAsGeoPackage = true;   // false: ESRI Shapefile
    bool writeDistanceRaster = true;
    // An R script (ggplot2) that redraws the distance histogram from question
    // 2 of the report, bin for bin — not a fresh binning of the raw distances,
    // the same one. On by default: it costs a few lines of text and turns a
    // block of ASCII bars into a figure someone can actually put in a paper.
    bool writeHistogramScript = true;

    quint32 seed = 20260819;       // fixed, so a run can be repeated exactly
};

// What Params::ecdfDistances falls back to. Round numbers a reader can hold in
// mind, spanning "on it" to "a long walk away".
inline QVector<double> kDefaultEcdfDistances()
{
    return {0.0, 100.0, 250.0, 500.0, 1000.0, 2500.0};
}

// One site.
struct SiteResult {
    double x = 0.0, y = 0.0;
    bool inside = true;      // within the raster's extent
    bool hasData = true;     // the cell under it is not NoData
    bool nearEdge = false;   // within one radius of the extent's boundary

    double distM = 0.0;      // metres to the nearest corridor cell
    double proxIndex = 0.0;  // 0-100: share of the neighbourhood that is corridor
    double enrichment = 0.0; // that share over the whole surface's; 1 = chance
    double intenIndex = 0.0; // 0-100: percentile of the log-weighted neighbourhood
    double rankSite = -1.0;  // rank of the cell under the site, -1 if NoData
    double coverage = 1.0;   // fraction of the disc that had data

    QString cls;             // ON_CORRIDOR / NEAR_THIN / DIFFUSE / OFF
    QStringList fields;      // the input layer's own attributes, as text
};

// The distances, and the null they are judged against.
//
// Kept apart from RadiusStats because none of it depends on the radius: the
// distance to the nearest corridor is a property of the site and the surface,
// and asking for it at four radii would print the same number four times. It
// is also why the null lives here — the null tests the distance, so running it
// once per radius would spend the replicates to reach the same answer again.
struct DistanceStats {
    double median = 0.0;
    double d10 = 0.0, d25 = 0.0, d75 = 0.0, d90 = 0.0;
    double iqr = 0.0;

    QVector<double> ecdfAt;      // the rungs actually used, in metres
    QVector<double> ecdfShare;   // fraction of sites within each

    bool nullDone = false;
    int nullUsed = 0;            // replicates that placed successfully
    bool nullShifted = true;     // false: the pattern could not be shifted

    double nullMedian = 0.0;     // "118 m against 240 m expected"
    double nullLo = 0.0, nullHi = 0.0;   // 2.5th / 97.5th percentile of the nulls
    // Observed over expected — the effect size to carry between periods. A
    // ratio rather than a z-score, because the null distribution of a distance
    // is skewed and floored at zero: sites sitting exactly on the corridors
    // come out at p = 0.001 and z = -1.2 at the same time, and only one of
    // those two numbers is telling the truth. 0.5 means "half as far as chance
    // would put them", in any period and on any surface.
    double ratio = 1.0;
    double p = 1.0;              // left-tailed: closer than chance is the claim
};

// The sample, at one radius. Only what actually changes with the radius.
//
// Mean *and* median for each, and the mean is the one to quote — which is the
// opposite of the usual advice, for a reason worth stating. Both references
// these measures carry are exact statements about a mean: the expected share
// of corridor around a point placed at random is the corridor's share of the
// whole surface, so mean enrichment is exactly 1 under chance; and mid-ranks
// make the average percentile of a surface exactly 50, so mean intensity is
// exactly 50. Neither identity holds for a median.
//
// The medians are kept beside them because on most surfaces they are far
// lower, and that gap is information rather than noise: corridors are thin,
// linear and clustered, so on a typical FETE the majority of locations have no
// corridor within reach at all and a majority of neighbourhoods are entirely
// dead. A median enrichment of 0 alongside a mean of 1 is what that looks
// like, and a reader who sees only one of the two numbers will misread it.
struct RadiusStats {
    double radius = 0.0;
    double meanProxIdx = 0.0, medianProxIdx = 0.0;      // 0-100
    double meanEnrich = 0.0, medianEnrich = 0.0;        // mean 1 = chance
    double meanIntenIdx = 0.0, medianIntenIdx = 0.0;    // 0-100, mean 50 = chance
    double shareWithin = 0.0;     // fraction of sites with a corridor inside the radius
};

struct Result {
    bool ok = false;
    QString error;

    // What was read
    int rasterWidth = 0, rasterHeight = 0;
    double cellSize = 0.0;
    QString crsName;
    double validCells = 0.0, totalCells = 0.0;

    // The corridor
    double thresholdValue = 0.0;      // the raw value the mode settled on
    double thresholdPercentile = 0.0; // where that value sits in the surface
    double corridorShare = 0.0;       // fraction of valid cells selected
    QString thresholdNote;            // Otsu's warning, when there is one

    // The sample
    int pointsRead = 0;
    int pointsOutside = 0;    // outside the extent
    int pointsNoData = 0;     // inside, but on a NoData cell
    int pointsUsed = 0;
    int pointsNearEdge = 0;
    int lowCoverage = 0;      // coverage below half

    QStringList fieldNames;
    QVector<SiteResult> sites;

    DistanceStats dist;       // radius-independent: the numbers to quote
    RadiusStats main;
    QVector<RadiusStats> sensitivity;

    // Where the outputs went
    QString csvPath, vectorPath, rasterPath, summaryPath, rScriptPath;

    QString report() const;
};

// The distance histogram from question 2 of the report — its bin width and
// per-bin counts, the last bin being the p90+ overflow. Shared by report()
// (drawn as ASCII bars) and writeHistogramRScript() (drawn as a ggplot2
// figure), so the two can never show a different histogram for the same run.
struct DistHistogram {
    double width = 0.0;
    static constexpr int kBins = 10;   // plus one more for the overflow bin
    QVector<int> counts;               // size kBins+1 once computed; empty if no sites
};
DistHistogram computeDistHistogram(const Result &res);

// Writes an R script (ggplot2) that redraws computeDistHistogram(res) as a bar
// chart. Self-contained: the bin counts are written into the script as literal
// vectors, so it runs with nothing else from this run at hand beyond an R
// installation with ggplot2.
bool writeHistogramRScript(const Result &res, const QString &path, QString *error);

// Running commentary for the log the interface shows: what was read, what the
// threshold turned out to be, how long each stage took. The report says what
// the answer is; this says what was done to reach it.
using LogSink = std::function<void(const QString &)>;

Result run(const Params &params, const LogSink &log = {});

// Exposed for the self-test: the exact Euclidean distance transform, in cells,
// of the "true" cells of `mask`. Squared distances are returned.
QVector<double> distanceTransformSquared(const QVector<quint8> &mask, int w, int h);

} // namespace Coherence
