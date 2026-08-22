#pragma once

#include <QString>
#include <QVector>

#include <functional>

// Comparing a computed route against a route that is actually known — a Roman
// road, a drover's track, a surveyed path.
//
// This is the step that turns a least-cost path from an illustration into a
// claim that can be wrong. Without it a model can only ever agree with itself.
//
// What is measured, and why these and not others:
//
//   * distances from one line to the other, in metres. Reported as a
//     distribution (median, 90th percentile, maximum) rather than a single
//     number, because a route can follow the real one closely for 9 km and
//     then take the wrong side of a hill for 1 km, and a mean hides exactly
//     that.
//   * the maximum of the two directions is the (discrete) Hausdorff distance:
//     the worst disagreement anywhere. Both directions are needed — a short
//     computed path lying on top of a long known one is close in one direction
//     and far in the other.
//   * the share of each line that runs within a tolerance of the other, which
//     is the question most often actually asked: "how much of it did we get
//     right?"
//
// Everything is measured in the units of the layers' own coordinate system, so
// both must be projected and in the same CRS. Degrees would make every distance
// meaningless, which is checked and refused rather than silently reported.
namespace RouteCompare {

struct Result {
    bool ok = false;
    QString error;

    // Sampled geometry, for the report
    int computedFeatures = 0;
    int knownFeatures = 0;
    double computedLength = 0.0;   // metres, planimetric
    double knownLength = 0.0;
    double sampleStep = 0.0;       // metres between sampled points

    // computed -> known
    double medianAToB = 0.0;
    double p90AToB = 0.0;
    double maxAToB = 0.0;
    double withinAToB = 0.0;       // fraction of the computed line within tolerance

    // known -> computed
    double medianBToA = 0.0;
    double p90BToA = 0.0;
    double maxBToA = 0.0;
    double withinBToA = 0.0;

    double hausdorff = 0.0;        // max(maxAToB, maxBToA)
    double tolerance = 0.0;        // what was asked for, echoed back

    // Human-readable, ready to show and to write next to the results.
    QString report() const;
};

// Running commentary, one line at a time, for the log the interface shows.
// The report says what the answer is; the log says what was read to arrive at
// it — which CRS each layer turned out to be in, how many parts and how many
// sample points — and that is what you need when the answer is surprising.
using LogSink = std::function<void(const QString &)>;

// `computedPath` is normally the LCPA paths shapefile; `knownPath` the route
// you are testing against. Any OGR-readable vector of lines will do for either.
// `toleranceMetres` sets what counts as "close". `log` may be left out, and
// nothing is then produced beyond the Result.
Result compare(const QString &computedPath, const QString &knownPath,
               double toleranceMetres, const LogSink &log = {});

} // namespace RouteCompare
