#include "routecompare.h"

#include "gdalapi.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QObject>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

struct Pt { double x, y; };
using Line = QVector<Pt>;

double segmentDistance(const Pt &p, const Pt &a, const Pt &b)
{
    const double vx = b.x - a.x, vy = b.y - a.y;
    const double len2 = vx * vx + vy * vy;
    if (len2 <= 0.0) {
        const double dx = p.x - a.x, dy = p.y - a.y;
        return std::sqrt(dx * dx + dy * dy);
    }
    // Projection of p onto the segment, clamped to its ends.
    double t = ((p.x - a.x) * vx + (p.y - a.y) * vy) / len2;
    t = std::clamp(t, 0.0, 1.0);
    const double dx = p.x - (a.x + t * vx);
    const double dy = p.y - (a.y + t * vy);
    return std::sqrt(dx * dx + dy * dy);
}

double distanceToLines(const Pt &p, const QVector<Line> &lines)
{
    double best = std::numeric_limits<double>::max();
    for (const Line &l : lines) {
        for (int i = 0; i + 1 < l.size(); ++i) {
            const double d = segmentDistance(p, l[i], l[i + 1]);
            if (d < best) best = d;
        }
        // A layer can legitimately hold a single-vertex "line"; without this
        // it would contribute no segments and be silently invisible.
        if (l.size() == 1) {
            const double dx = p.x - l[0].x, dy = p.y - l[0].y;
            best = std::min(best, std::sqrt(dx * dx + dy * dy));
        }
    }
    return best;
}

double totalLength(const QVector<Line> &lines)
{
    double sum = 0.0;
    for (const Line &l : lines) {
        for (int i = 0; i + 1 < l.size(); ++i) {
            const double dx = l[i + 1].x - l[i].x;
            const double dy = l[i + 1].y - l[i].y;
            sum += std::sqrt(dx * dx + dy * dy);
        }
    }
    return sum;
}

// Points every `step` metres along the lines, plus every vertex: a corner cut
// between two samples would otherwise never be measured.
QVector<Pt> densify(const QVector<Line> &lines, double step)
{
    QVector<Pt> out;
    for (const Line &l : lines) {
        if (l.isEmpty()) continue;
        out.append(l.first());
        for (int i = 0; i + 1 < l.size(); ++i) {
            const double dx = l[i + 1].x - l[i].x;
            const double dy = l[i + 1].y - l[i].y;
            const double len = std::sqrt(dx * dx + dy * dy);
            const int n = int(len / step);
            for (int k = 1; k <= n; ++k) {
                const double t = double(k) * step / len;
                out.append({ l[i].x + t * dx, l[i].y + t * dy });
            }
            out.append(l[i + 1]);
        }
    }
    return out;
}

void collectGeometry(OGRGeometryH g, QVector<Line> &out)
{
    GdalApi &api = GdalApi::instance();
    if (!g) return;
    const int type = GdalApi::flattenGeomType(api.G_GetGeometryType(g));
    if (type == GdalApi::WkbLineString) {
        const int n = api.G_GetPointCount(g);
        Line l;
        l.reserve(n);
        for (int i = 0; i < n; ++i) {
            double x = 0, y = 0, z = 0;
            api.G_GetPoint(g, i, &x, &y, &z);
            l.append({ x, y });
        }
        if (!l.isEmpty()) out.append(l);
        return;
    }
    // Multi-anything, and geometry collections: recurse.
    const int parts = api.G_GetGeometryCount(g);
    for (int i = 0; i < parts; ++i)
        collectGeometry(api.G_GetGeometryRef(g, i), out);
}

// Reads every line of the first layer. Returns false with `error` set.
// `crs` comes back as a human-readable name for the log — it is the first
// thing to check when two layers that ought to coincide come out kilometres
// apart — and is left empty when the layer carries no spatial reference.
bool readLines(const QString &path, QVector<Line> &out, int *features,
               bool *geographic, QString *crs, QString *error)
{
    GdalApi &api = GdalApi::instance();
    GDALDatasetH ds = api.OpenEx(path.toUtf8().constData(), GdalApi::OF_Vector,
                                 nullptr, nullptr, nullptr);
    if (!ds) {
        *error = QObject::tr("Cannot open %1").arg(QFileInfo(path).fileName());
        return false;
    }
    if (api.DatasetGetLayerCount(ds) < 1) {
        api.Close(ds);
        *error = QObject::tr("%1 contains no layer").arg(QFileInfo(path).fileName());
        return false;
    }
    OGRLayerH layer = api.DatasetGetLayer(ds, 0);

    // Degrees would make every distance below meaningless, so this is refused
    // rather than reported in units nobody can interpret.
    *geographic = false;
    crs->clear();
    if (api.L_GetSpatialRef) {
        if (OGRSpatialReferenceH srs = api.L_GetSpatialRef(layer)) {
            *geographic = api.OSRIsGeographic(srs) != 0;
            if (api.OSRGetName) {
                if (const char *name = api.OSRGetName(srs))
                    *crs = QString::fromUtf8(name);
            }
            // The authority code is what anyone actually quotes, so it is worth
            // the two extra calls when the driver can supply it.
            if (api.OSRGetAuthorityName && api.OSRGetAuthorityCode) {
                const char *auth = api.OSRGetAuthorityName(srs, nullptr);
                const char *code = api.OSRGetAuthorityCode(srs, nullptr);
                if (auth && code)
                    *crs += QStringLiteral(" (%1:%2)")
                                .arg(QString::fromUtf8(auth), QString::fromUtf8(code));
            }
        }
    }

    api.L_ResetReading(layer);
    *features = 0;
    while (OGRFeatureH f = api.L_GetNextFeature(layer)) {
        collectGeometry(api.F_GetGeometryRef(f), out);
        api.F_Destroy(f);
        ++(*features);
    }
    api.Close(ds);

    if (out.isEmpty()) {
        *error = QObject::tr("%1 holds no line geometry").arg(QFileInfo(path).fileName());
        return false;
    }
    return true;
}

struct Stats { double median, p90, max, within; };

Stats summarise(QVector<double> d, double tolerance)
{
    Stats s{ 0, 0, 0, 0 };
    if (d.isEmpty()) return s;
    std::sort(d.begin(), d.end());
    const auto at = [&](double q) {
        const int last = int(d.size()) - 1;
        const int i = std::clamp(int(q * last + 0.5), 0, last);
        return d[i];
    };
    s.median = at(0.5);
    s.p90 = at(0.9);
    s.max = d.last();
    int inside = 0;
    for (double v : d) if (v <= tolerance) ++inside;
    s.within = double(inside) / double(d.size());
    return s;
}

} // namespace

namespace RouteCompare {

Result compare(const QString &computedPath, const QString &knownPath,
               double toleranceMetres, const LogSink &log)
{
    Result r;
    r.tolerance = toleranceMetres;

    // Every failure below is reported through the log as well as through
    // Result::error: the summary is one line, and the log is where you can see
    // which of the two layers it was about.
    const auto say = [&log](const QString &line) { if (log) log(line); };
    const auto fail = [&](const QString &message) {
        r.error = message;
        say(message);
        return r;
    };

    QElapsedTimer clock;
    clock.start();

    GdalApi &api = GdalApi::instance();
    if (!api.isLoaded())
        return fail(QObject::tr("GDAL is not available; use \"Locate GDAL folder\" first."));

    QVector<Line> computed, known;
    bool geoA = false, geoB = false;
    QString crsA, crsB;

    say(QObject::tr("Reading computed routes: %1")
            .arg(QDir::toNativeSeparators(computedPath)));
    if (!readLines(computedPath, computed, &r.computedFeatures, &geoA, &crsA, &r.error)) {
        say(r.error);
        return r;
    }
    say(QObject::tr("  %1 feature(s), %2 line part(s), CRS: %3")
            .arg(r.computedFeatures)
            .arg(computed.size())
            .arg(crsA.isEmpty() ? QObject::tr("none declared") : crsA));

    say(QObject::tr("Reading known route: %1")
            .arg(QDir::toNativeSeparators(knownPath)));
    if (!readLines(knownPath, known, &r.knownFeatures, &geoB, &crsB, &r.error)) {
        say(r.error);
        return r;
    }
    say(QObject::tr("  %1 feature(s), %2 line part(s), CRS: %3")
            .arg(r.knownFeatures)
            .arg(known.size())
            .arg(crsB.isEmpty() ? QObject::tr("none declared") : crsB));

    if (geoA || geoB) {
        return fail(QObject::tr(
            "One of the layers is in geographic coordinates (degrees). Distances "
            "would not be in metres and the comparison would mean nothing — "
            "reproject both to the same projected CRS first."));
    }
    // Not fatal: two different projected CRSs can still be metres, and a
    // reprojection here would be guesswork. It is, however, the single most
    // likely explanation for a comparison that comes out absurd.
    if (!crsA.isEmpty() && !crsB.isEmpty() && crsA != crsB) {
        say(QObject::tr("  WARNING: the two layers declare different coordinate "
                        "systems (%1 vs %2). The distances below assume they are "
                        "in the same one.")
                .arg(crsA, crsB));
    }

    r.computedLength = totalLength(computed);
    r.knownLength = totalLength(known);
    say(QObject::tr("Lengths: computed %1 m, known %2 m")
            .arg(QString::number(r.computedLength, 'f', 1),
                 QString::number(r.knownLength, 'f', 1)));

    // Fine enough that the sampling itself is not what the numbers measure,
    // coarse enough to stay quick on a long route.
    r.sampleStep = std::max(1.0, toleranceMetres / 5.0);

    const QVector<Pt> sampledA = densify(computed, r.sampleStep);
    const QVector<Pt> sampledB = densify(known, r.sampleStep);
    say(QObject::tr("Sampling every %1 m: %2 point(s) on the computed routes, "
                    "%3 on the known route")
            .arg(QString::number(r.sampleStep, 'f', 1))
            .arg(sampledA.size())
            .arg(sampledB.size()));
    say(QObject::tr("Measuring distances in both directions..."));

    QVector<double> aToB, bToA;
    aToB.reserve(sampledA.size());
    bToA.reserve(sampledB.size());
    for (const Pt &p : sampledA) aToB.append(distanceToLines(p, known));
    for (const Pt &p : sampledB) bToA.append(distanceToLines(p, computed));

    const Stats sa = summarise(aToB, toleranceMetres);
    const Stats sb = summarise(bToA, toleranceMetres);
    r.medianAToB = sa.median; r.p90AToB = sa.p90; r.maxAToB = sa.max; r.withinAToB = sa.within;
    r.medianBToA = sb.median; r.p90BToA = sb.p90; r.maxBToA = sb.max; r.withinBToA = sb.within;
    r.hausdorff = std::max(sa.max, sb.max);
    r.ok = true;
    say(QObject::tr("Done in %1 s.")
            .arg(QString::number(clock.elapsed() / 1000.0, 'f', 2)));
    return r;
}

QString Result::report() const
{
    if (!ok)
        return QObject::tr("Comparison failed: %1").arg(error);

    const auto m = [](double v) { return QString::number(v, 'f', 1); };
    const auto pc = [](double v) { return QString::number(100.0 * v, 'f', 1); };

    return QObject::tr(
               "Computed route: %1 feature(s), %2 m long\n"
               "Known route:    %3 feature(s), %4 m long\n"
               "Sampled every %5 m; \"close\" means within %6 m.\n"
               "\n"
               // A single %: QString::arg() only treats %1..%99 as placeholders,
               // so there is nothing to escape here — %% would print twice.
               "Computed -> known   median %7 m, 90th pct %8 m, worst %9 m\n"
               "                    %10% of its length runs close to the known route\n"
               "Known -> computed   median %11 m, 90th pct %12 m, worst %13 m\n"
               "                    %14% of it was recovered by the model\n"
               "\n"
               "Hausdorff distance (worst disagreement either way): %15 m\n")
        .arg(computedFeatures).arg(m(computedLength))
        .arg(knownFeatures).arg(m(knownLength))
        .arg(m(sampleStep)).arg(m(tolerance))
        .arg(m(medianAToB), m(p90AToB), m(maxAToB), pc(withinAToB))
        .arg(m(medianBToA), m(p90BToA), m(maxBToA), pc(withinBToA))
        .arg(m(hausdorff));
}

} // namespace RouteCompare
