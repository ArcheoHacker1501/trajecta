#pragma once
// ============================================================================
// TRAJECTA - slope-dependent cost functions
//
// Every function here answers the same question: given one move between two
// cell centres, how much does it cost? The move is described by `dh_m`, the
// horizontal distance in metres, and `dz_m`, the signed elevation difference,
// and the slope is `S = dz_m / dh_m` — a tangent, not an angle, and signed, so
// every model below is anisotropic and A->B is not B->A.
//
// A note on why these return a cost and not a speed. A hiking function is
// published as a speed: how fast you cross a cell. Feeding that straight into
// a shortest-path search inverts the question, because the search minimises
// what you give it and would then avoid the *easiest* ground. White (2015)
// makes the point explicitly, and it is the reason the entry named after him
// is called "modified": the modification is the inversion into time,
// cost = distance / speed. There is no sense in also offering the raw hiking
// function — it would be the same model pointed the wrong way.
//
// The one exception is Herzog, which is an energy model and is already a cost:
// it returns kilojoules, not hours. cost_function_units() is what keeps the two
// apart, and every summary and manifest prints it.
//
// Shared by main_fete.cpp and main_lcpa.cpp so the two modes cannot drift.
// ============================================================================

#include <algorithm>
#include <cmath>

enum CostFunctionType {
    TOBLER_WHITE_2015 = 1,
    MARQUEZ_PEREZ_ET_AL_2017 = 2,
    IRMISCHER_CLARKE_2017 = 3,
    HERZOG_2013 = 4,
    CAMPBELL_2019_P5 = 5,
    CAMPBELL_2019_P50 = 6
};

namespace costfn {
inline constexpr double kPi = 3.14159265358979323846;
}

// What one unit of cost means. Everything but Herzog answers "how long", Herzog
// answers "how much effort", and the two must never be added together or
// compared. A cost raster whose unit is only implied is an error that
// propagates silently into every analysis downstream.
inline const char* cost_function_units(CostFunctionType cf) {
    return (cf == HERZOG_2013) ? "kJ/kg" : "hours";
}

// The published name, for the run manifest: "2" means nothing to somebody
// reading the record of an analysis a year later.
inline const char* cost_function_name(CostFunctionType cf) {
    switch (cf) {
    case TOBLER_WHITE_2015:        return "Modified Tobler's Hiking Function (White 2015)";
    case MARQUEZ_PEREZ_ET_AL_2017: return "Marquez-Perez et al. (2017)";
    case IRMISCHER_CLARKE_2017:    return "Irmischer & Clarke (2017), on-path male";
    case HERZOG_2013:              return "Herzog (2013) metabolic cost, after Minetti et al. (2002)";
    case CAMPBELL_2019_P5:         return "Campbell et al. (2019) Lorentz, 5th percentile";
    case CAMPBELL_2019_P50:        return "Campbell et al. (2019) Lorentz, 50th percentile";
    }
    return "unknown";
}

// ---- Tobler (1993), inverted into time as White (2015) describes ----
// v = 6 * exp(-3.5 * |S + 0.05|) km/h. The +0.05 puts the fastest walking on a
// 5% downhill, which is what the data show. This is the on-path form: Tobler's
// x0.6 factor for cross-country travel is not applied.
inline float tobler_white_2015(double dh_m, double dz_m) {
    const double sf = dz_m / dh_m;
    const double speed_kmh = 6.0 * std::exp(-3.5 * std::abs(sf + 0.05));
    const double safe_speed = std::max(speed_kmh, 1e-12);
    return (float)((dh_m / 1000.0) / safe_speed);
}

// ---- Marquez-Perez et al. (2017) ----
// v = 4.8 * exp(-5.3 * |0.7*S + 0.03|) km/h. Tobler recalibrated on GPS tracks
// from marked trails: slower overall, and slope hurts sooner.
inline float marquez_perez_et_al_2017(double dh_m, double dz_m) {
    const double sf = dz_m / dh_m;
    const double speed_kmh = 4.8 * std::exp(-5.3 * std::abs((sf * 0.7) + 0.03));
    const double safe_speed = std::max(speed_kmh, 1e-12);
    return (float)((dh_m / 1000.0) / safe_speed);
}

// ---- Irmischer & Clarke (2017), on-path male ----
// v = 0.11 + exp(-(S% + 5)^2 / 1800) m/s, with 1800 = 2*30^2. The paper gives
// four variants (male/female x on-path/off-path); this is one of them. The
// 0.11 m/s term is a floor, so the function never reaches zero speed.
inline float irmischer_clarke_2017(double dh_m, double dz_m) {
    const double sf = (dz_m / dh_m) * 100.0;
    const double speed_ms  = 0.11 + std::exp(-(sf + 5.0) * (sf + 5.0) / 1800.0);
    const double speed_kmh = speed_ms * 3.6;
    const double safe_speed = std::max(speed_kmh, 1e-12);
    return (float)((dh_m / 1000.0) / safe_speed);
}

// ---- Herzog (2013), sixth-degree fit to Minetti et al. (2002) ----
// Kilojoules per kilogram of walker per metre travelled. This is energy, not
// time, and it has the shape the treadmill measurements show and the speed
// models miss: a minimum at about a 10.5% downhill and a rise on *both* sides,
// because braking down a steep slope costs too.
//
// Minetti's data span roughly +/-45% slope. Beyond that the polynomial is an
// extrapolation: it stays positive and climbs steeply, which is right in
// direction but is no longer a measurement. The slope cut-off is the honest way
// to stay inside the calibrated range.
inline double herzog_2013_kj_per_kg_m(double s) {
    const double s2 = s * s, s3 = s2 * s, s4 = s3 * s, s5 = s4 * s, s6 = s5 * s;
    const double c = 1337.8 * s6 + 278.19 * s5 - 517.39 * s4 - 78.199 * s3
                   + 93.419 * s2 + 19.825 * s + 1.64;
    // The fit stays positive over any slope a walker could face, but a cost of
    // zero or less would break the shortest-path search outright.
    return (c > 1e-6) ? c : 1e-6;
}

inline float herzog_2013(double dh_m, double dz_m) {
    return (float)(herzog_2013_kj_per_kg_m(dz_m / dh_m) * dh_m);   // kJ/kg
}

// ---- Campbell et al. (2019), asymmetric Lorentz on travel rate ----
// Fitted to 421,247 GPS activities from 29,928 people. theta is the slope in
// DEGREES and the result is metres per second:
//
//     v = c / (pi*b*(1 + ((theta - a)/b)^2)) + d + e*theta
//
// The paper publishes one parameter set per percentile of the population. The
// two offered here are the 5th, which the authors recommend as representative
// of ordinary hiking, and the 50th, the median of a dataset that also contains
// joggers and runners. The fit is calibrated for |theta| < 30 degrees.
struct CampbellParams { double c, b, a, d, e; };

inline CampbellParams campbell_params(CostFunctionType cf) {
    if (cf == CAMPBELL_2019_P50) return { 63.660, 10.064, -2.171, 0.628, -0.00463 };
    return                              { 36.813, 14.041, -1.527, 0.320, -0.00273 };  // 5th
}

inline double campbell_2019_speed_ms(const CampbellParams& p, double theta_deg) {
    const double t = (theta_deg - p.a) / p.b;
    const double lorentz = p.c / (costfn::kPi * p.b * (1.0 + t * t));
    const double v = lorentz + p.d + p.e * theta_deg;
    return (v > 0.01) ? v : 0.01;   // the fit never reaches zero inside its range
}

inline float campbell_2019(CostFunctionType cf, double dh_m, double dz_m) {
    const double theta_deg = std::atan(dz_m / dh_m) * 180.0 / costfn::kPi;
    const double v_kmh = campbell_2019_speed_ms(campbell_params(cf), theta_deg) * 3.6;
    return (float)((dh_m / 1000.0) / v_kmh);                       // hours
}

inline float apply_cost_function(CostFunctionType cf, double dh_m, double dz_m) {
    switch (cf) {
        case MARQUEZ_PEREZ_ET_AL_2017: return marquez_perez_et_al_2017(dh_m, dz_m);
        case IRMISCHER_CLARKE_2017:    return irmischer_clarke_2017(dh_m, dz_m);
        case HERZOG_2013:              return herzog_2013(dh_m, dz_m);
        case CAMPBELL_2019_P5:
        case CAMPBELL_2019_P50:        return campbell_2019(cf, dh_m, dz_m);
        default:                       return tobler_white_2015(dh_m, dz_m);
    }
}

// ---- Slope cut-off ----
// A move is refused outright when it is steeper than the walker is willing to
// take, in either direction. This is a property of the *move*, not of the cell:
// a terrace can be entered from the side and not from below, which is how real
// terrain behaves, and it is why the test lives next to the cost rather than in
// the passability mask.
//
// The limits arrive in degrees, which is how people think about slopes, and are
// converted once into the tangents the engine actually compares against.
struct SlopeLimit {
    bool enabled = false;
    double up_tan = 0.0;     // tan(max uphill), positive
    double down_tan = 0.0;   // tan(max downhill), positive
};

inline SlopeLimit make_slope_limit(bool enabled, double up_deg, double down_deg) {
    SlopeLimit sl;
    sl.enabled = enabled;
    const double kDegToRad = costfn::kPi / 180.0;
    // 90 degrees would be a vertical wall and an infinite tangent; anything at
    // or above it means "no limit in this direction".
    sl.up_tan   = (up_deg   >= 89.9) ? 1e30 : std::tan(up_deg   * kDegToRad);
    sl.down_tan = (down_deg >= 89.9) ? 1e30 : std::tan(down_deg * kDegToRad);
    return sl;
}
