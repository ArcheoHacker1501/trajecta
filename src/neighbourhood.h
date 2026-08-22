#pragma once
// ============================================================================
// TRAJECTA - neighbourhood templates for the cost graph
//
// A neighbourhood is the set of moves a path may take from one cell. It cannot
// be any set of offsets: it has to be closed under the eight symmetries of the
// square (four rotations and their mirrors), otherwise some compass directions
// are cheaper to travel than others and every result leans that way. That is
// the real constraint behind the "you cannot just ask for 37 neighbours" rule.
//
// The offsets therefore come in whole orbits under that symmetry group. An
// orbit has eight members in general, and four when the direction lies on an
// axis (a, 0) or on a diagonal (a, a), because those map onto themselves. The
// reachable sizes are the running totals of those orbits:
//
//     8, 16, 24, 32, 40, 44, 48, 56, 64, 68, 76, 80, ...
//
// The order is the one the hand-written tables of v1.0.0 used, so 8, 16, 24 and
// 32 produce exactly the same offsets as before and results stay comparable:
// by growing Chebyshev ring, and inside a ring the primitive directions first
// (those with gcd(|dr|,|dc|) = 1), each group by increasing length.
//
// A note on the non-primitive directions, which the tables have always
// included from 24 upwards: (2,0) is two steps of (1,0), so it adds no new
// heading. It is not redundant, though - the cost of the long move is computed
// from the slope over the whole span, so it can cross a dip the two short
// moves would have to climb into. It buys smoothness at the price of ignoring
// what lies between the endpoints.
// ============================================================================

#include <algorithm>
#include <utility>
#include <vector>

struct Off { int dr; int dc; };

namespace neighbourhood {

// Below 8 a path cannot move diagonally at all; above 256 the per-cell work is
// out of proportion to anything it buys.
inline constexpr int kMin = 8;
inline constexpr int kMax = 256;

inline int gcdInt(int a, int b) {
    while (b != 0) { const int t = a % b; a = b; b = t; }
    return a < 0 ? -a : a;
}

// The images of (a, b) under the eight symmetries, minus what is already there.
inline void appendOrbit(int a, int b, std::vector<Off>& out) {
    const int cand[8][2] = { {a,b},{a,-b},{-a,b},{-a,-b},{b,a},{b,-a},{-b,a},{-b,-a} };
    for (const auto& c : cand) {
        bool seen = false;
        for (const Off& o : out) {
            if (o.dr == c[0] && o.dc == c[1]) { seen = true; break; }
        }
        if (!seen) out.push_back(Off{ c[0], c[1] });
    }
}

// Orbit representatives of one Chebyshev ring, in the order the tables used.
inline std::vector<std::pair<int, int>> ringOrder(int k) {
    std::vector<std::pair<int, int>> prim, rest;
    for (int j = 0; j <= k; ++j) {
        (gcdInt(k, j) == 1 ? prim : rest).push_back({ k, j });
    }
    const auto byLength = [](const std::pair<int, int>& p, const std::pair<int, int>& q) {
        return p.first * p.first + p.second * p.second
             < q.first * q.first + q.second * q.second;
    };
    std::sort(prim.begin(), prim.end(), byLength);
    std::sort(rest.begin(), rest.end(), byLength);
    prim.insert(prim.end(), rest.begin(), rest.end());
    return prim;
}

// Fills `out` with the largest admissible neighbourhood no bigger than
// `requested`, and returns how many directions that is. Asking for a number
// between two admissible sizes rounds down rather than up: a path is never
// given moves the user did not ask to pay for.
inline int build(int requested, std::vector<Off>& out) {
    out.clear();
    out.reserve(64);
    if (requested < kMin) requested = kMin;
    if (requested > kMax) requested = kMax;

    for (int k = 1; k <= kMax; ++k) {
        for (const auto& rep : ringOrder(k)) {
            std::vector<Off> trial = out;
            appendOrbit(rep.first, rep.second, trial);
            if (static_cast<int>(trial.size()) > requested) {
                return static_cast<int>(out.size());
            }
            out.swap(trial);
        }
    }
    return static_cast<int>(out.size());
}

// Every admissible size up to kMax, for error messages and help text.
inline std::vector<int> admissibleSizes() {
    std::vector<int> sizes;
    std::vector<Off> offs;
    for (int k = 1; k <= kMax; ++k) {
        for (const auto& rep : ringOrder(k)) {
            appendOrbit(rep.first, rep.second, offs);
            const int n = static_cast<int>(offs.size());
            if (n > kMax) return sizes;
            if (n >= kMin) sizes.push_back(n);
        }
    }
    return sizes;
}

// The largest admissible size not above `requested`, without building anything.
inline int snap(int requested) {
    std::vector<Off> offs;
    return build(requested, offs);
}

} // namespace neighbourhood
