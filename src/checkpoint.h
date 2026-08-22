#pragma once

// Checkpointing for the FETE propagation phase.
//
// A FETE run over a large DEM takes hours to days. The expensive part is a
// single loop over the source points, and the only state that loop accumulates
// is the density raster: every other array is either an input or is rebuilt
// deterministically from the inputs in seconds. So a checkpoint is exactly two
// things — how many sources are done, and the density so far.
//
// The loop is therefore run in blocks. At the end of a block every thread has
// finished every source in it (the barrier at the end of an `omp for`
// guarantees that), which is the only moment at which "sources 0..n-1 are
// complete" is a true statement in a dynamically scheduled parallel loop.
//
// Writing is timer-gated, not block-gated: on a small DEM a block is seconds
// and there is nothing worth saving; on a large one a block can be an hour.
//
// Two files are kept, written alternately. A new one is written in full under a
// temporary name and renamed into place — rename is atomic on both NTFS and
// POSIX — and only then is the older one removed. A machine that loses power
// mid-write therefore always leaves at least one complete checkpoint behind.
//
// Everything is off unless TRAJECTA_CHECKPOINT_DIR is set, so a run that does
// not ask for checkpoints behaves exactly as it did before this file existed.

#include <cstdint>
#include <cstdio>
#include <cstdlib>   // getenv, atof, atoi — used on every platform
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ckpt {

namespace fs = std::filesystem;

// Bumped whenever the layout below changes; an older file is then ignored
// rather than misread.
constexpr uint32_t kMagic = 0x54434B31;   // "TCK1"
// 2: the fingerprint gained the slope cut-off. A checkpoint written before it
// existed has a different layout, so it is refused rather than misread.
constexpr uint32_t kVersion = 2;

// Everything that has to match for a resumed run to produce the same numbers
// as an uninterrupted one. The cost surface, the slope and the multipliers are
// all rebuilt from these, so if they agree the rebuild agrees too.
struct Fingerprint {
    uint64_t demSize = 0;
    uint64_t demMTime = 0;
    uint64_t ptsSize = 0;
    uint64_t ptsMTime = 0;
    uint64_t modifiersSize = 0;
    uint64_t modifiersMTime = 0;
    uint64_t costRasterSize = 0;
    uint64_t costRasterMTime = 0;
    int64_t  cells = 0;          // N
    int32_t  rows = 0;
    int32_t  cols = 0;
    int32_t  sources = 0;        // P
    int32_t  neighbours = 0;
    int32_t  costFunction = 0;
    int32_t  bufferRadius = 0;
    int32_t  polylineBufferRadius = 0;
    double   barrierThreshold = 0.0;
    // Slope cut-off. Part of the fingerprint because resuming a run under a
    // different limit would splice together two different graphs.
    int32_t  slopeLimited = 0;
    double   maxSlopeUpDeg = 0.0;
    double   maxSlopeDownDeg = 0.0;
    uint64_t sourceHash = 0;     // over point_nodes, so the order is pinned too

    // Field by field, not memcmp: the struct has alignment padding and those
    // bytes are not guaranteed to be zero.
    bool operator==(const Fingerprint &o) const {
        return demSize == o.demSize && demMTime == o.demMTime
            && ptsSize == o.ptsSize && ptsMTime == o.ptsMTime
            && modifiersSize == o.modifiersSize && modifiersMTime == o.modifiersMTime
            && costRasterSize == o.costRasterSize && costRasterMTime == o.costRasterMTime
            && cells == o.cells && rows == o.rows && cols == o.cols
            && sources == o.sources && neighbours == o.neighbours
            && costFunction == o.costFunction && bufferRadius == o.bufferRadius
            && polylineBufferRadius == o.polylineBufferRadius
            && barrierThreshold == o.barrierThreshold
            && slopeLimited == o.slopeLimited
            && maxSlopeUpDeg == o.maxSlopeUpDeg
            && maxSlopeDownDeg == o.maxSlopeDownDeg
            && sourceHash == o.sourceHash;
    }
    bool operator!=(const Fingerprint &o) const { return !(*this == o); }
};

struct Header {
    uint32_t magic = kMagic;
    uint32_t version = kVersion;
    uint64_t sequence = 0;       // higher wins when both slots are readable
    int32_t  nextSource = 0;     // sources [0, nextSource) are complete
    int32_t  slot = 0;
    int64_t  unreachedPairs = 0;
    uint64_t densityChecksum = 0;
    Fingerprint fingerprint;
};

// Cheap, order-dependent, and good enough to catch a truncated or garbled
// payload. Not a security measure: nothing here is adversarial.
inline uint64_t fnv1a(const void *data, size_t bytes, uint64_t seed = 1469598103934665603ULL)
{
    const auto *p = static_cast<const unsigned char *>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

inline uint64_t fileSize(const std::string &path)
{
    if (path.empty())
        return 0;
    std::error_code ec;
    const auto n = fs::file_size(fs::path(path), ec);
    return ec ? 0 : static_cast<uint64_t>(n);
}

inline uint64_t fileMTime(const std::string &path)
{
    if (path.empty())
        return 0;
    std::error_code ec;
    const auto t = fs::last_write_time(fs::path(path), ec);
    return ec ? 0 : static_cast<uint64_t>(t.time_since_epoch().count());
}

// The per-user data directory, and never the install directory: Program Files
// (or /Applications) is not writable by a normal account, which is exactly the
// account a long analysis runs under.
//
// Every branch here has to match what the interface computes in
// Checkpoint::defaultDir(): the engine writes the checkpoint and the interface
// goes looking for it, so a disagreement would not fail loudly — every crash
// recovery would simply find nothing. The three pairings are
//
//   Windows   %LOCALAPPDATA%                == QStandardPaths::GenericConfigLocation
//   macOS     ~/Library/Application Support == QStandardPaths::GenericDataLocation
//   Linux     ~/.local/share                == QStandardPaths::GenericDataLocation
inline std::string defaultDir()
{
#ifdef _WIN32
    const char *base = std::getenv("LOCALAPPDATA");
    if (!base || !*base)
        base = std::getenv("APPDATA");
    if (base && *base)
        return (fs::path(base) / "Trajecta" / "checkpoints").string();
    return {};
#elif defined(__APPLE__)
    const char *home = std::getenv("HOME");
    if (home && *home)
        return (fs::path(home) / "Library" / "Application Support" / "Trajecta"
                / "checkpoints")
            .string();
    return {};
#else
    const char *home = std::getenv("HOME");
    if (home && *home)
        return (fs::path(home) / ".local" / "share" / "Trajecta" / "checkpoints").string();
    return {};
#endif
}

struct Config {
    bool enabled = false;
    std::string dir;
    double intervalSeconds = 1800.0;   // 30 minutes
    std::string resumePath;            // non-empty: resume from this file
    // How many sources make up one block. 1000 is small enough that the
    // barrier ending a block costs nothing measurable and large enough that
    // the check never gets in the way; the override exists so the test suite
    // can reach a block boundary on a dataset of a few hundred points.
    int blockSources = 1000;
};

inline Config configFromEnv()
{
    Config c;
    // The GUI sets these; an interactive user can set them in the shell. Absent
    // means off, which is the default the setting in the gear menu starts at.
    if (const char *dir = std::getenv("TRAJECTA_CHECKPOINT_DIR")) {
        if (*dir) {
            c.enabled = true;
            c.dir = dir;
        }
    }
    if (const char *mins = std::getenv("TRAJECTA_CHECKPOINT_MINUTES")) {
        const double m = std::atof(mins);
        if (m > 0.0)
            c.intervalSeconds = m * 60.0;
    }
    if (const char *resume = std::getenv("TRAJECTA_RESUME")) {
        if (*resume)
            c.resumePath = resume;
    }
    if (const char *block = std::getenv("TRAJECTA_CHECKPOINT_BLOCK")) {
        const int n = std::atoi(block);
        if (n > 0)
            c.blockSources = n;
    }
    return c;
}

inline std::string slotPath(const std::string &dir, int slot)
{
    return (fs::path(dir) / ("fete_checkpoint_" + std::to_string(slot) + ".tckpt")).string();
}

// Writes slot `slot` and, once it is safely in place, removes the other one.
// Returns false without disturbing the existing checkpoints if anything fails:
// a checkpoint that cannot be written must never take the run down with it.
inline bool save(const std::string &dir, int slot, uint64_t sequence,
                 const Fingerprint &fp, int nextSource, int64_t unreachedPairs,
                 const std::vector<uint64_t> &density, std::string *error = nullptr)
{
    std::error_code ec;
    fs::create_directories(fs::path(dir), ec);
    if (ec) {
        if (error) *error = ec.message();
        return false;
    }

    const std::string target = slotPath(dir, slot);
    const std::string temp = target + ".tmp";

    Header h;
    h.sequence = sequence;
    h.nextSource = nextSource;
    h.slot = slot;
    h.unreachedPairs = unreachedPairs;
    h.fingerprint = fp;
    h.densityChecksum = density.empty()
                            ? 0
                            : fnv1a(density.data(), density.size() * sizeof(uint64_t));

    {
        std::ofstream f(fs::path(temp), std::ios::binary | std::ios::trunc);
        if (!f) {
            if (error) *error = "cannot open " + temp;
            return false;
        }
        f.write(reinterpret_cast<const char *>(&h), sizeof(h));
        const uint64_t count = density.size();
        f.write(reinterpret_cast<const char *>(&count), sizeof(count));
        if (count)
            f.write(reinterpret_cast<const char *>(density.data()),
                    static_cast<std::streamsize>(count * sizeof(uint64_t)));
        f.flush();
        if (!f) {
            if (error) *error = "write failed (disk full?)";
            f.close();
            fs::remove(fs::path(temp), ec);
            return false;
        }
    }

    // rename() over an existing file fails on Windows; the target slot is the
    // one being replaced, so removing it first is safe — the *other* slot is
    // still a complete checkpoint at this instant.
    fs::remove(fs::path(target), ec);
    fs::rename(fs::path(temp), fs::path(target), ec);
    if (ec) {
        if (error) *error = ec.message();
        fs::remove(fs::path(temp), ec);
        return false;
    }

    // Only now is the previous one expendable.
    fs::remove(fs::path(slotPath(dir, slot == 0 ? 1 : 0)), ec);
    return true;
}

// Reads the header only. Used to find the newest valid checkpoint, and by the
// interface to describe one without loading a hundred megabytes of density.
inline bool peek(const std::string &path, Header *out)
{
    std::ifstream f(fs::path(path), std::ios::binary);
    if (!f)
        return false;
    Header h;
    f.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!f || h.magic != kMagic || h.version != kVersion)
        return false;
    if (out)
        *out = h;
    return true;
}

inline bool load(const std::string &path, Header *out, std::vector<uint64_t> *density)
{
    std::ifstream f(fs::path(path), std::ios::binary);
    if (!f)
        return false;
    Header h;
    f.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!f || h.magic != kMagic || h.version != kVersion)
        return false;
    uint64_t count = 0;
    f.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (!f)
        return false;
    // A corrupt length must not turn into a hundred-gigabyte allocation.
    if (count != static_cast<uint64_t>(h.fingerprint.cells))
        return false;
    std::vector<uint64_t> d(static_cast<size_t>(count));
    if (count)
        f.read(reinterpret_cast<char *>(d.data()),
               static_cast<std::streamsize>(count * sizeof(uint64_t)));
    if (!f)
        return false;
    const uint64_t sum = count ? fnv1a(d.data(), d.size() * sizeof(uint64_t)) : 0;
    if (sum != h.densityChecksum)
        return false;
    if (out)
        *out = h;
    if (density)
        *density = std::move(d);
    return true;
}

// The newest readable checkpoint in `dir`, or an empty string. Both slots are
// tried: the newer one can be the truncated one after a power cut.
inline std::string findLatest(const std::string &dir, Header *out = nullptr)
{
    std::string best;
    Header bestHeader;
    bool have = false;
    for (int slot = 0; slot < 2; ++slot) {
        Header h;
        const std::string p = slotPath(dir, slot);
        if (!peek(p, &h))
            continue;
        if (!have || h.sequence > bestHeader.sequence) {
            best = p;
            bestHeader = h;
            have = true;
        }
    }
    if (have && out)
        *out = bestHeader;
    return best;
}

// The newest checkpoint in `dir` that loads *completely*, which is not always
// the newest one on disk: a power cut between the rename and the moment the
// file system flushes the data leaves a file of the right length whose contents
// never arrived. The checksum in load() catches that, and the older slot is
// then still a good checkpoint — half an hour behind, and worth far more than
// starting the analysis again.
inline bool loadBest(const std::string &dir, Header *out,
                     std::vector<uint64_t> *density)
{
    // Newest first, so a healthy newest file is the one that gets used.
    int order[2] = {0, 1};
    Header h0, h1;
    const bool have0 = peek(slotPath(dir, 0), &h0);
    const bool have1 = peek(slotPath(dir, 1), &h1);
    if (have0 && have1 && h1.sequence > h0.sequence) {
        order[0] = 1;
        order[1] = 0;
    } else if (!have0 && have1) {
        order[0] = 1;
        order[1] = 0;
    }
    for (int i = 0; i < 2; ++i) {
        if (load(slotPath(dir, order[i]), out, density))
            return true;
    }
    return false;
}

inline void discard(const std::string &dir)
{
    std::error_code ec;
    for (int slot = 0; slot < 2; ++slot) {
        fs::remove(fs::path(slotPath(dir, slot)), ec);
        fs::remove(fs::path(slotPath(dir, slot) + ".tmp"), ec);
    }
}

} // namespace ckpt
