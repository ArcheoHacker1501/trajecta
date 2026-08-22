#pragma once

// The run manifest: a plain-text record, written next to the results, of what
// was computed and from what.
//
// The point is not logging — the console transcript already does that. The
// point is that a raster on disk, six months later, cannot answer "which DEM
// was this? which cost function? was the barrier on?". The manifest travels
// with the output and answers exactly that, which is what makes a result
// citable and a figure reproducible.
//
// Design notes:
//
//  * Plain text, `key: value`, grouped in sections. Readable in Notepad by the
//    person who will actually need it, and trivially parseable by a script.
//  * Every input file is recorded with its size, its modification time *and* a
//    content hash. Size and time alone cannot tell apart two DEMs of the same
//    extent; the hash can. Hashing is streamed, so a 2 GB DEM costs a couple of
//    seconds against a run measured in hours.
//  * Written at the *end* of the run, because half of what makes it worth
//    having — what was actually produced, how long it took, whether large pages
//    really engaged — is not known before then.
//  * A manifest that cannot be written never takes the run down with it: the
//    results are what matter, and they are already on disk by then.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#ifndef TRAJECTA_VERSION
#define TRAJECTA_VERSION "1.0.1"
#endif

namespace manifest {

namespace fs = std::filesystem;

// Local time, "2026-08-06 14:30:12". Local rather than UTC on purpose: the
// reader is the person who ran it, and they think in their own clock.
// `secondsAgo` walks the clock back, which is how the start of a run is
// recovered from a manifest that is only built once the run is over.
inline std::string nowLocal(double secondsAgo = 0.0)
{
    const auto now = std::chrono::system_clock::now()
                     - std::chrono::duration_cast<std::chrono::system_clock::duration>(
                           std::chrono::duration<double>(secondsAgo));
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

inline std::string fileTimeString(const fs::path &p)
{
    std::error_code ec;
    const auto ft = fs::last_write_time(p, ec);
    if (ec)
        return "unknown";
    // file_clock -> system_clock is only portable via clock_cast in C++20, so
    // this goes the long way round: the difference between the two clocks
    // measured once, applied to this timestamp.
    const auto sctp = std::chrono::system_clock::now()
                      + std::chrono::duration_cast<std::chrono::system_clock::duration>(
                            ft - std::filesystem::file_time_type::clock::now());
    const std::time_t t = std::chrono::system_clock::to_time_t(sctp);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

// FNV-1a over the file's bytes, streamed in 1 MB blocks. Not a cryptographic
// hash and not meant to be: it answers "is this the same file I used then?",
// where the alternative being guarded against is an honest mistake, not an
// adversary.
inline std::string hashFile(const std::string &path)
{
    if (path.empty())
        return "-";
    std::ifstream f(fs::path(path), std::ios::binary);
    if (!f)
        return "unreadable";
    uint64_t h = 1469598103934665603ULL;
    std::vector<char> buf(1u << 20);
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize got = f.gcount();
        for (std::streamsize i = 0; i < got; ++i) {
            h ^= static_cast<unsigned char>(buf[static_cast<size_t>(i)]);
            h *= 1099511628211ULL;
        }
    }
    std::ostringstream os;
    os << "fnv1a:" << std::hex << std::setw(16) << std::setfill('0') << h;
    return os.str();
}

inline std::string humanSize(uintmax_t bytes)
{
    std::ostringstream os;
    if (bytes >= (1ull << 30))
        os << std::fixed << std::setprecision(2) << (double(bytes) / (1ull << 30)) << " GB";
    else if (bytes >= (1ull << 20))
        os << std::fixed << std::setprecision(2) << (double(bytes) / (1ull << 20)) << " MB";
    else if (bytes >= 1024)
        os << std::fixed << std::setprecision(1) << (double(bytes) / 1024.0) << " KB";
    else
        os << bytes << " B";
    os << " (" << bytes << " bytes)";
    return os.str();
}

class Manifest {
public:
    // `mode` is what the user chose: "FETE", "LCPA", ...
    explicit Manifest(const std::string &mode)
        : mode_(mode), startClock_(std::chrono::steady_clock::now())
    {
    }

    // The run's own measurement of how long it took, in seconds. Without it the
    // manifest times itself — and since it is built at the very end of a run,
    // that reads as a fraction of a second no matter how long the analysis
    // actually took. The start time is derived by walking the clock back.
    void setElapsed(double seconds) { elapsed_ = seconds; }

    void section(const std::string &title)
    {
        body_ << "\n[" << title << "]\n";
    }

    void kv(const std::string &key, const std::string &value)
    {
        body_ << "  " << pad(key) << (value.empty() ? std::string("(none)") : value) << "\n";
    }

    void kv(const std::string &key, long long value)
    {
        body_ << "  " << pad(key) << value << "\n";
    }

    void kv(const std::string &key, double value, int decimals = 3)
    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(decimals) << value;
        body_ << "  " << pad(key) << os.str() << "\n";
    }

    void kv(const std::string &key, bool value)
    {
        body_ << "  " << pad(key) << (value ? "yes" : "no") << "\n";
    }

    // An input file, with everything needed to recognise it again later. A path
    // that was left empty (an optional input the user did not supply) is
    // recorded as such rather than skipped: "no cost modifiers" is itself part
    // of the record.
    void inputFile(const std::string &label, const std::string &path)
    {
        if (path.empty()) {
            kv(label, std::string("(not used)"));
            return;
        }
        std::error_code ec;
        const fs::path p(path);
        body_ << "  " << pad(label) << path << "\n";
        if (!fs::exists(p, ec)) {
            body_ << "  " << pad("") << "  ! file not found when the manifest was written\n";
            return;
        }
        body_ << "  " << pad("") << "  size:     " << humanSize(fs::file_size(p, ec)) << "\n";
        body_ << "  " << pad("") << "  modified: " << fileTimeString(p) << "\n";
        body_ << "  " << pad("") << "  hash:     " << hashFile(path) << "\n";
    }

    // An output file. Recorded only if it exists: the engine skips whatever was
    // left unnamed, and a manifest claiming a file that was never written would
    // be worse than one that stays quiet about it.
    void outputFile(const std::string &label, const std::string &path)
    {
        if (path.empty())
            return;
        std::error_code ec;
        const fs::path p(path);
        if (!fs::exists(p, ec))
            return;
        body_ << "  " << pad(label) << path << "\n";
        body_ << "  " << pad("") << "  size:     " << humanSize(fs::file_size(p, ec)) << "\n";
        body_ << "  " << pad("") << "  hash:     " << hashFile(path) << "\n";
        ++outputs_;
    }

    int outputCount() const { return outputs_; }

    // Writes the whole thing. Returns false and leaves `error` set on failure;
    // the caller is expected to warn and carry on.
    bool write(const std::string &path, std::string *error = nullptr) const
    {
        std::error_code ec;
        const fs::path p(path);
        if (p.has_parent_path())
            fs::create_directories(p.parent_path(), ec);

        std::ofstream f(p, std::ios::trunc);
        if (!f) {
            if (error)
                *error = "cannot open " + path;
            return false;
        }

        const double seconds =
            elapsed_ >= 0.0
                ? elapsed_
                : std::chrono::duration<double>(std::chrono::steady_clock::now() - startClock_)
                      .count();

        f << "Trajecta run manifest\n";
        f << "=====================\n";
        f << "  This file records how the results next to it were produced.\n";
        f << "  It is written automatically at the end of every run; the option\n";
        f << "  can be turned off in the interface or answered 'no' in the console.\n";
        f << "\n[run]\n";
        f << "  " << pad("Trajecta version") << TRAJECTA_VERSION << "\n";
        f << "  " << pad("mode") << mode_ << "\n";
        f << "  " << pad("started") << nowLocal(seconds) << "\n";
        f << "  " << pad("finished") << nowLocal() << "\n";
        f << "  " << pad("duration") << formatDuration(seconds) << "\n";
        f << body_.str();
        f.flush();
        if (!f) {
            if (error)
                *error = "write failed (disk full?)";
            return false;
        }
        return true;
    }

private:
    static std::string pad(const std::string &key)
    {
        // Two columns, so the values line up and the file can be skimmed.
        static const size_t width = 26;
        if (key.empty())
            return std::string(width, ' ');
        if (key.size() >= width - 2)
            return key + ": ";
        return key + ":" + std::string(width - key.size() - 1, ' ');
    }

    static std::string formatDuration(double seconds)
    {
        std::ostringstream os;
        const long long total = static_cast<long long>(seconds);
        const long long h = total / 3600, m = (total % 3600) / 60, s = total % 60;
        if (h > 0)
            os << h << " h " << m << " min " << s << " s";
        else if (m > 0)
            os << m << " min " << s << " s";
        else
            os << std::fixed << std::setprecision(1) << seconds << " s";
        os << "  (" << std::fixed << std::setprecision(1) << seconds << " seconds)";
        return os.str();
    }

    std::string mode_;
    std::chrono::steady_clock::time_point startClock_;
    double elapsed_ = -1.0;   // negative: fall back to timing this object
    std::ostringstream body_;
    int outputs_ = 0;
};

// Where the manifest goes: next to the results, named after the main output so
// that re-running an analysis replaces its own manifest and never someone
// else's. Falls back to a generic name when the run produced nothing named.
inline std::string pathFor(const std::string &outputDir, const std::string &primaryOutputName)
{
    const std::string base =
        primaryOutputName.empty() ? std::string("trajecta_run") : primaryOutputName;
    return (fs::path(outputDir) / (base + "_manifest.txt")).string();
}

} // namespace manifest
