#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <filesystem>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <queue>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cctype>
#include <atomic>
#include <random>
#include <omp.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <unistd.h>
#endif
#include "gdal_priv.h"
#include "ogrsf_frmts.h"

namespace fs = std::filesystem; 

// ========== GLOBAL SETTINGS ==========
bool g_verbose_mode = false;  // Global flag for verbose/debug output

// ========== LARGE MEMORY PAGES ==========
//
// The propagation phase is memory-latency bound, and a large part of that cost
// is address translation rather than data movement: with 4 KB pages a 1.2 GB
// working set needs ~300k page-table entries against a TLB that holds ~2k, so
// nearly every access pays a page walk on top of its cache miss. A 2 MB page is
// the same memory described by one entry instead of 512, which brings the TLB's
// reach from ~8 MB to ~4 GB and makes the walk one level shorter.
//
// This changes only how the OS *describes* the memory, never its contents: the
// same bytes live at the same virtual addresses, no arithmetic is affected, and
// the output is bit-for-bit identical either way. The feature is opt-in purely
// because it can *fail* (see below), not because it is risky.
//
// Two things can stop it, and both are handled by falling back silently to
// ordinary pages:
//   * SeLockMemoryPrivilege is not granted to the account. It is assigned to
//     nobody by default, so running elevated does NOT provide it; it has to be
//     granted once through the User Rights Assignment and picked up by a new
//     logon token.
//   * The allocation needs physically contiguous memory. On a machine that has
//     been up for a while it may simply not be available.
bool g_large_pages_requested = false;   // user asked for them
std::atomic<long long> g_large_pages_bytes{0};      // served by 2 MB pages
std::atomic<long long> g_small_pages_bytes{0};      // fell back to 4 KB
std::atomic<int> g_large_page_failures{0};

#ifdef _WIN32
// Turns on the privilege in this process's token. Returns false when the right
// was never assigned to the account, which is the default state of Windows.
bool enable_lock_memory_privilege() {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token))
        return false;
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = false;
    // Wide literal rather than SE_LOCK_MEMORY_NAME: the macro follows the
    // project's narrow-string setting, which would not match the W function.
    if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege",
                              &tp.Privileges[0].Luid)) {
        AdjustTokenPrivileges(token, FALSE, &tp, sizeof(tp), nullptr, nullptr);
        // AdjustTokenPrivileges reports success even when it changed nothing:
        // the real answer is in GetLastError().
        ok = (GetLastError() == ERROR_SUCCESS);
    }
    CloseHandle(token);
    return ok;
}

size_t large_page_size() {
    static const size_t sz = GetLargePageMinimum();   // 2 MB on x64
    return sz;
}
#endif

// Allocator that serves big blocks from large pages when they are available and
// from ordinary pages otherwise. Used through std::vector so the container
// semantics the algorithm relies on (reserve, push_back, clear) are untouched.
//
// Whether a block went through VirtualAlloc is decided from its size alone, so
// deallocate() reaches the same conclusion without having to remember anything.
template <class T>
struct LargePageAllocator {
    using value_type = T;

    LargePageAllocator() noexcept = default;
    template <class U>
    LargePageAllocator(const LargePageAllocator<U>&) noexcept {}

    // Below this, the 64 KB granularity of VirtualAlloc would waste more than
    // the page size saves.
    static constexpr size_t kMinVirtualAlloc = 1u << 20;   // 1 MB

    T* allocate(size_t n) {
        const size_t bytes = n * sizeof(T);
        if (bytes < kMinVirtualAlloc) {
            void* p = ::operator new(bytes);
            g_small_pages_bytes += (long long)bytes;
            return static_cast<T*>(p);
        }
#ifdef _WIN32
        if (g_large_pages_requested) {
            const size_t granularity = large_page_size();
            if (granularity > 0) {
                // Large-page allocations must be a whole number of large pages.
                const size_t rounded =
                    ((bytes + granularity - 1) / granularity) * granularity;
                void* p = VirtualAlloc(nullptr, rounded,
                                       MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                                       PAGE_READWRITE);
                if (p) {
                    g_large_pages_bytes += (long long)rounded;
                    return static_cast<T*>(p);
                }
                ++g_large_page_failures;   // fragmented, or privilege missing
            }
        }
        void* p = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT,
                               PAGE_READWRITE);
        if (!p) throw std::bad_alloc();
        g_small_pages_bytes += (long long)bytes;
        return static_cast<T*>(p);
#else
        void* p = ::operator new(bytes);
        g_small_pages_bytes += (long long)bytes;
        return static_cast<T*>(p);
#endif
    }

    void deallocate(T* p, size_t n) noexcept {
        const size_t bytes = n * sizeof(T);
        if (bytes < kMinVirtualAlloc) {
            ::operator delete(p);
            return;
        }
#ifdef _WIN32
        VirtualFree(p, 0, MEM_RELEASE);
#else
        ::operator delete(p);
#endif
    }

    template <class U>
    bool operator==(const LargePageAllocator<U>&) const noexcept { return true; }
    template <class U>
    bool operator!=(const LargePageAllocator<U>&) const noexcept { return false; }
};

// Shorthand for the per-thread buffers of the propagation phase.
template <class T>
using lp_vector = std::vector<T, LargePageAllocator<T>>;

// ========== LOGGING FUNCTIONS ==========
inline void info_print(const std::string& msg) {
    // Always print important messages
    std::cout << msg;
}

inline void debug_print(const std::string& msg) {
    // Print only if verbose mode is enabled
    if (g_verbose_mode) {
        std::cout << msg;
    }
}

// ========== HELP TEXT ==========
const char* HELP_TEXT = R"(
===============================================================================
                             TRAJECTA v1.0.0
                  A SPATIAL MOVEMENT ANALYSIS SOFTWARE
                       Developed by Stefano Apra'
              Institute for the Study of the Ancient World

                  FETE - From Everywhere to Everywhere

FETE (From-Everywhere-to-Everywhere) is a least-cost modeling tool that
computes anisotropic shortest paths across Digital Elevation Models (DEM). The
model was originally theorized by Devin A. White and Sarah B. Barber (2012).

REFERENCE:
  White, D. A., & Barber, S. B. (2012). Geospatial modeling of pedestrian transportation networks: A case study from precolumbian Oaxaca, Mexico. Journal of Archaeological Science, Volume 39, Issue 8, pp. 2684-2696 (https://doi.org/10.1016/j.jas.2012.04.017).

===============================================================================

MODES:
  1. FETE (From-Everywhere-to-Everywhere) [DEFAULT]
     Computes cost-distance density from EVERY sample point to ALL OTHER sample points.
     Useful for understanding general movement patterns and accessibility.
     Output: Density raster showing accumulated path usage.

  2. LCPA (Least-Cost Path Analysis)
     Computes optimal paths from A SINGLE origin to one or more destinations.
     Useful for finding specific routes with lowest traversal cost.
     Output: Path raster(s) showing optimal route(s).

  4. Sample points
     Writes a sample-points layer from a DEM and stops, without running any
     analysis. Same parameters and same code as the generation built into
     FETE, so the layer can be inspected first and then fed to a FETE run
     unchanged.
     Output: point shapefile.

IMPLEMENTED COST FUNCTIONS:
  1. Modified Tobler's Walking Function (White 2015) [DEFAULT]
  2. Marquez-Perez et al. (2017) Walking Function
  3. Irmischer and Clarke (2017) Walking Function

INPUT REQUIREMENTS:
  - DEM: GeoTIFF file (.tif/.tiff), must be georeferenced
  - Points: vector file with point geometry (.shp, .geojson/.json, .kml,
    .gml/.xml, or .csv with coordinate columns named x/y, lon/lat or
    easting/northing)
    In FETE mode the points can also be GENERATED from the DEM instead of
    imported: choose a density (a spacing in cells, or a target number of
    points) and an arrangement (regular grid or stratified random). The
    layer is written to the output folder as a shapefile and then used as
    the input of the analysis, so the exact input stays inspectable.
  - Cost modifiers (optional): polylines with a 'cost' field (.shp, .geojson,
    .kml, .gml/.xml, or .csv with a WKT geometry column)
  - CRS: DEM and Points MUST have the same coordinate system (user responsibility)
  - Bounds: All points must be contained within DEM extent

PARAMETERS:
  - Neighbours: Connectivity for calculating cumulative cost surface (8, 16, 24, 32, 64)
  - Slope Units: Automatic (based on cost function used)
  - Buffer Radius: Cells around path for density calculation
  - Barrier Threshold: cost multipliers >= threshold are treated as
    impassable barriers (default 1000; 0 = pure soft costs, much slower
    with extreme multipliers)
  - CPU Threads: Parallel processing threads (configured once at startup)
  - Max RAM: Memory limit for raster processing

OUTPUT:
  - Terrain slope raster (.tif)
  - Cost surface raster (.tif)
  - Accumulated path density (.tif)
  - Optimal path raster (.tif)

===============================================================================
)";

// ========== STRUCTURES ==========
struct Config {
    std::string dem_path;
    std::string pts_path;
    std::string out_dir;
    std::string cost_modifiers_path;  // Path to shapefile with cost modifiers
    std::string cost_raster_path;     // Path to raster (.tif) with cost multipliers
};

struct ConfigLCPA {
    std::string dem_path;
    std::string origin_path;
    std::string destinations_path;
    std::string out_dir;
    std::string cost_modifiers_path;  // Path to shapefile with cost modifiers
    std::string cost_raster_path;     // Path to raster (.tif) with cost multipliers
};

struct ValidationResult {
    bool is_valid;
    std::string error_message;
};

struct FETEOutput {
    bool success;
    std::string slope_path;
    std::string cost_path;
    std::string additional_cost_path;  // Additional cost surface from polylines
    std::string total_cost_path;       // Total cost surface (base * additional)
    std::string density_path;
    // 64-bit: per-cell density can exceed 2^32 on large point sets
    uint64_t max_density;
    uint64_t min_density;
    uint64_t avg_density;
    int nonzero_cells;
    int total_cells;
    double time_seconds;
    bool was_cancelled;
};

struct Off { int dr; int dc; };

// ========== UTILITY FUNCTIONS ==========

static inline std::string ltrim_copy(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    return s;
}

static inline std::string join_path(const std::string& dir, const std::string& file) {
    return (fs::path(dir) / file).string();
}

std::string get_cpu_model() {
#ifdef _WIN32
    std::string cpu_model = "Unknown";
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buffer[256];
        DWORD size = sizeof(buffer);
        if (RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr, (LPBYTE)buffer, &size) == ERROR_SUCCESS) {
            cpu_model = std::string(buffer);
        }
        RegCloseKey(hKey);
    }
    return cpu_model;
#else
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                return ltrim_copy(line.substr(pos + 1));
            }
        }
    }
    return "Unknown";
#endif
}

int64_t get_total_ram_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memStatus)) {
        return memStatus.ullTotalPhys / (1024 * 1024);
    }
    return 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return (int64_t)pages * (int64_t)page_size / (1024 * 1024);
    }
    return 0;
#endif
}

// Read a prompt answer from stdin. If the stream is closed (EOF from a
// redirected file, or the GUI dying and closing the pipe), exit immediately:
// a failed std::getline leaves the loop variables empty and every prompt loop
// would otherwise spin forever re-printing its error message.
void safe_getline(std::string& s) {
    if (!std::getline(std::cin, s)) {
        std::cout << "\nERROR: Input stream closed - exiting.\n";
        std::exit(1);
    }
}

// Handles the global "exit" command. A confirmed exit terminates the process
// right here; a declined one CLEARS the input, so the literal string "exit"
// can never leak into the field the prompt was asking for (it used to become
// the DEM path). Always returns false: the prompt simply sees empty input.
bool check_exit_command(std::string& input) {
    if (input == "exit" || input == "EXIT" || input == "Exit") {
        std::cout << "\nAre you sure you want to exit? (yes/no)\n> ";
        std::string confirm;
        safe_getline(confirm);
        if (confirm == "yes" || confirm == "YES" || confirm == "Yes" || confirm == "y" || confirm == "Y") {
            std::cout << "\nGoodbye!\n\n";
            exit(0);
        }
        input.clear();
    }
    return false;
}

void center_text(const std::string& text, int width = 70) {
    int padding = (width - (int)text.length()) / 2;
    if (padding < 0) padding = 0;
    std::cout << std::string(padding, ' ') << text << "\n";
}

void print_help();

bool check_help_command(const std::string& input) {
    if (input == "help" || input == "HELP" || input == "Help") {
        print_help();
        return true;
    }
    return false;
}

void print_green_success(const std::string& success) {
    std::cout << "\033[32m" << success << "\033[0m" << std::flush;
}

// Print question text in neon green (bright green ANSI)
void print_question(const std::string& text) {
    std::cout << "\033[92m" << text << "\033[0m";
}

// Print default setting text in bright yellow
void print_default(const std::string& text) {
    std::cout << "\033[93m" << text << "\033[0m";
}

// Enable ANSI escape codes on Windows 10+
void enable_ansi_colors() {
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
    // UTF-8 output for Unicode progress bars and charts
    SetConsoleOutputCP(CP_UTF8);
#endif
}

// ========== PERFORMANCE MONITORING (verbose mode) ==========

struct PerfSample {
    int iteration;          // completed iterations at this sample
    double wall_time;       // seconds since loop start
    double batch_seconds;   // wall time for this batch
    double iter_per_sec;    // throughput: batch_size / batch_seconds
    double cpu_percent;     // process CPU utilization 0-100%
    double ram_mb;          // process working set in MB
    double workset_mb;      // avg Dijkstra working set per source in MB (cache footprint)
};

// Process RAM usage in MB
static double get_process_ram_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::istringstream iss(line.substr(6));
            double kb; iss >> kb;
            return kb / 1024.0;
        }
    }
    return 0.0;
#endif
}

// Process CPU monitor - tracks delta between calls
struct CpuMonitor {
#ifdef _WIN32
    ULARGE_INTEGER prev_kernel, prev_user;
#endif
    double prev_wall;
    int num_processors;
    bool initialized;

    CpuMonitor() : prev_wall(0), num_processors(1), initialized(false) {
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        num_processors = si.dwNumberOfProcessors;
        prev_kernel.QuadPart = 0;
        prev_user.QuadPart = 0;
#else
        num_processors = std::max(1, (int)std::thread::hardware_concurrency());
#endif
    }

    double sample(double current_wall) {
#ifdef _WIN32
        FILETIME creation, exit_t, kernel, user;
        GetProcessTimes(GetCurrentProcess(), &creation, &exit_t, &kernel, &user);
        ULARGE_INTEGER k, u;
        k.LowPart = kernel.dwLowDateTime; k.HighPart = kernel.dwHighDateTime;
        u.LowPart = user.dwLowDateTime;   u.HighPart = user.dwHighDateTime;

        if (!initialized) {
            prev_kernel = k; prev_user = u; prev_wall = current_wall;
            initialized = true;
            return 0.0;
        }

        double dt_wall = current_wall - prev_wall;
        if (dt_wall < 0.001) return 0.0;

        // Process kernel+user time in seconds (100ns units)
        double dt_cpu = ((double)(k.QuadPart - prev_kernel.QuadPart) +
                         (double)(u.QuadPart - prev_user.QuadPart)) / 10000000.0;

        prev_kernel = k; prev_user = u; prev_wall = current_wall;

        // CPU% = (cpu_time / wall_time) / num_processors * 100
        return std::min(100.0, (dt_cpu / dt_wall) / num_processors * 100.0);
#else
        (void)current_wall;
        return 0.0;
#endif
    }
};

// Render a Unicode block chart to console (verbose diagnostics).
// Uses eighth-block characters for smooth column tops and ANSI colors.
static void print_ascii_chart(const std::string& title,
    const std::vector<double>& values,
    const std::string& x_label_start, const std::string& x_label_end,
    const std::string& y_unit = "",
    const char* bar_color = "\033[36m",
    int chart_width = 60, int chart_height = 12) {

    if (values.size() < 2) return;

    // Vertical eighth-blocks (1/8 .. 8/8)
    static const char* LEVELS[8] = {
        "\xE2\x96\x81", "\xE2\x96\x82", "\xE2\x96\x83", "\xE2\x96\x84",
        "\xE2\x96\x85", "\xE2\x96\x86", "\xE2\x96\x87", "\xE2\x96\x88" };
    const char* FULL = "\xE2\x96\x88";      // full block
    const char* VBAR = "\xE2\x94\x82";      // box drawing vertical
    const char* CORNER = "\xE2\x94\x94";    // box drawing corner
    const char* HBAR = "\xE2\x94\x80";      // box drawing horizontal
    const char* AXIS_COLOR = "\033[90m";
    const char* RESET = "\033[0m";
    const char* BOLD = "\033[1m";

    double v_min = *std::min_element(values.begin(), values.end());
    double v_max = *std::max_element(values.begin(), values.end());
    if (v_max <= v_min) { v_max = v_min + 1.0; }

    // Resample the series onto exactly chart_width columns. With at least one
    // sample per column each column averages its own slice; with fewer samples
    // than columns the intermediate columns interpolate between neighbours.
    // Binning by nearest column instead leaves every column no sample landed
    // on empty, which punches holes into the plot whenever a run produces
    // fewer samples than the chart is wide.
    const int n = (int)values.size();
    std::vector<double> bins(chart_width, 0.0);
    if (n >= chart_width) {
        for (int col = 0; col < chart_width; ++col) {
            int i0 = (int)((double)col * n / chart_width);
            int i1 = (int)((double)(col + 1) * n / chart_width) - 1;
            i0 = std::max(0, std::min(n - 1, i0));
            i1 = std::max(i0, std::min(n - 1, i1));
            double sum = 0.0;
            for (int i = i0; i <= i1; ++i) sum += values[i];
            bins[col] = sum / (i1 - i0 + 1);
        }
    } else {
        for (int col = 0; col < chart_width; ++col) {
            double pos = (double)col * (n - 1) / (chart_width - 1);
            int a = std::min((int)pos, n - 1);
            int b = std::min(a + 1, n - 1);
            bins[col] = values[a] + (values[b] - values[a]) * (pos - a);
        }
    }

    std::cout << "\n  " << BOLD << title << RESET;
    if (!y_unit.empty()) std::cout << " " << AXIS_COLOR << "[" << y_unit << "]" << RESET;
    std::cout << "\n";

    const int lw = 10; // y-axis label width

    // Each row is assembled in full and written once: GDAL/PROJ log to stderr
    // while this runs, and a row emitted in several pieces can be split by an
    // error line, which visually breaks the chart apart.
    for (int row = chart_height - 1; row >= 0; --row) {
        std::string line;
        line.reserve(chart_width * 3 + 64);

        // Y-axis labels at top, middle, bottom
        char buf[24];
        if (row == chart_height - 1 || row == 0 || row == chart_height / 2) {
            double label_val = (row == 0) ? v_min
                : v_min + (v_max - v_min) * (row + 1) / chart_height;
            if (label_val >= 1000.0) snprintf(buf, sizeof(buf), "%*.0f", lw - 1, label_val);
            else if (label_val >= 10.0) snprintf(buf, sizeof(buf), "%*.1f", lw - 1, label_val);
            else snprintf(buf, sizeof(buf), "%*.2f", lw - 1, label_val);
            line += AXIS_COLOR; line += buf; line += " "; line += VBAR; line += RESET;
        } else {
            line += std::string(lw - 1, ' ');
            line += AXIS_COLOR; line += " "; line += VBAR; line += RESET;
        }

        line += bar_color;
        for (int col = 0; col < chart_width; ++col) {
            double h = (bins[col] - v_min) / (v_max - v_min) * chart_height; // filled rows
            double cell = h - row;   // filled fraction of this row's cell
            if (cell >= 1.0) {
                line += FULL;
            } else if (cell > 0.0) {
                int lvl = (int)(cell * 8.0);
                if (lvl < 1) lvl = 1;
                line += LEVELS[lvl - 1];
            } else if (row == 0) {
                line += LEVELS[0];  // baseline: the column sits at the axis minimum
            } else {
                line += ' ';
            }
        }
        line += RESET;
        line += "\n";
        std::cout << line;
    }

    // X-axis
    std::cout << std::string(lw - 1, ' ') << AXIS_COLOR << " " << CORNER;
    for (int i = 0; i < chart_width; ++i) std::cout << HBAR;
    std::cout << RESET << "\n";

    int gap = chart_width - (int)x_label_start.size() - (int)x_label_end.size();
    if (gap < 1) gap = 1;
    std::cout << std::string(lw + 1, ' ') << AXIS_COLOR << x_label_start
              << std::string(gap, ' ') << x_label_end << RESET << "\n";
}

// Save performance data to CSV
static void save_perf_csv(const std::string& path, const std::vector<PerfSample>& samples) {
    std::ofstream f(path);
    if (!f.is_open()) return;
    f << "iteration,wall_time_s,batch_time_s,iter_per_sec,cpu_percent,ram_mb,workset_mb\n";
    for (auto& s : samples) {
        f << s.iteration << "," << std::fixed << std::setprecision(2)
          << s.wall_time << "," << std::setprecision(3) << s.batch_seconds << ","
          << std::setprecision(1) << s.iter_per_sec << ","
          << std::setprecision(1) << s.cpu_percent << ","
          << std::setprecision(1) << s.ram_mb << ","
          << std::setprecision(2) << s.workset_mb << "\n";
    }
    f.close();
}

// Unicode progress bar with percentage, counter, throughput and ETA.
// Pass elapsed_sec < 0 when no timing information is available.
void print_progress(int current, int total, double elapsed_sec = -1.0, int bar_width = 40) {
    if (total <= 0) return;
    double frac = (double)current / (double)total;
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;

    // Horizontal eighth-blocks (1/8 .. 7/8) for smooth bar growth
    static const char* PARTS[7] = {
        "\xE2\x96\x8F", "\xE2\x96\x8E", "\xE2\x96\x8D", "\xE2\x96\x8C",
        "\xE2\x96\x8B", "\xE2\x96\x8A", "\xE2\x96\x89" };
    const char* FULL = "\xE2\x96\x88";   // full block
    const char* EMPTY = "\xE2\x96\x91";  // light shade

    double cells = frac * bar_width;
    int full = (int)cells;
    int part = (int)((cells - (double)full) * 8.0);

    std::ostringstream out;
    char pct[16];
    snprintf(pct, sizeof(pct), "%5.1f%%", frac * 100.0);
    out << "\r\033[K\033[1m" << pct << "\033[0m \033[32m";
    for (int i = 0; i < full; ++i) out << FULL;
    int drawn = full;
    if (part > 0 && drawn < bar_width) { out << PARTS[part - 1]; drawn++; }
    out << "\033[90m";
    for (int i = drawn; i < bar_width; ++i) out << EMPTY;
    out << "\033[0m " << current << "/" << total;

    if (elapsed_sec > 0.0 && current > 0 && current < total) {
        double rate = current / elapsed_sec;
        double eta = (total - current) / std::max(rate, 1e-9);
        char tail[80];
        if (eta >= 3600.0) {
            snprintf(tail, sizeof(tail), " \xC2\xB7 %.1f it/s \xC2\xB7 ETA %d:%02d:%02d",
                rate, (int)(eta / 3600), ((int)eta % 3600) / 60, (int)eta % 60);
        } else {
            snprintf(tail, sizeof(tail), " \xC2\xB7 %.1f it/s \xC2\xB7 ETA %02d:%02d",
                rate, (int)(eta / 60), (int)eta % 60);
        }
        out << "\033[90m" << tail << "\033[0m";
    }

    std::cout << out.str();
    std::cout.flush();
}

// Where the last-used-paths config files live. Under the user profile, so
// console runs don't scatter fete_config.txt copies across working
// directories; falls back to the current directory if no profile is set.
std::string config_file_path(const char* name) {
#ifdef _WIN32
    const char* base = std::getenv("APPDATA");
    if (!base || !*base) return name;
    fs::path dir = fs::path(base) / "Trajecta";
#else
    const char* base = std::getenv("HOME");
    if (!base || !*base) return name;
    fs::path dir = fs::path(base) / ".config" / "trajecta";
#endif
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return name;
    return (dir / name).string();
}

void save_config(const Config& cfg) {
    std::ofstream file(config_file_path("fete_config.txt"));
    file << cfg.dem_path << "\n";
    file << cfg.pts_path << "\n";
    file << cfg.out_dir << "\n";
    file << cfg.cost_modifiers_path << "\n";
    file << cfg.cost_raster_path << "\n";
    file.close();
}

void save_config_lcpa(const ConfigLCPA& cfg) {
    std::ofstream file(config_file_path("lcpa_config.txt"));
    file << cfg.dem_path << "\n";
    file << cfg.origin_path << "\n";
    file << cfg.destinations_path << "\n";
    file << cfg.out_dir << "\n";
    file << cfg.cost_modifiers_path << "\n";
    file << cfg.cost_raster_path << "\n";
    file.close();
}

Config load_config() {
    Config cfg = { "", "", "", "", "" };
    std::ifstream file(config_file_path("fete_config.txt"));
    if (file.is_open()) {
        std::getline(file, cfg.dem_path);
        std::getline(file, cfg.pts_path);
        std::getline(file, cfg.out_dir);
        std::getline(file, cfg.cost_modifiers_path);
        std::getline(file, cfg.cost_raster_path);
        file.close();
    }
    return cfg;
}

ConfigLCPA load_config_lcpa() {
    ConfigLCPA cfg = { "", "", "", "", "", "" };
    std::ifstream file(config_file_path("lcpa_config.txt"));
    if (file.is_open()) {
        std::getline(file, cfg.dem_path);
        std::getline(file, cfg.origin_path);
        std::getline(file, cfg.destinations_path);
        std::getline(file, cfg.out_dir);
        std::getline(file, cfg.cost_modifiers_path);
        std::getline(file, cfg.cost_raster_path);
        file.close();
    }
    return cfg;
}

static inline std::string to_lower_copy(std::string s) {
    for (auto& c : s) c = (char)std::tolower(static_cast<unsigned char>(c));
    return s;
}

std::string get_file_extension(const std::string& path) {
    return fs::path(path).extension().string();
}

bool file_exists(const std::string& path) {
    return fs::exists(path);
}

// ========== VECTOR FORMAT SUPPORT ==========

std::string supported_vector_formats() {
    return ".shp, .geojson/.json, .kml, .gml/.xml, .csv";
}

bool is_supported_vector_format(const std::string& path) {
    std::string ext = to_lower_copy(get_file_extension(path));
    return ext == ".shp" || ext == ".csv" || ext == ".xml" || ext == ".gml" ||
           ext == ".kml" || ext == ".geojson" || ext == ".json";
}

// Open any supported vector format through OGR. For CSV, pass open options so
// point coordinates (x/y, lon/lat, easting/northing columns) or WKT geometry
// columns are recognized.
GDALDataset* open_vector_dataset(const std::string& path) {
    std::string ext = to_lower_copy(get_file_extension(path));
    if (ext == ".csv") {
        const char* csv_options[] = {
            "X_POSSIBLE_NAMES=x,lon,longitude,easting,x_coord,xcoord",
            "Y_POSSIBLE_NAMES=y,lat,latitude,northing,y_coord,ycoord",
            "GEOM_POSSIBLE_NAMES=geometry,geom,wkt,the_geom",
            "KEEP_GEOM_COLUMNS=NO",
            nullptr };
        return (GDALDataset*)GDALOpenEx(path.c_str(), GDAL_OF_VECTOR, nullptr, csv_options, nullptr);
    }
    return (GDALDataset*)GDALOpenEx(path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
}

// ========== RASTER OUTPUT HELPER ==========

// Create a single-band GeoTIFF and write data into it, with error checking.
// Outputs are DEFLATE-compressed and tiled (readable by any GIS; typically
// 3-10x smaller than the uncompressed rasters written before). Pass `nodata`
// to declare a NoData value on the band so impassable cells don't masquerade
// as legitimate zeros in GIS software.
bool write_gtiff_raster(const std::string& path, int ncols, int nrows,
    const double gt[6], const char* wkt, void* data, GDALDataType dtype,
    const double* nodata = nullptr) {
    GDALDriver* drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!drv) {
        std::cout << "ERROR: GTiff driver not available in this GDAL build\n";
        return false;
    }
    char** opts = nullptr;
    opts = CSLSetNameValue(opts, "COMPRESS", "DEFLATE");
    opts = CSLSetNameValue(opts, "PREDICTOR",
        (dtype == GDT_Float32 || dtype == GDT_Float64) ? "3" : "2");
    opts = CSLSetNameValue(opts, "TILED", "YES");
    opts = CSLSetNameValue(opts, "BIGTIFF", "IF_SAFER");
    GDALDataset* ds = drv->Create(path.c_str(), ncols, nrows, 1, dtype, opts);
    CSLDestroy(opts);
    if (!ds) {
        std::cout << "ERROR: Cannot create output raster: " << path << "\n";
        std::cout << "       Check that the output directory exists and is writable.\n";
        return false;
    }
    double gt_copy[6];
    for (int i = 0; i < 6; ++i) gt_copy[i] = gt[i];
    ds->SetGeoTransform(gt_copy);
    if (wkt) ds->SetProjection(wkt);
    if (nodata) ds->GetRasterBand(1)->SetNoDataValue(*nodata);
    CPLErr err = ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows,
        data, ncols, nrows, dtype, 0, 0);
    GDALClose(ds);
    if (err != CE_None) {
        std::cout << "ERROR: Failed writing raster data: " << path << "\n";
        return false;
    }
    return true;
}

// ========== SEGMENT CLIPPING (Liang-Barsky) ==========

// Clip a world-space segment to a bounding box. Returns false if the segment
// lies entirely outside. Keeps Bresenham spans bounded even when polyline
// vertices are far outside the raster (e.g. CRS mismatch).
static bool clip_segment_to_bounds(double& x1, double& y1, double& x2, double& y2,
    double xmin, double ymin, double xmax, double ymax) {
    double t0 = 0.0, t1 = 1.0;
    const double dx = x2 - x1, dy = y2 - y1;
    const double p[4] = { -dx, dx, -dy, dy };
    const double q[4] = { x1 - xmin, xmax - x1, y1 - ymin, ymax - y1 };
    for (int i = 0; i < 4; ++i) {
        if (p[i] == 0.0) {
            if (q[i] < 0.0) return false;
        } else {
            double t = q[i] / p[i];
            if (p[i] < 0.0) {
                if (t > t1) return false;
                if (t > t0) t0 = t;
            } else {
                if (t < t0) return false;
                if (t < t1) t1 = t;
            }
        }
    }
    const double nx1 = x1 + t0 * dx, ny1 = y1 + t0 * dy;
    const double nx2 = x1 + t1 * dx, ny2 = y1 + t1 * dy;
    x1 = nx1; y1 = ny1; x2 = nx2; y2 = ny2;
    return true;
}

// ========== VALIDATION FUNCTIONS ==========

ValidationResult validate_dem(const std::string& dem_path) {
    if (!file_exists(dem_path)) {
        return { false, "ERROR: DEM file not found: " + dem_path };
    }
    std::string ext = to_lower_copy(get_file_extension(dem_path));
    if (ext != ".tif" && ext != ".tiff") {
        return { false, "ERROR: DEM must be GeoTIFF (.tif/.tiff), found: " + ext };
    }
    GDALDataset* ds = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
    if (!ds) {
        return { false, "ERROR: Cannot open DEM file with GDAL" };
    }
    double gt[6];
    if (ds->GetGeoTransform(gt) != CE_None || gt[1] == 0.0 || gt[5] == 0.0) {
        GDALClose(ds);
        return { false, "ERROR: DEM is not georeferenced (no valid geotransform)" };
    }
    // Every coordinate<->pixel conversion in the engine assumes a north-up
    // grid; with rotation terms the points would be silently mapped wrong.
    if (std::abs(gt[2]) > 1e-12 || std::abs(gt[4]) > 1e-12) {
        GDALClose(ds);
        return { false, "ERROR: DEM has a rotated geotransform, which is not supported.\n"
                        "       Resample it north-up first (e.g. gdalwarp)." };
    }
    if (ds->GetProjectionRef() == nullptr || std::string(ds->GetProjectionRef()).empty()) {
        GDALClose(ds);
        return { false, "ERROR: DEM has no coordinate reference system (CRS)" };
    }
    GDALClose(ds);
    return { true, "" };
}

// Validation only ever looks at geometry, so tell OGR to skip every attribute
// field. On a shapefile this keeps the .dbf closed; on a 500k-feature layer
// that alone removes most of the per-feature cost.
static void ignore_attribute_fields(OGRLayer* layer) {
    OGRFeatureDefn* defn = layer->GetLayerDefn();
    if (!defn || defn->GetFieldCount() == 0) return;
    std::vector<const char*> ignored;
    ignored.reserve((size_t)defn->GetFieldCount() + 1);
    for (int i = 0; i < defn->GetFieldCount(); ++i) {
        ignored.push_back(defn->GetFieldDefn(i)->GetNameRef());
    }
    ignored.push_back(nullptr);
    layer->SetIgnoredFields(ignored.data());
}

// Points file: format, readability, "at least 2 point geometries" and "all
// points inside the DEM" in one place.
//
// Both answers usually come straight out of the layer header: a typed point
// layer knows its feature count, and most drivers (shapefile, GeoPackage,
// FlatGeobuf, ...) know their bounding box without reading a single feature.
// When that box sits inside the DEM no point can be outside it, so nothing has
// to be scanned at all. Only layers that cannot answer from metadata (CSV,
// GeoJSON) or whose extent crosses the DEM edge fall back to a feature scan --
// and that scan now runs once, not twice.
ValidationResult validate_points_against_dem(const std::string& dem_path,
                                             const std::string& pts_path) {
    if (!file_exists(pts_path)) {
        return { false, "ERROR: Points file not found: " + pts_path };
    }
    if (!is_supported_vector_format(pts_path)) {
        return { false, "ERROR: Points must be one of " + supported_vector_formats() +
                        ", found: " + get_file_extension(pts_path) };
    }
    GDALDataset* pts_ds = open_vector_dataset(pts_path);
    if (!pts_ds) {
        return { false, "ERROR: Cannot open points file with GDAL/OGR" };
    }
    OGRLayer* layer = pts_ds->GetLayer(0);
    if (!layer) {
        GDALClose(pts_ds);
        return { false, "ERROR: Points file has no readable layer" };
    }

    GDALDataset* dem_ds = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
    if (!dem_ds) {
        GDALClose(pts_ds);
        return { false, "ERROR: Cannot open files for bounds validation" };
    }
    const int ncols = dem_ds->GetRasterXSize();
    const int nrows = dem_ds->GetRasterYSize();
    double gt[6];
    dem_ds->GetGeoTransform(gt);
    GDALClose(dem_ds);
    const double dem_xmin = gt[0];
    const double dem_xmax = gt[0] + ncols * gt[1];
    const double dem_ymax = gt[3];
    const double dem_ymin = gt[3] + nrows * gt[5];

    const std::string count_hint =
        "\n       (for .csv, coordinate columns must be named x/y, lon/lat or easting/northing)";

    // ---- Header fast path ----
    bool count_known = false;
    if (wkbFlatten(layer->GetGeomType()) == wkbPoint) {
        GIntBig fc = layer->GetFeatureCount(FALSE);  // FALSE: only if free
        if (fc >= 0) {
            if (fc < 2) {
                GDALClose(pts_ds);
                return { false, "ERROR: Points file must contain at least 2 point "
                                "geometries, found: " + std::to_string(fc) + count_hint };
            }
            count_known = true;
        }
    }
    OGREnvelope env;
    bool bounds_known = false;
    if (layer->GetExtent(&env, FALSE) == OGRERR_NONE) {   // FALSE: only if free
        if (env.MinX >= dem_xmin && env.MaxX <= dem_xmax &&
            env.MinY >= dem_ymin && env.MaxY <= dem_ymax) {
            bounds_known = true;   // whole layer inside the DEM
        }
    }
    if (count_known && bounds_known) {
        GDALClose(pts_ds);
        return { true, "" };
    }

    // ---- Single scan: count point geometries and test bounds together ----
    ignore_attribute_fields(layer);
    long long point_count = 0;
    long long outside_points = 0;
    layer->ResetReading();
    OGRFeature* feat = nullptr;
    auto check = [&](const OGRPoint* p) {
        ++point_count;
        const double px = p->getX(), py = p->getY();
        if (px < dem_xmin || px > dem_xmax || py < dem_ymin || py > dem_ymax) {
            ++outside_points;
        }
    };
    while ((feat = layer->GetNextFeature()) != nullptr) {
        // Count actual point geometries (not just features): CSV rows without
        // recognized coordinate columns produce features with null geometry
        const OGRGeometry* geom = feat->GetGeometryRef();
        if (geom) {
            OGRwkbGeometryType gtype = wkbFlatten(geom->getGeometryType());
            if (gtype == wkbPoint) {
                check(geom->toPoint());
            } else if (gtype == wkbMultiPoint) {
                const OGRMultiPoint* mp = geom->toMultiPoint();
                for (int gi = 0; gi < mp->getNumGeometries(); gi++) {
                    check(mp->getGeometryRef(gi)->toPoint());
                }
            }
        }
        OGRFeature::DestroyFeature(feat);
    }
    GDALClose(pts_ds);

    if (!count_known && point_count < 2) {
        return { false, "ERROR: Points file must contain at least 2 point geometries, found: " +
                        std::to_string(point_count) + count_hint };
    }
    if (outside_points > 0) {
        return { false, "ERROR: " + std::to_string(outside_points) + " of " +
                        std::to_string(point_count) + " points are outside DEM extent" };
    }
    return { true, "" };
}

ValidationResult validate_all_inputs(const std::string& dem_path, const std::string& pts_path) {
    ValidationResult res = validate_dem(dem_path);
    if (!res.is_valid) return res;
    return validate_points_against_dem(dem_path, pts_path);
}

void print_help() {
    std::cout << HELP_TEXT;
}

// Output file names are asked BEFORE the computation runs; a character
// Windows forbids would otherwise surface only when the raster is written —
// for the density raster, after the whole propagation has finished.
bool valid_output_filename(const std::string& name) {
    return !name.empty() && name.find_first_of("\\/:*?\"<>|") == std::string::npos;
}

void print_filename_error() {
    std::cout << "ERROR: Filename cannot contain \\ / : * ? \" < > | characters\n";
}

// ========== SAMPLE POINT GENERATION ==========
//
// Optional alternative to importing a sample-points file: derive the points
// from the DEM itself. Everything here happens in main(), BEFORE the inputs
// are validated; the generated layer is written to disk and then validated and
// consumed exactly like a user-supplied file, so the analysis keeps a single
// code path and the user can always inspect the exact input that was used.
//
// When the points come from a file none of this is reached: no prompt, no DEM
// read, no allocation. run_fete() does not know this feature exists.

struct PointGenOptions {
    bool by_target_count = true;    // false: the spacing below is used as is
    int spacing = 10;               // one point every N cells, on both axes
    long long target_count = 5000;  // desired point count (by_target_count)
    bool random = false;            // false: regular grid, true: stratified random
    unsigned int seed = 1;
    int edge_buffer = 0;            // cells kept clear along each DEM border
    std::string layer_name;         // without extension; empty = self-documenting default
};

struct PointGenResult {
    bool ok = false;
    std::string path;      // the shapefile that was written
    long long count = 0;   // points actually created
    int spacing = 1;       // spacing used (derived, when a target count was given)
    std::string error;
};

// Largest dimension of the decimated read used to estimate how many DEM cells
// can host a point. A full band read would cost as much as the analysis' own;
// a <= 1000 px overview lands within a few percent in milliseconds. Trajecta
// Studio applies the very same rule for its live preview, so the spacing it
// shows is the spacing the engine derives.
static const int kGenPreviewDim = 1000;

double dem_passable_fraction(GDALRasterBand* band, int ncols, int nrows,
                             int has_nodata, float nodata_f) {
    const double scale = std::min(1.0, (double)kGenPreviewDim / std::max(ncols, nrows));
    const int dw = std::max(1, (int)std::lround(ncols * scale));
    const int dh = std::max(1, (int)std::lround(nrows * scale));
    std::vector<float> ov((size_t)dw * (size_t)dh);
    if (band->RasterIO(GF_Read, 0, 0, ncols, nrows, ov.data(), dw, dh,
                       GDT_Float32, 0, 0) != CE_None) {
        return -1.0;
    }
    long long good = 0;
    for (float v : ov) {
        if (!std::isnan(v) && v < 9999.0f && !(has_nodata && v == nodata_f)) ++good;
    }
    return (double)good / ((double)dw * (double)dh);
}

PointGenResult generate_sample_points(const std::string& dem_path,
                                      const std::string& out_dir,
                                      const PointGenOptions& opt) {
    PointGenResult res;
    res.spacing = std::max(1, opt.spacing);

    GDALDataset* ds = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
    if (!ds) {
        res.error = "ERROR: Cannot open the DEM to generate the sample points";
        return res;
    }
    GDALRasterBand* band = ds->GetRasterBand(1);
    const int ncols = ds->GetRasterXSize();
    const int nrows = ds->GetRasterYSize();
    double gt[6];
    if (!band || ncols <= 0 || nrows <= 0 || ds->GetGeoTransform(gt) != CE_None) {
        GDALClose(ds);
        res.error = "ERROR: The DEM has no usable raster band or geotransform";
        return res;
    }
    const char* wkt_raw = ds->GetProjectionRef();
    const std::string wkt = wkt_raw ? wkt_raw : "";
    int has_nodata = 0;
    const double nodata_val = band->GetNoDataValue(&has_nodata);
    const float nodata_f = (float)nodata_val;
    // Same passability rule as run_fete(): NaN, elevations >= 9999 and the
    // band NoData value are all cells a path can never cross.
    const auto passable = [&](float v) -> bool {
        return !std::isnan(v) && v < 9999.0f && !(has_nodata && v == nodata_f);
    };

    // Points within a few cells of the border have a truncated neighbourhood,
    // and every route leaving the raster is invisible to the analysis, so the
    // density near the edge is biased inwards. Keeping the sources off the
    // border is the cheap half of the remedy.
    const int buf = std::max(0, opt.edge_buffer);
    const int row_lo = buf, row_hi = nrows - buf;   // half-open
    const int col_lo = buf, col_hi = ncols - buf;
    if (row_hi - row_lo < 1 || col_hi - col_lo < 1) {
        GDALClose(ds);
        res.error = "ERROR: An edge buffer of " + std::to_string(buf) +
                    " cell(s) leaves no room on a " + std::to_string(ncols) +
                    "x" + std::to_string(nrows) + " DEM";
        return res;
    }

    if (opt.by_target_count) {
        if (opt.target_count < 2) {
            GDALClose(ds);
            res.error = "ERROR: The target number of points must be at least 2";
            return res;
        }
        const double frac = dem_passable_fraction(band, ncols, nrows, has_nodata, nodata_f);
        if (frac < 0.0) {
            GDALClose(ds);
            res.error = "ERROR: Failed reading the DEM while sizing the point grid";
            return res;
        }
        if (frac <= 0.0) {
            GDALClose(ds);
            res.error = "ERROR: The DEM has no valid cells to place sample points on";
            return res;
        }
        // Only the area points may actually land in counts towards the target.
        const double valid_cells =
            frac * (double)(col_hi - col_lo) * (double)(row_hi - row_lo);
        res.spacing = std::max(1, (int)std::llround(
            std::sqrt(valid_cells / (double)opt.target_count)));
    }
    const int step = res.spacing;

    std::string name = opt.layer_name;
    if (name.empty()) {
        name = "sample_points_" + std::string(opt.random ? "random" : "grid")
             + "_s" + std::to_string(step);
        if (buf > 0) name += "_b" + std::to_string(buf);
    }
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    const std::string path = join_path(out_dir, name + ".shp");

    GDALDriver* drv = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    if (!drv) {
        GDALClose(ds);
        res.error = "ERROR: The ESRI Shapefile driver is not available in this GDAL build";
        return res;
    }
    if (file_exists(path)) drv->Delete(path.c_str());
    GDALDataset* out = drv->Create(path.c_str(), 0, 0, 0, GDT_Unknown, nullptr);
    if (!out) {
        GDALClose(ds);
        res.error = "ERROR: Cannot create the generated points file: " + path +
                    "\n       (check that the output folder is writable and the "
                    "file is not open in a GIS)";
        return res;
    }
    // The generated points inherit the DEM's CRS, which makes a mismatch
    // between DEM and points structurally impossible in this mode.
    OGRSpatialReference srs;
    OGRSpatialReference* srs_ptr = nullptr;
    if (!wkt.empty() && srs.importFromWkt(wkt.c_str()) == OGRERR_NONE) srs_ptr = &srs;
    OGRLayer* layer = out->CreateLayer(name.c_str(), srs_ptr, wkbPoint, nullptr);
    if (!layer) {
        GDALClose(out);
        GDALClose(ds);
        res.error = "ERROR: Cannot create the point layer inside: " + path;
        return res;
    }
    OGRFieldDefn id_field("id", OFTInteger);
    layer->CreateField(&id_field);
    OGRFeatureDefn* defn = layer->GetLayerDefn();

    long long id = 0;
    auto emit_point = [&](int r, int c) {
        OGRFeature* f = OGRFeature::CreateFeature(defn);
        f->SetField(0, (int)id);
        // Cell centre, so the point maps back to exactly this cell when
        // run_fete converts it to a pixel again.
        OGRPoint pt(gt[0] + (c + 0.5) * gt[1], gt[3] + (r + 0.5) * gt[5]);
        f->SetGeometry(&pt);
        if (layer->CreateFeature(f) == OGRERR_NONE) ++id;
        OGRFeature::DestroyFeature(f);
    };

    bool io_ok = true;
    if (!opt.random) {
        // Regular grid: every step-th row and column, offset by half a step so
        // the pattern sits inside the raster instead of hugging its edges.
        // Only the rows that carry points are read.
        std::vector<float> row((size_t)ncols);
        for (int r = row_lo + step / 2; r < row_hi; r += step) {
            if (band->RasterIO(GF_Read, 0, r, ncols, 1, row.data(), ncols, 1,
                               GDT_Float32, 0, 0) != CE_None) {
                io_ok = false;
                break;
            }
            for (int c = col_lo + step / 2; c < col_hi; c += step) {
                if (passable(row[c])) emit_point(r, c);
            }
        }
    } else {
        // Stratified random: one point per step x step block. Unlike uniform
        // random sampling this reproduces the requested density exactly while
        // still breaking the regularity of the grid.
        std::mt19937 rng(opt.seed);
        const int band_rows = std::min(step, row_hi - row_lo);
        std::vector<float> block((size_t)ncols * (size_t)band_rows);
        for (int r0 = row_lo; r0 < row_hi && io_ok; r0 += step) {
            const int rows_here = std::min(step, row_hi - r0);
            if (band->RasterIO(GF_Read, 0, r0, ncols, rows_here, block.data(),
                               ncols, rows_here, GDT_Float32, 0, 0) != CE_None) {
                io_ok = false;
                break;
            }
            for (int c0 = col_lo; c0 < col_hi; c0 += step) {
                const int cols_here = std::min(step, col_hi - c0);
                std::uniform_int_distribution<int> pick_r(0, rows_here - 1);
                std::uniform_int_distribution<int> pick_c(0, cols_here - 1);
                int hit_r = -1, hit_c = -1;
                // A few random draws, then an exhaustive sweep: a block that is
                // mostly NoData still contributes if it holds any valid cell.
                const int tries = std::min(rows_here * cols_here, 12);
                for (int t = 0; t < tries; ++t) {
                    const int rr = pick_r(rng), cc = pick_c(rng);
                    if (passable(block[(size_t)rr * ncols + (c0 + cc)])) {
                        hit_r = rr; hit_c = cc;
                        break;
                    }
                }
                for (int rr = 0; rr < rows_here && hit_r < 0; ++rr) {
                    for (int cc = 0; cc < cols_here; ++cc) {
                        if (passable(block[(size_t)rr * ncols + (c0 + cc)])) {
                            hit_r = rr; hit_c = cc;
                            break;
                        }
                    }
                }
                if (hit_r >= 0) emit_point(r0 + hit_r, c0 + hit_c);
            }
        }
    }
    GDALClose(out);
    GDALClose(ds);

    if (!io_ok) {
        res.error = "ERROR: Failed reading the DEM while generating the sample points";
        return res;
    }
    res.count = id;
    res.path = path;
    res.ok = true;
    return res;
}

// Asks the generation parameters. Shared by FETE (where the layer feeds the
// analysis straight away) and by the standalone "sample points" mode (where it
// is written and nothing else happens), so the two ask literally the same
// questions in the same order -- which is also what lets Trajecta Studio drive
// both with a single set of prompt rules.
//
// Returns false when the user asked to exit.
bool ask_point_generation_options(PointGenOptions& opt) {
    while (true) {
        print_question("\nHow do you want to express the point density?\n");
        std::cout << "  1) Point spacing: one point every N cells\n";
        std::cout << "  2) Target number of points "; print_default("[DEFAULT]"); std::cout << "\n";
        std::cout << "  "; print_default("Leave blank for default (2)"); std::cout << "\n";
        std::cout << "> ";
        std::string input;
        safe_getline(input);
        if (check_exit_command(input)) return false;
        if (check_help_command(input)) continue;
        if (input == "1") { opt.by_target_count = false; break; }
        if (input.empty() || input == "2") { opt.by_target_count = true; break; }
        std::cout << "ERROR: Please enter 1 or 2\n";
    }

    if (!opt.by_target_count) {
        while (true) {
            print_question("\nEnter point spacing in cells:\n");
            std::cout << "  1 = one point per DEM cell (only sane on very small rasters),\n";
            std::cout << "  N = one point every N rows AND every N columns.\n";
            std::cout << "  "; print_default("Leave blank for default (10)"); std::cout << "\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return false;
            if (check_help_command(input)) continue;
            if (input.empty()) { opt.spacing = 10; break; }
            try {
                int v = std::stoi(input);
                if (v < 1) { std::cout << "ERROR: Spacing must be at least 1\n"; continue; }
                opt.spacing = v;
                break;
            }
            catch (...) {
                std::cout << "ERROR: Please enter a whole number of cells\n";
            }
        }
    }
    else {
        while (true) {
            print_question("\nEnter the target number of points:\n");
            std::cout << "  The spacing is derived from it and reported below;\n";
            std::cout << "  the actual count is close to, not exactly, the target.\n";
            std::cout << "  "; print_default("Leave blank for default (5000)"); std::cout << "\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return false;
            if (check_help_command(input)) continue;
            if (input.empty()) { opt.target_count = 5000; break; }
            try {
                long long v = std::stoll(input);
                if (v < 2) { std::cout << "ERROR: At least 2 points are needed\n"; continue; }
                opt.target_count = v;
                break;
            }
            catch (...) {
                std::cout << "ERROR: Please enter a whole number of points\n";
            }
        }
    }

    while (true) {
        print_question("\nSelect point arrangement:\n");
        std::cout << "  1) Regular grid "; print_default("[DEFAULT]"); std::cout << "\n";
        std::cout << "  2) Stratified random (one random cell per block, same density)\n";
        std::cout << "  "; print_default("Leave blank for default (1)"); std::cout << "\n";
        std::cout << "> ";
        std::string input;
        safe_getline(input);
        if (check_exit_command(input)) return false;
        if (check_help_command(input)) continue;
        if (input.empty() || input == "1") { opt.random = false; break; }
        if (input == "2") { opt.random = true; break; }
        std::cout << "ERROR: Please enter 1 or 2\n";
    }

    if (opt.random) {
        while (true) {
            print_question("\nEnter the random seed:\n");
            std::cout << "  The same seed always produces the same points, so a run\n";
            std::cout << "  can be reproduced exactly.\n";
            std::cout << "  "; print_default("Leave blank for default (1)"); std::cout << "\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return false;
            if (check_help_command(input)) continue;
            if (input.empty()) { opt.seed = 1; break; }
            try {
                long long v = std::stoll(input);
                if (v < 0) { std::cout << "ERROR: The seed cannot be negative\n"; continue; }
                opt.seed = (unsigned int)v;
                break;
            }
            catch (...) {
                std::cout << "ERROR: Please enter a whole number\n";
            }
        }
    }

    while (true) {
        print_question("\nEnter edge buffer in cells:\n");
        std::cout << "  Points are kept at least this far from the DEM border.\n";
        std::cout << "  Near the edge a source has fewer neighbours to move to, and\n";
        std::cout << "  any route that would leave the raster is invisible, so the\n";
        std::cout << "  density there is biased inwards.\n";
        std::cout << "  "; print_default("Leave blank for default (0, no buffer)"); std::cout << "\n";
        std::cout << "> ";
        std::string input;
        safe_getline(input);
        if (check_exit_command(input)) return false;
        if (check_help_command(input)) continue;
        if (input.empty()) { opt.edge_buffer = 0; break; }
        try {
            int v = std::stoi(input);
            if (v < 0) { std::cout << "ERROR: The buffer cannot be negative\n"; continue; }
            opt.edge_buffer = v;
            break;
        }
        catch (...) {
            std::cout << "ERROR: Please enter a whole number of cells\n";
        }
    }

    while (true) {
        print_question("\nEnter filename for the generated points layer (without extension):\n");
        std::cout << "  A shapefile with this name is written into the output folder\n";
        std::cout << "  and then used as the input of the analysis.\n";
        std::cout << "  "; print_default("Leave blank for an automatic name"); std::cout << "\n";
        std::cout << "> ";
        std::string input;
        safe_getline(input);
        if (check_exit_command(input)) return false;
        if (check_help_command(input)) continue;
        if (input.empty()) { opt.layer_name.clear(); break; }
        if (!valid_output_filename(input)) { print_filename_error(); continue; }
        if (input.length() >= 4 && to_lower_copy(input.substr(input.length() - 4)) == ".shp") {
            input = input.substr(0, input.length() - 4);
        }
        opt.layer_name = input;
        break;
    }
    return true;
}

// Mode 4: write a sample-points layer and stop. Same DEM prompt, same
// generation questions and the same generate_sample_points() call as a FETE run
// in generate mode, so the layer produced here is bit-for-bit the layer a FETE
// run with those parameters would consume. Trajecta Studio uses this to let the
// user look at the points in the Viewer before committing to an analysis.
int run_points_mode() {
    std::cout << std::string(70, '=') << "\n";
    center_text("SAMPLE POINTS MODE");
    center_text("Generate a point layer from a DEM");
    std::cout << std::string(70, '=') << "\n\n";

    GDALAllRegister();
    OGRRegisterAll();

    Config saved_config = load_config();
    std::string dem_path = saved_config.dem_path;
    std::string out_dir = saved_config.out_dir;

    while (true) {
        while (true) {
            print_question("\nEnter path to DEM file (.tif):\n");
            if (!dem_path.empty()) {
                std::cout << "  "; print_default("Default: " + dem_path); std::cout << "\n";
            }
            std::cout << "  Example: C:\\path\\to\\DEM.tif\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return 0;
            if (check_help_command(input)) continue;
            if (!input.empty()) dem_path = input;
            if (dem_path.empty()) {
                std::cout << "ERROR: DEM path cannot be empty!\n";
                continue;
            }
            ValidationResult dem_ok = validate_dem(dem_path);
            if (!dem_ok.is_valid) {
                std::cout << dem_ok.error_message << "\n";
                dem_path = "";
                continue;
            }
            break;
        }

        while (true) {
            print_question("\nEnter output directory for the points layer:\n");
            if (!out_dir.empty()) {
                std::cout << "  "; print_default("Default: " + out_dir); std::cout << "\n";
            }
            std::cout << "  Example: C:\\path\\to\\Results\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return 0;
            if (check_help_command(input)) continue;
            if (!input.empty()) out_dir = input;
            if (!out_dir.empty()) break;
            std::cout << "ERROR: Output directory cannot be empty!\n";
        }

        PointGenOptions gen_opts;
        if (!ask_point_generation_options(gen_opts)) return 0;

        std::cout << "\nGenerating sample points from the DEM...\n";
        auto gen_start = std::chrono::high_resolution_clock::now();
        PointGenResult res = generate_sample_points(dem_path, out_dir, gen_opts);
        auto gen_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - gen_start).count();

        if (!res.ok) {
            std::cout << res.error << "\n";
            std::cout << "Please correct the parameters and try again.\n\n";
            continue;
        }
        if (res.count < 2) {
            std::cout << "ERROR: The generated layer holds " << res.count
                      << " point(s); FETE needs at least 2.\n";
            std::cout << "       Use a smaller spacing or a larger target count.\n";
            std::cout << "Please correct the parameters and try again.\n\n";
            continue;
        }

        // Saved as the FETE default so the next analysis starts from exactly
        // this layer.
        Config to_save = { dem_path, res.path, out_dir,
                           saved_config.cost_modifiers_path,
                           saved_config.cost_raster_path };
        save_config(to_save);

        print_green_success("Sample points successfully generated!\n");
        std::cout << "\nOutput Summary:\n";
        std::cout << "  Total time: " << std::fixed << std::setprecision(2) << gen_time << " sec\n";
        std::cout << "  Points: " << res.count << "\n";
        std::cout << "  Arrangement: "
                  << (gen_opts.random ? "stratified random" : "regular grid") << "\n";
        std::cout << "  Spacing: one point every " << res.spacing << " cell(s)\n";
        if (gen_opts.random) std::cout << "  Seed: " << gen_opts.seed << "\n";
        std::cout << "  Edge buffer: " << gen_opts.edge_buffer << " cell(s)\n";
        std::cout << "\nOutput Files:\n";
        std::cout << "  - " << res.path << "\n";
        if (res.count > 200000) {
            std::cout << "\n  WARNING: FETE cost grows with the square of the point count; "
                      << res.count << " points\n";
            std::cout << "           will take a very long time to compute.\n";
        }

        print_question("\nGenerate another sample points layer? (yes/no)\n"); std::cout << "> ";
        std::string again;
        safe_getline(again);
        if (check_exit_command(again)) return 0;
        if (again == "no" || again == "n" || again.empty()) {
            print_question("\nExit program? (yes/no)\n"); std::cout << "> ";
            std::string exit_choice;
            safe_getline(exit_choice);
            if (check_exit_command(exit_choice)) return 0;
            if (exit_choice == "yes" || exit_choice == "y") {
                std::cout << "\nGoodbye!\n\n";
                break;
            }
        }
    }
    return 0;
}

// ========== NEIGHBOR OFFSET ARRAYS ==========
static inline int idx(int r, int c, int ncols) { return r * ncols + c; }
static inline void idx2coord(int index, int ncols, int& r, int& c) { r = index / ncols; c = index % ncols; }

static const Off OFFS_8[8] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1} };
static const Off OFFS_16[16] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}, {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1} };
static const Off OFFS_24[24] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}, {-2,-2},{-2,-1},{-2,0},{-2,1},{-2,2},{0,-2},{0,2}, {2,-2},{2,-1},{2,0},{2,1},{2,2}, {-1,-2},{-1,2},{1,-2},{1,2} };
static const Off OFFS_32[32] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}, {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}, {-2,-2},{-2,0},{-2,2},{0,-2},{0,2},{2,-2},{2,0},{2,2}, {-3,-1},{-3,1},{-1,-3},{-1,3},{1,-3},{1,3},{3,-1},{3,1} };
static const Off OFFS_64[64] = { {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}, {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}, {-2,-2},{-2,0},{-2,2},{0,-2},{0,2},{2,-2},{2,0},{2,2}, {-3,-1},{-3,1},{-1,-3},{-1,3},{1,-3},{1,3},{3,-1},{3,1}, {-3,-2},{-3,0},{-3,2},{-2,-3},{-2,3},{0,-3},{0,3},{2,-3},{2,3},{3,-2},{3,0},{3,2}, {-3,-3},{-3,3},{3,-3},{3,3} };

// ========== COST FUNCTIONS ==========
enum CostFunctionType { TOBLER_WHITE_2015 = 1, MARQUEZ_PEREZ_ET_AL_2017 = 2, IRMISCHER_CLARKE_2017 = 3 };

static inline float tobler_white_2015(double dh_m, double dz_m) {
    const double sf = dz_m / dh_m;
    const double speed_kmh = 6.0 * std::exp(-3.5 * std::abs(sf + 0.05));
    const double safe_speed = std::max(speed_kmh, 1e-12);
    return (float)((dh_m / 1000.0) / safe_speed);
}

static inline float marquez_perez_et_al_2017(double dh_m, double dz_m) {
    const double sf = dz_m / dh_m;
    const double speed_kmh = 4.8 * std::exp(-5.3 * std::abs((sf * 0.7) + 0.03));
    const double safe_speed = std::max(speed_kmh, 1e-12);
    return (float)((dh_m / 1000.0) / safe_speed);
}

static inline float irmischer_clarke_2017(double dh_m, double dz_m) {
    const double sf = (dz_m / dh_m) * 100.0;
    const double speed_ms  = 0.11 + std::exp(-(sf + 5.0) * (sf + 5.0) / 1800.0);
    const double speed_kmh = speed_ms * 3.6;
    const double safe_speed = std::max(speed_kmh, 1e-12);
    return (float)((dh_m / 1000.0) / safe_speed);
}

static inline float apply_cost_function(CostFunctionType cf, double dh_m, double dz_m) {
    switch (cf) {
        case MARQUEZ_PEREZ_ET_AL_2017: return marquez_perez_et_al_2017(dh_m, dz_m);
        case IRMISCHER_CLARKE_2017: return irmischer_clarke_2017(dh_m, dz_m);
        default:                       return tobler_white_2015(dh_m, dz_m);
    }
}

static inline bool world_to_pixel_northup(double x, double y, const double gt[6], int& col, int& row) {
    if (std::abs(gt[2]) > 1e-12 || std::abs(gt[4]) > 1e-12) return false;
    col = (int)std::floor((x - gt[0]) / gt[1]);
    row = (int)std::floor((y - gt[3]) / gt[5]);
    return true;
}

// ========== COST MODIFIERS RASTERIZATION ==========

/**
 * Rasterize polylines with cost multipliers onto a raster grid
 *
 * @param polylines_path Path to shapefile containing polylines with 'cost' field
 * @param nrows Number of rows in the raster
 * @param ncols Number of columns in the raster
 * @param gt Geotransform array from GDAL
 * @param buffer_cells Buffer radius in cells (applied to each side of polyline)
 * @param max_threads Maximum number of OpenMP threads
 * @return Vector of float cost multipliers (1.0 = no modifier, >1.0 = increased
 *         cost, <1.0 = reduced cost, e.g. roads)
 *
 * Overlap rule: penalties (>= 1) combine with max(), discounts (< 1) with
 * min(), and the two are multiplied together at the end. This is
 * order-independent and immune to a feature's buffer overlapping itself
 * (which is why plain per-cell multiplication cannot be used).
 */
std::vector<float> rasterize_polylines_with_costs(
    const std::string& polylines_path,
    int nrows, int ncols,
    const double gt[6],
    int buffer_cells,
    int max_threads) {

    int N = nrows * ncols;
    std::vector<float> cost_raster(N, 1.0f);  // penalties: combined with max()
    std::vector<float> discount;              // multipliers < 1: combined with min(), allocated on first use

    std::cout << "Reading polylines from vector file...\n";

    // Open vector file (shapefile, GeoJSON, KML, GML/XML, CSV with WKT)
    GDALDataset* polylines_ds = open_vector_dataset(polylines_path);
    if (!polylines_ds) {
        std::cout << "ERROR: Cannot open polylines file: " << polylines_path << "\n";
        return cost_raster;
    }

    OGRLayer* layer = polylines_ds->GetLayer(0);
    if (!layer) {
        std::cout << "ERROR: Cannot read layer from polylines file\n";
        GDALClose(polylines_ds);
        return cost_raster;
    }

    // FALSE: only if the driver knows it from the header. Forcing the count
    // would scan the whole file a second time just for the progress bar.
    GIntBig feature_count = layer->GetFeatureCount(FALSE);
    if (feature_count >= 0)
        info_print("Found " + std::to_string((long long)feature_count) + " polyline features\n");

    // Print CRS information (debug only)
    const OGRSpatialReference* layer_srs = layer->GetSpatialRef();
    if (layer_srs) {
        char* srs_wkt = nullptr;
        layer_srs->exportToWkt(&srs_wkt);
        debug_print("Polylines CRS: " + (srs_wkt ? std::string(srs_wkt).substr(0, 80) : "NULL") + "...\n");
        CPLFree(srs_wkt);
    } else {
        debug_print("WARNING: Polylines have no CRS defined!\n");
    }

    // Print DEM bounds (debug only)
    double dem_xmin = gt[0];
    double dem_xmax = gt[0] + ncols * gt[1];
    double dem_ymax = gt[3];
    double dem_ymin = gt[3] + nrows * gt[5];
    debug_print("DEM bounds: X[" + std::to_string(dem_xmin) + " to " + std::to_string(dem_xmax) +
                "], Y[" + std::to_string(dem_ymin) + " to " + std::to_string(dem_ymax) + "]\n");

    // Check if 'cost' field exists
    OGRFeatureDefn* layer_defn = layer->GetLayerDefn();
    int cost_field_idx = layer_defn->GetFieldIndex("cost");
    if (cost_field_idx == -1) {
        std::cout << "ERROR: 'cost' field not found in polylines file!\n";
        std::cout << "Available fields: ";
        for (int i = 0; i < layer_defn->GetFieldCount(); ++i) {
            std::cout << layer_defn->GetFieldDefn(i)->GetNameRef();
            if (i < layer_defn->GetFieldCount() - 1) std::cout << ", ";
        }
        std::cout << "\n";
        GDALClose(polylines_ds);
        return cost_raster;
    }

    debug_print("'cost' field found at index " + std::to_string(cost_field_idx) + "\n");
    info_print("Rasterizing polylines with buffer = " + std::to_string(buffer_cells) + " cells per side...\n");

    // Process each feature
    layer->ResetReading();
    OGRFeature* feat = nullptr;
    int processed = 0;
    int features_with_valid_points = 0;
    int total_points_processed = 0;
    int points_inside_bounds = 0;

    // Rasterize one linestring: clip each segment to the DEM bounds, then
    // Bresenham with buffer dilation. Clipping keeps the pixel walk bounded
    // even when vertices lie far outside the raster (e.g. CRS mismatch).
    auto rasterize_line = [&](OGRLineString* line, float cost_multiplier, bool& had_valid_points) {
        int num_points = line->getNumPoints();
        for (int i = 0; i < num_points - 1; ++i) {
            double x1 = line->getX(i);
            double y1 = line->getY(i);
            double x2 = line->getX(i + 1);
            double y2 = line->getY(i + 1);

            total_points_processed += 2;

            if (!clip_segment_to_bounds(x1, y1, x2, y2, dem_xmin, dem_ymin, dem_xmax, dem_ymax)) {
                continue;  // segment entirely outside the DEM
            }

            int col1, row1, col2, row2;
            if (!world_to_pixel_northup(x1, y1, gt, col1, row1)) continue;
            if (!world_to_pixel_northup(x2, y2, gt, col2, row2)) continue;

            // Clipped coordinates can land exactly on the max edge
            col1 = std::max(0, std::min(ncols - 1, col1));
            row1 = std::max(0, std::min(nrows - 1, row1));
            col2 = std::max(0, std::min(ncols - 1, col2));
            row2 = std::max(0, std::min(nrows - 1, row2));

            points_inside_bounds += 2;
            had_valid_points = true;

            // Bresenham's line algorithm to rasterize the segment
            int dx = std::abs(col2 - col1);
            int dy = std::abs(row2 - row1);
            int sx = (col1 < col2) ? 1 : -1;
            int sy = (row1 < row2) ? 1 : -1;
            int err = dx - dy;

            int col = col1;
            int row = row1;

            while (true) {
                // Apply cost to current pixel and buffer area
                for (int br = -buffer_cells; br <= buffer_cells; ++br) {
                    for (int bc = -buffer_cells; bc <= buffer_cells; ++bc) {
                        int r = row + br;
                        int c = col + bc;

                        if (r >= 0 && r < nrows && c >= 0 && c < ncols) {
                            int cell_idx = idx(r, c, ncols);
                            if (cost_multiplier >= 1.0f) {
                                // Penalties: strongest overlapping feature wins
                                cost_raster[cell_idx] = std::max(cost_raster[cell_idx], cost_multiplier);
                            } else {
                                // Discounts (roads, tracks): cheapest wins.
                                // A max() against the 1.0 background would
                                // silently discard every multiplier below 1.
                                if (discount.empty()) discount.assign(N, 1.0f);
                                discount[cell_idx] = std::min(discount[cell_idx], cost_multiplier);
                            }
                        }
                    }
                }

                if (col == col2 && row == row2) break;

                int e2 = 2 * err;
                if (e2 > -dy) {
                    err -= dy;
                    col += sx;
                }
                if (e2 < dx) {
                    err += dx;
                    row += sy;
                }
            }
        }
    };

    while ((feat = layer->GetNextFeature()) != nullptr) {
        processed++;
        // Redrawing the bar for EVERY feature turns console I/O into the
        // bottleneck on large vector files; every 100 is smooth enough.
        if (processed % 100 == 0 || (GIntBig)processed == feature_count) {
            if (feature_count > 0) {
                print_progress(processed, (int)std::min<GIntBig>(
                    feature_count, std::numeric_limits<int>::max()));
            } else {
                // Driver could not give a free feature count (CSV, GeoJSON):
                // no percentage is possible, but show that work is happening.
                std::cout << "\r\033[K  Features processed: " << processed << std::flush;
            }
        }

        // Get cost multiplier from 'cost' field
        float cost_multiplier = (float)feat->GetFieldAsDouble(cost_field_idx);
        if (cost_multiplier <= 0.0f) {
            cost_multiplier = 1.0f;  // Zero/negative make no physical sense: neutral
        }

        // Debug: Print first feature info
        if (processed == 1) {
            debug_print("\nFirst feature - cost multiplier: " + std::to_string(cost_multiplier) + "\n");
        }

        OGRGeometry* geom = feat->GetGeometryRef();
        if (!geom) {
            OGRFeature::DestroyFeature(feat);
            continue;
        }

        bool feature_had_valid_points = false;

        // Handle different geometry types (LineString, MultiLineString)
        OGRwkbGeometryType geom_type = wkbFlatten(geom->getGeometryType());

        if (geom_type == wkbLineString) {
            rasterize_line(geom->toLineString(), cost_multiplier, feature_had_valid_points);
        }
        else if (geom_type == wkbMultiLineString) {
            OGRMultiLineString* multiline = geom->toMultiLineString();
            for (int j = 0; j < multiline->getNumGeometries(); ++j) {
                rasterize_line((OGRLineString*)multiline->getGeometryRef(j),
                    cost_multiplier, feature_had_valid_points);
            }
        }

        if (feature_had_valid_points) {
            features_with_valid_points++;
        }

        OGRFeature::DestroyFeature(feat);
    }

    if (processed > 0) std::cout << "\n";  // close the progress line

    // Compose discounts into the penalty raster (order-independent)
    if (!discount.empty()) {
        for (int i = 0; i < N; ++i) cost_raster[i] *= discount[i];
    }

    // Count cells with cost modifiers (penalties or discounts)
    int modified_cells = 0;
    for (int i = 0; i < N; ++i) {
        if (cost_raster[i] != 1.0f) {
            modified_cells++;
        }
    }

    // Always show modified cells count (important info)
    info_print("  Modified cells: " + std::to_string(modified_cells) + " / " + std::to_string(N) + " (" +
               std::to_string(100.0 * modified_cells / N) + "%)\n");

    // Detailed statistics (debug only)
    debug_print("\nPolylines rasterization statistics:\n");
    debug_print("  Features processed: " + std::to_string(processed) + "\n");
    debug_print("  Features with valid points inside DEM: " + std::to_string(features_with_valid_points) + "\n");
    debug_print("  Total points processed: " + std::to_string(total_points_processed) + "\n");
    debug_print("  Points inside DEM bounds: " + std::to_string(points_inside_bounds) + "\n");

    if (modified_cells == 0) {
        info_print("\nWARNING: No cells were modified!\n");
        info_print("Possible causes:\n");
        info_print("  1. Polylines are outside DEM bounds (check CRS match)\n");
        info_print("  2. All 'cost' values are exactly 1.0 (or missing/invalid)\n");
        info_print("  3. Coordinate transformation failed\n");
    }

    GDALClose(polylines_ds);
    return cost_raster;
}

// ========== MAIN FETE ALGORITHM ==========

FETEOutput run_fete(const std::string& dem_path, const std::string& pts_path, const std::string& out_dir,
    const std::string& slope_filename, const std::string& cost_filename,
    const std::string& output_filename, int buffer_radius, int max_threads, int64_t max_ram_mb,
    int num_neighbours, bool slope_in_degrees, CostFunctionType cost_function,
    const std::string& cost_modifiers_path = "", int polyline_buffer_radius = 0,
    const std::string& cost_raster_path = "",
    const std::string& additional_cost_filename = "", const std::string& total_cost_filename = "",
    double barrier_threshold = 1000.0) {

    FETEOutput output = { false, "", "", "", "", "", 0, 0, 0, 0, 0, 0.0, false };
    auto global_start = std::chrono::high_resolution_clock::now();

    // Ensure the output directory exists: GDAL Create returns null otherwise
    {
        std::error_code ec;
        fs::create_directories(out_dir, ec);
        if (ec) {
            std::cout << "ERROR: Cannot create output directory: " << out_dir
                      << " (" << ec.message() << ")\n";
            return output;
        }
    }

    int nrows = 0, ncols = 0, N = 0;
    double gt[6] = { 0 };
    const char* wkt = nullptr;
    std::vector<float> dem;
    GDALDataset* dem_ds = nullptr;

    const Off* current_offs = OFFS_16;
    int num_offs = 16;

    switch (num_neighbours) {
    case 8:  current_offs = OFFS_8;  num_offs = 8;  break;
    case 16: current_offs = OFFS_16; num_offs = 16; break;
    case 24: current_offs = OFFS_24; num_offs = 24; break;
    case 32: current_offs = OFFS_32; num_offs = 32; break;
    case 64: current_offs = OFFS_64; num_offs = 64; break;
    default: current_offs = OFFS_16; num_offs = 16; break;
    }

    std::cout << "Reading DEM...\n";
    auto step1_start = std::chrono::high_resolution_clock::now();

    dem_ds = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
    if (!dem_ds) {
        std::cout << "ERROR: Cannot open DEM file: " << dem_path << "\n";
        return output;
    }

    GDALRasterBand* band = dem_ds->GetRasterBand(1);
    ncols = dem_ds->GetRasterXSize();
    nrows = dem_ds->GetRasterYSize();
    int64_t N64 = (int64_t)nrows * (int64_t)ncols;
    if (N64 > (int64_t)std::numeric_limits<int>::max()) {
        std::cout << "ERROR: DEM has " << N64 << " cells, exceeding the maximum supported ("
                  << std::numeric_limits<int>::max() << "). Resample the DEM to a coarser resolution.\n";
        GDALClose(dem_ds);
        return output;
    }
    N = (int)N64;
    dem_ds->GetGeoTransform(gt);
    wkt = dem_ds->GetProjectionRef();

    dem.resize(N);
    if (band->RasterIO(GF_Read, 0, 0, ncols, nrows, dem.data(), ncols, nrows, GDT_Float32, 0, 0) != CE_None) {
        std::cout << "ERROR: Failed to read DEM data\n";
        GDALClose(dem_ds);
        return output;
    }

    // Build passability mask: NaN, DEM >= 9999 and the band NoData value
    // are all treated as impassable
    int has_nodata = 0;
    const double nodata_val = band->GetNoDataValue(&has_nodata);
    const float nodata_f = (float)nodata_val;
    std::vector<uint8_t> passable(N, 1);
    int impassable_count = 0;
    // Elevation range over passable cells (needed to size the edge-cost LUT)
    float z_min = std::numeric_limits<float>::infinity();
    float z_max = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < N; ++i) {
        float v = dem[i];
        if (std::isnan(v) || v >= 9999.0f || (has_nodata && v == nodata_f)) {
            passable[i] = 0;
            impassable_count++;
        } else {
            if (v < z_min) z_min = v;
            if (v > z_max) z_max = v;
        }
    }
    if (!(z_max >= z_min)) { z_min = 0.0f; z_max = 0.0f; }

    std::cout << "DEM read: " << nrows << "x" << ncols << " (" << N << " cells)\n";
    if (has_nodata) {
        std::cout << "  NoData value: " << nodata_val << "\n";
    }
    if (impassable_count > 0) {
        std::cout << "  Impassable cells (NoData or DEM >= 9999): " << impassable_count
                  << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * impassable_count / N) << "%)\n";
    }
    auto step1_end = std::chrono::high_resolution_clock::now();
    auto step1_time = std::chrono::duration<double>(step1_end - step1_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << step1_time << " sec\n";

    // Realistic memory estimate.
    // Shared peak (write phase): dem 4 + passable 1 + multipliers 4 +
    // density 8 + Float64 write buffer 8 = 25 bytes/cell.
    // Dijkstra phase: 17 shared (dem+passable+multipliers+density) plus, per
    // thread: cost 4 + predecessor 4 + visited 1 + path_count 4 +
    // visit_order 4 + touched 4 + heap reserve ~2 = 23 bytes/cell.
    int64_t bytes_shared_peak = 25LL * N;
    int64_t bytes_dijkstra = 17LL * N + 23LL * N * max_threads;
    int64_t estimated_ram = std::max(bytes_shared_peak, bytes_dijkstra) / (1024 * 1024);
    if (estimated_ram > max_ram_mb) {
        std::cout << "ERROR: Estimated peak memory use is ~" << estimated_ram << " MB with "
                  << max_threads << " threads, but max allowed is " << max_ram_mb << " MB\n";
        std::cout << "       Reduce the number of CPU threads or increase the RAM limit.\n";
        GDALClose(dem_ds);
        return output;
    }

    std::cout << "\nReading sample points...\n";
    auto step2_start = std::chrono::high_resolution_clock::now();

    std::vector<int> point_nodes;
    int P = 0;
    GDALDataset* pts_ds = open_vector_dataset(pts_path);
    if (!pts_ds) {
        std::cout << "ERROR: Cannot open points file: " << pts_path << "\n";
        GDALClose(dem_ds);
        return output;
    }

    OGRLayer* layer = pts_ds->GetLayer(0);
    layer->ResetReading();
    const double dem_xmax_edge = gt[0] + ncols * gt[1];
    const double dem_ymin_edge = gt[3] + nrows * gt[5];
    auto add_point = [&](double x, double y) {
        int col, row;
        if (!world_to_pixel_northup(x, y, gt, col, row)) return;
        // Points exactly on the max edge belong to the last pixel
        if (col == ncols && x <= dem_xmax_edge) col = ncols - 1;
        if (row == nrows && y >= dem_ymin_edge) row = nrows - 1;
        if (row >= 0 && row < nrows && col >= 0 && col < ncols) {
            point_nodes.push_back(idx(row, col, ncols));
        }
    };
    OGRFeature* feat = nullptr;
    while ((feat = layer->GetNextFeature()) != nullptr) {
        OGRGeometry* geom = feat->GetGeometryRef();
        if (geom) {
            OGRwkbGeometryType gtype = wkbFlatten(geom->getGeometryType());
            if (gtype == wkbPoint) {
                OGRPoint* p = geom->toPoint();
                add_point(p->getX(), p->getY());
            } else if (gtype == wkbMultiPoint) {
                OGRMultiPoint* mp = geom->toMultiPoint();
                for (int gi = 0; gi < mp->getNumGeometries(); gi++) {
                    OGRPoint* p = mp->getGeometryRef(gi)->toPoint();
                    add_point(p->getX(), p->getY());
                }
            }
        }
        OGRFeature::DestroyFeature(feat);
    }
    GDALClose(pts_ds);

    P = (int)point_nodes.size();
    std::cout << "Points read: " << P << "\n";
    if (P < 2) {
        std::cout << "ERROR: FETE requires at least 2 sample points\n";
        GDALClose(dem_ds);
        return output;
    }
    auto step2_end = std::chrono::high_resolution_clock::now();
    auto step2_time = std::chrono::duration<double>(step2_end - step2_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << step2_time << " sec\n";

    std::cout << "\nCalculating slope (" << (slope_in_degrees ? "degrees" : "percentage") << ")...\n";
    auto step2a_start = std::chrono::high_resolution_clock::now();

    const double res_x = gt[1];
    const double res_y = std::abs(gt[5]);

    std::vector<float> slope_data(N, 0.0f);

    // No collapse(2): MSVC's OpenMP 2.0 ignores the clause anyway (C4849) and
    // parallelizing the row loop alone is enough with thousands of rows.
#pragma omp parallel for num_threads(max_threads)
    for (int r = 1; r < nrows - 1; ++r) {
        for (int c = 1; c < ncols - 1; ++c) {
            int center = idx(r, c, ncols);
            // Skip cells whose stencil touches NoData/impassable cells
            if (!passable[center] ||
                !passable[idx(r, c + 1, ncols)] || !passable[idx(r, c - 1, ncols)] ||
                !passable[idx(r + 1, c, ncols)] || !passable[idx(r - 1, c, ncols)]) {
                continue;
            }
            float dz_dx = (dem[idx(r, c + 1, ncols)] - dem[idx(r, c - 1, ncols)]) / (2.0f * res_x);
            float dz_dy = (dem[idx(r + 1, c, ncols)] - dem[idx(r - 1, c, ncols)]) / (2.0f * res_y);
            float gradient = std::sqrt(dz_dx * dz_dx + dz_dy * dz_dy);

            if (slope_in_degrees) {
                slope_data[center] = std::atan(gradient) * 180.0f / 3.14159265f;
            }
            else {
                slope_data[center] = gradient * 100.0f;
            }
        }
    }

    // Impassable cells carry no slope: mark them NoData in the output so GIS
    // software doesn't display them as flat terrain.
    const float kOutNoData = -9999.0f;
    const double kOutNoDataD = -9999.0;
#pragma omp parallel for num_threads(max_threads)
    for (int i = 0; i < N; ++i)
        if (!passable[i]) slope_data[i] = kOutNoData;

    std::string slope_path = join_path(out_dir, slope_filename + ".tif");
    if (!write_gtiff_raster(slope_path, ncols, nrows, gt, wkt, slope_data.data(),
                            GDT_Float32, &kOutNoDataD)) {
        GDALClose(dem_ds);
        return output;
    }

    std::cout << "Slope saved\n";
    auto step2a_end = std::chrono::high_resolution_clock::now();
    auto step2a_time = std::chrono::duration<double>(step2a_end - step2a_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << step2a_time << " sec\n";

    std::cout << "\nCalculating cost surface (" << num_neighbours << "-connectivity)...\n";
    auto step2b_start = std::chrono::high_resolution_clock::now();

    std::vector<float> cost_surface(N, 0.0f);

#pragma omp parallel for num_threads(max_threads)
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            int from_idx = idx(r, c, ncols);
            if (!passable[from_idx]) continue;
            float z_from = dem[from_idx];
            float total_cost = 0.0f;
            int valid_neighbors = 0;

            for (int k = 0; k < num_offs; ++k) {
                int nr = r + current_offs[k].dr;
                int nc = c + current_offs[k].dc;
                if (nr < 0 || nr >= nrows || nc < 0 || nc >= ncols) continue;

                int to_idx = idx(nr, nc, ncols);
                if (!passable[to_idx]) continue;
                double dh = std::sqrt((current_offs[k].dr * res_y) * (current_offs[k].dr * res_y) +
                    (current_offs[k].dc * res_x) * (current_offs[k].dc * res_x));
                double dz = (double)dem[to_idx] - (double)z_from;
                total_cost += apply_cost_function(cost_function, dh, dz);
                valid_neighbors++;
            }

            if (valid_neighbors > 0) {
                cost_surface[from_idx] = total_cost / (float)valid_neighbors;
            }
        }
    }

#pragma omp parallel for num_threads(max_threads)
    for (int i = 0; i < N; ++i)
        if (!passable[i]) cost_surface[i] = kOutNoData;

    std::string cost_path = join_path(out_dir, cost_filename + ".tif");
    if (!write_gtiff_raster(cost_path, ncols, nrows, gt, wkt, cost_surface.data(),
                            GDT_Float32, &kOutNoDataD)) {
        GDALClose(dem_ds);
        return output;
    }

    std::cout << "Base cost surface saved\n";
    auto step2b_end = std::chrono::high_resolution_clock::now();
    auto step2b_time = std::chrono::duration<double>(step2b_end - step2b_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << step2b_time << " sec\n";

    // ========== APPLY COST MODIFIERS IF SPECIFIED ==========
    std::vector<float> cost_multipliers(N, 1.0f);  // Default: no modifiers (all 1.0)
    std::string additional_cost_path = "";
    std::string total_cost_path = "";
    bool has_any_modifiers = false;

    // --- Apply shapefile-based cost modifiers (.shp) ---
    if (!cost_modifiers_path.empty()) {
        std::cout << "\n" << std::string(70, '-') << "\n";
        std::cout << "Applying cost modifiers from polylines (.shp)...\n";
        std::cout << std::string(70, '-') << "\n";

        auto step2c_start = std::chrono::high_resolution_clock::now();

        // Rasterize polylines with cost multipliers
        cost_multipliers = rasterize_polylines_with_costs(
            cost_modifiers_path, nrows, ncols, gt, polyline_buffer_radius, max_threads);
        has_any_modifiers = true;

        auto step2c_end = std::chrono::high_resolution_clock::now();
        auto step2c_time = std::chrono::duration<double>(step2c_end - step2c_start).count();
        std::cout << "Shapefile cost modifiers applied successfully\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << step2c_time << " sec\n";
        std::cout << std::string(70, '-') << "\n";
    }

    // --- Apply raster-based cost modifiers (.tif) ---
    if (!cost_raster_path.empty()) {
        std::cout << "\n" << std::string(70, '-') << "\n";
        std::cout << "Applying cost modifiers from raster (.tif)...\n";
        std::cout << std::string(70, '-') << "\n";

        auto step2d_start = std::chrono::high_resolution_clock::now();

        GDALDataset* raster_ds = (GDALDataset*)GDALOpen(cost_raster_path.c_str(), GA_ReadOnly);
        if (!raster_ds) {
            std::cout << "WARNING: Cannot open cost raster: " << cost_raster_path << " - skipping\n";
        } else {
            int r_cols = raster_ds->GetRasterXSize();
            int r_rows = raster_ds->GetRasterYSize();

            if (r_cols != ncols || r_rows != nrows) {
                std::cout << "WARNING: Cost raster dimensions (" << r_cols << "x" << r_rows
                          << ") do not match DEM (" << ncols << "x" << nrows << ") - skipping\n";
                GDALClose(raster_ds);
            } else {
                std::vector<float> raster_multipliers(N, 1.0f);
                GDALRasterBand* band = raster_ds->GetRasterBand(1);
                CPLErr read_err = band->RasterIO(GF_Read, 0, 0, ncols, nrows,
                    raster_multipliers.data(), ncols, nrows, GDT_Float32, 0, 0);
                GDALClose(raster_ds);

                if (read_err != CE_None) {
                    std::cout << "WARNING: Failed to read cost raster data - skipping\n";
                } else {
                    // Multiply into cost_multipliers (stacks with shapefile modifiers)
                    int applied_cells = 0;
#pragma omp parallel for num_threads(max_threads) reduction(+:applied_cells)
                    for (int i = 0; i < N; ++i) {
                        float val = raster_multipliers[i];
                        // Treat NoData (NaN) and values <= 0 as neutral (1.0)
                        if (std::isnan(val) || val <= 0.0f) val = 1.0f;
                        if (val != 1.0f) applied_cells++;
                        cost_multipliers[i] *= val;
                    }
                    has_any_modifiers = true;

                    std::cout << "Raster cost modifiers applied: " << applied_cells
                              << " cells modified out of " << N << "\n";
                }
            }
        }

        auto step2d_end = std::chrono::high_resolution_clock::now();
        auto step2d_time = std::chrono::duration<double>(step2d_end - step2d_start).count();
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << step2d_time << " sec\n";
        std::cout << std::string(70, '-') << "\n";
    }

    // --- Save combined cost surfaces ---
    if (has_any_modifiers) {
        // Save additional cost surface (combined multipliers)
        additional_cost_path = join_path(out_dir, additional_cost_filename + ".tif");
        if (!write_gtiff_raster(additional_cost_path, ncols, nrows, gt, wkt, cost_multipliers.data(), GDT_Float32)) {
            GDALClose(dem_ds);
            return output;
        }
        std::cout << "Additional cost surface saved: " << additional_cost_path << "\n";

        // Multiply base cost surface by cost multipliers to get total cost surface
        std::cout << "Calculating total cost surface (base * multipliers)...\n";
        std::vector<float> total_cost_surface(N);

#pragma omp parallel for num_threads(max_threads)
        for (int i = 0; i < N; ++i) {
            // cost_surface holds NoData on impassable cells: don't multiply it
            total_cost_surface[i] = passable[i] ? cost_surface[i] * cost_multipliers[i]
                                                : kOutNoData;
        }

        // Save total cost surface
        total_cost_path = join_path(out_dir, total_cost_filename + ".tif");
        if (!write_gtiff_raster(total_cost_path, ncols, nrows, gt, wkt,
                                total_cost_surface.data(), GDT_Float32, &kOutNoDataD)) {
            GDALClose(dem_ds);
            return output;
        }
        std::cout << "Total cost surface saved: " << total_cost_path << "\n";
    }

    // Slope and base cost surface are informational outputs only: free them
    // before the propagation phase to shrink the working set
    std::vector<float>().swap(slope_data);
    std::vector<float>().swap(cost_surface);

    // ---- Treat extreme cost multipliers as hard barriers ----
    // With very large multipliers (e.g. 999999) any point lying on a modified
    // cell forces Dijkstra to expand to enormous cost levels, settling the
    // ENTIRE raster for every source before early termination can fire.
    // Marking those cells impassable keeps searches tight and matches the
    // physical intent of an "impassable" obstacle.
    if (has_any_modifiers && barrier_threshold > 0.0) {
        const float thr = (float)barrier_threshold;
        int barrier_cells = 0;
        for (int i = 0; i < N; ++i) {
            if (passable[i] && cost_multipliers[i] >= thr) {
                passable[i] = 0;
                barrier_cells++;
            }
        }
        if (barrier_cells > 0) {
            std::cout << "Barrier cells (multiplier >= " << barrier_threshold << "): "
                      << barrier_cells << " marked impassable\n";
        }
    }

    // Exclude points on impassable cells (NoData or barrier): they can never
    // be reached, and they would disable early termination (every search
    // would flood the whole raster)
    {
        std::vector<int> usable;
        usable.reserve(point_nodes.size());
        int excluded = 0;
        for (int node : point_nodes) {
            if (passable[node]) usable.push_back(node);
            else excluded++;
        }
        if (excluded > 0) {
            std::cout << "WARNING: " << excluded
                      << " point(s) fall on impassable cells (NoData or barrier) and were excluded\n";
        }
        point_nodes.swap(usable);
        P = (int)point_nodes.size();
        if (P < 2) {
            std::cout << "ERROR: fewer than 2 usable sample points remain\n";
            GDALClose(dem_ds);
            return output;
        }
        std::cout << "Usable points: " << P << "\n";
    }

    std::cout << "\nLooping propagation algorithm...\n";
    auto step3_start = std::chrono::high_resolution_clock::now();

    // Reset the counters so the line printed after the phase describes THIS run
    // and not whatever an earlier mode left behind.
    g_large_pages_bytes = 0;
    g_small_pages_bytes = 0;
    g_large_page_failures = 0;

    const float INF = std::numeric_limits<float>::infinity();
    // 64-bit accumulation: with P points a cell can collect up to ~P^2 path
    // transits (2.7e11 for 500k points), far past the 4.29e9 limit of the
    // uint32 counters used before, which wrapped around silently and turned
    // the BUSIEST corridors into low-density cells.
    std::vector<uint64_t> fete_density(N, 0);

    // ---- OPT-1: Pre-compute edge horizontal distances per offset ----
    // Eliminates sqrt() from inner Dijkstra loop (~100M calls/source)
    std::vector<double> precomp_inv_dh(num_offs);
    std::vector<double> precomp_dh_div6000(num_offs);
    std::vector<double> precomp_dh_div4800(num_offs);
    for (int k = 0; k < num_offs; ++k) {
        double dh = std::sqrt((current_offs[k].dr * res_y) * (current_offs[k].dr * res_y) +
            (current_offs[k].dc * res_x) * (current_offs[k].dc * res_x));
        precomp_inv_dh[k] = 1.0 / dh;
        precomp_dh_div6000[k] = dh / 6000.0;
        precomp_dh_div4800[k] = dh / 4800.0;
    }

    // ---- OPT-2: Pre-compute flat neighbor offsets ----
    // Replaces nr*ncols+nc multiply with v+flat_off addition
    std::vector<int> flat_off(num_offs);
    for (int k = 0; k < num_offs; ++k) {
        flat_off[k] = current_offs[k].dr * ncols + current_offs[k].dc;
    }

    // ---- OPT-16: Edge-cost lookup table ----
    // The edge cost is scale_k * g(sf) with sf = dz/dh: g() is the only
    // transcendental part and is evaluated billions of times per run.
    // Tabulating g(sf) over the DEM's slope range and interpolating linearly
    // replaces exp() with two cached loads; relative error is ~1e-7, far
    // below the empirical uncertainty of the cost functions.
    // Set TRAJECTA_EXACT_COST=1 to force the exact exp() path (validation).
    const bool use_lut = (std::getenv("TRAJECTA_EXACT_COST") == nullptr);
    const int LUT_N = 65536;
    std::vector<float> cost_lut(LUT_N + 1, 0.0f);
    std::vector<double> lut_scale(num_offs);
    double lut_sf_min = 0.0, lut_inv_step = 0.0;
    {
        double max_inv_dh = 0.0;
        for (int k = 0; k < num_offs; ++k) {
            max_inv_dh = std::max(max_inv_dh, precomp_inv_dh[k]);
            if (cost_function == TOBLER_WHITE_2015)
                lut_scale[k] = precomp_dh_div6000[k];          // cost = (dh/6000)*g
            else if (cost_function == MARQUEZ_PEREZ_ET_AL_2017)
                lut_scale[k] = precomp_dh_div4800[k];          // cost = (dh/4800)*g
            else
                lut_scale[k] = precomp_dh_div6000[k] * 6.0;    // dh/1000; cost = (dh/1000)*g
        }
        double dz_span = ((double)z_max > (double)z_min) ? (double)z_max - (double)z_min : 1.0;
        double sf_span = dz_span * max_inv_dh * (1.0 + 1e-9) + 1e-12;
        lut_sf_min = -sf_span;
        double step = (2.0 * sf_span) / LUT_N;
        lut_inv_step = 1.0 / step;
        for (int i = 0; i <= LUT_N; ++i) {
            double sf = lut_sf_min + step * i;
            double g;
            if (cost_function == TOBLER_WHITE_2015) {
                g = std::exp(std::min(3.5 * std::abs(sf + 0.05), 80.0));
            } else if (cost_function == MARQUEZ_PEREZ_ET_AL_2017) {
                g = std::exp(std::min(5.3 * std::abs(sf * 0.7 + 0.03), 80.0));
            } else {
                double s = sf * 100.0;
                double speed_ms  = 0.11 + std::exp(-(s + 5.0) * (s + 5.0) / 1800.0);
                double speed_kmh = std::max(speed_ms * 3.6, 1e-12);
                g = 1.0 / speed_kmh;
            }
            cost_lut[i] = (float)g;
        }
        debug_print("Edge-cost LUT: " + std::to_string(LUT_N) + " entries, sf range +/-" +
                    std::to_string(sf_span) + (use_lut ? "" : " (DISABLED: exact mode)") + "\n");
    }

    // ---- OPT-3: Build is_point_node lookup for O(1) early-termination checks ----
    std::vector<char> is_point_node(N, 0);
    int num_unique_points = 0;
    for (int i = 0; i < P; ++i) {
        if (!is_point_node[point_nodes[i]]) {
            is_point_node[point_nodes[i]] = 1;
            num_unique_points++;
        }
    }

    // ---- OPT-4: Pre-compute buffer dilation offsets ----
    struct BufOff { int dr; int dc; };
    std::vector<BufOff> buf_offs;
    for (int bdr = -buffer_radius; bdr <= buffer_radius; ++bdr) {
        for (int bdc = -buffer_radius; bdc <= buffer_radius; ++bdc) {
            if (bdr == 0 && bdc == 0) continue;
            buf_offs.push_back({ bdr, bdc });
        }
    }
    const int num_buf_offs = (int)buf_offs.size();

    // ---- OPT-5: Detect if cost multipliers are active ----
    const bool has_multipliers = has_any_modifiers;

    // ---- OPT-17: Buffer smoothing as separable box filter ----
    // Dilating during the write phase costs (2b+1)^2 atomic adds per path
    // cell; summing the plain density over a (2b+1)x(2b+1) box afterwards is
    // mathematically identical and costs two O(N) passes.
    // Set TRAJECTA_LEGACY_BUFFER=1 to restore the per-cell dilation path.
    const bool use_boxfilter = (buffer_radius > 0) &&
                               (std::getenv("TRAJECTA_LEGACY_BUFFER") == nullptr);

    // Raw pointer aliases for cache-friendly access inside hot loops
    const float* dem_ptr = dem.data();
    const uint8_t* passable_ptr = passable.data();
    const float* cm_ptr = cost_multipliers.data();
    const int* pn_ptr = point_nodes.data();
    const char* ipn_ptr = is_point_node.data();
    const double* inv_dh_ptr = precomp_inv_dh.data();
    const double* dh6000_ptr = precomp_dh_div6000.data();
    const double* dh4800_ptr = precomp_dh_div4800.data();
    const float* lut_ptr = cost_lut.data();
    const double* lscale_ptr = lut_scale.data();
    const int* flat_off_ptr = flat_off.data();
    const BufOff* buf_ptr = buf_offs.data();
    uint64_t* density_ptr = fete_density.data();

    // ---- PERFORMANCE MONITORING (verbose mode only) ----
    const int perf_batch_size = 100;
    std::vector<PerfSample> perf_samples;
    std::atomic<int> global_completed{0};
    std::atomic<long long> global_touched{0};  // settled cells, for cache-footprint chart
    std::atomic<long long> unreached_pairs{0}; // source-to-point paths that do not exist
    CpuMonitor cpu_monitor;
    double perf_prev_wall = 0.0;
    long long perf_prev_touched = 0;
    int perf_prev_count = 0;
    if (g_verbose_mode) {
        perf_samples.reserve(P / perf_batch_size + 2);
        cpu_monitor.sample(0.0);  // initialize baseline
    }

    // ---- OPT-6: Split parallel region from for loop ----
    // Allocate per-thread buffers ONCE, reuse across all iterations
#pragma omp parallel num_threads(max_threads)
    {
        // Per-thread buffers allocated once (eliminates ~400K malloc/free per array)
        // lp_vector: served by 2 MB pages when the user enabled them and the OS
        // can provide them, by ordinary pages otherwise. Identical contents and
        // identical results either way; see LargePageAllocator.
        lp_vector<float> cumulative_cost(N, INF);
        lp_vector<int> predecessor(N, -1);
        lp_vector<char> visited(N, 0);        // OPT-7: char instead of bit-packed bool
        lp_vector<uint32_t> path_count(N, 0); // For Brandes backward propagation

        // OPT-9: Vector-as-heap with reserve (avoids reallocation)
        using pq_entry = std::pair<float, int>;
        auto pq_greater = [](const pq_entry& a, const pq_entry& b) { return a.first > b.first; };
        lp_vector<pq_entry> pq_vec;
        pq_vec.reserve(N / 4);

        // OPT-10: Track visited/touched cells for smart reset
        lp_vector<int> visit_order;
        lp_vector<int> touched;
        visit_order.reserve(N);
        touched.reserve(N);

        bool first_iteration = true;

#pragma omp for schedule(dynamic)
        for (int source_idx = 0; source_idx < P; ++source_idx) {
            const int source = pn_ptr[source_idx];

            // ---- OPT-10: Smart reset - only touch cells modified last iteration ----
            if (!first_iteration) {
                for (int t : touched) {
                    cumulative_cost[t] = INF;
                    predecessor[t] = -1;
                    visited[t] = 0;
                }
                // path_count is reset in the density-write loop below
            }
            first_iteration = false;
            visit_order.clear();
            touched.clear();
            pq_vec.clear();

            // Initialize Dijkstra source
            cumulative_cost[source] = 0.0f;
            touched.push_back(source);
            pq_vec.push_back({ 0.0f, source });

            // ---- OPT-3: Early termination counter ----
            int dest_remaining = num_unique_points;

            // ========== DIJKSTRA ==========
            while (!pq_vec.empty()) {
                std::pop_heap(pq_vec.begin(), pq_vec.end(), pq_greater);
                auto [cost, v] = pq_vec.back();
                pq_vec.pop_back();

                if (visited[v]) continue;
                visited[v] = 1;
                visit_order.push_back(v);
                if (cost >= INF) break;

                // Early termination: stop when all sample points reached
                if (ipn_ptr[v]) {
                    --dest_remaining;
                    if (dest_remaining == 0) break;
                }

                // OPT-11: Avoid idx2coord division; use subtract instead of modulo
                int r = v / ncols;
                int c = v - r * ncols;

                // OPT-12: Hoist dem[v] read outside the offset loop
                float dem_v = dem_ptr[v];

                for (int k = 0; k < num_offs; ++k) {
                    // OPT-13: Unsigned bounds check halves comparisons
                    int nr = r + current_offs[k].dr;
                    int nc = c + current_offs[k].dc;
                    if ((unsigned)nr >= (unsigned)nrows || (unsigned)nc >= (unsigned)ncols) continue;

                    // OPT-2: Flat offset instead of multiply; OPT-14: compute u BEFORE visited check
                    int u = v + flat_off_ptr[k];
                    if (visited[u]) continue;
                    if (!passable_ptr[u]) continue;  // Skip NoData/impassable cells

                    // OPT-1: Pre-computed dh eliminates sqrt; reformulated Tobler eliminates divisions
                    double dz = (double)dem_ptr[u] - (double)dem_v;
                    double sf = dz * inv_dh_ptr[k];
                    float edge_cost;
                    if (use_lut) {
                        // OPT-16: table lookup + linear interpolation replaces exp()
                        double pos = (sf - lut_sf_min) * lut_inv_step;
                        if (pos < 0.0) pos = 0.0;
                        else if (pos > (double)LUT_N) pos = (double)LUT_N;
                        int li = (int)pos;
                        if (li >= LUT_N) li = LUT_N - 1;
                        float frac = (float)(pos - (double)li);
                        float g = lut_ptr[li] + frac * (lut_ptr[li + 1] - lut_ptr[li]);
                        edge_cost = (float)(lscale_ptr[k] * g);
                    } else if (cost_function == TOBLER_WHITE_2015) {
                        double arg = 3.5 * std::abs(sf + 0.05);
                        edge_cost = (float)(dh6000_ptr[k] * std::exp(std::min(arg, 80.0)));
                    } else if (cost_function == MARQUEZ_PEREZ_ET_AL_2017) {
                        double arg = 5.3 * std::abs(sf * 0.7 + 0.03);
                        edge_cost = (float)(dh4800_ptr[k] * std::exp(std::min(arg, 80.0)));
                    } else {
                        double s = sf * 100.0;
                        double speed_ms  = 0.11 + std::exp(-(s + 5.0) * (s + 5.0) / 1800.0);
                        double speed_kmh = std::max(speed_ms * 3.6, 1e-12);
                        edge_cost = (float)((dh6000_ptr[k] * 6000.0) / (speed_kmh * 1000.0));
                    }

                    // OPT-5: Skip multiply when no cost modifiers loaded
                    if (has_multipliers) edge_cost *= cm_ptr[u];

                    // `cost` equals cumulative_cost[v] once v is settled: reuse
                    // the popped value instead of reloading the array
                    float new_cost = cost + edge_cost;

                    if (new_cost < cumulative_cost[u]) {
                        if (cumulative_cost[u] >= INF) {
                            touched.push_back(u);   // OPT-10: Track for smart reset
                        }
                        cumulative_cost[u] = new_cost;
                        predecessor[u] = v;
                        pq_vec.push_back({ new_cost, u });
                        std::push_heap(pq_vec.begin(), pq_vec.end(), pq_greater);
                    }
                }
            }

            // Points never settled have no path from this source (separated
            // by barriers or NoData); track for the summary note
            if (dest_remaining > 0) {
                unreached_pairs += dest_remaining;
            }

            // ========== OPT-8: BRANDES BACKWARD PROPAGATION ==========
            // Replaces P-1 random predecessor-chain walks with ONE linear sweep.
            // path_count[v] = number of shortest paths from source passing through v.

            // Initialize: each destination contributes 1 (handles duplicates via ++)
            for (int d = 0; d < P; ++d) {
                int dest = pn_ptr[d];
                if (dest != source && visited[dest]) {
                    path_count[dest]++;
                }
            }

            // Backward sweep: far-to-near through visit order
            const int visit_count = (int)visit_order.size();
            for (int i = visit_count - 1; i >= 0; --i) {
                int v = visit_order[i];
                int pred = predecessor[v];
                if (pred >= 0) {
                    path_count[pred] += path_count[v];
                }
            }

            // ========== DENSITY WRITE WITH ATOMIC UPDATES ==========
            // OPT-15: Eliminates local_density array (25MB) AND critical-section O(N) merge
            for (int i = 0; i < visit_count; ++i) {
                int v = visit_order[i];
                uint32_t pc = path_count[v];
                path_count[v] = 0;  // Reset for next iteration (smart reset)
                if (pc == 0) continue;

                // Atomic self-increment
#pragma omp atomic
                density_ptr[v] += pc;

                // OPT-4 (legacy path): per-cell buffer dilation; with OPT-17
                // the dilation happens once at the end as a box filter
                if (!use_boxfilter) {
                    int vr = v / ncols;
                    int vc = v - vr * ncols;
                    for (int b = 0; b < num_buf_offs; ++b) {
                        int br = vr + buf_ptr[b].dr;
                        int bc = vc + buf_ptr[b].dc;
                        if ((unsigned)br < (unsigned)nrows && (unsigned)bc < (unsigned)ncols) {
#pragma omp atomic
                            density_ptr[br * ncols + bc] += pc;
                        }
                    }
                }
            }

            // Track completion and collect performance data
            if (g_verbose_mode) {
                global_touched += (long long)visit_count;
            }
            int my_count = ++global_completed;

            if (my_count % perf_batch_size == 0 || my_count == P) {
#pragma omp critical
                {
                    // Threads claim their iteration number atomically but reach
                    // this section in any order, so a batch can arrive after a
                    // later one. Timing such a straggler against the newer
                    // state yields negative deltas, which is what made the
                    // throughput and cache-footprint charts saw-tooth. Drop
                    // out-of-order arrivals and derive the batch size from the
                    // counters instead of assuming perf_batch_size, so the
                    // window a sample covers is always the real one.
                    if (my_count > perf_prev_count) {
                        double now_wall = std::chrono::duration<double>(
                            std::chrono::high_resolution_clock::now() - step3_start).count();
                        print_progress(my_count, P, now_wall);

                        if (g_verbose_mode) {
                            int batch_sz = my_count - perf_prev_count;
                            double batch_sec = now_wall - perf_prev_wall;
                            double ips = (batch_sec > 0.001) ? batch_sz / batch_sec : 0.0;
                            double cpu = cpu_monitor.sample(now_wall);
                            double ram = get_process_ram_mb();
                            // Working set per source: settled cells x per-thread bytes
                            // (cost 4 + predecessor 4 + visited 1 + path_count 4)
                            long long touched_now = global_touched.load();
                            long long d_touched = touched_now - perf_prev_touched;
                            if (d_touched < 0) d_touched = 0;
                            double avg_cells = (double)d_touched / batch_sz;
                            double ws_mb = avg_cells * 13.0 / (1024.0 * 1024.0);
                            perf_prev_touched = touched_now;
                            perf_samples.push_back({ my_count, now_wall, batch_sec, ips, cpu, ram, ws_mb });
                            perf_prev_wall = now_wall;
                        }
                        perf_prev_count = my_count;
                    }
                }
            }
        } // end omp for
    } // end omp parallel

    std::cout << "\n";
    auto step3_end = std::chrono::high_resolution_clock::now();
    auto step3_time = std::chrono::duration<double>(step3_end - step3_start).count();
    std::cout << "\n";
    std::cout << "Completed in " << std::fixed << std::setprecision(2) << step3_time << " seconds\n";

    // What actually happened, not what was asked for: the allocation can fail
    // even when the privilege is in place, and the user has to be able to tell
    // whether the setting did anything.
    if (g_large_pages_requested) {
        const double lp_mb = g_large_pages_bytes.load() / (1024.0 * 1024.0);
        const double sp_mb = g_small_pages_bytes.load() / (1024.0 * 1024.0);
        if (g_large_pages_bytes.load() > 0) {
            std::cout << "Large pages: ACTIVE - " << std::fixed << std::setprecision(0)
                      << lp_mb << " MB on "
                      << (g_large_pages_bytes.load() / (long long)large_page_size())
                      << " large pages";
            if (g_large_page_failures.load() > 0)
                std::cout << " (" << g_large_page_failures.load()
                          << " allocation(s) fell back, " << sp_mb << " MB)";
            std::cout << "\n";
        } else {
            std::cout << "Large pages: REQUESTED BUT UNAVAILABLE - using 4 KB pages"
                      << " (" << g_large_page_failures.load()
                      << " allocation attempt(s) refused)\n";
            std::cout << "  Either the 'Lock pages in memory' privilege is not granted to\n";
            std::cout << "  this account (grant it, then sign out and back in), or memory is\n";
            std::cout << "  too fragmented to find contiguous 2 MB blocks (restart the PC).\n";
            std::cout << "  Results are unaffected either way.\n";
        }
    }
    if (unreached_pairs.load() > 0) {
        std::cout << "NOTE: " << unreached_pairs.load()
                  << " source-to-point paths do not exist (points separated by barriers or NoData)\n";
    }

    // ---- OPT-17: apply buffer smoothing as a separable box filter ----
    if (use_boxfilter) {
        auto bf_start = std::chrono::high_resolution_clock::now();
        const int b = buffer_radius;
        std::vector<uint64_t> row_sum((size_t)N);

        // Horizontal pass: sliding-window sum over columns [c-b, c+b]
#pragma omp parallel for num_threads(max_threads)
        for (int r = 0; r < nrows; ++r) {
            const uint64_t* src = fete_density.data() + (size_t)r * ncols;
            uint64_t* dst = row_sum.data() + (size_t)r * ncols;
            uint64_t s = 0;
            int init_end = std::min(b, ncols - 1);
            for (int c = 0; c <= init_end; ++c) s += src[c];
            for (int c = 0; c < ncols; ++c) {
                dst[c] = s;
                int add = c + b + 1;
                int sub = c - b;
                if (add < ncols) s += src[add];
                if (sub >= 0) s -= src[sub];
            }
        }

        // Vertical pass: sliding-window sum over rows [r-b, r+b]
#pragma omp parallel for num_threads(max_threads)
        for (int c = 0; c < ncols; ++c) {
            uint64_t s = 0;
            int init_end = std::min(b, nrows - 1);
            for (int r = 0; r <= init_end; ++r) s += row_sum[(size_t)r * ncols + c];
            for (int r = 0; r < nrows; ++r) {
                uint64_t v64 = s;
                int add = r + b + 1;
                int sub = r - b;
                if (add < nrows) s += row_sum[(size_t)add * ncols + c];
                if (sub >= 0) s -= row_sum[(size_t)sub * ncols + c];
                fete_density[(size_t)r * ncols + c] = v64;
            }
        }

        auto bf_end = std::chrono::high_resolution_clock::now();
        std::cout << "Buffer smoothing (box filter, radius " << b << "): "
                  << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double>(bf_end - bf_start).count() << " sec\n";
    }

    // ---- PERFORMANCE CHARTS (verbose mode) ----
    if (g_verbose_mode && perf_samples.size() >= 2) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "  PERFORMANCE DIAGNOSTICS\n";
        std::cout << std::string(70, '=') << "\n";

        // Summary stats
        double total_wall = perf_samples.back().wall_time;
        double avg_ips = (double)P / total_wall;
        double min_ips = perf_samples[0].iter_per_sec, max_ips = perf_samples[0].iter_per_sec;
        double avg_cpu = 0.0, avg_ram = 0.0, avg_ws = 0.0;
        for (auto& s : perf_samples) {
            min_ips = std::min(min_ips, s.iter_per_sec);
            max_ips = std::max(max_ips, s.iter_per_sec);
            avg_cpu += s.cpu_percent;
            avg_ram += s.ram_mb;
            avg_ws += s.workset_mb;
        }
        avg_cpu /= perf_samples.size();
        avg_ram /= perf_samples.size();
        avg_ws /= perf_samples.size();

        std::cout << "\n  Total iterations: " << P << "\n";
        std::cout << "  Total wall time:  " << std::fixed << std::setprecision(1) << total_wall << " s\n";
        std::cout << "  Avg throughput:   " << std::setprecision(1) << avg_ips << " iter/s\n";
        std::cout << "  Min throughput:   " << std::setprecision(1) << min_ips << " iter/s\n";
        std::cout << "  Max throughput:   " << std::setprecision(1) << max_ips << " iter/s\n";
        std::cout << "  Avg CPU:          " << std::setprecision(1) << avg_cpu << "%\n";
        std::cout << "  Avg RAM:          " << std::setprecision(0) << avg_ram << " MB\n";
        std::cout << "  Avg working set:  " << std::setprecision(1) << avg_ws << " MB/source\n";

        // Chart 1: Throughput over time
        std::vector<double> throughput_vals;
        for (auto& s : perf_samples) throughput_vals.push_back(s.iter_per_sec);
        char x_end_buf[32];
        snprintf(x_end_buf, sizeof(x_end_buf), "iter %d", P);
        print_ascii_chart("Throughput over time", throughput_vals,
            "iter 0", x_end_buf, "iter/s", "\033[36m");

        // Chart 2: CPU utilization
        std::vector<double> cpu_vals;
        for (auto& s : perf_samples) cpu_vals.push_back(s.cpu_percent);
        print_ascii_chart("CPU utilization over time", cpu_vals,
            "iter 0", x_end_buf, "%", "\033[32m");

        // Chart 3: RAM usage
        std::vector<double> ram_vals;
        for (auto& s : perf_samples) ram_vals.push_back(s.ram_mb);
        print_ascii_chart("Process RAM over time", ram_vals,
            "iter 0", x_end_buf, "MB", "\033[35m");

        // Chart 4: cache/working-set footprint. True hardware cache occupancy
        // is not observable from user space; this charts the per-source
        // Dijkstra working set (settled cells x per-thread bytes), i.e. the
        // data volume competing for the CPU caches over time.
        std::vector<double> ws_vals;
        for (auto& s : perf_samples) ws_vals.push_back(s.workset_mb);
        print_ascii_chart("Cache footprint over time (working set per source)", ws_vals,
            "iter 0", x_end_buf, "MB", "\033[33m");

        // Save CSV
        std::string perf_csv_path = join_path(out_dir, output_filename + "_perf.csv");
        save_perf_csv(perf_csv_path, perf_samples);
        std::cout << "\n  Performance data saved: " << perf_csv_path << "\n";
        std::cout << std::string(70, '=') << "\n";
    }

    std::cout << "\nWriting density raster...\n";

    std::string fete_density_path = join_path(out_dir, output_filename + ".tif");
    {
        // Written as Float64: counts stay exact up to 2^53 (far above any
        // possible density) and every GIS reads the format; GTiff UInt64
        // support would need GDAL >= 3.5 on the reader's side too.
        std::vector<double> density_out(N);
#pragma omp parallel for num_threads(max_threads)
        for (int i = 0; i < N; ++i) density_out[i] = (double)fete_density[i];
        if (!write_gtiff_raster(fete_density_path, ncols, nrows, gt, wkt,
                                density_out.data(), GDT_Float64)) {
            GDALClose(dem_ds);
            return output;
        }
    }

    std::cout << "Density raster saved\n";

    uint64_t max_density = 0;
    uint64_t min_density = UINT64_MAX;
    uint64_t total_density = 0;
    int nonzero_cells = 0;

    for (int i = 0; i < N; ++i) {
        if (fete_density[i] > 0) {
            max_density = std::max(max_density, fete_density[i]);
            min_density = std::min(min_density, fete_density[i]);
            total_density += fete_density[i];
            nonzero_cells++;
        }
    }

    auto global_end = std::chrono::high_resolution_clock::now();
    double global_time = std::chrono::duration<double>(global_end - global_start).count();

    output.success = true;
    output.slope_path = slope_path;
    output.cost_path = cost_path;
    output.additional_cost_path = additional_cost_path;
    output.total_cost_path = total_cost_path;
    output.density_path = fete_density_path;
    output.max_density = max_density;
    output.min_density = (nonzero_cells > 0) ? min_density : 0;
    output.avg_density = (nonzero_cells > 0) ? total_density / nonzero_cells : 0;
    output.nonzero_cells = nonzero_cells;
    output.total_cells = N;
    output.time_seconds = global_time;

    GDALClose(dem_ds);

    return output;
}

// ========== MAIN PROGRAM ==========

// Forward declaration - LCPA mode from main_lcpa.cpp
int run_lcpa_mode();
int run_interp_mode();

// Point GDAL and PROJ at the data shipped beside the executable.
//
// The installer now carries its own GDAL, so it must also carry GDAL's data
// directory and PROJ's proj.db: without proj.db PROJ cannot identify a CRS and
// every run fills with "Cannot find proj.db", quietly degrading the projection
// metadata of the outputs. Doing this here rather than in the GUI means it also
// holds when trajecta.exe is run on its own.
//
// Anything the user has already set wins, so a deliberate override still works.
void use_bundled_gdal_data() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    if (!GetModuleFileNameW(nullptr, buf, MAX_PATH))
        return;
    std::error_code ec;
    const fs::path root = fs::path(buf).parent_path();

    const fs::path gdal_data = root / "share" / "gdal";
    if (std::getenv("GDAL_DATA") == nullptr && fs::exists(gdal_data, ec))
        CPLSetConfigOption("GDAL_DATA", gdal_data.string().c_str());

    const fs::path proj_dir = root / "share" / "proj";
    if (std::getenv("PROJ_DATA") == nullptr && std::getenv("PROJ_LIB") == nullptr
        && fs::exists(proj_dir / "proj.db", ec)) {
        const std::string p = proj_dir.string();
        const char* paths[] = { p.c_str(), nullptr };
        OSRSetPROJSearchPaths(paths);
    }
#endif
}

int main(int argc, char* argv[]) {
    use_bundled_gdal_data();
    // enable_ansi_colors() is all that's needed for the colored output;
    // the old system("color 0F") spawned a cmd.exe and permanently
    // repainted the user's terminal.
    enable_ansi_colors();

    // Check for help command
    if (argc > 1 && std::string(argv[1]) == "help") {
        print_help();
        return 0;
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    center_text("TRAJECTA v1.0.0 - A SPATIAL MOVEMENT ANALYSIS SOFTWARE");
    center_text("by Stefano Apra, ISAW - NYU");
    std::cout << std::string(70, '=') << "\n";
    std::cout << "You can type 'help' at any prompt for instructions\n";
    std::cout << "Type 'exit' at any prompt to quit (with confirmation)\n";
    std::cout << "Press Ctrl+C to cancel the execution (Windows default)\n";
    std::cout << std::string(70, '=') << "\n\n";

    // ===== CHOOSE COMPUTATION MODE =====
    print_question("Choose computation mode:\n");
    std::cout << "  1) FETE (From Everywhere to Everywhere) "; print_default("[DEFAULT]"); std::cout << "\n";
    std::cout << "  2) LCPA (Least-Cost Path Analysis)\n";
    std::cout << "  3) NNI (Natural Neighbour Interpolation)\n";
    std::cout << "  4) Sample points (generate a point layer from a DEM)\n\n";

    int mode = 1;  // Default to FETE
    while (true) {
        std::cout << "> ";
        std::string mode_input;
        safe_getline(mode_input);

        if (mode_input == "exit" || mode_input == "EXIT" || mode_input == "Exit") {
            std::cout << "\nGoodbye!\n\n";
            return 0;
        }
        if (mode_input == "help" || mode_input == "HELP" || mode_input == "Help") {
            std::cout << "\nEnter 1 for FETE, 2 for LCPA, 3 for NNI or 4 to generate sample points\n\n";
            continue;
        }

        try {
            int choice = std::stoi(mode_input);
            if (choice >= 1 && choice <= 4) {
                mode = choice;
                break;
            }
            else {
                std::cout << "ERROR: Please enter 1, 2, 3 or 4\n";
            }
        }
        catch (...) {
            if (mode_input.empty()) {
                mode = 1;  // Default to FETE if empty
                break;
            }
            std::cout << "ERROR: Invalid input. Enter 1, 2, 3 or 4\n";
        }
    }

    std::cout << "\n";

    // ===== VERBOSE MODE SELECTION =====
    print_question("Enable detailed debug output? (yes/no):\n");
    std::cout << "  yes - Show detailed logging for troubleshooting\n";
    std::cout << "  no  - Show only progress bars and summaries "; print_default("[DEFAULT]"); std::cout << "\n";

    while (true) {
        std::cout << "> ";
        std::string verbose_input;
        safe_getline(verbose_input);

        if (verbose_input == "exit" || verbose_input == "EXIT" || verbose_input == "Exit") {
            std::cout << "\nGoodbye!\n\n";
            return 0;
        }
        if (verbose_input == "help" || verbose_input == "HELP" || verbose_input == "Help") {
            std::cout << "\nEnter 'yes' for detailed output or 'no' for compact output\n\n";
            continue;
        }

        if (verbose_input.empty() || verbose_input == "no" || verbose_input == "n" ||
            verbose_input == "NO" || verbose_input == "No" || verbose_input == "N") {
            g_verbose_mode = false;
            std::cout << "Verbose mode: OFF (compact output)\n\n";
            break;
        }
        else if (verbose_input == "yes" || verbose_input == "y" ||
                 verbose_input == "YES" || verbose_input == "Yes" || verbose_input == "Y") {
            g_verbose_mode = true;
            std::cout << "Verbose mode: ON (detailed output)\n\n";
            break;
        }
        else {
            std::cout << "ERROR: Please enter 'yes' or 'no'\n";
        }
    }

    // ===== EXECUTE SELECTED MODE =====
    if (mode == 2) {
        // Launch LCPA mode
        return run_lcpa_mode();
    }
    if (mode == 3) {
        // Launch NNI mode (natural neighbour interpolation)
        return run_interp_mode();
    }
    if (mode == 4) {
        // Write a sample-points layer and stop -- no analysis
        return run_points_mode();
    }

    // Continue with FETE mode (default)
    std::cout << std::string(70, '=') << "\n";
    center_text("FETE MODE");
    center_text("From Everywhere to Everywhere");
    std::cout << std::string(70, '=') << "\n\n";

    GDALAllRegister();
    OGRRegisterAll();

    // Load previous config
    Config saved_config = load_config();

    int max_available_threads = omp_get_max_threads();
    std::string cpu_model = get_cpu_model();
    int64_t total_ram_mb = get_total_ram_mb();

    std::cout << "System Information:\n";
    std::cout << "  Available CPU threads: " << max_available_threads << "\n";
    std::cout << "  CPU Model: " << cpu_model << "\n";
    std::cout << "  Total RAM: " << (total_ram_mb / 1024) << " GB\n\n";

    int max_threads = std::max(1, max_available_threads - 4);
    int64_t max_ram_mb = 4096;

    std::string dem_path = saved_config.dem_path;
    std::string pts_path = saved_config.pts_path;
    std::string out_dir = saved_config.out_dir;
    std::string cost_modifiers_path = saved_config.cost_modifiers_path;
    std::string cost_raster_path = saved_config.cost_raster_path;
    std::string output_filename;
    std::string slope_filename;
    std::string cost_filename;
    std::string additional_cost_filename;
    std::string total_cost_filename;
    int buffer_radius = 0;
    int polyline_buffer_radius = 0;
    double barrier_threshold = 1000.0;
    int num_neighbours = 16;
    bool slope_in_degrees = true;
    CostFunctionType cost_function = TOBLER_WHITE_2015;
    // Sample points source. Left at false, the engine behaves exactly as it
    // always has: the points come from a file and nothing below is executed.
    bool generate_points = false;
    PointGenOptions gen_opts;

    // Every FETE run (including re-runs) starts from here: thread selection,
    // RAM limit, then the full input/parameter configuration
    while (true) {
        // ===== CPU THREADS =====
        while (true) {
            std::string threads_input;
            print_question("Enter maximum CPU threads to use (1-" + std::to_string(max_available_threads) + "):\n");
            std::cout << "  Recommended: " << std::max(1, max_available_threads - 4) << " (reserve 4 cores for system)\n";
            std::cout << "> ";
            safe_getline(threads_input);

            if (check_exit_command(threads_input)) {
                return 0;
            }
            if (check_help_command(threads_input)) {
                continue;
            }

            try {
                max_threads = std::stoi(threads_input);
                if (max_threads < 1) max_threads = 1;
                if (max_threads > max_available_threads) max_threads = max_available_threads;
            }
            catch (...) {
                max_threads = std::max(1, max_available_threads - 4);
            }
            break;
        }

        std::cout << "Using " << max_threads << " threads\n\n";

        // ===== RAM LIMIT =====
        while (true) {
            std::string ram_input;
            print_question("Enter maximum RAM to allocate (MB):\n");
            std::cout << "  Total available: " << total_ram_mb << " MB (~" << (total_ram_mb / 1024) << " GB)\n";
            std::cout << "  Example: 4096 (for 4 GB), 8192 (for 8 GB)\n";
            std::cout << "  Recommended: ~60% of available RAM\n";
            std::cout << "> ";
            safe_getline(ram_input);

            if (check_exit_command(ram_input)) {
                return 0;
            }
            if (check_help_command(ram_input)) {
                continue;
            }

            try {
                max_ram_mb = std::stoll(ram_input);
                if (max_ram_mb < 512) max_ram_mb = 512;
            }
            catch (...) {
                max_ram_mb = 4096;
            }
            break;
        }

        std::cout << "Using maximum " << max_ram_mb << " MB RAM\n";

        // ---- Large memory pages (opt-in) ----
        while (true) {
            print_question("\nUse large memory pages? (yes/no):\n");
            std::cout << "  2 MB pages instead of 4 KB cut the address-translation cost of\n";
            std::cout << "  the propagation phase. Typically 15-30% faster on big DEMs.\n";
            std::cout << "  Results are bit-for-bit identical either way.\n";
            std::cout << "  Requires the Windows 'Lock pages in memory' privilege; falls back\n";
            std::cout << "  to normal pages automatically when unavailable.\n";
            std::cout << "  Default: NO\n";
            std::cout << "> ";
            std::string lp_input;
            safe_getline(lp_input);

            if (check_exit_command(lp_input)) {
                return 0;
            }
            if (check_help_command(lp_input)) {
                continue;
            }
            const std::string lp = to_lower_copy(ltrim_copy(lp_input));
            if (lp.empty() || lp == "no" || lp == "n") {
                g_large_pages_requested = false;
                break;
            }
            if (lp == "yes" || lp == "y") {
                g_large_pages_requested = true;
                break;
            }
            std::cout << "ERROR: Please answer yes or no\n";
        }

        if (g_large_pages_requested) {
#ifdef _WIN32
            if (!enable_lock_memory_privilege()) {
                std::cout << "WARNING: the 'Lock pages in memory' privilege is not granted to\n";
                std::cout << "         this account, so large pages cannot be used. Grant it in\n";
                std::cout << "         Local Security Policy, then sign out and back in.\n";
                std::cout << "         Continuing with normal 4 KB pages.\n";
            } else {
                std::cout << "Large memory pages: requested (page size "
                          << (large_page_size() / 1024) << " KB)\n";
            }
#else
            std::cout << "Large memory pages: not supported on this platform, ignoring\n";
            g_large_pages_requested = false;
#endif
        }

        std::cout << std::string(70, '=') << "\n\n";
        std::cout << "\n";

        {
            // DEM Path
            while (true) {
                print_question("Enter path to DEM file (.tif):\n");
                if (!dem_path.empty()) {
                    std::cout << "  "; print_default("Default: " + dem_path); std::cout << "\n";
                }
                std::cout << "  Example: C:\\path\\to\\DEM.tif\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (!input.empty()) dem_path = input;
                if (!dem_path.empty()) break;
                std::cout << "ERROR: DEM path cannot be empty!\n";
            }

            // Sample points: import a file (as always) or generate them from
            // the DEM. In import mode the whole generation branch below is
            // skipped: no extra question, no extra work.
            while (true) {
                print_question("\nSample points source:\n");
                std::cout << "  1) Import from an existing file "; print_default("[DEFAULT]"); std::cout << "\n";
                std::cout << "  2) Generate them from the DEM\n";
                std::cout << "  "; print_default("Leave blank for default (1)"); std::cout << "\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (input.empty() || input == "1") { generate_points = false; break; }
                if (input == "2") { generate_points = true; break; }
                std::cout << "ERROR: Please enter 1 or 2\n";
            }

            if (!generate_points) {
                // Points Path
                while (true) {
                    print_question("\nEnter path to sample points file (" + supported_vector_formats() + "):\n");
                    if (!pts_path.empty()) {
                        std::cout << "  "; print_default("Default: " + pts_path); std::cout << "\n";
                    }
                    std::cout << "  Example: C:\\path\\to\\Points.shp (or .csv with x/y, lon/lat columns)\n";
                    std::cout << "> ";
                    std::string input;
                    safe_getline(input);
                    if (check_exit_command(input)) return 0;
                    if (check_help_command(input)) continue;
                    if (!input.empty()) pts_path = input;
                    if (!pts_path.empty()) break;
                    std::cout << "ERROR: Points path cannot be empty!\n";
                }
            }
            else {
                gen_opts = PointGenOptions();   // never inherit a previous run
                if (!ask_point_generation_options(gen_opts)) return 0;
            }

            // Output Directory
            while (true) {
                print_question("\nEnter output directory for results:\n");
                if (!out_dir.empty()) {
                    std::cout << "  "; print_default("Default: " + out_dir); std::cout << "\n";
                }
                std::cout << "  Example: C:\\path\\to\\Results\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (!input.empty()) out_dir = input;
                if (!out_dir.empty()) break;
                std::cout << "ERROR: Output directory cannot be empty!\n";
            }

            // Cost Modifiers - Ask if user wants to add additional cost modifiers
            bool add_cost_modifiers = false;
            while (true) {
                print_question("\nDo you want to add additional cost modifiers? (yes/no):\n");
                std::cout << "  Cost modifiers allow you to increase traversal costs for specific\n";
                std::cout << "  features such as rivers, restricted areas, or difficult terrain.\n";
                std::cout << "  "; print_default("Default: NO"); std::cout << "\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;

                if (input.empty() || input == "no" || input == "n" || input == "NO" || input == "No") {
                    add_cost_modifiers = false;
                    cost_modifiers_path = "";
                    cost_raster_path = "";
                    break;
                }
                else if (input == "yes" || input == "y" || input == "YES" || input == "Yes") {
                    add_cost_modifiers = true;
                    break;
                }
                else {
                    std::cout << "ERROR: Please enter 'yes' or 'no'\n";
                }
            }

            // If user wants to add cost modifiers, ask for vector file then raster
            if (add_cost_modifiers) {
                // --- Step 1: Ask for polylines vector file ---
                while (true) {
                    print_question("\nEnter path to cost modifiers vector file (" + supported_vector_formats() + "):\n");
                    if (!cost_modifiers_path.empty()) {
                        std::cout << "  "; print_default("Default: " + cost_modifiers_path); std::cout << "\n";
                    }
                    std::cout << "  Example: C:\\path\\to\\rivers.shp\n";
                    std::cout << "  Note: The file should contain polylines with a 'cost' field\n";
                    std::cout << "        containing float values (cost multipliers, e.g., 1.5, 2.0, etc.)\n";
                    std::cout << "        For .csv, geometry must be in a WKT column (geometry/wkt)\n";
                    std::cout << "  Leave blank to skip vector modifiers\n";
                    std::cout << "> ";
                    std::string input;
                    safe_getline(input);
                    if (check_exit_command(input)) return 0;
                    if (input.empty()) {
                        cost_modifiers_path = "";  // blank = skip
                        break;
                    }
                    if (!file_exists(input)) {
                        std::cout << "ERROR: File not found: " << input << "\n";
                        continue;
                    }
                    if (!is_supported_vector_format(input)) {
                        std::cout << "ERROR: Unsupported format. Use one of " << supported_vector_formats() << "\n";
                        continue;
                    }
                    cost_modifiers_path = input;
                    break;
                }

                // Ask for buffer radius only if shapefile was provided
                if (!cost_modifiers_path.empty()) {
                    print_question("\nSelect buffer radius (cells) for polyline rasterization:\n");
                    std::cout << "  The buffer ensures the algorithm doesn't 'jump' across features.\n";
                    std::cout << "  Each cell of buffer is applied on each side of the polyline.\n";
                    std::cout << "  0) No buffer\n";
                    std::cout << "  1) 1 cell per side\n";
                    std::cout << "  2) 2 cells per side (safer for 16-connectivity) "; print_default("[DEFAULT]"); std::cout << "\n";
                    std::cout << "  3) 3 cells per side\n";
                    std::cout << "  "; print_default("Leave blank for default (2)"); std::cout << "\n";
                    std::cout << "> ";
                    std::string buffer_input;
                    safe_getline(buffer_input);
                    if (check_exit_command(buffer_input)) return 0;

                    try {
                        int choice = std::stoi(buffer_input);
                        if (choice >= 0) polyline_buffer_radius = choice;
                    }
                    catch (...) {
                        polyline_buffer_radius = 2;
                    }
                    std::cout << "Polyline buffer set to " << polyline_buffer_radius << " cell(s) per side.\n";
                }

                // --- Step 2: Ask for raster (.tif) ---
                while (true) {
                    print_question("\nEnter path to cost modifiers raster (.tif):\n");
                    if (!cost_raster_path.empty()) {
                        std::cout << "  "; print_default("Default: " + cost_raster_path); std::cout << "\n";
                    }
                    std::cout << "  Example: C:\\path\\to\\landcover_costs.tif\n";
                    std::cout << "  Note: The raster should have the same dimensions as the DEM.\n";
                    std::cout << "        Cell values are cost multipliers (e.g., 1.0 = no change, 2.0 = double cost)\n";
                    std::cout << "  Leave blank to skip raster modifiers\n";
                    std::cout << "> ";
                    std::string input;
                    safe_getline(input);
                    if (check_exit_command(input)) return 0;
                    if (input.empty()) {
                        cost_raster_path = "";  // blank = skip
                        break;
                    }
                    if (!file_exists(input)) {
                        std::cout << "ERROR: File not found: " << input << "\n";
                        continue;
                    }
                    cost_raster_path = input;
                    break;
                }

                // If user left both blank, reset add_cost_modifiers
                if (cost_modifiers_path.empty() && cost_raster_path.empty()) {
                    std::cout << "No cost modifiers specified, continuing without modifiers.\n";
                }

                // --- Step 3: Barrier threshold ---
                if (!cost_modifiers_path.empty() || !cost_raster_path.empty()) {
                    print_question("\nTreat extreme cost multipliers as impassable barriers? Enter threshold:\n");
                    std::cout << "  Cells whose multiplier is >= the threshold become hard barriers\n";
                    std::cout << "  (excluded from movement, points on them are skipped).\n";
                    std::cout << "  Recommended when obstacles use very large multipliers (e.g. 999999):\n";
                    std::cout << "  without a threshold, every source floods the entire raster and the\n";
                    std::cout << "  computation slows down dramatically.\n";
                    std::cout << "  Enter 0 to disable (treat all multipliers as soft costs)\n";
                    std::cout << "  "; print_default("Leave blank for default (1000)"); std::cout << "\n";
                    std::cout << "> ";
                    std::string input;
                    safe_getline(input);
                    if (check_exit_command(input)) return 0;
                    try {
                        double v = std::stod(input);
                        barrier_threshold = (v > 0.0) ? v : 0.0;
                    }
                    catch (...) {
                        barrier_threshold = 1000.0;
                    }
                    if (barrier_threshold > 0.0) {
                        std::cout << "Barrier threshold: multipliers >= " << barrier_threshold
                                  << " are treated as impassable\n";
                    } else {
                        std::cout << "Barrier threshold disabled: all multipliers are soft costs\n";
                    }
                }
            }

            // Generate the sample points now that the output directory is
            // known. The layer is written to disk first and only then read
            // back, so it goes through the very same validation as an imported
            // file and stays on disk as a record of what the run consumed.
            if (generate_points) {
                std::cout << "\nGenerating sample points from the DEM...\n";
                auto gen_start = std::chrono::high_resolution_clock::now();
                PointGenResult gen_res = generate_sample_points(dem_path, out_dir, gen_opts);
                auto gen_time = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - gen_start).count();
                if (!gen_res.ok) {
                    std::cout << gen_res.error << "\n";
                    std::cout << "Please correct the parameters and try again.\n\n";
                    continue;
                }
                if (gen_res.count < 2) {
                    std::cout << "ERROR: The generated layer holds " << gen_res.count
                              << " point(s); FETE needs at least 2.\n";
                    std::cout << "       Use a smaller spacing or a larger target count.\n";
                    std::cout << "Please correct the parameters and try again.\n\n";
                    continue;
                }
                pts_path = gen_res.path;
                std::cout << "Generated " << gen_res.count << " points ("
                          << (gen_opts.random ? "stratified random" : "regular grid")
                          << ", one every " << gen_res.spacing << " cell(s)";
                if (gen_opts.random) std::cout << ", seed " << gen_opts.seed;
                if (gen_opts.edge_buffer > 0)
                    std::cout << ", " << gen_opts.edge_buffer << "-cell edge buffer";
                std::cout << ") in " << std::fixed << std::setprecision(3)
                          << gen_time << " sec\n";
                std::cout << "  Saved to: " << pts_path << "\n";
                if (gen_res.count > 200000) {
                    std::cout << "  WARNING: FETE cost grows with the square of the point "
                                 "count; " << gen_res.count << " points\n";
                    std::cout << "           will take a very long time to compute.\n";
                }
            }

            // Validate inputs BEFORE asking for filenames
            std::cout << "\nValidating inputs...\n";
            auto val_start = std::chrono::high_resolution_clock::now();
            ValidationResult val_result = validate_all_inputs(dem_path, pts_path);
            auto val_time = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - val_start).count();
            if (!val_result.is_valid) {
                std::cout << val_result.error_message << "\n";
                std::cout << "Please correct the paths and try again.\n\n";
                dem_path = "";
                pts_path = "";
                continue;
            }
            std::cout << "Validation successful! (" << std::fixed
                      << std::setprecision(3) << val_time << " sec)\n";

            // Parameters configuration.
            // Each prompt runs in its own loop: 'help' re-asks THIS question.
            // A bare `continue` here used to hit the outer configuration loop
            // and silently restart the whole setup from the thread count.
            while (true) {
                print_question("\nSelect number of neighbours for cost surface calculation:\n");
                std::cout << "  1) 8-connectivity (3x3 grid)\n";
                std::cout << "  2) 16-connectivity (knight moves) "; print_default("[DEFAULT]"); std::cout << "\n";
                std::cout << "  3) 24-connectivity (extended)\n";
                std::cout << "  4) 32-connectivity (more extended)\n";
                std::cout << "  5) 64-connectivity (full extended)\n";
                std::cout << "  "; print_default("Leave blank for default (16)"); std::cout << "\n";
                std::cout << "> ";
                std::string neighbours_input;
                safe_getline(neighbours_input);
                if (check_exit_command(neighbours_input)) return 0;
                if (check_help_command(neighbours_input)) continue;

                try {
                    int choice = std::stoi(neighbours_input);
                    switch (choice) {
                    case 1: num_neighbours = 8;  break;
                    case 2: num_neighbours = 16; break;
                    case 3: num_neighbours = 24; break;
                    case 4: num_neighbours = 32; break;
                    case 5: num_neighbours = 64; break;
                    default: num_neighbours = 16; break;
                    }
                }
                catch (...) {
                    num_neighbours = 16;
                }
                break;
            }

            while (true) {
                print_question("\nSelect cost function:\n");
                std::cout << "  1) Modified Tobler's Function (White 2015) "; print_default("[DEFAULT]"); std::cout << "\n";
                std::cout << "  2) Marquez-Perez et al. (2017)\n";
                std::cout << "  3) Irmischer and Clarke (2017)\n";
                std::cout << "  "; print_default("Leave blank for default"); std::cout << "\n";
                std::cout << "> ";
                std::string cf_input;
                safe_getline(cf_input);
                if (check_exit_command(cf_input)) return 0;
                if (check_help_command(cf_input)) continue;

                try {
                    int choice = std::stoi(cf_input);
                    if (choice == 2)      cost_function = MARQUEZ_PEREZ_ET_AL_2017;
                    else if (choice == 3) cost_function = IRMISCHER_CLARKE_2017;
                    else                  cost_function = TOBLER_WHITE_2015;
                }
                catch (...) {
                    cost_function = TOBLER_WHITE_2015;
                }
                break;
            }

            slope_in_degrees = (cost_function == TOBLER_WHITE_2015);

            while (true) {
                print_question("\nSelect buffer radius (cells) for path smoothing:\n");
                std::cout << "  0) No buffer "; print_default("[DEFAULT]"); std::cout << "\n";
                std::cout << "  1) 1 cell on each side\n";
                std::cout << "  2) 2 cells on each side\n";
                std::cout << "  3) 3 cells on each side\n";
                std::cout << "  "; print_default("Leave blank for default (0)"); std::cout << "\n";
                std::cout << "> ";
                std::string buffer_input;
                safe_getline(buffer_input);
                if (check_exit_command(buffer_input)) return 0;
                if (check_help_command(buffer_input)) continue;

                try {
                    int choice = std::stoi(buffer_input);
                    if (choice >= 0) buffer_radius = choice;
                }
                catch (...) {
                    buffer_radius = 0;
                }
                break;
            }

            // Output filenames
            while (true) {
                std::cout << "\nEnter slope raster filename (without extension):\n";
                std::cout << "  Example: slope\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (input.empty()) {
                    std::cout << "ERROR: Slope filename cannot be empty!\n";
                    continue;
                }
                if (!valid_output_filename(input)) { print_filename_error(); continue; }
                slope_filename = input;
                if (slope_filename.length() >= 4 && slope_filename.substr(slope_filename.length() - 4) == ".tif") {
                    slope_filename = slope_filename.substr(0, slope_filename.length() - 4);
                }
                break;
            }

            while (true) {
                std::cout << "\nEnter base cost surface raster filename (without extension):\n";
                std::cout << "  This is the cost surface calculated from slope * cost function\n";
                std::cout << "  Example: cost_surface\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (input.empty()) {
                    std::cout << "ERROR: Cost surface filename cannot be empty!\n";
                    continue;
                }
                if (!valid_output_filename(input)) { print_filename_error(); continue; }
                cost_filename = input;
                if (cost_filename.length() >= 4 && cost_filename.substr(cost_filename.length() - 4) == ".tif") {
                    cost_filename = cost_filename.substr(0, cost_filename.length() - 4);
                }
                break;
            }

            // If cost modifiers were added, ask for additional and total cost
            // surface filenames. Must match run_fete's own condition: it
            // writes both rasters whenever ANY modifier is active, so gating
            // on the vector path alone left the names empty and produced a
            // file literally called ".tif" in raster-only runs.
            if (!cost_modifiers_path.empty() || !cost_raster_path.empty()) {
                while (true) {
                    std::cout << "\nEnter additional cost surface raster filename (without extension):\n";
                    std::cout << "  This is the rasterized polylines with cost multipliers\n";
                    std::cout << "  Example: cost_surface_additional\n";
                    std::cout << "> ";
                    std::string input;
                    safe_getline(input);
                    if (check_exit_command(input)) return 0;
                    if (check_help_command(input)) continue;
                    if (input.empty()) {
                        std::cout << "ERROR: Additional cost surface filename cannot be empty!\n";
                        continue;
                    }
                    if (!valid_output_filename(input)) { print_filename_error(); continue; }
                    additional_cost_filename = input;
                    if (additional_cost_filename.length() >= 4 && additional_cost_filename.substr(additional_cost_filename.length() - 4) == ".tif") {
                        additional_cost_filename = additional_cost_filename.substr(0, additional_cost_filename.length() - 4);
                    }
                    break;
                }

                while (true) {
                    std::cout << "\nEnter total cost surface raster filename (without extension):\n";
                    std::cout << "  This is the final cost surface (base * additional)\n";
                    std::cout << "  Example: cost_surface_total\n";
                    std::cout << "> ";
                    std::string input;
                    safe_getline(input);
                    if (check_exit_command(input)) return 0;
                    if (check_help_command(input)) continue;
                    if (input.empty()) {
                        std::cout << "ERROR: Total cost surface filename cannot be empty!\n";
                        continue;
                    }
                    if (!valid_output_filename(input)) { print_filename_error(); continue; }
                    total_cost_filename = input;
                    if (total_cost_filename.length() >= 4 && total_cost_filename.substr(total_cost_filename.length() - 4) == ".tif") {
                        total_cost_filename = total_cost_filename.substr(0, total_cost_filename.length() - 4);
                    }
                    break;
                }
            }

            while (true) {
                std::cout << "\nEnter output density raster filename (without extension):\n";
                std::cout << "  Example: FETE_density\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (input.empty()) {
                    std::cout << "ERROR: Output density filename cannot be empty!\n";
                    continue;
                }
                if (!valid_output_filename(input)) { print_filename_error(); continue; }
                output_filename = input;
                if (output_filename.length() >= 4 && output_filename.substr(output_filename.length() - 4) == ".tif") {
                    output_filename = output_filename.substr(0, output_filename.length() - 4);
                }
                break;
            }

        }

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "Configuration:\n";
        std::cout << "  DEM: " << dem_path << "\n";
        std::cout << "  Points: " << pts_path
                  << (generate_points ? "  (generated from the DEM)" : "") << "\n";
        std::cout << "  Output dir: " << out_dir << "\n";
        if (!cost_modifiers_path.empty()) {
            std::cout << "  Cost modifiers: " << cost_modifiers_path << "\n";
            std::cout << "  Polyline buffer: " << polyline_buffer_radius << " cells per side\n";
        }
        if (!cost_modifiers_path.empty() || !cost_raster_path.empty()) {
            if (barrier_threshold > 0.0)
                std::cout << "  Barrier threshold: multipliers >= " << barrier_threshold << " impassable\n";
            else
                std::cout << "  Barrier threshold: disabled (soft costs)\n";
        }
        std::cout << "  Slope filename: " << slope_filename << ".tif\n";
        std::cout << "  Base cost filename: " << cost_filename << ".tif\n";
        if (!cost_modifiers_path.empty() || !cost_raster_path.empty()) {
            std::cout << "  Additional cost filename: " << additional_cost_filename << ".tif\n";
            std::cout << "  Total cost filename: " << total_cost_filename << ".tif\n";
        }
        std::cout << "  Density filename: " << output_filename << ".tif\n";
        std::cout << "  Buffer radius: " << buffer_radius << " cells\n";
        std::cout << "  Neighbours: " << num_neighbours << "-connectivity\n";
        std::cout << "  Slope units: " << (slope_in_degrees ? "degrees" : "percentage") << "\n";
        std::cout << "  Cost function: " << (cost_function == TOBLER_WHITE_2015 ? "Modified Tobler (White 2015)" : (cost_function == MARQUEZ_PEREZ_ET_AL_2017 ? "Marquez-Perez et al. (2017)" : "Irmischer and Clarke (2017)")) << "\n";
        std::cout << "  Max threads: " << max_threads << "\n";
        std::cout << "  Max RAM: " << max_ram_mb << " MB\n";
        std::cout << std::string(70, '=') << "\n\n";

        FETEOutput result = run_fete(dem_path, pts_path, out_dir, slope_filename, cost_filename, output_filename,
            buffer_radius, max_threads, max_ram_mb, num_neighbours, slope_in_degrees, cost_function,
            cost_modifiers_path, polyline_buffer_radius, cost_raster_path,
            additional_cost_filename, total_cost_filename, barrier_threshold);

        if (result.success) {
            // Save config
            Config to_save = { dem_path, pts_path, out_dir, cost_modifiers_path, cost_raster_path };
            save_config(to_save);

            // Print success message in green
            print_green_success("FETE successfully computed!\n");
            std::cout << "\nOutput Summary:\n";
            std::cout << "  Total time: " << std::fixed << std::setprecision(2) << result.time_seconds << " sec\n";
            std::cout << "  Max density: " << result.max_density << "\n";
            std::cout << "  Min density: " << result.min_density << "\n";
            std::cout << "  Avg density: " << result.avg_density << "\n";
            std::cout << "  Non-zero cells: " << result.nonzero_cells << "/" << result.total_cells << "\n";
            std::cout << "\nOutput Files:\n";
            if (generate_points) {
                std::cout << "  - " << pts_path << " (generated sample points)\n";
            }
            std::cout << "  - " << result.slope_path << "\n";
            std::cout << "  - " << result.cost_path << " (base cost surface)\n";
            if (!result.additional_cost_path.empty()) {
                std::cout << "  - " << result.additional_cost_path << " (additional cost multipliers)\n";
                std::cout << "  - " << result.total_cost_path << " (total cost surface)\n";
            }
            std::cout << "  - " << result.density_path << "\n";
        }

        print_question("\nRun another FETE computation? (yes/no)\n"); std::cout << "> ";
        std::string again;
        safe_getline(again);
        if (check_exit_command(again)) return 0;

        if (again == "no" || again == "n") {
            print_question("\nExit program? (yes/no)\n"); std::cout << "> ";
            std::string exit_choice;
            safe_getline(exit_choice);
            if (check_exit_command(exit_choice)) return 0;

            if (exit_choice == "yes" || exit_choice == "y") {
                std::cout << "\nGoodbye!\n\n";
                break;
            }
        }
    }

    return 0;
}
