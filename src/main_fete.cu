#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <filesystem>
#include <cmath>
#include <cstdint>
#include <queue>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cctype>
#include <omp.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "gdal_priv.h"
#include "ogrsf_frmts.h"

namespace fs = std::filesystem;

// ========== GLOBAL SETTINGS ==========
bool g_verbose_mode = false;  // Global flag for verbose/debug output

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
            TRAJECTA - A SPATIAL MOVEMENT ANALYSIS SOFTWARE
                     Developed by Stefano Apra
              Institute for the Study of the Ancient World

                      From Everywhere to Everywhere

FETE (From Everywhere to Everywhere) is a least-cost modeling tool that
computes anisotropic shortest paths across Digital Elevation Models (DEM). The
model was originally theorized by Devin A. White and Sarah B. Barber (2012).

REFERENCE:
  White, D. A., & Barber, S. B. (2012). Geospatial modeling of pedestrian
  transportation networks for emergency response. Transactions in GIS, 16(2),
  147-166. https://doi.org/10.1016/j.jas.2012.04.017

===============================================================================

MODES:
  1. FETE (From Everywhere to Everywhere) [DEFAULT]
     Computes cost-distance density from ALL sample points to the entire DEM.
     Useful for understanding general movement patterns and accessibility.
     Output: Density raster showing accumulated path usage.

  2. LCPA (Least-Cost Path Analysis) [DEVELOPMENT]
     Computes optimal paths from a single origin to one or more destinations.
     Useful for finding specific routes with lowest traversal cost.
     Output: Path raster(s) showing optimal route(s).

COST FUNCTIONS:
  1. Modified Tobler's Walking Function (White 2015) [DEFAULT]
     - Based on Tobler (1993) empirical walking speed model
     - Accounts for slope to compute travel cost
     - Formula: cost = distance / (6 * exp(-3.5 * |slope + 0.05|))
     - Converts speed (km/h) to travel time cost
     - Ensures: LOW slope = LOW cost, HIGH slope = HIGH cost

INPUT REQUIREMENTS:
  - DEM: GeoTIFF file (.tif), must be georeferenced
  - Points: ESRI Shapefile (.shp), point geometry
  - CRS: DEM and Points MUST have the same coordinate system (user responsibility)
  - Bounds: All points must be contained within DEM extent

PARAMETERS:
  - Neighbours: Connectivity for calculating cumulative cost surface (8, 16, 24, 32, 64)
  - Slope Units: Degrees or Percentage
  - Buffer Radius: Cells around path for density calculation
  - CPU Threads: Parallel processing threads (configured once at startup)
  - Max RAM: Memory limit for raster processing

OUTPUT:
  - slope_[name].tif: Terrain slope raster
  - cost_surface_[name].tif: Cost surface raster
  - density_[name].tif: (FETE) Accumulated path density
  - path_[name].tif: (LCPA - future) Optimal path raster

===============================================================================
)";

// ========== STRUCTURES ==========
struct Config {
    std::string dem_path;
    std::string pts_path;
    std::string out_dir;
    std::string cost_modifiers_path;  // Path to shapefile with cost modifiers
};

struct ConfigLCPA {
    std::string dem_path;
    std::string origin_path;
    std::string destinations_path;
    std::string out_dir;
    std::string cost_modifiers_path;  // Path to shapefile with cost modifiers
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
    uint32_t max_density;
    uint32_t min_density;
    uint32_t avg_density;
    int nonzero_cells;
    int total_cells;
    double time_seconds;
    bool was_cancelled;
};

struct Off { int dr; int dc; };

// ========== GPU STRUCTURES ==========
struct GPUInfo {
    bool available;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory_mb;
    size_t free_memory_mb;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int clock_rate_mhz;
};

struct MemoryRequirement {
    size_t dem_mb;
    size_t cost_surface_mb;
    size_t graph_csr_mb;
    size_t working_memory_mb;
    size_t safety_margin_mb;
    size_t total_mb;
};

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

bool check_exit_command(const std::string& input) {
    if (input == "exit" || input == "EXIT" || input == "Exit") {
        std::cout << "\nAre you sure you want to exit? (yes/no)\n> ";
        std::string confirm;
        std::getline(std::cin, confirm);
        if (confirm == "yes" || confirm == "YES" || confirm == "Yes" || confirm == "y" || confirm == "Y") {
            std::cout << "\nGoodbye!\n\n";
            exit(0);
        }
        return false;
    }
    return false;
}

void center_text(const std::string& text, int width = 70) {
    int padding = (width - text.length()) / 2;
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
#ifdef _WIN32
    system("color 0A");
    std::cout << success << std::flush;
    system("color 0F");
    std::cout << std::flush;
#else
    std::cout << "\033[32m" << success << "\033[0m" << std::flush;
#endif
}

void print_progress(int current, int total, int bar_width = 45) {
    double percentage = (double)current / total;
    int filled = (int)(percentage * bar_width);

    std::cout << "\r[" << current << "/" << total << "] ";
    std::cout << std::string(filled, 219);
    std::cout << std::string(bar_width - filled, 177);
    std::cout << " " << (int)(percentage * 100) << "%";
    std::cout.flush();
}

void save_config(const Config& cfg) {
    std::ofstream file("fete_config.txt");
    file << cfg.dem_path << "\n";
    file << cfg.pts_path << "\n";
    file << cfg.out_dir << "\n";
    file << cfg.cost_modifiers_path << "\n";
    file.close();
}

void save_config_lcpa(const ConfigLCPA& cfg) {
    std::ofstream file("lcpa_config.txt");
    file << cfg.dem_path << "\n";
    file << cfg.origin_path << "\n";
    file << cfg.destinations_path << "\n";
    file << cfg.out_dir << "\n";
    file << cfg.cost_modifiers_path << "\n";
    file.close();
}

Config load_config() {
    Config cfg = { "", "", "", "" };
    std::ifstream file("fete_config.txt");
    if (file.is_open()) {
        std::getline(file, cfg.dem_path);
        std::getline(file, cfg.pts_path);
        std::getline(file, cfg.out_dir);
        std::getline(file, cfg.cost_modifiers_path);
        file.close();
    }
    return cfg;
}

ConfigLCPA load_config_lcpa() {
    ConfigLCPA cfg = { "", "", "", "", "" };
    std::ifstream file("lcpa_config.txt");
    if (file.is_open()) {
        std::getline(file, cfg.dem_path);
        std::getline(file, cfg.origin_path);
        std::getline(file, cfg.destinations_path);
        std::getline(file, cfg.out_dir);
        std::getline(file, cfg.cost_modifiers_path);
        file.close();
    }
    return cfg;
}

// ========== GPU FUNCTIONS ==========

GPUInfo get_gpu_info() {
    GPUInfo info;
    info.available = false;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        return info;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    info.available = true;
    info.name = std::string(prop.name);
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;

    // Clock rate field name changed in CUDA versions
    // For CUDA 13.1+, we just set a default value since the exact field name varies
    info.clock_rate_mhz = 0;  // Will be shown as 0 MHz (not critical for memory checks)

    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    info.free_memory_mb = free_bytes / (1024 * 1024);

    return info;
}

void print_gpu_info(const GPUInfo& info) {
    if (!info.available) {
        std::cout << "GPU Status: NOT AVAILABLE\n";
        std::cout << "  No NVIDIA CUDA-capable GPU detected\n";
        return;
    }

    std::cout << "GPU Information:\n";
    std::cout << "  Device: " << info.name << "\n";
    std::cout << "  Compute Capability: " << info.compute_capability_major
        << "." << info.compute_capability_minor << "\n";
    std::cout << "  Total Memory: " << info.total_memory_mb << " MB (~"
        << (info.total_memory_mb / 1024) << " GB)\n";
    std::cout << "  Free Memory: " << info.free_memory_mb << " MB (~"
        << (info.free_memory_mb / 1024) << " GB)\n";
    std::cout << "  Multiprocessors: " << info.multiprocessor_count << "\n";
    std::cout << "  Max threads/block: " << info.max_threads_per_block << "\n";
    std::cout << "  Clock rate: " << info.clock_rate_mhz << " MHz\n";
}

MemoryRequirement calculate_memory_requirement(int nrows, int ncols, int num_neighbours) {
    MemoryRequirement req;

    int64_t N = (int64_t)nrows * ncols;
    double avg_edges_per_node = num_neighbours * 0.97;  // ~3% bordi hanno meno vicini
    int64_t E = (int64_t)(N * avg_edges_per_node);

    req.dem_mb = (N * sizeof(float)) / (1024 * 1024);
    req.cost_surface_mb = (N * sizeof(float)) / (1024 * 1024);

    // Grafo CSR
    size_t row_offsets_bytes = (N + 1) * sizeof(int);
    size_t column_indices_bytes = E * sizeof(int);
    size_t edge_weights_bytes = E * sizeof(float);
    req.graph_csr_mb = (row_offsets_bytes + column_indices_bytes + edge_weights_bytes) / (1024 * 1024);

    // Working memory (dist, pred, density, buffers)
    req.working_memory_mb = (N * sizeof(float) +  // dist
        N * sizeof(int) +     // pred
        N * sizeof(uint32_t) +  // density
        100 * 1024 * 1024) / (1024 * 1024);  // buffers

    req.safety_margin_mb = 200;  // 200 MB margin

    req.total_mb = req.dem_mb + req.cost_surface_mb + req.graph_csr_mb +
        req.working_memory_mb + req.safety_margin_mb;

    return req;
}

bool check_gpu_memory_sufficient(const GPUInfo& gpu_info, const MemoryRequirement& req) {
    return req.total_mb <= gpu_info.free_memory_mb;
}

void print_memory_requirement(const MemoryRequirement& req) {
    std::cout << "\nMemory Requirement Analysis:\n";
    std::cout << "  DEM:                    " << req.dem_mb << " MB\n";
    std::cout << "  Cost Surface:           " << req.cost_surface_mb << " MB\n";
    std::cout << "  Graph CSR:              " << req.graph_csr_mb << " MB\n";
    std::cout << "  Working Memory:         " << req.working_memory_mb << " MB\n";
    std::cout << "  Safety Margin:          " << req.safety_margin_mb << " MB\n";
    std::cout << "  " << std::string(40, '-') << "\n";
    std::cout << "  TOTAL REQUIRED:         " << req.total_mb << " MB (~"
        << (req.total_mb / 1024.0) << " GB)\n";
}

void print_memory_error_diagnostic(const GPUInfo& gpu_info, const MemoryRequirement& req,
    int nrows, int ncols, int num_neighbours) {
    std::cout << "\n" << std::string(70, '!') << "\n";
    std::cout << "ERROR: INSUFFICIENT GPU MEMORY\n";
    std::cout << std::string(70, '!') << "\n\n";

    std::cout << "Your DEM is TOO LARGE for the available GPU memory!\n\n";

    std::cout << "Current Configuration:\n";
    std::cout << "  DEM Size: " << nrows << " x " << ncols << " = "
        << ((int64_t)nrows * ncols / 1000000) << "M cells\n";
    std::cout << "  Connectivity: " << num_neighbours << "-neighbours\n";
    std::cout << "  GPU: " << gpu_info.name << "\n";
    std::cout << "  GPU Memory Available: " << gpu_info.free_memory_mb << " MB (~"
        << (gpu_info.free_memory_mb / 1024.0) << " GB)\n\n";

    print_memory_requirement(req);

    std::cout << "\nMemory Deficit: " << (req.total_mb - gpu_info.free_memory_mb)
        << " MB\n\n";

    std::cout << "SOLUTIONS:\n\n";

    // Calculate max DEM size for current GPU
    double scale_factor = sqrt((double)gpu_info.free_memory_mb / req.total_mb);
    int max_dim = (int)(std::min(nrows, ncols) * scale_factor * 0.95);  // 5% safety

    std::cout << "1) REDUCE DEM SIZE\n";
    std::cout << "   Current: " << nrows << "x" << ncols << "\n";
    std::cout << "   Maximum for your GPU: ~" << max_dim << "x" << max_dim << "\n";
    std::cout << "   Action: Resample/clip your DEM to smaller dimensions\n\n";

    if (num_neighbours > 8) {
        MemoryRequirement req_8 = calculate_memory_requirement(nrows, ncols, 8);
        std::cout << "2) REDUCE CONNECTIVITY\n";
        std::cout << "   Current: " << num_neighbours << "-connectivity (" << req.total_mb << " MB)\n";
        std::cout << "   With 8-connectivity: " << req_8.total_mb << " MB\n";
        if (req_8.total_mb <= gpu_info.free_memory_mb) {
            std::cout << "   ✅ This WOULD FIT! Consider using 8-connectivity\n\n";
        }
        else {
            std::cout << "   ❌ Still wouldn't fit. Also reduce DEM size.\n\n";
        }
    }

    std::cout << "3) USE CPU MODE\n";
    std::cout << "   The CPU version has no memory limits (uses RAM)\n";
    std::cout << "   It will be slower but can handle any DEM size\n\n";

    std::cout << "4) USE A LARGER GPU\n";
    std::cout << "   Recommended GPUs for this DEM:\n";
    if (req.total_mb <= 8192) {
        std::cout << "   - RTX 3070 Ti / RTX 4070 (8 GB)\n";
    }
    if (req.total_mb <= 16384) {
        std::cout << "   - RTX 4080 / V100-16GB (16 GB)\n";
    }
    if (req.total_mb <= 24576) {
        std::cout << "   - RTX 4090 (24 GB)\n";
    }
    if (req.total_mb > 24576) {
        std::cout << "   - V100-32GB (32 GB)\n";
        std::cout << "   - A100-40GB/80GB (40/80 GB)\n";
    }

    std::cout << "\n" << std::string(70, '!') << "\n";
}

std::string get_file_extension(const std::string& path) {
    size_t pos = path.find_last_of(".");
    if (pos == std::string::npos) return "";
    return path.substr(pos);
}

bool file_exists(const std::string& path) {
    return fs::exists(path);
}

// ========== VALIDATION FUNCTIONS ==========

ValidationResult validate_dem(const std::string& dem_path) {
    if (!file_exists(dem_path)) {
        return { false, "ERROR: DEM file not found: " + dem_path };
    }
    std::string ext = get_file_extension(dem_path);
    if (ext != ".tif" && ext != ".TIF") {
        return { false, "ERROR: DEM must be GeoTIFF (.tif), found: " + ext };
    }
    GDALDataset* ds = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
    if (!ds) {
        return { false, "ERROR: Cannot open DEM file with GDAL" };
    }
    double gt[6];
    ds->GetGeoTransform(gt);
    if (gt[1] == 0.0 && gt[5] == 0.0) {
        GDALClose(ds);
        return { false, "ERROR: DEM is not georeferenced (no valid geotransform)" };
    }
    if (ds->GetProjectionRef() == nullptr || std::string(ds->GetProjectionRef()).empty()) {
        GDALClose(ds);
        return { false, "ERROR: DEM has no coordinate reference system (CRS)" };
    }
    GDALClose(ds);
    return { true, "" };
}

ValidationResult validate_points(const std::string& pts_path) {
    if (!file_exists(pts_path)) {
        return { false, "ERROR: Points shapefile not found: " + pts_path };
    }
    std::string ext = get_file_extension(pts_path);
    if (ext != ".shp" && ext != ".SHP") {
        return { false, "ERROR: Points must be shapefile (.shp), found: " + ext };
    }
    GDALDataset* ds = (GDALDataset*)GDALOpenEx(pts_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (!ds) {
        return { false, "ERROR: Cannot open points shapefile with OGR" };
    }
    OGRLayer* layer = ds->GetLayer(0);
    if (!layer || layer->GetFeatureCount() == 0) {
        GDALClose(ds);
        return { false, "ERROR: Points shapefile is empty or invalid" };
    }
    GDALClose(ds);
    return { true, "" };
}

ValidationResult validate_points_bounds(const std::string& dem_path, const std::string& pts_path) {
    GDALDataset* dem_ds = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
    GDALDataset* pts_ds = (GDALDataset*)GDALOpenEx(pts_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (!dem_ds || !pts_ds) {
        if (dem_ds) GDALClose(dem_ds);
        if (pts_ds) GDALClose(pts_ds);
        return { false, "ERROR: Cannot open files for bounds validation" };
    }
    int ncols = dem_ds->GetRasterXSize();
    int nrows = dem_ds->GetRasterYSize();
    double gt[6];
    dem_ds->GetGeoTransform(gt);
    double dem_xmin = gt[0];
    double dem_xmax = gt[0] + ncols * gt[1];
    double dem_ymax = gt[3];
    double dem_ymin = gt[3] + nrows * gt[5];
    OGRLayer* layer = pts_ds->GetLayer(0);
    int total_points = 0;
    int outside_points = 0;
    layer->ResetReading();
    OGRFeature* feat = nullptr;
    while ((feat = layer->GetNextFeature()) != nullptr) {
        total_points++;
        OGRGeometry* geom = feat->GetGeometryRef();
        if (geom && wkbFlatten(geom->getGeometryType()) == wkbPoint) {
            OGRPoint* p = geom->toPoint();
            if (p->getX() < dem_xmin || p->getX() > dem_xmax || p->getY() < dem_ymin || p->getY() > dem_ymax) {
                outside_points++;
            }
        }
        OGRFeature::DestroyFeature(feat);
    }
    GDALClose(dem_ds);
    GDALClose(pts_ds);
    if (outside_points > 0) {
        if (outside_points == total_points) {
            return { false, "ERROR: ALL points are outside DEM extent" };
        }
        else {
            return { false, "ERROR: " + std::to_string(outside_points) + " of " + std::to_string(total_points) + " points are outside DEM extent" };
        }
    }
    return { true, "" };
}

ValidationResult validate_all_inputs(const std::string& dem_path, const std::string& pts_path) {
    ValidationResult res = validate_dem(dem_path);
    if (!res.is_valid) return res;
    res = validate_points(pts_path);
    if (!res.is_valid) return res;
    res = validate_points_bounds(dem_path, pts_path);
    if (!res.is_valid) return res;
    return { true, "" };
}

void print_help() {
    std::cout << HELP_TEXT;
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
enum CostFunctionType { TOBLER_WHITE_2015 = 1 };

static inline float tobler_white_2015(double dh_m, double dz_m) {
    const double sf = dz_m / dh_m;
    const double speed_kmh = 6.0 * std::exp(-3.5 * std::abs(sf + 0.05));
    const double safe_speed = std::max(speed_kmh, 1e-12);
    return (float)((dh_m / 1000.0) / safe_speed);
}

static inline float apply_cost_function(CostFunctionType cf, double dh_m, double dz_m) {
    return tobler_white_2015(dh_m, dz_m);
}

static inline bool world_to_pixel_northup(double x, double y, const double gt[6], int& col, int& row) {
    if (std::abs(gt[2]) > 1e-12 || std::abs(gt[4]) > 1e-12) return false;
    col = (int)std::floor((x - gt[0]) / gt[1]);
    row = (int)std::floor((y - gt[3]) / gt[5]);
    return true;
}

// ========== FORWARD DECLARATIONS ==========

// GPU version (implemented in main_fete_gpu.cu)
extern FETEOutput run_fete_gpu(const std::string& dem_path, const std::string& pts_path, const std::string& out_dir,
    const std::string& slope_filename, const std::string& cost_filename,
    const std::string& output_filename, int buffer_radius, int max_threads, int64_t max_ram_mb,
    int num_neighbours, bool slope_in_degrees, CostFunctionType cost_function, int num_streams);

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
 * @return Vector of float cost multipliers (1.0 = no modifier, >1.0 = increased cost)
 */
std::vector<float> rasterize_polylines_with_costs(
    const std::string& polylines_path,
    int nrows, int ncols,
    const double gt[6],
    int buffer_cells,
    int max_threads) {

    int N = nrows * ncols;
    std::vector<float> cost_raster(N, 1.0f);  // Initialize all cells to 1.0 (no modifier)

    std::cout << "Reading polylines from shapefile...\n";

    // Open shapefile
    GDALDataset* polylines_ds = (GDALDataset*)GDALOpenEx(polylines_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (!polylines_ds) {
        std::cout << "ERROR: Cannot open polylines shapefile: " << polylines_path << "\n";
        return cost_raster;
    }

    OGRLayer* layer = polylines_ds->GetLayer(0);
    if (!layer) {
        std::cout << "ERROR: Cannot read layer from shapefile\n";
        GDALClose(polylines_ds);
        return cost_raster;
    }

    int feature_count = layer->GetFeatureCount();
    info_print("Found " + std::to_string(feature_count) + " polyline features\n");

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
        std::cout << "ERROR: 'cost' field not found in shapefile!\n";
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

    while ((feat = layer->GetNextFeature()) != nullptr) {
        processed++;
        print_progress(processed, feature_count);

        // Get cost multiplier from 'cost' field
        float cost_multiplier = feat->GetFieldAsDouble(cost_field_idx);
        if (cost_multiplier <= 0.0f) {
            cost_multiplier = 1.0f;  // Default to no modifier if invalid
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
            OGRLineString* line = geom->toLineString();
            int num_points = line->getNumPoints();

            // Rasterize line by iterating through line segments
            for (int i = 0; i < num_points - 1; ++i) {
                double x1 = line->getX(i);
                double y1 = line->getY(i);
                double x2 = line->getX(i + 1);
                double y2 = line->getY(i + 1);

                total_points_processed += 2;

                // Debug: Print first point coordinates
                if (processed == 1 && i == 0) {
                    debug_print("First segment coords: (" + std::to_string(x1) + ", " + std::to_string(y1) +
                               ") -> (" + std::to_string(x2) + ", " + std::to_string(y2) + ")\n");
                }

                // Convert to pixel coordinates
                int col1, row1, col2, row2;
                if (!world_to_pixel_northup(x1, y1, gt, col1, row1)) {
                    if (processed == 1 && i == 0) {
                        debug_print("WARNING: First point conversion failed!\n");
                    }
                    continue;
                }
                if (!world_to_pixel_northup(x2, y2, gt, col2, row2)) {
                    if (processed == 1 && i == 0) {
                        debug_print("WARNING: Second point conversion failed!\n");
                    }
                    continue;
                }

                // Check if points are inside raster bounds
                if (row1 >= 0 && row1 < nrows && col1 >= 0 && col1 < ncols) {
                    points_inside_bounds++;
                    feature_had_valid_points = true;
                }
                if (row2 >= 0 && row2 < nrows && col2 >= 0 && col2 < ncols) {
                    points_inside_bounds++;
                    feature_had_valid_points = true;
                }

                // Debug: Print first converted pixel coordinates
                if (processed == 1 && i == 0) {
                    debug_print("Converted to pixels: (" + std::to_string(col1) + ", " + std::to_string(row1) +
                               ") -> (" + std::to_string(col2) + ", " + std::to_string(row2) + ")\n");
                    debug_print("Raster size: " + std::to_string(ncols) + "x" + std::to_string(nrows) + "\n");
                }

                // Bresenham's line algorithm to rasterize line segment
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
                                // Take maximum cost if multiple features overlap
                                cost_raster[cell_idx] = std::max(cost_raster[cell_idx], cost_multiplier);
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
        }
        else if (geom_type == wkbMultiLineString) {
            OGRMultiLineString* multiline = geom->toMultiLineString();
            for (int j = 0; j < multiline->getNumGeometries(); ++j) {
                OGRLineString* line = (OGRLineString*)multiline->getGeometryRef(j);
                int num_points = line->getNumPoints();

                for (int i = 0; i < num_points - 1; ++i) {
                    double x1 = line->getX(i);
                    double y1 = line->getY(i);
                    double x2 = line->getX(i + 1);
                    double y2 = line->getY(i + 1);

                    int col1, row1, col2, row2;
                    if (!world_to_pixel_northup(x1, y1, gt, col1, row1)) continue;
                    if (!world_to_pixel_northup(x2, y2, gt, col2, row2)) continue;

                    int dx = std::abs(col2 - col1);
                    int dy = std::abs(row2 - row1);
                    int sx = (col1 < col2) ? 1 : -1;
                    int sy = (row1 < row2) ? 1 : -1;
                    int err = dx - dy;

                    int col = col1;
                    int row = row1;

                    while (true) {
                        for (int br = -buffer_cells; br <= buffer_cells; ++br) {
                            for (int bc = -buffer_cells; bc <= buffer_cells; ++bc) {
                                int r = row + br;
                                int c = col + bc;

                                if (r >= 0 && r < nrows && c >= 0 && c < ncols) {
                                    int cell_idx = idx(r, c, ncols);
                                    cost_raster[cell_idx] = std::max(cost_raster[cell_idx], cost_multiplier);
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
            }
        }

        if (feature_had_valid_points) {
            features_with_valid_points++;
        }

        OGRFeature::DestroyFeature(feat);
    }

    // Count cells with cost modifiers
    int modified_cells = 0;
    for (int i = 0; i < N; ++i) {
        if (cost_raster[i] > 1.0f) {
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
        info_print("  2. All 'cost' values are <= 1.0\n");
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
    const std::string& additional_cost_filename = "", const std::string& total_cost_filename = "") {

    FETEOutput output = { false, "", "", "", "", "", 0, 0, 0, 0, 0, 0.0, false };
    auto global_start = std::chrono::high_resolution_clock::now();

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
    N = nrows * ncols;
    dem_ds->GetGeoTransform(gt);
    wkt = dem_ds->GetProjectionRef();

    dem.resize(N);
    band->RasterIO(GF_Read, 0, 0, ncols, nrows, dem.data(), ncols, nrows, GDT_Float32, 0, 0);

    std::cout << "DEM read: " << nrows << "x" << ncols << " (" << N << " cells)\n";
    auto step1_end = std::chrono::high_resolution_clock::now();
    auto step1_time = std::chrono::duration<double>(step1_end - step1_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << step1_time << " sec\n";

    int64_t estimated_ram = (int64_t)N * sizeof(float) / (1024 * 1024);
    if (estimated_ram > max_ram_mb) {
        std::cout << "ERROR: DEM requires ~" << estimated_ram << " MB, but max allowed is " << max_ram_mb << " MB\n";
        GDALClose(dem_ds);
        return output;
    }

    std::cout << "\nReading sample points...\n";
    auto step2_start = std::chrono::high_resolution_clock::now();

    std::vector<int> point_nodes;
    int P = 0;
    GDALDataset* pts_ds = (GDALDataset*)GDALOpenEx(pts_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (!pts_ds) {
        std::cout << "ERROR: Cannot open points file: " << pts_path << "\n";
        GDALClose(dem_ds);
        return output;
    }

    OGRLayer* layer = pts_ds->GetLayer(0);
    layer->ResetReading();
    OGRFeature* feat = nullptr;
    while ((feat = layer->GetNextFeature()) != nullptr) {
        OGRGeometry* geom = feat->GetGeometryRef();
        if (geom && wkbFlatten(geom->getGeometryType()) == wkbPoint) {
            OGRPoint* p = geom->toPoint();
            int col, row;
            if (world_to_pixel_northup(p->getX(), p->getY(), gt, col, row)) {
                if (row >= 0 && row < nrows && col >= 0 && col < ncols) {
                    point_nodes.push_back(idx(row, col, ncols));
                }
            }
        }
        OGRFeature::DestroyFeature(feat);
    }

    P = (int)point_nodes.size();
    std::cout << "Points read: " << P << "\n";
    auto step2_end = std::chrono::high_resolution_clock::now();
    auto step2_time = std::chrono::duration<double>(step2_end - step2_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << step2_time << " sec\n";

    std::cout << "\nCalculating slope (" << (slope_in_degrees ? "degrees" : "percentage") << ")...\n";
    auto step2a_start = std::chrono::high_resolution_clock::now();

    const double res_x = gt[1];
    const double res_y = std::abs(gt[5]);

    GDALDriver* gtiff_drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    std::vector<float> slope_data(N, 0.0f);

#pragma omp parallel for collapse(2) num_threads(max_threads)
    for (int r = 1; r < nrows - 1; ++r) {
        for (int c = 1; c < ncols - 1; ++c) {
            int center = idx(r, c, ncols);
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

    std::string slope_path = join_path(out_dir, slope_filename + ".tif");
    GDALDataset* slope_ds = gtiff_drv->Create(slope_path.c_str(), ncols, nrows, 1, GDT_Float32, nullptr);
    slope_ds->SetGeoTransform(gt);
    slope_ds->SetProjection(wkt);
    slope_ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows, slope_data.data(), ncols, nrows, GDT_Float32, 0, 0);
    GDALClose(slope_ds);

    std::cout << "Slope saved\n";
    auto step2a_end = std::chrono::high_resolution_clock::now();
    auto step2a_time = std::chrono::duration<double>(step2a_end - step2a_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << step2a_time << " sec\n";

    std::cout << "\nCalculating cost surface (" << num_neighbours << "-connectivity)...\n";
    auto step2b_start = std::chrono::high_resolution_clock::now();

    std::vector<float> cost_surface(N, 0.0f);

#pragma omp parallel for collapse(2) num_threads(max_threads)
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            int from_idx = idx(r, c, ncols);
            float z_from = dem[from_idx];
            float total_cost = 0.0f;
            int valid_neighbors = 0;

            for (int k = 0; k < num_offs; ++k) {
                int nr = r + current_offs[k].dr;
                int nc = c + current_offs[k].dc;
                if (nr < 0 || nr >= nrows || nc < 0 || nc >= ncols) continue;

                int to_idx = idx(nr, nc, ncols);
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

    std::string cost_path = join_path(out_dir, cost_filename + ".tif");
    GDALDataset* cost_ds = gtiff_drv->Create(cost_path.c_str(), ncols, nrows, 1, GDT_Float32, nullptr);
    cost_ds->SetGeoTransform(gt);
    cost_ds->SetProjection(wkt);
    cost_ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows, cost_surface.data(), ncols, nrows, GDT_Float32, 0, 0);
    GDALClose(cost_ds);

    std::cout << "Base cost surface saved\n";
    auto step2b_end = std::chrono::high_resolution_clock::now();
    auto step2b_time = std::chrono::duration<double>(step2b_end - step2b_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << step2b_time << " sec\n";

    // ========== APPLY COST MODIFIERS IF SPECIFIED ==========
    std::vector<float> cost_multipliers(N, 1.0f);  // Default: no modifiers (all 1.0)
    std::string additional_cost_path = "";
    std::string total_cost_path = "";

    if (!cost_modifiers_path.empty()) {
        std::cout << "\n" << std::string(70, '-') << "\n";
        std::cout << "Applying cost modifiers from polylines...\n";
        std::cout << std::string(70, '-') << "\n";

        auto step2c_start = std::chrono::high_resolution_clock::now();

        // Rasterize polylines with cost multipliers
        cost_multipliers = rasterize_polylines_with_costs(
            cost_modifiers_path, nrows, ncols, gt, polyline_buffer_radius, max_threads);

        // Save additional cost surface (this is the multipliers raster)
        additional_cost_path = join_path(out_dir, additional_cost_filename + ".tif");
        GDALDataset* additional_cost_ds = gtiff_drv->Create(additional_cost_path.c_str(), ncols, nrows, 1, GDT_Float32, nullptr);
        additional_cost_ds->SetGeoTransform(gt);
        additional_cost_ds->SetProjection(wkt);
        additional_cost_ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows, cost_multipliers.data(), ncols, nrows, GDT_Float32, 0, 0);
        GDALClose(additional_cost_ds);
        std::cout << "Additional cost surface saved: " << additional_cost_path << "\n";

        // Multiply base cost surface by cost multipliers to get total cost surface
        std::cout << "Calculating total cost surface (base * multipliers)...\n";
        std::vector<float> total_cost_surface(N);

#pragma omp parallel for num_threads(max_threads)
        for (int i = 0; i < N; ++i) {
            total_cost_surface[i] = cost_surface[i] * cost_multipliers[i];
        }

        // Save total cost surface
        total_cost_path = join_path(out_dir, total_cost_filename + ".tif");
        GDALDataset* total_cost_ds = gtiff_drv->Create(total_cost_path.c_str(), ncols, nrows, 1, GDT_Float32, nullptr);
        total_cost_ds->SetGeoTransform(gt);
        total_cost_ds->SetProjection(wkt);
        total_cost_ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows, total_cost_surface.data(), ncols, nrows, GDT_Float32, 0, 0);
        GDALClose(total_cost_ds);
        std::cout << "Total cost surface saved: " << total_cost_path << "\n";

        auto step2c_end = std::chrono::high_resolution_clock::now();
        auto step2c_time = std::chrono::duration<double>(step2c_end - step2c_start).count();
        std::cout << "Cost modifiers applied successfully\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << step2c_time << " sec\n";
        std::cout << std::string(70, '-') << "\n";
    }

    std::cout << "\nLooping propagation algorithm...\n";
    auto step3_start = std::chrono::high_resolution_clock::now();

    const float INF = std::numeric_limits<float>::infinity();
    std::vector<uint32_t> fete_density(N, 0);

#pragma omp parallel for num_threads(max_threads) schedule(dynamic)
    for (int source_idx = 0; source_idx < P; ++source_idx) {
        const int source = point_nodes[source_idx];
        int source_r, source_c;
        idx2coord(source, ncols, source_r, source_c);

#pragma omp critical
        {
            print_progress(source_idx + 1, P);
        }

        std::vector<float> cumulative_cost(N, INF);
        std::vector<int> predecessor(N, -1);
        std::vector<bool> visited(N, false);

        using pq_entry = std::pair<float, int>;
        std::priority_queue<pq_entry, std::vector<pq_entry>, std::greater<pq_entry>> pq;

        cumulative_cost[source] = 0.0f;
        pq.push({ 0.0f, source });

        while (!pq.empty()) {
            auto [cost, v] = pq.top();
            pq.pop();

            if (visited[v]) continue;
            visited[v] = true;
            if (cost >= INF) break;

            int r, c;
            idx2coord(v, ncols, r, c);

            for (int k = 0; k < num_offs; ++k) {
                int nr = r + current_offs[k].dr;
                int nc = c + current_offs[k].dc;
                if (nr < 0 || nr >= nrows || nc < 0 || nc >= ncols) continue;
                if (visited[idx(nr, nc, ncols)]) continue;

                int u = idx(nr, nc, ncols);
                double dh = std::sqrt((current_offs[k].dr * res_y) * (current_offs[k].dr * res_y) +
                    (current_offs[k].dc * res_x) * (current_offs[k].dc * res_x));
                double dz = (double)dem[u] - (double)dem[v];
                float edge_cost = apply_cost_function(cost_function, dh, dz);

                // Apply cost multiplier to destination node (if cost modifiers are active)
                edge_cost *= cost_multipliers[u];

                float new_cost = cumulative_cost[v] + edge_cost;

                if (new_cost < cumulative_cost[u]) {
                    cumulative_cost[u] = new_cost;
                    predecessor[u] = v;
                    pq.push({ new_cost, u });
                }
            }
        }

        std::vector<uint32_t> local_density(N, 0);

        for (int dest_idx = 0; dest_idx < P; ++dest_idx) {
            int dest = point_nodes[dest_idx];
            if (dest == source || cumulative_cost[dest] >= INF) continue;

            std::vector<int> path;
            int current = dest;
            while (current != -1 && current != source) {
                path.push_back(current);
                current = predecessor[current];
            }
            path.push_back(source);
            std::reverse(path.begin(), path.end());

            for (int node : path) {
                int r, c;
                idx2coord(node, ncols, r, c);

                local_density[node]++;

                for (int dr = -buffer_radius; dr <= buffer_radius; ++dr) {
                    for (int dc = -buffer_radius; dc <= buffer_radius; ++dc) {
                        if (dr == 0 && dc == 0) continue;

                        int nr = r + dr;
                        int nc = c + dc;

                        if (nr >= 0 && nr < nrows && nc >= 0 && nc < ncols) {
                            local_density[idx(nr, nc, ncols)]++;
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            for (int i = 0; i < N; ++i) {
                if (local_density[i] > 0) {
                    fete_density[i] += local_density[i];
                }
            }
        }
    }

    std::cout << "\n";
    auto step3_end = std::chrono::high_resolution_clock::now();
    auto step3_time = std::chrono::duration<double>(step3_end - step3_start).count();
    std::cout << "Completed in " << std::fixed << std::setprecision(2) << step3_time << " seconds\n";

    std::cout << "\nWriting density raster...\n";

    std::string fete_density_path = join_path(out_dir, output_filename + ".tif");
    GDALDataset* fete_density_ds = gtiff_drv->Create(fete_density_path.c_str(), ncols, nrows, 1, GDT_UInt32, nullptr);
    fete_density_ds->SetGeoTransform(gt);
    fete_density_ds->SetProjection(wkt);
    fete_density_ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows, fete_density.data(), ncols, nrows, GDT_UInt32, 0, 0);
    GDALClose(fete_density_ds);

    std::cout << "Density raster saved\n";

    uint32_t max_density = 0;
    uint32_t min_density = UINT32_MAX;
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

    GDALClose(pts_ds);
    GDALClose(dem_ds);

    return output;
}

// ========== MAIN PROGRAM ==========

// Forward declaration - LCPA mode from main_lcpa.cu
int run_lcpa_mode();

int main(int argc, char* argv[]) {
    system("color 0F");

    // Check for help command
    if (argc > 1 && std::string(argv[1]) == "help") {
        print_help();
        return 0;
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    center_text("TRAJECTA - SPATIAL MOVEMENT ANALYSIS");
    center_text("by Stefano Apra, ISAW - NYU");
    std::cout << std::string(70, '=') << "\n";
    std::cout << "You can type 'help' at any prompt for instructions\n";
    std::cout << "Type 'exit' at any prompt to quit (with confirmation)\n";
    std::cout << "Press Ctrl+C to cancel the execution (Windows default)\n";
    std::cout << std::string(70, '=') << "\n\n";

    // ===== CHOOSE COMPUTATION MODE =====
    std::cout << "Choose computation mode:\n";
    std::cout << "  1) FETE (From Everywhere to Everywhere)\n";
    std::cout << "  2) LCPA (Least-Cost Path Analysis)\n\n";

    int mode = 1;  // Default to FETE
    while (true) {
        std::cout << "> ";
        std::string mode_input;
        std::getline(std::cin, mode_input);

        if (mode_input == "exit" || mode_input == "EXIT" || mode_input == "Exit") {
            std::cout << "\nGoodbye!\n\n";
            return 0;
        }
        if (mode_input == "help" || mode_input == "HELP" || mode_input == "Help") {
            std::cout << "\nEnter 1 for FETE or 2 for LCPA\n\n";
            continue;
        }

        try {
            int choice = std::stoi(mode_input);
            if (choice == 1 || choice == 2) {
                mode = choice;
                break;
            }
            else {
                std::cout << "ERROR: Please enter 1 or 2\n";
            }
        }
        catch (...) {
            if (mode_input.empty()) {
                mode = 1;  // Default to FETE if empty
                break;
            }
            std::cout << "ERROR: Invalid input. Enter 1 or 2\n";
        }
    }

    std::cout << "\n";

    // ===== VERBOSE MODE SELECTION =====
    std::cout << "Enable detailed debug output? (yes/no):\n";
    std::cout << "  yes - Show detailed logging for troubleshooting\n";
    std::cout << "  no  - Show only progress bars and summaries [DEFAULT]\n";

    while (true) {
        std::cout << "> ";
        std::string verbose_input;
        std::getline(std::cin, verbose_input);

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

    int max_threads = 10;
    while (true) {
        std::string threads_input;
        std::cout << "Enter maximum CPU threads to use (1-" << max_available_threads << "):\n";
        std::cout << "  Recommended: " << std::max(1, max_available_threads - 4) << " (reserve 4 cores for system)\n";
        std::cout << "> ";
        std::getline(std::cin, threads_input);

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
            break;
        }
        catch (...) {
            max_threads = std::max(1, max_available_threads - 4);
            break;
        }
    }

    std::cout << "Using " << max_threads << " threads\n\n";

    int64_t max_ram_mb = 4096;
    while (true) {
        std::string ram_input;
        std::cout << "Enter maximum RAM to allocate (MB):\n";
        std::cout << "  Total available: " << total_ram_mb << " MB (~" << (total_ram_mb / 1024) << " GB)\n";
        std::cout << "  Example: 4096 (for 4 GB), 8192 (for 8 GB)\n";
        std::cout << "  Recommended: ~60% of available RAM\n";
        std::cout << "> ";
        std::getline(std::cin, ram_input);

        if (check_exit_command(ram_input)) {
            return 0;
        }
        if (check_help_command(ram_input)) {
            continue;
        }

        try {
            max_ram_mb = std::stoll(ram_input);
            if (max_ram_mb < 512) max_ram_mb = 512;
            break;
        }
        catch (...) {
            max_ram_mb = 4096;
            break;
        }
    }

    std::cout << "Using maximum " << max_ram_mb << " MB RAM\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::string dem_path = saved_config.dem_path;
    std::string pts_path = saved_config.pts_path;
    std::string out_dir = saved_config.out_dir;
    std::string cost_modifiers_path = saved_config.cost_modifiers_path;
    std::string output_filename;
    std::string slope_filename;
    std::string cost_filename;
    std::string additional_cost_filename;
    std::string total_cost_filename;
    int buffer_radius = 0;
    int polyline_buffer_radius = 0;
    int num_neighbours = 16;
    bool slope_in_degrees = true;
    CostFunctionType cost_function = TOBLER_WHITE_2015;
    bool first_run = true;

    while (true) {
        std::cout << "\n";

        if (first_run) {
            // DEM Path
            while (true) {
                std::cout << "Enter path to DEM file (.tif):\n";
                if (!dem_path.empty()) {
                    std::cout << "  Default: " << dem_path << "\n";
                }
                std::cout << "  Example: C:\\path\\to\\DEM.tif\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (!input.empty()) dem_path = input;
                if (!dem_path.empty()) break;
                std::cout << "ERROR: DEM path cannot be empty!\n";
            }

            // Points Path
            while (true) {
                std::cout << "\nEnter path to sample points shapefile (.shp):\n";
                if (!pts_path.empty()) {
                    std::cout << "  Default: " << pts_path << "\n";
                }
                std::cout << "  Example: C:\\path\\to\\Points.shp\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (!input.empty()) pts_path = input;
                if (!pts_path.empty()) break;
                std::cout << "ERROR: Points path cannot be empty!\n";
            }

            // Output Directory
            while (true) {
                std::cout << "\nEnter output directory for results:\n";
                if (!out_dir.empty()) {
                    std::cout << "  Default: " << out_dir << "\n";
                }
                std::cout << "  Example: C:\\path\\to\\Results\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (!input.empty()) out_dir = input;
                if (!out_dir.empty()) break;
                std::cout << "ERROR: Output directory cannot be empty!\n";
            }

            // Cost Modifiers - Ask if user wants to add additional cost modifiers
            bool add_cost_modifiers = false;
            while (true) {
                std::cout << "\nDo you want to add additional cost modifiers? (yes/no):\n";
                std::cout << "  Cost modifiers allow you to increase traversal costs for specific\n";
                std::cout << "  features such as rivers, restricted areas, or difficult terrain.\n";
                std::cout << "  Default: no\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;

                if (input.empty() || input == "no" || input == "n" || input == "NO" || input == "No") {
                    add_cost_modifiers = false;
                    cost_modifiers_path = "";
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

            // If user wants to add cost modifiers, ask for the shapefile path
            if (add_cost_modifiers) {
                while (true) {
                    std::cout << "\nEnter path to cost modifiers shapefile (.shp):\n";
                    if (!cost_modifiers_path.empty()) {
                        std::cout << "  Default: " << cost_modifiers_path << "\n";
                    }
                    std::cout << "  Example: C:\\path\\to\\rivers.shp\n";
                    std::cout << "  Note: The shapefile should contain polylines with a 'cost' field\n";
                    std::cout << "        containing float values (cost multipliers, e.g., 1.5, 2.0, etc.)\n";
                    std::cout << "> ";
                    std::string input;
                    std::getline(std::cin, input);
                    if (check_exit_command(input)) return 0;
                    if (check_help_command(input)) continue;
                    if (!input.empty()) cost_modifiers_path = input;
                    if (!cost_modifiers_path.empty()) break;
                    std::cout << "ERROR: Cost modifiers path cannot be empty!\n";
                }

                // Ask for buffer radius for polylines
                std::cout << "\nSelect buffer radius (cells) for polyline rasterization:\n";
                std::cout << "  The buffer ensures the algorithm doesn't 'jump' across features.\n";
                std::cout << "  Each cell of buffer is applied on each side of the polyline.\n";
                std::cout << "  0) No buffer\n";
                std::cout << "  1) 1 cell per side\n";
                std::cout << "  2) 2 cells per side (safer for 16-connectivity) [DEFAULT]\n";
                std::cout << "  3) 3 cells per side\n";
                std::cout << "  Leave blank for default (2)\n";
                std::cout << "> ";
                std::string buffer_input;
                std::getline(std::cin, buffer_input);
                if (check_exit_command(buffer_input)) return 0;
                if (check_help_command(buffer_input)) continue;

                try {
                    int choice = std::stoi(buffer_input);
                    if (choice >= 0) polyline_buffer_radius = choice;
                }
                catch (...) {
                    polyline_buffer_radius = 2;
                }

                std::cout << "\nPolyline buffer set to " << polyline_buffer_radius << " cell(s) per side.\n";
            }

            // Validate inputs BEFORE asking for filenames
            std::cout << "\nValidating inputs...\n";
            ValidationResult val_result = validate_all_inputs(dem_path, pts_path);
            if (!val_result.is_valid) {
                std::cout << val_result.error_message << "\n";
                std::cout << "Please correct the paths and try again.\n\n";
                dem_path = "";
                pts_path = "";
                continue;
            }
            std::cout << "Validation successful!\n";

            // Parameters configuration
            std::cout << "\nSelect number of neighbours for cost surface calculation:\n";
            std::cout << "  1) 8-connectivity (3x3 grid)\n";
            std::cout << "  2) 16-connectivity (knight moves) [DEFAULT]\n";
            std::cout << "  3) 24-connectivity (extended)\n";
            std::cout << "  4) 32-connectivity (more extended)\n";
            std::cout << "  5) 64-connectivity (full extended)\n";
            std::cout << "  Leave blank for default (16)\n";
            std::cout << "> ";
            std::string neighbours_input;
            std::getline(std::cin, neighbours_input);
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

            std::cout << "\nSelect cost function:\n";
            std::cout << "  1) Modified Tobler's Function (White 2015) [DEFAULT]\n";
            std::cout << "  Leave blank for default\n";
            std::cout << "> ";
            std::string cf_input;
            std::getline(std::cin, cf_input);
            if (check_exit_command(cf_input)) return 0;
            if (check_help_command(cf_input)) continue;

            try {
                int choice = std::stoi(cf_input);
                if (choice == 1) cost_function = TOBLER_WHITE_2015;
            }
            catch (...) {
                cost_function = TOBLER_WHITE_2015;
            }

            std::cout << "\nSelect slope units:\n";
            std::cout << "  1) Degrees [DEFAULT]\n";
            std::cout << "  2) Percentage\n";
            std::cout << "  Leave blank for default (degrees)\n";
            std::cout << "> ";
            std::string slope_units_input;
            std::getline(std::cin, slope_units_input);
            if (check_exit_command(slope_units_input)) return 0;
            if (check_help_command(slope_units_input)) continue;

            try {
                int choice = std::stoi(slope_units_input);
                slope_in_degrees = (choice != 2);
            }
            catch (...) {
                slope_in_degrees = true;
            }

            std::cout << "\nSelect buffer radius (cells) for path smoothing:\n";
            std::cout << "  0) No buffer [DEFAULT]\n";
            std::cout << "  1) 1 cell on each side\n";
            std::cout << "  2) 2 cells on each side\n";
            std::cout << "  3) 3 cells on each side\n";
            std::cout << "  Leave blank for default (0)\n";
            std::cout << "> ";
            std::string buffer_input;
            std::getline(std::cin, buffer_input);
            if (check_exit_command(buffer_input)) return 0;
            if (check_help_command(buffer_input)) continue;

            try {
                int choice = std::stoi(buffer_input);
                if (choice >= 0) buffer_radius = choice;
            }
            catch (...) {
                buffer_radius = 0;
            }

            // Output filenames
            while (true) {
                std::cout << "\nEnter slope raster filename (without extension):\n";
                std::cout << "  Example: slope_degrees\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (input.empty()) {
                    std::cout << "ERROR: Slope filename cannot be empty!\n";
                    continue;
                }
                slope_filename = input;
                if (slope_filename.length() >= 4 && slope_filename.substr(slope_filename.length() - 4) == ".tif") {
                    slope_filename = slope_filename.substr(0, slope_filename.length() - 4);
                }
                break;
            }

            while (true) {
                std::cout << "\nEnter base cost surface raster filename (without extension):\n";
                std::cout << "  This is the cost surface calculated from slope * cost function\n";
                std::cout << "  Example: cost_surface_base\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (input.empty()) {
                    std::cout << "ERROR: Cost surface filename cannot be empty!\n";
                    continue;
                }
                cost_filename = input;
                if (cost_filename.length() >= 4 && cost_filename.substr(cost_filename.length() - 4) == ".tif") {
                    cost_filename = cost_filename.substr(0, cost_filename.length() - 4);
                }
                break;
            }

            // If cost modifiers were added, ask for additional and total cost surface filenames
            if (!cost_modifiers_path.empty()) {
                while (true) {
                    std::cout << "\nEnter additional cost surface raster filename (without extension):\n";
                    std::cout << "  This is the rasterized polylines with cost multipliers\n";
                    std::cout << "  Example: cost_surface_additional\n";
                    std::cout << "> ";
                    std::string input;
                    std::getline(std::cin, input);
                    if (check_exit_command(input)) return 0;
                    if (check_help_command(input)) continue;
                    if (input.empty()) {
                        std::cout << "ERROR: Additional cost surface filename cannot be empty!\n";
                        continue;
                    }
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
                    std::getline(std::cin, input);
                    if (check_exit_command(input)) return 0;
                    if (check_help_command(input)) continue;
                    if (input.empty()) {
                        std::cout << "ERROR: Total cost surface filename cannot be empty!\n";
                        continue;
                    }
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
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (input.empty()) {
                    std::cout << "ERROR: Output density filename cannot be empty!\n";
                    continue;
                }
                output_filename = input;
                if (output_filename.length() >= 4 && output_filename.substr(output_filename.length() - 4) == ".tif") {
                    output_filename = output_filename.substr(0, output_filename.length() - 4);
                }
                break;
            }

            first_run = false;
        }
        else {
            // Subsequent runs - ask for filenames only
            while (true) {
                std::cout << "Enter slope raster filename (without extension):\n";
                std::cout << "  Example: slope_degrees\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (input.empty()) {
                    std::cout << "ERROR: Slope filename cannot be empty!\n";
                    continue;
                }
                slope_filename = input;
                if (slope_filename.length() >= 4 && slope_filename.substr(slope_filename.length() - 4) == ".tif") {
                    slope_filename = slope_filename.substr(0, slope_filename.length() - 4);
                }
                break;
            }

            while (true) {
                std::cout << "\nEnter base cost surface raster filename (without extension):\n";
                std::cout << "  This is the cost surface calculated from slope * cost function\n";
                std::cout << "  Example: cost_surface_base\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (input.empty()) {
                    std::cout << "ERROR: Cost surface filename cannot be empty!\n";
                    continue;
                }
                cost_filename = input;
                if (cost_filename.length() >= 4 && cost_filename.substr(cost_filename.length() - 4) == ".tif") {
                    cost_filename = cost_filename.substr(0, cost_filename.length() - 4);
                }
                break;
            }

            // If cost modifiers were added, ask for additional and total cost surface filenames
            if (!cost_modifiers_path.empty()) {
                while (true) {
                    std::cout << "\nEnter additional cost surface raster filename (without extension):\n";
                    std::cout << "  This is the rasterized polylines with cost multipliers\n";
                    std::cout << "  Example: cost_surface_additional\n";
                    std::cout << "> ";
                    std::string input;
                    std::getline(std::cin, input);
                    if (check_exit_command(input)) return 0;
                    if (check_help_command(input)) continue;
                    if (input.empty()) {
                        std::cout << "ERROR: Additional cost surface filename cannot be empty!\n";
                        continue;
                    }
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
                    std::getline(std::cin, input);
                    if (check_exit_command(input)) return 0;
                    if (check_help_command(input)) continue;
                    if (input.empty()) {
                        std::cout << "ERROR: Total cost surface filename cannot be empty!\n";
                        continue;
                    }
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
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (input.empty()) {
                    std::cout << "ERROR: Output density filename cannot be empty!\n";
                    continue;
                }
                output_filename = input;
                if (output_filename.length() >= 4 && output_filename.substr(output_filename.length() - 4) == ".tif") {
                    output_filename = output_filename.substr(0, output_filename.length() - 4);
                }
                break;
            }
        }

        // ========== GPU DETECTION AND MODE SELECTION ==========
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "System Information:\n";
        std::cout << "  Available CPU threads: " << max_threads << "\n";
        std::cout << "  Total RAM: " << (get_total_ram_mb() / 1024) << " GB\n\n";

        GPUInfo gpu_info = get_gpu_info();
        print_gpu_info(gpu_info);
        std::cout << std::string(70, '=') << "\n\n";

        // Ask CPU or GPU
        bool use_gpu = false;
        std::cout << "Select processing mode:\n";
        std::cout << "  1) CPU [DEFAULT]\n";

        if (gpu_info.available) {
            std::cout << "  2) GPU (NVIDIA CUDA) - " << gpu_info.name << "\n";
        }
        else {
            std::cout << "  2) GPU (NVIDIA CUDA) - [GPU not detected or incompatible]\n";
        }

        std::cout << "> ";
        std::string mode_input;
        std::getline(std::cin, mode_input);
        if (check_exit_command(mode_input)) return 0;
        if (check_help_command(mode_input)) continue;

        try {
            int choice = std::stoi(mode_input);
            if (choice == 2) {
                if (gpu_info.available) {
                    // Check if DEM will fit in GPU memory
                    // Need to open DEM to get dimensions
                    GDALDataset* temp_ds = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
                    if (temp_ds) {
                        int temp_ncols = temp_ds->GetRasterXSize();
                        int temp_nrows = temp_ds->GetRasterYSize();
                        GDALClose(temp_ds);

                        MemoryRequirement mem_req = calculate_memory_requirement(temp_nrows, temp_ncols, num_neighbours);

                        print_memory_requirement(mem_req);
                        std::cout << "\nGPU Memory Available: " << gpu_info.free_memory_mb << " MB\n";

                        if (check_gpu_memory_sufficient(gpu_info, mem_req)) {
                            std::cout << "✅ GPU has sufficient memory!\n\n";
                            use_gpu = true;
                        }
                        else {
                            print_memory_error_diagnostic(gpu_info, mem_req, temp_nrows, temp_ncols, num_neighbours);
                            std::cout << "\nAutomatically falling back to CPU mode...\n\n";
                            use_gpu = false;
                        }
                    }
                    else {
                        std::cout << "\nERROR: Cannot open DEM to check dimensions.\n";
                        std::cout << "Falling back to CPU mode.\n\n";
                        use_gpu = false;
                    }
                }
                else {
                    std::cout << "\n" << std::string(70, '!') << "\n";
                    std::cout << "ERROR: GPU mode selected but no compatible GPU detected!\n";
                    std::cout << "Possible reasons:\n";
                    std::cout << "  - No NVIDIA GPU installed\n";
                    std::cout << "  - NVIDIA drivers not installed\n";
                    std::cout << "  - CUDA toolkit not properly configured\n";
                    std::cout << "\nAutomatically falling back to CPU mode...\n";
                    std::cout << std::string(70, '!') << "\n\n";
                    use_gpu = false;
                }
            }
        }
        catch (...) {
            use_gpu = false;
        }

        // ========== CUDA STREAMS SELECTION (GPU only) ==========
        int num_streams = 4;  // Default
        if (use_gpu) {
            // Calculate max streams based on available memory
            GDALDataset* temp_ds = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
            int temp_N = 0;
            size_t per_stream_mb = 0;
            int max_streams_by_memory = 1;  // Default cap
            
            if (temp_ds) {
                temp_N = temp_ds->GetRasterXSize() * temp_ds->GetRasterYSize();
                GDALClose(temp_ds);
                per_stream_mb = (temp_N * sizeof(float) + temp_N * sizeof(int)) / (1024 * 1024);
                
                // Calculate how many streams fit in remaining GPU memory
                MemoryRequirement mem_req = calculate_memory_requirement(
                    (int)std::sqrt(temp_N), (int)std::sqrt(temp_N), num_neighbours);
                size_t remaining_mb = gpu_info.free_memory_mb - mem_req.total_mb;
                if (per_stream_mb > 0) {
                    max_streams_by_memory = std::max(1, (int)(remaining_mb / per_stream_mb));
                }
            }

            std::cout << "\n" << std::string(70, '-') << "\n";
            std::cout << "CUDA Streams Configuration:\n";
            std::cout << "  Memory per stream: ~" << per_stream_mb << " MB\n";
            std::cout << "  Max streams (by memory): " << max_streams_by_memory << "\n";
            std::cout << "  Recommended: 4-16 streams\n";
            std::cout << std::string(70, '-') << "\n";
            std::cout << "Enter number of CUDA streams [DEFAULT: 4]:\n> ";

            std::string streams_input;
            std::getline(std::cin, streams_input);
            if (check_exit_command(streams_input)) return 0;

            if (!streams_input.empty()) {
                try {
                    int requested = std::stoi(streams_input);
                    if (requested < 1) {
                        std::cout << "WARNING: Minimum 1 stream required. Using 1.\n";
                        num_streams = 1;
                    }
                    else if (requested > max_streams_by_memory) {
                        std::cout << "WARNING: Requested " << requested << " streams exceeds memory limit.\n";
                        std::cout << "Using maximum: " << max_streams_by_memory << " streams.\n";
                        num_streams = max_streams_by_memory;
                    }
                    else {
                        num_streams = requested;
                    }
                }
                catch (...) {
                    std::cout << "Invalid input. Using default: 4 streams.\n";
                    num_streams = 4;
                }
            }
            std::cout << "Using " << num_streams << " CUDA streams.\n\n";
        }

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "Configuration:\n";
        std::cout << "  DEM: " << dem_path << "\n";
        std::cout << "  Points: " << pts_path << "\n";
        std::cout << "  Output dir: " << out_dir << "\n";
        if (!cost_modifiers_path.empty()) {
            std::cout << "  Cost modifiers: " << cost_modifiers_path << "\n";
            std::cout << "  Polyline buffer: " << polyline_buffer_radius << " cells per side\n";
        }
        std::cout << "  Slope filename: " << slope_filename << ".tif\n";
        std::cout << "  Base cost filename: " << cost_filename << ".tif\n";
        if (!cost_modifiers_path.empty()) {
            std::cout << "  Additional cost filename: " << additional_cost_filename << ".tif\n";
            std::cout << "  Total cost filename: " << total_cost_filename << ".tif\n";
        }
        std::cout << "  Density filename: " << output_filename << ".tif\n";
        std::cout << "  Buffer radius: " << buffer_radius << " cells\n";
        std::cout << "  Neighbours: " << num_neighbours << "-connectivity\n";
        std::cout << "  Slope units: " << (slope_in_degrees ? "degrees" : "percentage") << "\n";
        std::cout << "  Cost function: Modified Tobler (White 2015)\n";
        std::cout << "  Max threads: " << max_threads << "\n";
        std::cout << "  Max RAM: " << max_ram_mb << " MB\n";
        std::cout << "  Processing mode: " << (use_gpu ? "GPU" : "CPU") << "\n";
        if (use_gpu) {
            std::cout << "  CUDA streams: " << num_streams << "\n";
        }
        std::cout << std::string(70, '=') << "\n\n";

        FETEOutput result;
        if (use_gpu) {
            result = run_fete_gpu(dem_path, pts_path, out_dir, slope_filename, cost_filename, output_filename,
                buffer_radius, max_threads, max_ram_mb, num_neighbours, slope_in_degrees, cost_function, num_streams);
        }
        else {
            result = run_fete(dem_path, pts_path, out_dir, slope_filename, cost_filename, output_filename,
                buffer_radius, max_threads, max_ram_mb, num_neighbours, slope_in_degrees, cost_function,
                cost_modifiers_path, polyline_buffer_radius, additional_cost_filename, total_cost_filename);
        }

        if (result.success) {
            // Save config
            Config to_save = { dem_path, pts_path, out_dir, cost_modifiers_path };
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
            std::cout << "  - " << result.slope_path << "\n";
            std::cout << "  - " << result.cost_path << " (base cost surface)\n";
            if (!result.additional_cost_path.empty()) {
                std::cout << "  - " << result.additional_cost_path << " (additional cost multipliers)\n";
                std::cout << "  - " << result.total_cost_path << " (total cost surface)\n";
            }
            std::cout << "  - " << result.density_path << "\n";
        }

        std::cout << "\nRun another FETE computation? (yes/no)\n> ";
        std::string again;
        std::getline(std::cin, again);
        if (check_exit_command(again)) return 0;

        if (again == "no" || again == "n") {
            std::cout << "\nExit program? (yes/no)\n> ";
            std::string exit_choice;
            std::getline(std::cin, exit_choice);
            if (check_exit_command(exit_choice)) return 0;

            if (exit_choice == "yes" || exit_choice == "y") {
                std::cout << "\nGoodbye!\n\n";
                break;
            }
        }
    }

    return 0;
}
