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
#include <omp.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#ifdef _WIN32
#include <windows.h>
#endif
#include "gdal_priv.h"
#include "ogrsf_frmts.h"

namespace fs = std::filesystem;

static inline std::string join_path(const std::string& dir, const std::string& file) {
    return (fs::path(dir) / file).string();
}

// ========== GLOBAL SETTINGS ==========
extern bool g_verbose_mode;  // Defined in main_fete.cu, shared across files

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

// ========== HELP TEXT FOR LCPA ONLY ==========
const char* HELP_TEXT_LCPA = R"(
===============================================================================
            TRAJECTA - A SPATIAL MOVEMENT ANALYSIS SOFTWARE
                     Developed by Stefano Apra
              Institute for the Study of the Ancient World
                    
                    Least-Cost Path Analysis (LCPA)

LCPA computes the optimal path(s) from a single origin point to one or more
destination points across a Digital Elevation Model (DEM). It calculates
the route with the lowest cumulative cost based on terrain-dependent costs.

===============================================================================

MODE: Least-Cost Path Analysis (LCPA)

INPUT REQUIREMENTS:
  - DEM: GeoTIFF file (.tif), must be georeferenced
  - Origin: ESRI Shapefile (.shp) with exactly 1 point
  - Destinations: ESRI Shapefile (.shp) with 1+ points
  - CRS: All files MUST have the same coordinate system

PARAMETERS:
  - Neighbours: Connectivity (8, 16, 24, 32, 64)
  - Slope Units: Degrees or Percentage
  - Buffer Radius: Cells around path for visualization
  - Cost Function: Modified Tobler's Walking Function
  - CPU Threads: Parallel processing threads
  - Max RAM: Memory limit for processing

OUTPUT:
  - slope_[name].tif: Terrain slope raster
  - cost_surface_[name].tif: Cost surface raster
  - path_raster_[name].tif: Raster showing computed paths
  - path_lines_[name].shp: Polyline shapefile with path geometries

===============================================================================
)";

// ========== STRUCTURES (from main.cu + LCPA-specific) ==========
struct Config {
    std::string dem_path;
    std::string pts_path;
    std::string out_dir;
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

struct LCPAOutput {
    bool success;
    std::string slope_path;
    std::string cost_path;
    std::string additional_cost_path;  // Additional cost surface from polylines
    std::string total_cost_path;       // Total cost surface (base * additional)
    std::string path_raster_path;
    std::string path_lines_path;
    int num_destinations;
    int total_path_cells;
    double total_cost;
    double time_seconds;
};

struct Off { int dr; int dc; };

// ========== HELPER FUNCTIONS ==========
static inline int idx(int r, int c, int ncols) { return r * ncols + c; }
static inline void idx2coord(int index, int ncols, int& r, int& c) { r = index / ncols; c = index % ncols; }

// ========== NEIGHBOR OFFSET ARRAYS ==========
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

// ========== FORWARD DECLARATIONS (functions from main.cu) ==========
extern void print_help();
extern void center_text(const std::string& text, int width = 70);
extern bool check_help_command(const std::string& input);
extern bool check_exit_command(const std::string& input);
extern std::string get_cpu_model();
extern int64_t get_total_ram_mb();
extern std::string get_file_extension(const std::string& path);
extern bool file_exists(const std::string& path);
extern void print_progress(int current, int total, int bar_width = 45);
extern void print_green_success(const std::string& success);
extern void save_config(const Config& cfg);
extern Config load_config();
extern void save_config_lcpa(const ConfigLCPA& cfg);
extern ConfigLCPA load_config_lcpa();
extern ValidationResult validate_dem(const std::string& dem_path);
extern std::vector<float> rasterize_polylines_with_costs(const std::string& polylines_path,
    int nrows, int ncols, const double gt[6], int buffer_cells, int max_threads);

// ========== LCPA-SPECIFIC FUNCTIONS ==========

int count_points_in_shapefile(const std::string& shp_path) {
    GDALDataset* ds = (GDALDataset*)GDALOpenEx(shp_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (!ds) return -1;
    OGRLayer* layer = ds->GetLayer(0);
    if (!layer) {
        GDALClose(ds);
        return -1;
    }
    int count = layer->GetFeatureCount();
    GDALClose(ds);
    return count;
}

ValidationResult validate_origin_shapefile(const std::string& shp_path) {
    if (!file_exists(shp_path)) {
        return { false, "ERROR: Origin shapefile not found: " + shp_path };
    }
    std::string ext = get_file_extension(shp_path);
    if (ext != ".shp" && ext != ".SHP") {
        return { false, "ERROR: Origin must be shapefile (.shp), found: " + ext };
    }
    int count = count_points_in_shapefile(shp_path);
    if (count == -1) {
        return { false, "ERROR: Cannot open origin shapefile with OGR" };
    }
    if (count != 1) {
        return { false, "ERROR: Origin shapefile must contain EXACTLY 1 point, found: " + std::to_string(count) };
    }
    return { true, "" };
}

ValidationResult validate_destinations_shapefile(const std::string& shp_path) {
    if (!file_exists(shp_path)) {
        return { false, "ERROR: Destinations shapefile not found: " + shp_path };
    }
    std::string ext = get_file_extension(shp_path);
    if (ext != ".shp" && ext != ".SHP") {
        return { false, "ERROR: Destinations must be shapefile (.shp), found: " + ext };
    }
    int count = count_points_in_shapefile(shp_path);
    if (count == -1) {
        return { false, "ERROR: Cannot open destinations shapefile with OGR" };
    }
    if (count < 1) {
        return { false, "ERROR: Destinations shapefile must contain at least 1 point, found: " + std::to_string(count) };
    }
    return { true, "" };
}

bool get_point_coordinates(const std::string& shp_path, double& x, double& y) {
    GDALDataset* ds = (GDALDataset*)GDALOpenEx(shp_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (!ds) return false;
    OGRLayer* layer = ds->GetLayer(0);
    if (!layer) {
        GDALClose(ds);
        return false;
    }
    layer->ResetReading();
    OGRFeature* feat = layer->GetNextFeature();
    if (!feat) {
        GDALClose(ds);
        return false;
    }
    OGRGeometry* geom = feat->GetGeometryRef();
    if (!geom || wkbFlatten(geom->getGeometryType()) != wkbPoint) {
        OGRFeature::DestroyFeature(feat);
        GDALClose(ds);
        return false;
    }
    OGRPoint* point = geom->toPoint();
    x = point->getX();
    y = point->getY();
    OGRFeature::DestroyFeature(feat);
    GDALClose(ds);
    return true;
}

bool get_all_destination_coordinates(const std::string& shp_path, std::vector<std::pair<double, double>>& coords) {
    GDALDataset* ds = (GDALDataset*)GDALOpenEx(shp_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (!ds) return false;
    OGRLayer* layer = ds->GetLayer(0);
    if (!layer) {
        GDALClose(ds);
        return false;
    }
    coords.clear();
    layer->ResetReading();
    OGRFeature* feat = nullptr;
    while ((feat = layer->GetNextFeature()) != nullptr) {
        OGRGeometry* geom = feat->GetGeometryRef();
        if (geom && wkbFlatten(geom->getGeometryType()) == wkbPoint) {
            OGRPoint* point = geom->toPoint();
            coords.push_back({ point->getX(), point->getY() });
        }
        OGRFeature::DestroyFeature(feat);
    }
    GDALClose(ds);
    return !coords.empty();
}

bool convert_geo_to_pixel(const std::string& dem_path, double x, double y, int& pixel_idx, int& ncols) {
    GDALDataset* ds = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
    if (!ds) return false;

    double gt[6];
    ds->GetGeoTransform(gt);
    ncols = ds->GetRasterXSize();
    int nrows = ds->GetRasterYSize();

    int col = (int)std::floor((x - gt[0]) / gt[1]);
    int row = (int)std::floor((y - gt[3]) / gt[5]);

    GDALClose(ds);

    if (col < 0 || col >= ncols || row < 0 || row >= nrows) {
        return false;  // Point outside DEM extent
    }

    pixel_idx = row * ncols + col;
    return true;
}


// ========== LCPA ALGORITHM ==========

LCPAOutput run_lcpa(const std::string& dem_path, const std::string& pts_path, const std::string& out_dir,
    const std::string& slope_filename, const std::string& cost_filename,
    const std::string& path_raster_filename, const std::string& path_lines_filename,
    int origin_idx, const std::vector<int>& destination_indices,
    int buffer_radius, int max_threads, int64_t max_ram_mb,
    int num_neighbours, bool slope_in_degrees, CostFunctionType cost_function,
    const std::string& cost_modifiers_path = "", int polyline_buffer_radius = 0,
    const std::string& additional_cost_filename = "", const std::string& total_cost_filename = "") {

    LCPAOutput output = { false, "", "", "", "", "", "", 0, 0, 0.0, 0.0 };
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

    std::cout << "\nRunning Dijkstra algorithm from origin to destinations...\n";
    auto step3_start = std::chrono::high_resolution_clock::now();

    const float INF = std::numeric_limits<float>::infinity();
    std::vector<float> cumulative_cost(N, INF);
    std::vector<int> predecessor(N, -1);
    std::vector<bool> visited(N, false);
    std::vector<uint32_t> path_raster(N, 0);

    using pq_entry = std::pair<float, int>;
    std::priority_queue<pq_entry, std::vector<pq_entry>, std::greater<pq_entry>> pq;

    cumulative_cost[origin_idx] = 0.0f;
    pq.push({ 0.0f, origin_idx });

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

    std::cout << "Dijkstra completed\n";
    auto step3_end = std::chrono::high_resolution_clock::now();
    auto step3_time = std::chrono::duration<double>(step3_end - step3_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << step3_time << " seconds\n";

    std::cout << "\nTracing paths and generating output...\n";

    // Create shapefile for paths
    std::string shp_path = join_path(out_dir, path_lines_filename + ".shp");
    OGRSpatialReference osr;
    osr.importFromWkt(wkt);

    GDALDriver* shp_drv = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    GDALDataset* shp_ds = shp_drv->Create(shp_path.c_str(), 0, 0, 0, GDT_Unknown, nullptr);
    OGRLayer* layer = shp_ds->CreateLayer("paths", &osr, wkbLineString, nullptr);

    OGRFieldDefn oField("PathID", OFTInteger);
    layer->CreateField(&oField);

    int path_count = 0;
    int total_path_cells = 0;
    double total_cost_accumulated = 0.0;

    // For each destination, trace back the path
    for (int dest_idx : destination_indices) {
        if (cumulative_cost[dest_idx] >= INF) {
            std::cout << "Warning: Destination at index " << dest_idx << " is unreachable\n";
            continue;
        }

        std::vector<int> path;
        int current = dest_idx;
        while (current != -1 && current != origin_idx) {
            path.push_back(current);
            current = predecessor[current];
        }
        path.push_back(origin_idx);
        std::reverse(path.begin(), path.end());

        // Add to raster
        for (int node : path) {
            path_raster[node]++;

            // Buffer around path
            int r, c;
            idx2coord(node, ncols, r, c);
            for (int dr = -buffer_radius; dr <= buffer_radius; ++dr) {
                for (int dc = -buffer_radius; dc <= buffer_radius; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int nr = r + dr;
                    int nc = c + dc;
                    if (nr >= 0 && nr < nrows && nc >= 0 && nc < ncols) {
                        path_raster[idx(nr, nc, ncols)]++;
                    }
                }
            }
        }

        total_path_cells += path.size();
        total_cost_accumulated += cumulative_cost[dest_idx];

        // Add polyline to shapefile
        OGRLineString line;
        for (int node : path) {
            int r, c;
            idx2coord(node, ncols, r, c);
            double x = gt[0] + c * gt[1];
            double y = gt[3] + r * gt[5];
            line.addPoint(x, y);
        }

        OGRFeature* feature = OGRFeature::CreateFeature(layer->GetLayerDefn());
        feature->SetField("PathID", path_count++);
        feature->SetGeometry(&line);
        layer->CreateFeature(feature);
        OGRFeature::DestroyFeature(feature);
    }

    GDALClose(shp_ds);

    // Write path raster
    std::string path_raster_path = join_path(out_dir, path_raster_filename + ".tif");
    GDALDataset* path_ds = gtiff_drv->Create(path_raster_path.c_str(), ncols, nrows, 1, GDT_UInt32, nullptr);
    path_ds->SetGeoTransform(gt);
    path_ds->SetProjection(wkt);
    path_ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows, path_raster.data(), ncols, nrows, GDT_UInt32, 0, 0);
    GDALClose(path_ds);

    std::cout << "Paths saved\n";

    auto global_end = std::chrono::high_resolution_clock::now();
    double global_time = std::chrono::duration<double>(global_end - global_start).count();

    output.success = true;
    output.slope_path = slope_path;
    output.cost_path = cost_path;
    output.additional_cost_path = additional_cost_path;
    output.total_cost_path = total_cost_path;
    output.path_raster_path = path_raster_path;
    output.path_lines_path = shp_path;
    output.num_destinations = (int)destination_indices.size();
    output.total_path_cells = total_path_cells;
    output.total_cost = total_cost_accumulated;
    output.time_seconds = global_time;

    GDALClose(dem_ds);

    return output;
}

// ========== MAIN PROGRAM ==========

int run_lcpa_mode() {
    system("color 0F");


    std::cout << "\n" << std::string(70, '=') << "\n";
    center_text("TRAJECTA - A SPATIAL MOVEMENT ANALYSIS SOFTWARE", 70);
    center_text("Least-Cost Path Analysis (LCPA)", 70);
    center_text("by Stefano Apra, ISAW - NYU", 70);
    std::cout << std::string(70, '=') << "\n";
    std::cout << "You can type 'help' at any prompt for instructions\n";
    std::cout << "Type 'exit' at any prompt to quit (with confirmation)\n";
    std::cout << "Press Ctrl+C to cancel the execution (Windows default)\n";
    std::cout << std::string(70, '=') << "\n\n";

    GDALAllRegister();
    OGRRegisterAll();

    // Load previous config
    // Load previous config
    ConfigLCPA saved_config = load_config_lcpa();

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
    std::string out_dir = saved_config.out_dir;
    std::string origin_shp_path = saved_config.origin_path;  // Origin shapefile path
    std::string destinations_shp_path = saved_config.destinations_path;  // Destinations shapefile path
    std::string cost_modifiers_path = saved_config.cost_modifiers_path;
    std::string slope_filename;
    std::string cost_filename;
    std::string additional_cost_filename;
    std::string total_cost_filename;
    std::string path_raster_filename;
    std::string path_lines_filename;
    double origin_x = 0.0, origin_y = 0.0;  // Origin coordinates from shapefile
    std::vector<std::pair<double, double>> destination_coords;  // Destination coordinates from shapefile
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

            // Origin Shapefile (with exactly 1 point)
            while (true) {
                std::cout << "\nEnter path to ORIGIN shapefile (.shp with exactly 1 point):\n";
                if (!origin_shp_path.empty()) {
                    std::cout << "  Default: " << origin_shp_path << "\n";
                }
                std::cout << "  Example: C:\\path\\to\\Origin.shp\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (!input.empty()) origin_shp_path = input;
                if (origin_shp_path.empty()) {
                    std::cout << "ERROR: Origin shapefile path cannot be empty!\n";
                    continue;
                }

                ValidationResult val = validate_origin_shapefile(origin_shp_path);
                if (!val.is_valid) {
                    std::cout << val.error_message << "\n";
                    continue;
                }
                break;
            }

            // Destinations Shapefile (with 1+ points)
            while (true) {
                std::cout << "\nEnter path to DESTINATIONS shapefile (.shp with 1+ points):\n";
                if (!destinations_shp_path.empty()) {
                    std::cout << "  Default: " << destinations_shp_path << "\n";
                }
                std::cout << "  Example: C:\\path\\to\\Destinations.shp\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (!input.empty()) destinations_shp_path = input;
                if (destinations_shp_path.empty()) {
                    std::cout << "ERROR: Destinations shapefile path cannot be empty!\n";
                    continue;
                }

                ValidationResult val = validate_destinations_shapefile(destinations_shp_path);
                if (!val.is_valid) {
                    std::cout << val.error_message << "\n";
                    continue;
                }
                break;
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

            // Validate DEM
            std::cout << "\nValidating DEM...\n";
            ValidationResult dem_val = validate_dem(dem_path);
            if (!dem_val.is_valid) {
                std::cout << dem_val.error_message << "\n";
                std::cout << "Please correct the DEM path and try again.\n\n";
                dem_path = "";
                continue;
            }
            std::cout << "Validation successful!\n";

            // Extract origin coordinates
            if (!get_point_coordinates(origin_shp_path, origin_x, origin_y)) {
                std::cout << "ERROR: Cannot extract coordinates from origin shapefile!\n";
                continue;
            }

            // Extract destination coordinates
            if (!get_all_destination_coordinates(destinations_shp_path, destination_coords)) {
                std::cout << "ERROR: Cannot extract coordinates from destinations shapefile!\n";
                continue;
            }

            std::cout << "Extracted " << destination_coords.size() << " destination coordinate(s)\n";

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

            // Output filenames - FIRST RUN (with Examples and Help)
            while (true) {
                std::cout << "\nEnter slope raster filename (without extension):\n";
                std::cout << "  Example: slope_degrees\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
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
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
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
                    if (check_help_command(input)) {
                        std::cout << HELP_TEXT_LCPA;
                        continue;
                    }
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
                    if (check_help_command(input)) {
                        std::cout << HELP_TEXT_LCPA;
                        continue;
                    }
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
                std::cout << "\nEnter path raster filename (without extension):\n";
                std::cout << "  Example: raster_lcps\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
                if (input.empty()) {
                    std::cout << "ERROR: Path raster filename cannot be empty!\n";
                    continue;
                }
                path_raster_filename = input;
                if (path_raster_filename.length() >= 4 && path_raster_filename.substr(path_raster_filename.length() - 4) == ".tif") {
                    path_raster_filename = path_raster_filename.substr(0, path_raster_filename.length() - 4);
                }
                break;
            }

            while (true) {
                std::cout << "\nEnter path lines shapefile filename (without extension):\n";
                std::cout << "  (This will contain polyline geometries of the paths)\n";
                std::cout << "  Example: LCPS_vectors\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
                if (input.empty()) {
                    std::cout << "ERROR: Path lines filename cannot be empty!\n";
                    continue;
                }
                path_lines_filename = input;
                if (path_lines_filename.length() >= 4 && path_lines_filename.substr(path_lines_filename.length() - 4) == ".shp") {
                    path_lines_filename = path_lines_filename.substr(0, path_lines_filename.length() - 4);
                }
                break;
            }
        }
        else {
            // Subsequent runs - only ask for output filenames
            while (true) {
                std::cout << "Enter slope raster filename (without extension):\n";
                std::cout << "  Example: slope_degrees\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
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
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
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
                    if (check_help_command(input)) {
                        std::cout << HELP_TEXT_LCPA;
                        continue;
                    }
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
                    if (check_help_command(input)) {
                        std::cout << HELP_TEXT_LCPA;
                        continue;
                    }
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
                std::cout << "\nEnter path raster filename (without extension):\n";
                std::cout << "  Example: LCPS_raster\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
                if (input.empty()) {
                    std::cout << "ERROR: Path raster filename cannot be empty!\n";
                    continue;
                }
                path_raster_filename = input;
                if (path_raster_filename.length() >= 4 && path_raster_filename.substr(path_raster_filename.length() - 4) == ".tif") {
                    path_raster_filename = path_raster_filename.substr(0, path_raster_filename.length() - 4);
                }
                break;
            }

            while (true) {
                std::cout << "\nEnter path lines shapefile filename (without extension):\n";
                std::cout << "  Example: LCPS_vectors\n";
                std::cout << "> ";
                std::string input;
                std::getline(std::cin, input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
                if (input.empty()) {
                    std::cout << "ERROR: Path lines filename cannot be empty!\n";
                    continue;
                }
                path_lines_filename = input;
                if (path_lines_filename.length() >= 4 && path_lines_filename.substr(path_lines_filename.length() - 4) == ".shp") {
                    path_lines_filename = path_lines_filename.substr(0, path_lines_filename.length() - 4);
                }
                break;
            }
        }

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "Configuration:\n";
        std::cout << "  DEM: " << dem_path << "\n";
        std::cout << "  Origin shapefile: " << origin_shp_path << "\n";
        std::cout << "  Destinations shapefile: " << destinations_shp_path << "\n";
        std::cout << "  Output dir: " << out_dir << "\n";
        std::cout << "  Origin coordinates: (" << origin_x << ", " << origin_y << ")\n";
        std::cout << "  Destination coordinates: " << destination_coords.size() << " point(s)\n";
        std::cout << "  Slope filename: " << slope_filename << ".tif\n";
        std::cout << "  Cost filename: " << cost_filename << ".tif\n";
        std::cout << "  Path raster filename: " << path_raster_filename << ".tif\n";
        std::cout << "  Path lines filename: " << path_lines_filename << ".shp\n";
        std::cout << "  Buffer radius: " << buffer_radius << " cells\n";
        std::cout << "  Neighbours: " << num_neighbours << "-connectivity\n";
        std::cout << "  Slope units: " << (slope_in_degrees ? "degrees" : "percentage") << "\n";
        std::cout << "  Cost function: Modified Tobler (White 2015)\n";
        std::cout << "  Max threads: " << max_threads << "\n";
        std::cout << "  Max RAM: " << max_ram_mb << " MB\n";
        std::cout << std::string(70, '=') << "\n\n";

        // Convert geographic coordinates to pixel indices
        int origin_idx = -1;
        int ncols = 0;
        int dummy;  // For ncols

        if (!convert_geo_to_pixel(dem_path, origin_x, origin_y, origin_idx, ncols)) {
            std::cout << "ERROR: Origin point is outside DEM extent!\n";
            continue;
        }

        std::vector<int> destination_indices;
        for (const auto& coord : destination_coords) {
            int dest_idx = -1;
            if (!convert_geo_to_pixel(dem_path, coord.first, coord.second, dest_idx, dummy)) {
                std::cout << "WARNING: Destination point (" << coord.first << ", " << coord.second << ") is outside DEM extent, skipping...\n";
                continue;
            }
            destination_indices.push_back(dest_idx);
        }

        if (destination_indices.empty()) {
            std::cout << "ERROR: All destination points are outside DEM extent!\n";
            continue;
        }

        LCPAOutput result = run_lcpa(dem_path, destinations_shp_path, out_dir, slope_filename, cost_filename,
            path_raster_filename, path_lines_filename,
            origin_idx, destination_indices,
            buffer_radius, max_threads, max_ram_mb,
            num_neighbours, slope_in_degrees, cost_function,
            cost_modifiers_path, polyline_buffer_radius, additional_cost_filename, total_cost_filename);

        if (result.success) {
            ConfigLCPA to_save = { dem_path, origin_shp_path, destinations_shp_path, out_dir, cost_modifiers_path };
            save_config_lcpa(to_save);

            print_green_success("LCPA successfully computed!\n");
            std::cout << "\nOutput Summary:\n";
            std::cout << "  Total time: " << std::fixed << std::setprecision(2) << result.time_seconds << " sec\n";
            std::cout << "  Destinations processed: " << result.num_destinations << "\n";
            std::cout << "  Total path cells: " << result.total_path_cells << "\n";
            std::cout << "  Total cost accumulated: " << std::fixed << std::setprecision(2) << result.total_cost << "\n";
            std::cout << "\nOutput Files:\n";
            std::cout << "  - " << result.slope_path << "\n";
            std::cout << "  - " << result.cost_path << " (base cost surface)\n";
            if (!result.additional_cost_path.empty()) {
                std::cout << "  - " << result.additional_cost_path << " (additional cost multipliers)\n";
                std::cout << "  - " << result.total_cost_path << " (total cost surface)\n";
            }
            std::cout << "  - " << result.path_raster_path << "\n";
            std::cout << "  - " << result.path_lines_path << "\n";
        }

        std::cout << "\nRun another LCPA computation? (yes/no)\n> ";
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
