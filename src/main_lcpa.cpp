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

#include "manifest.h"
#include "neighbourhood.h"   // defines Off and builds the offset tables
#include "costfunctions.h"   // the cost models and the slope cut-off

namespace fs = std::filesystem;

static inline std::string join_path(const std::string& dir, const std::string& file) {
    return (fs::path(dir) / file).string();
}

// ========== GLOBAL SETTINGS ==========
extern bool g_verbose_mode;  // Defined in main_fete.cpp, shared across files
extern bool g_write_manifest;  // Same: one setting, one flag, both modes

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
                             TRAJECTA v1.0.1
                  A SPATIAL MOVEMENT ANALYSIS SOFTWARE
                       Developed by Stefano Apra'
              Institute for the Study of the Ancient World
                    
                    Least-Cost Path Analysis (LCPA)

LCPA computes the optimal path(s) from a single origin point to one or more
destination points across a Digital Elevation Model (DEM). It calculates
the route with the lowest cumulative cost based on terrain-dependent costs.

===============================================================================

MODE: Least-Cost Path Analysis (LCPA)

INPUT REQUIREMENTS:
  - DEM: GeoTIFF file (.tif/.tiff), must be georeferenced
  - Origin: Vector file with exactly 1 point
  - Destinations: Vector file with 1+ points
  - CRS: All files MUST have the same coordinate system

PARAMETERS:
  - Neighbours: Connectivity (8, 16, 24, 32, 64)
  - Slope Units: Automatic (degrees for Tobler; percentage for Marquez-Perez and Irmischer-Clarke)
  - Buffer Radius: Cells around path for visualization
  - Cost Function: Modified Tobler / Marquez-Perez et al. / Irmischer and Clarke
  - CPU Threads: Parallel processing threads
  - Max RAM: Memory limit for processing

OUTPUT:
  - slope_[name].tif: Terrain slope raster
  - cost_surface_[name].tif: Cost surface raster
  - path_raster_[name].tif: Raster showing computed paths
  - path_lines_[name].shp: Polyline shapefile with path geometries

===============================================================================
)";

// ========== STRUCTURES (shared with main_fete.cpp + LCPA-specific) ==========
// NOTE: struct definitions shared across translation units must stay
// token-identical to the ones in main_fete.cpp (ODR)
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

struct LCPAOutput {
    bool success;
    std::string slope_path;
    std::string cost_path;
    std::string additional_cost_path;  // Additional cost surface from polylines
    std::string total_cost_path;       // Total cost surface (base * additional)
    std::string path_raster_path;
    std::string path_lines_path;
    std::string corridor_path;          // empty unless the corridor was asked for
    int num_destinations;
    int total_path_cells;
    double total_cost;
    double time_seconds;
};

// ========== HELPER FUNCTIONS ==========
static inline int idx(int r, int c, int ncols) { return r * ncols + c; }
static inline void idx2coord(int index, int ncols, int& r, int& c) { r = index / ncols; c = index % ncols; }

// Off and the offset tables come from neighbourhood.h, shared with FETE so the
// two modes cannot drift apart. Defined in main_fete.cpp:
bool ask_custom_neighbours(int& num_neighbours);
bool ask_slope_limit(bool& enabled, double& up_deg, double& down_deg);

// ========== COST FUNCTIONS ==========
// Defined in costfunctions.h, shared with FETE.

static inline bool world_to_pixel_northup(double x, double y, const double gt[6], int& col, int& row) {
    if (std::abs(gt[2]) > 1e-12 || std::abs(gt[4]) > 1e-12) return false;
    col = (int)std::floor((x - gt[0]) / gt[1]);
    row = (int)std::floor((y - gt[3]) / gt[5]);
    return true;
}

// ========== FORWARD DECLARATIONS (functions from main_fete.cpp) ==========
extern void print_help();
extern void center_text(const std::string& text, int width = 70);
extern bool check_help_command(const std::string& input);
extern bool check_exit_command(std::string& input);
extern bool valid_output_filename(const std::string& name);
extern void print_filename_error();
extern void safe_getline(std::string& s);
extern std::string get_cpu_model();
extern int64_t get_total_ram_mb();
extern std::string get_file_extension(const std::string& path);
extern bool file_exists(const std::string& path);
extern void print_green_success(const std::string& success);
extern void print_question(const std::string& text);
extern void print_default(const std::string& text);
extern void enable_ansi_colors();
extern void save_config_lcpa(const ConfigLCPA& cfg);
extern ConfigLCPA load_config_lcpa();
extern ValidationResult validate_dem(const std::string& dem_path);
extern std::vector<float> rasterize_polylines_with_costs(const std::string& polylines_path,
    int nrows, int ncols, const double gt[6], int buffer_cells, int max_threads);
extern std::string supported_vector_formats();
extern bool is_supported_vector_format(const std::string& path);
extern GDALDataset* open_vector_dataset(const std::string& path);
extern bool write_gtiff_raster(const std::string& path, int ncols, int nrows,
    const double gt[6], const char* wkt, void* data, GDALDataType dtype,
    const double* nodata = nullptr);
// Defined in main_fete.cpp. The last argument is what a resumed run inherited
// from its checkpoint; LCPA never resumes, so it stays 0.
void print_progress(int current, int total, double elapsed_sec, int bar_width,
                    int done_before);

// ========== LCPA-SPECIFIC FUNCTIONS ==========

// Count point geometries (Point/MultiPoint) in any supported vector file
int count_points_in_shapefile(const std::string& shp_path) {
    GDALDataset* ds = open_vector_dataset(shp_path);
    if (!ds) return -1;
    OGRLayer* layer = ds->GetLayer(0);
    if (!layer) {
        GDALClose(ds);
        return -1;
    }
    int count = 0;
    layer->ResetReading();
    OGRFeature* feat = nullptr;
    while ((feat = layer->GetNextFeature()) != nullptr) {
        OGRGeometry* geom = feat->GetGeometryRef();
        if (geom) {
            OGRwkbGeometryType gtype = wkbFlatten(geom->getGeometryType());
            if (gtype == wkbPoint) count++;
            else if (gtype == wkbMultiPoint) count += geom->toMultiPoint()->getNumGeometries();
        }
        OGRFeature::DestroyFeature(feat);
    }
    GDALClose(ds);
    return count;
}

ValidationResult validate_origin_shapefile(const std::string& shp_path) {
    if (!file_exists(shp_path)) {
        return { false, "ERROR: Origin file not found: " + shp_path };
    }
    if (!is_supported_vector_format(shp_path)) {
        return { false, "ERROR: Origin must be one of " + supported_vector_formats() +
                        ", found: " + get_file_extension(shp_path) };
    }
    int count = count_points_in_shapefile(shp_path);
    if (count == -1) {
        return { false, "ERROR: Cannot open origin file with GDAL/OGR" };
    }
    if (count != 1) {
        return { false, "ERROR: Origin file must contain EXACTLY 1 point, found: " + std::to_string(count) };
    }
    return { true, "" };
}

ValidationResult validate_destinations_shapefile(const std::string& shp_path) {
    if (!file_exists(shp_path)) {
        return { false, "ERROR: Destinations file not found: " + shp_path };
    }
    if (!is_supported_vector_format(shp_path)) {
        return { false, "ERROR: Destinations must be one of " + supported_vector_formats() +
                        ", found: " + get_file_extension(shp_path) };
    }
    int count = count_points_in_shapefile(shp_path);
    if (count == -1) {
        return { false, "ERROR: Cannot open destinations file with GDAL/OGR" };
    }
    if (count < 1) {
        return { false, "ERROR: Destinations file must contain at least 1 point, found: " + std::to_string(count) +
                        "\n       (for .csv, coordinate columns must be named x/y, lon/lat or easting/northing)" };
    }
    return { true, "" };
}

bool get_point_coordinates(const std::string& shp_path, double& x, double& y) {
    GDALDataset* ds = open_vector_dataset(shp_path);
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
    if (!geom) {
        OGRFeature::DestroyFeature(feat);
        GDALClose(ds);
        return false;
    }
    OGRwkbGeometryType gtype = wkbFlatten(geom->getGeometryType());
    OGRPoint* point = nullptr;
    if (gtype == wkbPoint) {
        point = geom->toPoint();
    } else if (gtype == wkbMultiPoint) {
        OGRMultiPoint* mp = geom->toMultiPoint();
        if (mp->getNumGeometries() > 0)
            point = mp->getGeometryRef(0)->toPoint();
    }
    if (!point) {
        OGRFeature::DestroyFeature(feat);
        GDALClose(ds);
        return false;
    }
    x = point->getX();
    y = point->getY();
    OGRFeature::DestroyFeature(feat);
    GDALClose(ds);
    return true;
}

bool get_all_destination_coordinates(const std::string& shp_path, std::vector<std::pair<double, double>>& coords) {
    GDALDataset* ds = open_vector_dataset(shp_path);
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
        if (geom) {
            OGRwkbGeometryType gtype = wkbFlatten(geom->getGeometryType());
            if (gtype == wkbPoint) {
                OGRPoint* point = geom->toPoint();
                coords.push_back({ point->getX(), point->getY() });
            } else if (gtype == wkbMultiPoint) {
                OGRMultiPoint* mp = geom->toMultiPoint();
                for (int gi = 0; gi < mp->getNumGeometries(); gi++) {
                    OGRPoint* point = mp->getGeometryRef(gi)->toPoint();
                    coords.push_back({ point->getX(), point->getY() });
                }
            }
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

    // Points exactly on the max edge belong to the last pixel
    if (col == ncols && x <= gt[0] + ncols * gt[1]) col = ncols - 1;
    if (row == nrows && y >= gt[3] + nrows * gt[5]) row = nrows - 1;

    if (col < 0 || col >= ncols || row < 0 || row >= nrows) {
        return false;  // Point outside DEM extent
    }

    pixel_idx = row * ncols + col;
    return true;
}


// ========== LCPA ALGORITHM ==========

LCPAOutput run_lcpa(const std::string& dem_path, const std::string& out_dir,
    const std::string& slope_filename, const std::string& cost_filename,
    const std::string& path_raster_filename, const std::string& path_lines_filename,
    int origin_idx, const std::vector<int>& destination_indices,
    int buffer_radius, int max_threads, int64_t max_ram_mb,
    int num_neighbours, bool slope_in_degrees, CostFunctionType cost_function,
    const std::string& cost_modifiers_path = "", int polyline_buffer_radius = 0,
    const std::string& cost_raster_path = "",
    const std::string& additional_cost_filename = "", const std::string& total_cost_filename = "",
    double barrier_threshold = 1000.0,
    // Only the manifest needs these: the computation works from the point
    // indices above, but the record has to name the files they came from.
    const std::string& origin_path = "", const std::string& destinations_path = "",
    bool slope_limit_enabled = false,
    double max_slope_up_deg = 90.0, double max_slope_down_deg = 90.0,
    bool want_corridor = false, double corridor_width_percent = 10.0,
    const std::string& corridor_filename = "cost_corridor") {

    const SlopeLimit slope_limit =
        make_slope_limit(slope_limit_enabled, max_slope_up_deg, max_slope_down_deg);

    LCPAOutput output = { false, "", "", "", "", "", "", "", 0, 0, 0.0, 0.0 };
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

    std::vector<Off> offs_storage;
    const int num_offs = neighbourhood::build(num_neighbours, offs_storage);
    const Off* current_offs = offs_storage.data();
    if (num_offs != num_neighbours) {
        std::cout << "NOTE: " << num_neighbours << " is not an admissible neighbourhood size; "
                  << "using " << num_offs << " directions.\n";
        num_neighbours = num_offs;   // so the summary and the manifest agree
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
    for (int i = 0; i < N; ++i) {
        float v = dem[i];
        if (std::isnan(v) || v >= 9999.0f || (has_nodata && v == nodata_f)) {
            passable[i] = 0;
            impassable_count++;
        }
    }
    std::cout << "DEM read: " << nrows << "x" << ncols << " (" << N << " cells)\n";
    if (has_nodata) {
        std::cout << "  NoData value: " << nodata_val << "\n";
    }
    if (impassable_count > 0) {
        std::cout << "  Impassable cells (NoData or DEM >= 9999): " << impassable_count << " (" << std::fixed << std::setprecision(1) << (100.0 * impassable_count / N) << "%)\n";
    }
    auto step1_end = std::chrono::high_resolution_clock::now();
    auto step1_time = std::chrono::duration<double>(step1_end - step1_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << step1_time << " sec\n";

    // Realistic memory estimate: dem, slope, cost surface, multipliers,
    // cumulative cost, predecessor, path raster (~4 bytes each) + masks
    int64_t estimated_ram = 30LL * N / (1024 * 1024);
    if (estimated_ram > max_ram_mb) {
        std::cout << "ERROR: Estimated peak memory use is ~" << estimated_ram
                  << " MB, but max allowed is " << max_ram_mb << " MB\n";
        GDALClose(dem_ds);
        return output;
    }

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

    // An empty filename means "do not save this output". The raster is still
    // computed — later stages read it from memory — only the write is skipped.
    std::string slope_path;
    if (!slope_filename.empty()) {
        slope_path = join_path(out_dir, slope_filename + ".tif");
        if (!write_gtiff_raster(slope_path, ncols, nrows, gt, wkt, slope_data.data(),
                                GDT_Float32, &kOutNoDataD)) {
            GDALClose(dem_ds);
            return output;
        }
        std::cout << "Slope saved\n";
    } else {
        std::cout << "Slope computed (not saved)\n";
    }
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
                // A move the walker would refuse must not enter the average
                // either, or the cost surface would advertise a route the
                // search will never take.
                if (slope_limit.enabled) {
                    if (dz > slope_limit.up_tan * dh) continue;
                    if (-dz > slope_limit.down_tan * dh) continue;
                }
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

    std::string cost_path;
    if (!cost_filename.empty()) {
        cost_path = join_path(out_dir, cost_filename + ".tif");
        if (!write_gtiff_raster(cost_path, ncols, nrows, gt, wkt, cost_surface.data(),
                                GDT_Float32, &kOutNoDataD)) {
            GDALClose(dem_ds);
            return output;
        }
        std::cout << "Base cost surface saved\n";
    } else {
        std::cout << "Base cost surface computed (not saved)\n";
    }
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
        if (!additional_cost_filename.empty()) {
            additional_cost_path = join_path(out_dir, additional_cost_filename + ".tif");
            if (!write_gtiff_raster(additional_cost_path, ncols, nrows, gt, wkt, cost_multipliers.data(), GDT_Float32)) {
                GDALClose(dem_ds);
                return output;
            }
            std::cout << "Additional cost surface saved: " << additional_cost_path << "\n";
        }

        // The total cost surface is a pure output: cost_multipliers, not this
        // product, is what the barrier pass and Dijkstra read. So when it is
        // not saved the multiplication is skipped entirely.
        if (!total_cost_filename.empty()) {
            std::cout << "Calculating total cost surface (base * multipliers)...\n";
            std::vector<float> total_cost_surface(N);

#pragma omp parallel for num_threads(max_threads)
            for (int i = 0; i < N; ++i) {
                // cost_surface holds NoData on impassable cells: don't multiply it
                total_cost_surface[i] = passable[i] ? cost_surface[i] * cost_multipliers[i]
                                                    : kOutNoData;
            }

            total_cost_path = join_path(out_dir, total_cost_filename + ".tif");
            if (!write_gtiff_raster(total_cost_path, ncols, nrows, gt, wkt,
                                    total_cost_surface.data(), GDT_Float32, &kOutNoDataD)) {
                GDALClose(dem_ds);
                return output;
            }
            std::cout << "Total cost surface saved: " << total_cost_path << "\n";
        }
    }

    // Slope and base cost surface are informational outputs only: free them
    // before the search phase to shrink the working set
    std::vector<float>().swap(slope_data);
    std::vector<float>().swap(cost_surface);

    std::cout << "\nRunning Dijkstra algorithm from origin to destinations...\n";
    auto step3_start = std::chrono::high_resolution_clock::now();

    // ---- Treat extreme cost multipliers as hard barriers ----
    // Very large multipliers (e.g. 999999) mean "impassable" in practice;
    // marking them as such keeps the search from flooding the whole raster
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

    // Validate origin/destinations against the passability mask
    if (!passable[origin_idx]) {
        std::cout << "ERROR: Origin point falls on an impassable/NoData cell\n";
        GDALClose(dem_ds);
        return output;
    }
    std::vector<int> usable_destinations;
    usable_destinations.reserve(destination_indices.size());
    for (int d : destination_indices) {
        if (passable[d]) usable_destinations.push_back(d);
    }
    if (usable_destinations.size() < destination_indices.size()) {
        std::cout << "WARNING: " << (destination_indices.size() - usable_destinations.size())
                  << " destination(s) fall on impassable/NoData cells and were excluded\n";
    }
    if (usable_destinations.empty()) {
        std::cout << "ERROR: All destinations fall on impassable/NoData cells\n";
        GDALClose(dem_ds);
        return output;
    }

    const float INF = std::numeric_limits<float>::infinity();
    std::vector<float> cumulative_cost(N, INF);
    std::vector<int> predecessor(N, -1);
    std::vector<uint32_t> path_raster(N, 0);

    // Early termination: stop as soon as every destination is settled
    std::vector<char> is_dest(N, 0);
    int dest_remaining = 0;
    for (int d : usable_destinations) {
        if (!is_dest[d]) { is_dest[d] = 1; dest_remaining++; }
    }

    using pq_entry = std::pair<float, int>;

    // One relaxation body for both directions.
    //
    // `reverse` matters because the cost is anisotropic: walking uphill is not
    // what walking downhill costs, so the cost *from* a cell to the destination
    // is not the cost from the destination to that cell. A forward run answers
    // "what does it cost to reach c from the source"; a reverse run has to
    // answer "what does it cost to get from c to the source", and that means
    // pricing the move u -> v while walking the graph outward from v. The
    // elevation difference is taken the other way round, and the cost modifier
    // belongs to v, which is where that move lands.
    //
    // Getting this wrong would still produce a plausible-looking corridor, just
    // a wrong one, which is why it is spelled out here.
    const auto run_dijkstra = [&](int source, bool reverse,
                                  std::vector<float>& cost_out,
                                  std::vector<int>* pred_out,
                                  bool stop_at_destinations) {
        std::vector<bool> seen(N, false);
        std::priority_queue<pq_entry, std::vector<pq_entry>, std::greater<pq_entry>> queue;
        cost_out.assign(N, INF);
        if (pred_out) pred_out->assign(N, -1);
        int remaining = stop_at_destinations ? dest_remaining : -1;

        cost_out[source] = 0.0f;
        queue.push({ 0.0f, source });

        while (!queue.empty()) {
            auto [cost, v] = queue.top();
            queue.pop();

            if (seen[v]) continue;
            seen[v] = true;
            if (cost >= INF) break;

            if (stop_at_destinations && is_dest[v]) {
                --remaining;
                if (remaining == 0) break;
            }

            int r, c;
            idx2coord(v, ncols, r, c);

            for (int k = 0; k < num_offs; ++k) {
                int nr = r + current_offs[k].dr;
                int nc = c + current_offs[k].dc;
                if (nr < 0 || nr >= nrows || nc < 0 || nc >= ncols) continue;

                int u = idx(nr, nc, ncols);
                if (seen[u]) continue;
                if (!passable[u]) continue;
                double dh = std::sqrt((current_offs[k].dr * res_y) * (current_offs[k].dr * res_y) +
                    (current_offs[k].dc * res_x) * (current_offs[k].dc * res_x));
                // Forward: the move is v -> u. Reverse: it is u -> v.
                double dz = reverse ? (double)dem[v] - (double)dem[u]
                                    : (double)dem[u] - (double)dem[v];
                if (slope_limit.enabled) {
                    if (dz > slope_limit.up_tan * dh) continue;
                    if (-dz > slope_limit.down_tan * dh) continue;
                }
                float edge_cost = apply_cost_function(cost_function, dh, dz);
                // The multiplier belongs to the cell the move lands on.
                edge_cost *= cost_multipliers[reverse ? v : u];

                float new_cost = cost_out[v] + edge_cost;
                if (new_cost < cost_out[u]) {
                    cost_out[u] = new_cost;
                    if (pred_out) (*pred_out)[u] = v;
                    queue.push({ new_cost, u });
                }
            }
        }
    };

    run_dijkstra(origin_idx, /*reverse=*/false, cumulative_cost, &predecessor,
                 /*stop_at_destinations=*/true);

    std::cout << "Dijkstra completed\n";
    auto step3_end = std::chrono::high_resolution_clock::now();
    auto step3_time = std::chrono::duration<double>(step3_end - step3_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << step3_time << " seconds\n";

    std::cout << "\nTracing paths and generating output...\n";

    // Either path output can be skipped by leaving its filename empty, but
    // never both at once — run_lcpa_mode refuses a blank pair, since that
    // would compute the paths and then throw them away.
    const bool want_lines = !path_lines_filename.empty();
    const bool want_raster = !path_raster_filename.empty();

    // Create shapefile for paths
    std::string shp_path;
    GDALDataset* shp_ds = nullptr;
    OGRLayer* layer = nullptr;
    if (want_lines) {
        shp_path = join_path(out_dir, path_lines_filename + ".shp");
        OGRSpatialReference osr;
        osr.importFromWkt(wkt);

        GDALDriver* shp_drv = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
        if (!shp_drv) {
            std::cout << "ERROR: ESRI Shapefile driver not available in this GDAL build\n";
            GDALClose(dem_ds);
            return output;
        }
        shp_ds = shp_drv->Create(shp_path.c_str(), 0, 0, 0, GDT_Unknown, nullptr);
        if (!shp_ds) {
            std::cout << "ERROR: Cannot create path lines shapefile: " << shp_path << "\n";
            std::cout << "       Check that the output directory is writable and the file is not open elsewhere.\n";
            GDALClose(dem_ds);
            return output;
        }
        layer = shp_ds->CreateLayer("paths", &osr, wkbLineString, nullptr);
        if (!layer) {
            std::cout << "ERROR: Cannot create layer in path lines shapefile\n";
            GDALClose(shp_ds);
            GDALClose(dem_ds);
            return output;
        }

        // What each line is, not just that it exists. A polyline on its own
        // cannot say where it started, where it ended or what it cost, and that
        // is exactly what anyone opening the layer in a GIS wants to know.
        // Shapefile field names are capped at 10 characters by the format.
        struct FieldSpec { const char* name; OGRFieldType type; int width; int precision; };
        const FieldSpec fields[] = {
            { "PathID",   OFTInteger, 0,  0 },   // 0-based, matches the raster order
            { "OriginX",  OFTReal,    24, 6 },   // map units of the DEM's CRS
            { "OriginY",  OFTReal,    24, 6 },
            { "DestX",    OFTReal,    24, 6 },
            { "DestY",    OFTReal,    24, 6 },
            { "OriginRC", OFTString,  24, 0 },   // "row,col", for tracing back
            { "DestRC",   OFTString,  24, 0 },
            { "TotalCost", OFTReal,   24, 6 },   // in CostUnits, below
            { "CostUnits", OFTString, 12, 0 },   // "hours" or "kJ/kg"
            { "Length_m",  OFTReal,   24, 3 },   // along the polyline, 3D-free
            { "Cells",     OFTInteger, 0, 0 },   // number of cells traversed
        };
        for (const FieldSpec& f : fields) {
            OGRFieldDefn def(f.name, f.type);
            if (f.width > 0) def.SetWidth(f.width);
            if (f.precision > 0) def.SetPrecision(f.precision);
            layer->CreateField(&def);
        }
    }

    int path_count = 0;
    int total_path_cells = 0;
    double total_cost_accumulated = 0.0;

    // For each destination, trace back the path
    for (int dest_idx : usable_destinations) {
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
        // A broken predecessor chain would otherwise get the origin appended
        // anyway, drawing a line from the wrong start point.
        if (current == -1) {
            std::cout << "Warning: incomplete path for destination at index "
                      << dest_idx << ", skipping\n";
            continue;
        }
        path.push_back(origin_idx);
        std::reverse(path.begin(), path.end());

        // Add to raster. Nothing else reads path_raster, so when it is not
        // being saved the accumulation (and its buffer sweep) is skipped.
        if (want_raster) {
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
        }

        total_path_cells += path.size();
        total_cost_accumulated += cumulative_cost[dest_idx];

        // Add polyline to shapefile. (c+0.5, r+0.5) is the CELL CENTER:
        // without the half-cell offset every vertex sat on the top-left
        // corner and the whole shapefile was shifted ~half a resolution
        // step NW of the path raster.
        if (layer) {
            OGRLineString line;
            double length_m = 0.0;
            double prev_x = 0.0, prev_y = 0.0;
            bool have_prev = false;
            for (int node : path) {
                int r, c;
                idx2coord(node, ncols, r, c);
                double x = gt[0] + (c + 0.5) * gt[1];
                double y = gt[3] + (r + 0.5) * gt[5];
                line.addPoint(x, y);
                if (have_prev) {
                    const double dx = x - prev_x, dy = y - prev_y;
                    length_m += std::sqrt(dx * dx + dy * dy);
                }
                prev_x = x;
                prev_y = y;
                have_prev = true;
            }

            // Planimetric length, in the map units of the DEM's CRS: it is the
            // distance walked on the map, not along the terrain, so on steep
            // ground the real walk is longer. The cost already accounts for the
            // climb; this field does not, and should not be read as if it did.
            int orow, ocol, drow, dcol;
            idx2coord(origin_idx, ncols, orow, ocol);
            idx2coord(dest_idx, ncols, drow, dcol);
            const double ox = gt[0] + (ocol + 0.5) * gt[1];
            const double oy = gt[3] + (orow + 0.5) * gt[5];
            const double dx_ = gt[0] + (dcol + 0.5) * gt[1];
            const double dy_ = gt[3] + (drow + 0.5) * gt[5];

            OGRFeature* feature = OGRFeature::CreateFeature(layer->GetLayerDefn());
            feature->SetField("PathID", path_count);
            feature->SetField("OriginX", ox);
            feature->SetField("OriginY", oy);
            feature->SetField("DestX", dx_);
            feature->SetField("DestY", dy_);
            feature->SetField("OriginRC",
                              (std::to_string(orow) + "," + std::to_string(ocol)).c_str());
            feature->SetField("DestRC",
                              (std::to_string(drow) + "," + std::to_string(dcol)).c_str());
            feature->SetField("TotalCost", (double)cumulative_cost[dest_idx]);
            feature->SetField("CostUnits", cost_function_units(cost_function));
            feature->SetField("Length_m", length_m);
            feature->SetField("Cells", (int)path.size());
            feature->SetGeometry(&line);
            layer->CreateFeature(feature);
            OGRFeature::DestroyFeature(feature);
        }
        // Counted whether or not the shapefile is written: the run report
        // reads it, and PathID keeps the same numbering as before.
        ++path_count;
    }

    if (shp_ds)
        GDALClose(shp_ds);

    // ---- Cost corridor ----
    // A least-cost path is one pixel wide and says nothing about how much
    // choice there was. The corridor does: for every cell it asks what a
    // detour through that cell would cost, as
    //
    //     excess(c) = ( cost(origin -> c) + cost(c -> dest) - best ) / best
    //
    // expressed as a percentage. Zero on the optimal path itself. A narrow
    // corridor means the terrain dictated the route; a wide one means the path
    // drawn on the map is one of many nearly equal options and should not be
    // read as *the* route.
    //
    // The second term needs the cost *to* the destination, which on anisotropic
    // terrain is a different question from the cost *from* it — hence the
    // reverse run.
    std::string corridor_path;
    if (want_corridor && !usable_destinations.empty()) {
        std::cout << "\nComputing cost corridor (" << corridor_width_percent
                  << "% above the optimum)...\n";
        const auto corridor_start = std::chrono::high_resolution_clock::now();

        std::vector<float> best_excess(N, INF);
        std::vector<float> to_dest;
        int done = 0;
        for (int dest_idx : usable_destinations) {
            const float best = cumulative_cost[dest_idx];
            if (!(best < INF) || !(best > 0.0f)) continue;   // unreachable, or degenerate
            run_dijkstra(dest_idx, /*reverse=*/true, to_dest, nullptr,
                         /*stop_at_destinations=*/false);
            for (int i = 0; i < N; ++i) {
                if (!passable[i]) continue;
                const float there = cumulative_cost[i];
                const float back = to_dest[i];
                if (!(there < INF) || !(back < INF)) continue;
                const float excess = ((there + back) - best) / best * 100.0f;
                if (excess < best_excess[i]) best_excess[i] = excess;
            }
            ++done;
            print_progress(done, (int)usable_destinations.size(), -1.0, 40, 0);
        }
        std::cout << "\n";

        // Outside the chosen width the answer is "not in the corridor", which
        // is nodata rather than a very large number: a GIS then draws nothing
        // there instead of stretching the palette to fit the whole map.
        const float kCorridorNoData = -9999.0f;
        std::vector<float> corridor(N, kCorridorNoData);
        int64_t inside = 0;
        for (int i = 0; i < N; ++i) {
            float excess = best_excess[i];
            if (!(excess < INF)) continue;
            // On the optimal path itself the two halves should add up to
            // exactly the best cost, so the excess is zero. In float they add
            // up to very slightly less, and a strict "must not be negative"
            // test threw away precisely the cells the corridor is built
            // around. The negative values are rounding, not information.
            if (excess < 0.0f) excess = 0.0f;
            if (excess <= (float)corridor_width_percent) {
                corridor[i] = excess;
                ++inside;
            }
        }

        corridor_path = join_path(out_dir, corridor_filename + ".tif");
        const double nodata_value = kCorridorNoData;
        if (!write_gtiff_raster(corridor_path, ncols, nrows, gt, wkt, corridor.data(),
                                GDT_Float32, &nodata_value)) {
            GDALClose(dem_ds);
            return output;
        }
        const double corridor_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - corridor_start).count();
        std::cout << "Corridor saved: " << inside << " cells within "
                  << corridor_width_percent << "% of the optimum ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * (double)inside / (double)N) << "% of the raster), "
                  << std::setprecision(2) << corridor_time << " s\n";
    }

    // Write path raster
    std::string path_raster_path;
    if (want_raster) {
        path_raster_path = join_path(out_dir, path_raster_filename + ".tif");
        if (!write_gtiff_raster(path_raster_path, ncols, nrows, gt, wkt, path_raster.data(), GDT_UInt32)) {
            GDALClose(dem_ds);
            return output;
        }
        std::cout << "Paths saved\n";
    } else {
        std::cout << "Path raster not saved (skipped)\n";
    }

    auto global_end = std::chrono::high_resolution_clock::now();
    double global_time = std::chrono::duration<double>(global_end - global_start).count();

    output.success = true;
    output.slope_path = slope_path;
    output.cost_path = cost_path;
    output.additional_cost_path = additional_cost_path;
    output.total_cost_path = total_cost_path;
    output.path_raster_path = path_raster_path;
    output.path_lines_path = shp_path;
    output.corridor_path = corridor_path;
    output.num_destinations = (int)usable_destinations.size();
    output.total_path_cells = total_path_cells;
    output.total_cost = total_cost_accumulated;
    output.time_seconds = global_time;

    // ---- Run manifest ----
    // See the FETE end of run for why this sits here rather than at the start.
    if (g_write_manifest) {
        manifest::Manifest mf("LCPA");
        // The run measured itself; the manifest must not measure the
        // fraction of a second it takes to build.
        mf.setElapsed(global_time);

        mf.section("inputs");
        mf.inputFile("DEM", dem_path);
        mf.inputFile("origin", origin_path);
        mf.inputFile("destinations", destinations_path);
        mf.inputFile("cost modifiers (vector)", cost_modifiers_path);
        mf.inputFile("cost modifiers (raster)", cost_raster_path);

        mf.section("settings");
        mf.kv("neighbours", (long long)num_neighbours);
        mf.kv("cost function", std::string(cost_function_name(cost_function)));
        mf.kv("cost units", std::string(cost_function_units(cost_function)));
        mf.kv("slope units", std::string(slope_in_degrees ? "degrees" : "percent"));
        if (want_corridor) {
            mf.kv("cost corridor width %", corridor_width_percent);
        } else {
            mf.kv("cost corridor", std::string("(not used)"));
        }
        if (slope_limit.enabled) {
            mf.kv("max uphill slope", max_slope_up_deg);
            mf.kv("max downhill slope", max_slope_down_deg);
        } else {
            mf.kv("slope cut-off", std::string("(not used)"));
        }
        mf.kv("path smoothing buffer", (long long)buffer_radius);
        if (!cost_modifiers_path.empty())
            mf.kv("polyline buffer", (long long)polyline_buffer_radius);
        const bool barriers_on = barrier_threshold > 0.0;
        mf.kv("impassable barriers", barriers_on);
        if (barriers_on)
            mf.kv("barrier threshold", barrier_threshold, 1);

        mf.section("hardware");
        mf.kv("CPU", get_cpu_model());
        mf.kv("threads used", (long long)max_threads);
        mf.kv("RAM ceiling (MB)", (long long)max_ram_mb);

        mf.section("results");
        mf.kv("grid", std::to_string(ncols) + " x " + std::to_string(nrows)
                          + " = " + std::to_string((long long)ncols * nrows) + " cells");
        mf.kv("destinations requested", (long long)destination_indices.size());
        mf.kv("destinations reached", (long long)usable_destinations.size());
        mf.kv("paths written", (long long)path_count);
        mf.kv("total path cells", (long long)total_path_cells);
        mf.kv("total accumulated cost", total_cost_accumulated, 2);
        mf.kv("computation time (s)", global_time, 1);

        mf.section("outputs");
        mf.outputFile("paths raster", path_raster_path);
        mf.outputFile("paths vector", shp_path);
        mf.outputFile("slope", slope_path);
        mf.outputFile("cost surface", cost_path);
        mf.outputFile("additional cost", additional_cost_path);
        mf.outputFile("total cost", total_cost_path);

        // Named after whichever main output the user asked for: the raster is
        // optional here, and so is the shapefile, but never both.
        const std::string primary = !path_raster_filename.empty() ? path_raster_filename
                                                                  : path_lines_filename;
        const std::string mf_path = manifest::pathFor(out_dir, primary);
        std::string mf_error;
        if (mf.write(mf_path, &mf_error)) {
            std::cout << "Run manifest: " << mf_path << "\n";
        } else {
            std::cout << "WARNING: the run manifest could not be written ("
                      << mf_error << ")\n";
        }
    }

    GDALClose(dem_ds);

    return output;
}

// ========== MAIN PROGRAM ==========

int run_lcpa_mode() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    center_text("TRAJECTA v1.0.1 - A SPATIAL MOVEMENT ANALYSIS SOFTWARE", 70);
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
    ConfigLCPA saved_config = load_config_lcpa();

    int max_available_threads = omp_get_max_threads();
    std::string cpu_model = get_cpu_model();
    int64_t total_ram_mb = get_total_ram_mb();

    std::cout << "System Information:\n";
    std::cout << "  Available CPU threads: " << max_available_threads << "\n";
    std::cout << "  CPU Model: " << cpu_model << "\n";
    std::cout << "  Total RAM: " << (total_ram_mb / 1024) << " GB\n\n";

    int max_threads = std::max(1, max_available_threads - 4);
    int64_t max_ram_mb = 8192;

    std::string dem_path = saved_config.dem_path;
    std::string out_dir = saved_config.out_dir;
    std::string origin_shp_path = saved_config.origin_path;  // Origin shapefile path
    std::string destinations_shp_path = saved_config.destinations_path;  // Destinations shapefile path
    std::string cost_modifiers_path = saved_config.cost_modifiers_path;
    std::string cost_raster_path = saved_config.cost_raster_path;
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
    double barrier_threshold = 1000.0;
    int num_neighbours = 16;
    bool slope_in_degrees = true;
    // Slope cut-off, off unless asked for: a limit nobody chose would
    // silently change every result.
    bool slope_limit_enabled = false;
    double max_slope_up_deg = 30.0;
    double max_slope_down_deg = 30.0;
    // Cost corridor: off unless asked for, one extra search per destination.
    bool want_corridor = false;
    double corridor_width_percent = 10.0;
    std::string corridor_filename = "cost_corridor";
    CostFunctionType cost_function = TOBLER_WHITE_2015;

    // Every LCPA run (including re-runs) starts from here: thread selection,
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
            std::cout << "  Recommended: at least 8192 MB. The analysis needs far less memory\n";
            std::cout << "  than most machines have, and a higher ceiling does not speed it up.\n";
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
                max_ram_mb = 8192;
            }
            break;
        }

        std::cout << "Using maximum " << max_ram_mb << " MB RAM\n";

        // ---- Run manifest (opt-out) ----
        // Same question, same wording and same default as FETE: it is the same
        // setting, and the interface drives both through one checkbox.
        while (true) {
            print_question("\nWrite a run manifest next to the results? (yes/no):\n");
            std::cout << "  A text file recording every input, setting and output of this\n";
            std::cout << "  run, so the results can be traced back and reproduced later.\n";
            std::cout << "  Costs a few seconds: each input file is hashed.\n";
            std::cout << "  Default: YES\n";
            std::cout << "> ";
            std::string mf_input;
            safe_getline(mf_input);

            if (check_exit_command(mf_input)) {
                return 0;
            }
            if (check_help_command(mf_input)) {
                continue;
            }
            // Normalised here rather than through main_fete.cpp's helpers,
            // which are file-local: leading spaces and capitals are the two
            // ways a typed answer differs from the expected one.
            std::string mf;
            for (char c : mf_input) {
                if (mf.empty() && (c == ' ' || c == '\t')) continue;
                mf += (char)std::tolower((unsigned char)c);
            }
            while (!mf.empty() && (mf.back() == ' ' || mf.back() == '\t' || mf.back() == '\r'))
                mf.pop_back();
            if (mf.empty() || mf == "yes" || mf == "y") {
                g_write_manifest = true;
                break;
            }
            if (mf == "no" || mf == "n") {
                g_write_manifest = false;
                break;
            }
            std::cout << "ERROR: Please answer yes or no\n";
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

            // Origin file (with exactly 1 point)
            while (true) {
                print_question("\nEnter path to ORIGIN file with exactly 1 point (" + supported_vector_formats() + "):\n");
                if (!origin_shp_path.empty()) {
                    std::cout << "  "; print_default("Default: " + origin_shp_path); std::cout << "\n";
                }
                std::cout << "  Example: C:\\path\\to\\Origin.shp (or .csv with x/y, lon/lat columns)\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (!input.empty()) origin_shp_path = input;
                if (origin_shp_path.empty()) {
                    std::cout << "ERROR: Origin file path cannot be empty!\n";
                    continue;
                }

                ValidationResult val = validate_origin_shapefile(origin_shp_path);
                if (!val.is_valid) {
                    std::cout << val.error_message << "\n";
                    continue;
                }
                break;
            }

            // Destinations file (with 1+ points)
            while (true) {
                print_question("\nEnter path to DESTINATIONS file with 1+ points (" + supported_vector_formats() + "):\n");
                if (!destinations_shp_path.empty()) {
                    std::cout << "  "; print_default("Default: " + destinations_shp_path); std::cout << "\n";
                }
                std::cout << "  Example: C:\\path\\to\\Destinations.shp (or .csv with x/y, lon/lat columns)\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) continue;
                if (!input.empty()) destinations_shp_path = input;
                if (destinations_shp_path.empty()) {
                    std::cout << "ERROR: Destinations file path cannot be empty!\n";
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

                // If user left both blank, inform
                if (cost_modifiers_path.empty() && cost_raster_path.empty()) {
                    std::cout << "No cost modifiers specified, continuing without modifiers.\n";
                }

                // --- Step 3: Barrier threshold ---
                if (!cost_modifiers_path.empty() || !cost_raster_path.empty()) {
                    print_question("\nTreat extreme cost multipliers as impassable barriers? Enter threshold:\n");
                    std::cout << "  Cells whose multiplier is >= the threshold become hard barriers\n";
                    std::cout << "  (excluded from movement, points on them are skipped).\n";
                    std::cout << "  Recommended when obstacles use very large multipliers (e.g. 999999):\n";
                    std::cout << "  without a threshold the search floods the entire raster and the\n";
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

            // Parameters configuration.
            // Each prompt runs in its own loop: 'help' re-asks THIS question
            // instead of restarting the whole setup from the thread count.
            while (true) {
                print_question("\nSelect number of neighbours for cost surface calculation:\n");
                std::cout << "  1) 8-connectivity (3x3 grid)\n";
                std::cout << "  2) 16-connectivity (knight moves) "; print_default("[DEFAULT]"); std::cout << "\n";
                std::cout << "  3) 24-connectivity (extended)\n";
                std::cout << "  4) 32-connectivity (more extended)\n";
                std::cout << "  5) 64-connectivity (full extended)\n";
                std::cout << "  6) Custom (enter the number of directions yourself)\n";
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
                    case 6: {
                        if (!ask_custom_neighbours(num_neighbours)) return 0;
                        break;
                    }
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
                std::cout << "  3) Irmischer and Clarke (2017), on-path male\n";
                std::cout << "  4) Herzog (2013) metabolic cost  -- energy, kJ/kg, not hours\n";
                std::cout << "  5) Campbell et al. (2019), 5th percentile (ordinary hiking)\n";
                std::cout << "  6) Campbell et al. (2019), 50th percentile (includes joggers)\n";
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
                    else if (choice == 4) cost_function = HERZOG_2013;
                    else if (choice == 5) cost_function = CAMPBELL_2019_P5;
                    else if (choice == 6) cost_function = CAMPBELL_2019_P50;
                    else                  cost_function = TOBLER_WHITE_2015;
                }
                catch (...) {
                    cost_function = TOBLER_WHITE_2015;
                }
                break;
            }

            if (cost_function == HERZOG_2013) {
                std::cout << "NOTE: this cost function measures energy. Every cost in this run,\n";
                std::cout << "      including the cost surfaces, is in kJ per kg of walker,\n";
                std::cout << "      not in hours, and cannot be compared with the other models.\n";
            }

            if (!ask_slope_limit(slope_limit_enabled, max_slope_up_deg, max_slope_down_deg))
                return 0;

            // Only the unit of the exported slope raster: Tobler is usually read
            // in degrees and Campbell is defined in them, the others in percent.
            slope_in_degrees = (cost_function == TOBLER_WHITE_2015
                                || cost_function == CAMPBELL_2019_P5
                                || cost_function == CAMPBELL_2019_P50);

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

            // Output filenames - FIRST RUN (with Examples and Help)
            while (true) {
                std::cout << "\nEnter slope raster filename (without extension):\n";
                std::cout << "  Example: slope\n";
                std::cout << "  Leave blank to skip this output\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
                if (input.empty()) {
                    slope_filename.clear();  // computed, but not written
                    break;
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
                std::cout << "  Leave blank to skip this output\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
                if (input.empty()) {
                    cost_filename.clear();  // computed, but not written
                    break;
                }
                if (!valid_output_filename(input)) { print_filename_error(); continue; }
                cost_filename = input;
                if (cost_filename.length() >= 4 && cost_filename.substr(cost_filename.length() - 4) == ".tif") {
                    cost_filename = cost_filename.substr(0, cost_filename.length() - 4);
                }
                break;
            }

            // If cost modifiers were added, ask for additional and total cost
            // surface filenames. Must match run_lcpa's own condition: it
            // writes both rasters whenever ANY modifier is active, so gating
            // on the vector path alone left the names empty and produced a
            // file literally called ".tif" in raster-only runs.
            if (!cost_modifiers_path.empty() || !cost_raster_path.empty()) {
                while (true) {
                    std::cout << "\nEnter additional cost surface raster filename (without extension):\n";
                    std::cout << "  This is the rasterized polylines with cost multipliers\n";
                    std::cout << "  Example: cost_surface_additional\n";
                    std::cout << "  Leave blank to skip this output\n";
                    std::cout << "> ";
                    std::string input;
                    safe_getline(input);
                    if (check_exit_command(input)) return 0;
                    if (check_help_command(input)) {
                        std::cout << HELP_TEXT_LCPA;
                        continue;
                    }
                    if (input.empty()) {
                        additional_cost_filename.clear();  // computed, but not written
                        break;
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
                    std::cout << "  Leave blank to skip this output\n";
                    std::cout << "> ";
                    std::string input;
                    safe_getline(input);
                    if (check_exit_command(input)) return 0;
                    if (check_help_command(input)) {
                        std::cout << HELP_TEXT_LCPA;
                        continue;
                    }
                    if (input.empty()) {
                        total_cost_filename.clear();  // not computed and not written
                        break;
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
                std::cout << "\nEnter path raster filename (without extension):\n";
                std::cout << "  Example: raster_lcps\n";
                std::cout << "  Leave blank to skip this output\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
                if (input.empty()) {
                    // Allowed on its own: the path lines prompt that follows
                    // refuses to leave both path outputs blank.
                    path_raster_filename.clear();
                    break;
                }
                if (!valid_output_filename(input)) { print_filename_error(); continue; }
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
                if (!path_raster_filename.empty())
                    std::cout << "  Leave blank to skip this output\n";
                else
                    std::cout << "  Required: the path raster was skipped\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
                if (input.empty()) {
                    // Skipping both would compute every path and then discard
                    // it, which is never what the user meant.
                    if (path_raster_filename.empty()) {
                        std::cout << "ERROR: at least one of the path raster and the path lines "
                                     "shapefile must be saved!\n";
                        continue;
                    }
                    path_lines_filename.clear();
                    break;
                }
                if (!valid_output_filename(input)) { print_filename_error(); continue; }
                path_lines_filename = input;
                if (path_lines_filename.length() >= 4 && path_lines_filename.substr(path_lines_filename.length() - 4) == ".shp") {
                    path_lines_filename = path_lines_filename.substr(0, path_lines_filename.length() - 4);
                }
                break;
            }
        }

        // ===== COST CORRIDOR =====
        while (true) {
            print_question("\nAlso compute the cost corridor? (yes/no):\n");
            std::cout << "  A least-cost path is one pixel wide and cannot say how much\n";
            std::cout << "  choice there was. The corridor shows every cell a detour could\n";
            std::cout << "  pass through for a given extra cost: narrow means the terrain\n";
            std::cout << "  dictated the route, wide means the line is one of many.\n";
            std::cout << "  It costs one extra search per destination.\n";
            std::cout << "  "; print_default("Default: no"); std::cout << "\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return 0;
            if (check_help_command(input)) { std::cout << HELP_TEXT_LCPA; continue; }
            // Same normalisation as the manifest prompt above: main_fete.cpp's
            // helpers are file-local, and leading spaces and capitals are the
            // two ways a typed answer differs from the expected one.
            std::string a;
            for (char c : input) {
                if (a.empty() && (c == ' ' || c == '\t')) continue;
                a += (char)std::tolower((unsigned char)c);
            }
            while (!a.empty() && (a.back() == ' ' || a.back() == '\t' || a.back() == '\r'))
                a.pop_back();
            if (a.empty() || a == "no" || a == "n") { want_corridor = false; break; }
            if (a != "yes" && a != "y") {
                std::cout << "ERROR: Please enter 'yes' or 'no'\n";
                continue;
            }
            want_corridor = true;
            break;
        }
        if (want_corridor) {
            while (true) {
                print_question("\nCorridor width, as a percentage above the optimum (1-500):\n");
                std::cout << "  5 keeps only what is nearly as cheap as the best route;\n";
                std::cout << "  25 shows the broad band of plausible alternatives.\n";
                std::cout << "  "; print_default("Default: 10"); std::cout << "\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) { std::cout << HELP_TEXT_LCPA; continue; }
                if (input.empty()) { corridor_width_percent = 10.0; break; }
                try {
                    const double v = std::stod(input);
                    if (v < 1.0 || v > 500.0) {
                        std::cout << "ERROR: Enter a percentage between 1 and 500\n";
                        continue;
                    }
                    corridor_width_percent = v;
                    break;
                } catch (...) {
                    std::cout << "ERROR: Invalid number\n";
                }
            }
            while (true) {
                std::cout << "\nEnter cost corridor raster filename (without extension):\n";
                std::cout << "  Example: cost_corridor\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) { std::cout << HELP_TEXT_LCPA; continue; }
                if (input.empty()) { corridor_filename = "cost_corridor"; break; }
                if (!valid_output_filename(input)) { print_filename_error(); continue; }
                corridor_filename = input;
                if (corridor_filename.length() >= 4
                    && corridor_filename.substr(corridor_filename.length() - 4) == ".tif") {
                    corridor_filename = corridor_filename.substr(0, corridor_filename.length() - 4);
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
        // An empty name means the file is not written at all, so say so rather
        // than printing a bare ".tif".
        auto print_output_name = [](const char* label, const std::string& name,
                                    const char* ext) {
            std::cout << "  " << label << ": "
                      << (name.empty() ? std::string("not saved") : name + ext) << "\n";
        };
        print_output_name("Slope filename", slope_filename, ".tif");
        print_output_name("Cost filename", cost_filename, ".tif");
        if (!cost_modifiers_path.empty() || !cost_raster_path.empty()) {
            print_output_name("Additional cost filename", additional_cost_filename, ".tif");
            print_output_name("Total cost filename", total_cost_filename, ".tif");
        }
        print_output_name("Path raster filename", path_raster_filename, ".tif");
        print_output_name("Path lines filename", path_lines_filename, ".shp");
        std::cout << "  Buffer radius: " << buffer_radius << " cells\n";
        std::cout << "  Neighbours: " << num_neighbours << "-connectivity\n";
        std::cout << "  Slope units: " << (slope_in_degrees ? "degrees" : "percentage") << "\n";
        std::cout << "  Cost function: " << cost_function_name(cost_function) << "\n";
        std::cout << "  Cost units: " << cost_function_units(cost_function) << "\n";
        if (slope_limit_enabled) {
            std::cout << "  Slope cut-off: refuse above " << max_slope_up_deg
                      << " deg uphill / " << max_slope_down_deg << " deg downhill\n";
        } else {
            std::cout << "  Slope cut-off: none\n";
        }
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

        LCPAOutput result = run_lcpa(dem_path, out_dir, slope_filename, cost_filename,
            path_raster_filename, path_lines_filename,
            origin_idx, destination_indices,
            buffer_radius, max_threads, max_ram_mb,
            num_neighbours, slope_in_degrees, cost_function,
            cost_modifiers_path, polyline_buffer_radius, cost_raster_path,
            additional_cost_filename, total_cost_filename, barrier_threshold,
            origin_shp_path, destinations_shp_path,
            slope_limit_enabled, max_slope_up_deg, max_slope_down_deg,
            want_corridor, corridor_width_percent, corridor_filename);

        if (result.success) {
            ConfigLCPA to_save = { dem_path, origin_shp_path, destinations_shp_path, out_dir, cost_modifiers_path, cost_raster_path };
            save_config_lcpa(to_save);

            print_green_success("LCPA successfully computed!\n");
            std::cout << "\nOutput Summary:\n";
            std::cout << "  Total time: " << std::fixed << std::setprecision(2) << result.time_seconds << " sec\n";
            std::cout << "  Destinations processed: " << result.num_destinations << "\n";
            std::cout << "  Total path cells: " << result.total_path_cells << "\n";
            std::cout << "  Total cost accumulated: " << std::fixed << std::setprecision(2) << result.total_cost << "\n";
            std::cout << "\nOutput Files:\n";
            // A skipped output leaves its path empty: list only what was
            // actually written, testing each one on its own.
            if (!result.slope_path.empty())
                std::cout << "  - " << result.slope_path << "\n";
            if (!result.cost_path.empty())
                std::cout << "  - " << result.cost_path << " (base cost surface)\n";
            if (!result.additional_cost_path.empty())
                std::cout << "  - " << result.additional_cost_path << " (additional cost multipliers)\n";
            if (!result.total_cost_path.empty())
                std::cout << "  - " << result.total_cost_path << " (total cost surface)\n";
            if (!result.path_raster_path.empty())
                std::cout << "  - " << result.path_raster_path << "\n";
            if (!result.path_lines_path.empty())
                std::cout << "  - " << result.path_lines_path << "\n";
        }

        print_question("\nRun another LCPA computation? (yes/no)\n"); std::cout << "> ";
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
