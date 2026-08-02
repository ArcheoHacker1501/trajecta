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
extern bool g_verbose_mode;  // Defined in main_fete.cpp, shared across files

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
                             TRAJECTA v1.0.0
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
        case IRMISCHER_CLARKE_2017:    return irmischer_clarke_2017(dh_m, dz_m);
        default:                       return tobler_white_2015(dh_m, dz_m);
    }
}

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
    double barrier_threshold = 1000.0) {

    LCPAOutput output = { false, "", "", "", "", "", "", 0, 0, 0.0, 0.0 };
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
        additional_cost_path = join_path(out_dir, additional_cost_filename + ".tif");
        if (!write_gtiff_raster(additional_cost_path, ncols, nrows, gt, wkt, cost_multipliers.data(), GDT_Float32)) {
            GDALClose(dem_ds);
            return output;
        }
        std::cout << "Additional cost surface saved: " << additional_cost_path << "\n";

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
    std::vector<bool> visited(N, false);
    std::vector<uint32_t> path_raster(N, 0);

    // Early termination: stop as soon as every destination is settled
    std::vector<char> is_dest(N, 0);
    int dest_remaining = 0;
    for (int d : usable_destinations) {
        if (!is_dest[d]) { is_dest[d] = 1; dest_remaining++; }
    }

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

        if (is_dest[v]) {
            --dest_remaining;
            if (dest_remaining == 0) break;
        }

        int r, c;
        idx2coord(v, ncols, r, c);

        for (int k = 0; k < num_offs; ++k) {
            int nr = r + current_offs[k].dr;
            int nc = c + current_offs[k].dc;
            if (nr < 0 || nr >= nrows || nc < 0 || nc >= ncols) continue;

            int u = idx(nr, nc, ncols);
            if (visited[u]) continue;
            if (!passable[u]) continue;
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
    if (!shp_drv) {
        std::cout << "ERROR: ESRI Shapefile driver not available in this GDAL build\n";
        GDALClose(dem_ds);
        return output;
    }
    GDALDataset* shp_ds = shp_drv->Create(shp_path.c_str(), 0, 0, 0, GDT_Unknown, nullptr);
    if (!shp_ds) {
        std::cout << "ERROR: Cannot create path lines shapefile: " << shp_path << "\n";
        std::cout << "       Check that the output directory is writable and the file is not open elsewhere.\n";
        GDALClose(dem_ds);
        return output;
    }
    OGRLayer* layer = shp_ds->CreateLayer("paths", &osr, wkbLineString, nullptr);
    if (!layer) {
        std::cout << "ERROR: Cannot create layer in path lines shapefile\n";
        GDALClose(shp_ds);
        GDALClose(dem_ds);
        return output;
    }

    OGRFieldDefn oField("PathID", OFTInteger);
    layer->CreateField(&oField);

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

        // Add polyline to shapefile. (c+0.5, r+0.5) is the CELL CENTER:
        // without the half-cell offset every vertex sat on the top-left
        // corner and the whole shapefile was shifted ~half a resolution
        // step NW of the path raster.
        OGRLineString line;
        for (int node : path) {
            int r, c;
            idx2coord(node, ncols, r, c);
            double x = gt[0] + (c + 0.5) * gt[1];
            double y = gt[3] + (r + 0.5) * gt[5];
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
    if (!write_gtiff_raster(path_raster_path, ncols, nrows, gt, wkt, path_raster.data(), GDT_UInt32)) {
        GDALClose(dem_ds);
        return output;
    }

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
    output.num_destinations = (int)usable_destinations.size();
    output.total_path_cells = total_path_cells;
    output.total_cost = total_cost_accumulated;
    output.time_seconds = global_time;

    GDALClose(dem_ds);

    return output;
}

// ========== MAIN PROGRAM ==========

int run_lcpa_mode() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    center_text("TRAJECTA v1.0.0 - A SPATIAL MOVEMENT ANALYSIS SOFTWARE", 70);
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
    int64_t max_ram_mb = 4096;

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

            // Output filenames - FIRST RUN (with Examples and Help)
            while (true) {
                std::cout << "\nEnter slope raster filename (without extension):\n";
                std::cout << "  Example: slope\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
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
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
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
            // surface filenames. Must match run_lcpa's own condition: it
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
                    if (check_help_command(input)) {
                        std::cout << HELP_TEXT_LCPA;
                        continue;
                    }
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
                    if (check_help_command(input)) {
                        std::cout << HELP_TEXT_LCPA;
                        continue;
                    }
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
                std::cout << "\nEnter path raster filename (without extension):\n";
                std::cout << "  Example: raster_lcps\n";
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
                if (input.empty()) {
                    std::cout << "ERROR: Path raster filename cannot be empty!\n";
                    continue;
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
                std::cout << "> ";
                std::string input;
                safe_getline(input);
                if (check_exit_command(input)) return 0;
                if (check_help_command(input)) {
                    std::cout << HELP_TEXT_LCPA;
                    continue;
                }
                if (input.empty()) {
                    std::cout << "ERROR: Path lines filename cannot be empty!\n";
                    continue;
                }
                if (!valid_output_filename(input)) { print_filename_error(); continue; }
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
        std::cout << "  Cost function: " << (cost_function == TOBLER_WHITE_2015 ? "Modified Tobler (White 2015)" : (cost_function == MARQUEZ_PEREZ_ET_AL_2017 ? "Marquez-Perez et al. (2017)" : "Irmischer and Clarke (2017)")) << "\n";
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
            additional_cost_filename, total_cost_filename, barrier_threshold);

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
            std::cout << "  - " << result.slope_path << "\n";
            std::cout << "  - " << result.cost_path << " (base cost surface)\n";
            if (!result.additional_cost_path.empty()) {
                std::cout << "  - " << result.additional_cost_path << " (additional cost multipliers)\n";
                std::cout << "  - " << result.total_cost_path << " (total cost surface)\n";
            }
            std::cout << "  - " << result.path_raster_path << "\n";
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
