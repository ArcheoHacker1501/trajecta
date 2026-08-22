// ============================================================================
// TRAJECTA - NNI mode: Natural Neighbour Interpolation (discrete Sibson)
//
// Takes a density raster (typically the FETE output), treats every cell at or
// above a threshold as a sample point, and interpolates a smooth continuous
// surface over the whole grid with the discrete Sibson method (Park et al.
// 2006): the value at a query cell is the average of the nearest-sample
// values of every cell the query would "steal" from the existing discrete
// Voronoi diagram. Runs entirely on the raster grid - no triangulation, no
// external dependencies - and parallelizes with OpenMP.
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <filesystem>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <omp.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "gdal_priv.h"

namespace fs = std::filesystem;

// ---- shared helpers defined in main_fete.cpp ----
extern bool g_verbose_mode;
void print_question(const std::string& text);
void print_default(const std::string& text);
void print_green_success(const std::string& success);
void center_text(const std::string& text, int width);
bool check_exit_command(std::string& input);
void safe_getline(std::string& s);
// Defined in main_fete.cpp. The last parameter is what a resumed FETE run
// inherited from its checkpoint; interpolation never resumes, so it stays 0.
void print_progress(int current, int total, double elapsed_sec, int bar_width,
                    int done_before);
bool write_gtiff_raster(const std::string& path, int ncols, int nrows,
    const double gt[6], const char* wkt, void* data, GDALDataType dtype,
    const double* nodata = nullptr);
std::string config_file_path(const char* name);

namespace {

inline std::string interp_join_path(const std::string& dir, const std::string& file) {
    return (fs::path(dir) / file).string();
}

// Small inline help; the global 'help' command belongs to the FETE/LCPA flows.
bool interp_help(const std::string& input, const char* text) {
    if (input == "help" || input == "HELP" || input == "Help") {
        std::cout << "\n" << text << "\n\n";
        return true;
    }
    return false;
}

struct InterpConfig {
    std::string input_path;
    std::string out_dir;
};

InterpConfig load_interp_config() {
    InterpConfig cfg;
    std::ifstream f(config_file_path("interp_config.txt"));
    if (f.is_open()) {
        std::getline(f, cfg.input_path);
        std::getline(f, cfg.out_dir);
    }
    return cfg;
}

void save_interp_config(const InterpConfig& cfg) {
    std::ofstream f(config_file_path("interp_config.txt"));
    f << cfg.input_path << "\n" << cfg.out_dir << "\n";
}

// Exact Euclidean nearest-sample transform (Felzenszwalb & Huttenlocher 2004,
// extended to carry the source cell). For every cell: squared distance to the
// nearest sample cell and that sample's linear index.
void nearest_sample_transform(const std::vector<uint8_t>& is_sample,
    int ncols, int nrows,
    std::vector<float>& dist2, std::vector<int32_t>& src)
{
    const double INF = 1e30;
    const size_t N = (size_t)ncols * nrows;
    dist2.assign(N, 0.0f);
    src.assign(N, -1);

    // Phase 1 - per column: vertical distance to the nearest sample in that
    // column and the row it sits on (-1 if the column has no sample).
    std::vector<int32_t> vrow(N, -1);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < ncols; ++c) {
        int last = -1;
        for (int r = 0; r < nrows; ++r) {           // downward sweep
            if (is_sample[(size_t)r * ncols + c]) last = r;
            vrow[(size_t)r * ncols + c] = last;
        }
        last = -1;
        for (int r = nrows - 1; r >= 0; --r) {      // upward sweep
            if (is_sample[(size_t)r * ncols + c]) last = r;
            const size_t i = (size_t)r * ncols + c;
            const int down = vrow[i];
            if (last >= 0 && (down < 0 || (r - down) > (last - r)))
                vrow[i] = last;
        }
    }

    // Phase 2 - per row: 1D lower envelope of the parabolas
    // y(c) = (c - c')^2 + g(c')^2 (Felzenszwalb & Huttenlocher), keeping the
    // winning column. Columns without samples carry a huge-but-finite offset,
    // so they only win where no real sample is reachable.
    const double ZINF = std::numeric_limits<double>::infinity();
#pragma omp parallel
    {
        std::vector<double> g2(ncols);
        std::vector<int> v(ncols);           // columns of the envelope parabolas
        std::vector<double> z(ncols + 1);    // boundaries between parabolas
#pragma omp for schedule(static)
        for (int r = 0; r < nrows; ++r) {
            for (int c = 0; c < ncols; ++c) {
                const int sr = vrow[(size_t)r * ncols + c];
                g2[c] = (sr < 0) ? INF : (double)(r - sr) * (r - sr);
            }
            int k = 0;
            v[0] = 0;
            z[0] = -ZINF;
            z[1] = ZINF;
            for (int c = 1; c < ncols; ++c) {
                const auto intersect = [&](int p) {
                    return ((g2[c] + (double)c * c) - (g2[p] + (double)p * p))
                           / (2.0 * (c - p));
                };
                double s = intersect(v[k]);
                while (s <= z[k]) {          // z[0] = -inf guarantees k >= 0
                    --k;
                    s = intersect(v[k]);
                }
                ++k;
                v[k] = c;
                z[k] = s;
                z[k + 1] = ZINF;
            }
            k = 0;
            for (int c = 0; c < ncols; ++c) {
                while (z[k + 1] < (double)c) ++k;
                const int vc = v[k];
                const size_t i = (size_t)r * ncols + c;
                const int sr = vrow[(size_t)r * ncols + vc];
                if (sr < 0) {                // no reachable sample at all
                    dist2[i] = (float)INF;
                    src[i] = -1;
                } else {
                    dist2[i] = (float)((double)(c - vc) * (c - vc) + g2[vc]);
                    src[i] = sr * ncols + vc;
                }
            }
        }
    }
}

} // namespace

// Entry point called from main() when the user picks mode 3.
int run_interp_mode() {
    std::cout << std::string(70, '=') << "\n";
    center_text("NNI MODE", 70);
    center_text("Natural Neighbour Interpolation (discrete Sibson)", 70);
    std::cout << std::string(70, '=') << "\n\n";

    GDALAllRegister();

    InterpConfig saved = load_interp_config();

    const int max_available_threads = omp_get_max_threads();
    int max_threads = std::max(1, max_available_threads - 4);

    // ===== CPU THREADS =====
    while (true) {
        print_question("Enter maximum CPU threads to use (1-" + std::to_string(max_available_threads) + "):\n");
        std::cout << "  Recommended: " << std::max(1, max_available_threads - 4) << " (reserve 4 cores for system)\n";
        std::cout << "> ";
        std::string input;
        safe_getline(input);
        if (check_exit_command(input)) return 0;
        if (interp_help(input, "Number of parallel threads used for the interpolation.")) continue;
        try {
            max_threads = std::stoi(input);
        } catch (...) {
            max_threads = std::max(1, max_available_threads - 4);
        }
        max_threads = std::max(1, std::min(max_threads, max_available_threads));
        break;
    }
    std::cout << "Using " << max_threads << " threads\n\n";
    omp_set_num_threads(max_threads);

    while (true) {  // one full interpolation per loop turn
        std::string input_path = saved.input_path;
        std::string out_dir = saved.out_dir;

        // ===== INPUT DENSITY RASTER =====
        GDALDataset* ds = nullptr;
        while (true) {
            print_question("Enter path to density raster file (.tif):\n");
            if (!input_path.empty()) {
                std::cout << "  "; print_default("Default: " + input_path); std::cout << "\n";
            }
            std::cout << "  Example: C:\\path\\to\\FETE_density.tif\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return 0;
            if (interp_help(input, "The raster whose non-zero cells are interpolated,\n"
                                   "typically the FETE density output.")) continue;
            if (!input.empty()) input_path = input;
            if (input_path.empty()) {
                std::cout << "ERROR: Input raster path cannot be empty!\n";
                continue;
            }
            ds = (GDALDataset*)GDALOpen(input_path.c_str(), GA_ReadOnly);
            if (!ds) {
                std::cout << "ERROR: Cannot open raster: " << input_path << "\n";
                input_path.clear();
                continue;
            }
            break;
        }

        const int ncols = ds->GetRasterXSize();
        const int nrows = ds->GetRasterYSize();
        const size_t N = (size_t)ncols * nrows;
        double gt[6] = { 0, 1, 0, 0, 0, 1 };
        ds->GetGeoTransform(gt);
        const char* wkt = ds->GetProjectionRef();

        std::cout << "Raster loaded: " << ncols << " x " << nrows << " cells\n\n";

        if (N > (size_t)std::numeric_limits<int32_t>::max()) {
            std::cout << "ERROR: Raster too large for NNI mode (max "
                      << std::numeric_limits<int32_t>::max() << " cells)\n";
            GDALClose(ds);
            return 1;
        }

        // ===== OUTPUT DIRECTORY =====
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
            if (interp_help(input, "Folder where the interpolated raster is written.")) continue;
            if (!input.empty()) out_dir = input;
            if (out_dir.empty()) {
                std::cout << "ERROR: Output directory cannot be empty!\n";
                continue;
            }
            // Create it NOW: FETE/LCPA do the same, and failing here beats
            // failing after the whole interpolation has run.
            std::error_code ec;
            fs::create_directories(out_dir, ec);
            if (ec) {
                std::cout << "ERROR: Cannot create output directory: " << out_dir
                          << " (" << ec.message() << ")\n";
                out_dir.clear();
                continue;
            }
            break;
        }

        // ===== READ BAND & SAMPLE THRESHOLD =====
        std::cout << "\nLoading density raster...\n";
        std::vector<float> value(N);
        GDALRasterBand* band = ds->GetRasterBand(1);
        if (band->RasterIO(GF_Read, 0, 0, ncols, nrows, value.data(),
                           ncols, nrows, GDT_Float32, 0, 0) != CE_None) {
            std::cout << "ERROR: Failed reading raster data\n";
            GDALClose(ds);
            return 1;
        }
        int has_nodata = 0;
        const double nodata = band->GetNoDataValue(&has_nodata);
        GDALClose(ds);

        double min_threshold = 1.0;
        while (true) {
            print_question("\nEnter minimum density value for sample cells:\n");
            std::cout << "  Cells at or above this value are used as sample points.\n";
            std::cout << "  "; print_default("Default: 1 (every non-zero density cell)"); std::cout << "\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return 0;
            if (interp_help(input, "Cells whose density is >= this value become the sample\n"
                                   "points the surface is interpolated from.")) continue;
            if (input.empty()) {
                min_threshold = 1.0;
                break;
            }
            try {
                min_threshold = std::stod(input);
                break;
            } catch (...) {
                std::cout << "ERROR: Invalid number\n";
            }
        }

        // Sampling every cell of an already-dense raster leaves nothing to
        // interpolate (every sample keeps its value): the spacing subsamples
        // the raster on a regular grid so the surface is generalized.
        int spacing = 1;
        bool preserve_maxima = false;
        std::vector<uint8_t> is_sample(N, 0);
        size_t n_samples = 0;
        while (true) {
            print_question("\nEnter sample spacing in cells (1 = every cell):\n");
            std::cout << "  Samples are taken every N cells on a regular grid. Larger\n";
            std::cout << "  values give a smoother, more generalized surface.\n";
            std::cout << "  "; print_default("Default: 1"); std::cout << "\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return 0;
            if (interp_help(input, "With spacing N only cells on an N x N grid (whose value\n"
                                   "reaches the threshold) become samples; everything else\n"
                                   "is interpolated between them.")) continue;
            if (!input.empty()) {
                try {
                    spacing = std::stoi(input);
                } catch (...) {
                    std::cout << "ERROR: Invalid number\n";
                    continue;
                }
                if (spacing < 1) {
                    std::cout << "ERROR: Spacing must be at least 1\n";
                    continue;
                }
            } else {
                spacing = 1;
            }

            // Only worth asking once there is subsampling to lose peaks to.
            if (spacing > 1) {
                while (true) {
                    print_question("\nPreserve local peaks? (yes/no):\n");
                    std::cout << "  Also keeps, in every " << spacing << "x" << spacing
                              << " block, the cell that holds the block's highest\n";
                    std::cout << "  value, so intense junctions survive the subsampling.\n";
                    std::cout << "  "; print_default("Default: no"); std::cout << "\n";
                    std::cout << "> ";
                    std::string pk;
                    safe_getline(pk);
                    if (check_exit_command(pk)) return 0;
                    if (interp_help(pk, "A regular grid keeps whichever cell happens to sit on\n"
                                        "it, which on a thin, spiky input is rarely the\n"
                                        "interesting one. This adds the real maximum of each\n"
                                        "block: at most one extra sample per block.")) continue;
                    if (pk.empty() || pk == "no" || pk == "n" || pk == "NO" || pk == "No") {
                        preserve_maxima = false;
                        break;
                    }
                    if (pk == "yes" || pk == "y" || pk == "YES" || pk == "Yes") {
                        preserve_maxima = true;
                        break;
                    }
                    std::cout << "ERROR: Please enter 'yes' or 'no'\n";
                }
            } else {
                preserve_maxima = false;
            }

            n_samples = 0;
            size_t n_valid = 0;
            for (int r = 0; r < nrows; ++r) {
                for (int c = 0; c < ncols; ++c) {
                    const size_t i = (size_t)r * ncols + c;
                    const bool nd = has_nodata && value[i] == (float)nodata;
                    if (!nd) ++n_valid;
                    is_sample[i] = (!nd && (r % spacing == 0) && (c % spacing == 0)
                                    && value[i] >= (float)min_threshold) ? 1 : 0;
                    if (is_sample[i]) ++n_samples;
                }
            }

            // Peak preservation. A regular grid keeps whichever cell happens to
            // sit on it, which on a thin, spiky input like a FETE density is
            // rarely the interesting one: the junctions are exactly the cells
            // that get dropped, and the surface loses its highest values
            // altogether. This adds, per spacing x spacing block, the cell that
            // actually holds the block's maximum. At most one sample per block,
            // and nothing at all when spacing is 1.
            if (preserve_maxima && spacing > 1) {
                size_t added = 0;
                for (int r0 = 0; r0 < nrows; r0 += spacing) {
                    for (int c0 = 0; c0 < ncols; c0 += spacing) {
                        const int r1 = std::min(r0 + spacing, nrows);
                        const int c1 = std::min(c0 + spacing, ncols);
                        float best = -std::numeric_limits<float>::max();
                        size_t best_i = (size_t)-1;
                        for (int r = r0; r < r1; ++r) {
                            for (int c = c0; c < c1; ++c) {
                                const size_t i = (size_t)r * ncols + c;
                                if (has_nodata && value[i] == (float)nodata) continue;
                                if (value[i] > best) { best = value[i]; best_i = i; }
                            }
                        }
                        if (best_i != (size_t)-1 && best >= (float)min_threshold
                            && !is_sample[best_i]) {
                            is_sample[best_i] = 1;
                            ++n_samples;
                            ++added;
                        }
                    }
                }
                std::cout << "Peak preservation added " << added << " sample cell(s)\n";
            }
            if (n_samples < 3) {
                std::cout << "ERROR: Only " << n_samples << " cells qualify as samples "
                          << "(threshold " << min_threshold << ", spacing " << spacing
                          << "). Lower the threshold or the spacing.\n";
                continue;
            }

            std::cout << "Sample cells: " << n_samples << " / " << N << "\n";
            if (n_valid > 0 && n_samples * 2 > n_valid) {
                std::cout << "WARNING: " << (100 * n_samples / n_valid) << "% of the valid cells "
                          << "qualify as samples.\n";
                std::cout << "WARNING: With so many samples the interpolated raster will be "
                          << "nearly identical\n";
                std::cout << "WARNING: to the input. Increase the sample spacing (e.g. 4) or "
                          << "the threshold\n";
                std::cout << "WARNING: to obtain a smoother, more generalized surface.\n";
            }
            break;
        }

        // ===== MAX SEARCH RADIUS =====
        int max_radius = 0;
        while (true) {
            print_question("\nEnter maximum search radius in cells (0 = unlimited):\n");
            std::cout << "  Limits how far the interpolation reaches into empty areas.\n";
            std::cout << "  Beyond it, cells take the value of their nearest sample.\n";
            std::cout << "  "; print_default("Default: 0 (unlimited)"); std::cout << "\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return 0;
            if (interp_help(input, "A cap keeps the computation fast when large parts of the\n"
                                   "raster are far away from any sample.")) continue;
            if (input.empty()) { max_radius = 0; break; }
            try {
                max_radius = std::stoi(input);
                if (max_radius < 0) { std::cout << "ERROR: Radius cannot be negative\n"; continue; }
                break;
            } catch (...) {
                std::cout << "ERROR: Invalid number\n";
            }
        }

        // ===== OUTPUT FILENAME =====
        std::string default_name = fs::path(input_path).stem().string() + "_NNI";
        std::string output_name;
        while (true) {
            print_question("\nEnter interpolated raster filename (without extension):\n");
            std::cout << "  "; print_default("Default: " + default_name); std::cout << "\n";
            std::cout << "> ";
            std::string input;
            safe_getline(input);
            if (check_exit_command(input)) return 0;
            if (interp_help(input, "Name of the output GeoTIFF, written in the output directory.")) continue;
            output_name = input.empty() ? default_name : input;
            if (output_name.find_first_of("\\/:*?\"<>|") != std::string::npos) {
                std::cout << "ERROR: Filename contains invalid characters\n";
                continue;
            }
            break;
        }

        // ===== CONFIG SUMMARY =====
        std::cout << "\n" << std::string(70, '=') << "\n";
        center_text("NNI CONFIGURATION", 70);
        std::cout << std::string(70, '=') << "\n";
        std::cout << "  Input raster: " << input_path << "\n";
        std::cout << "  Output directory: " << out_dir << "\n";
        std::cout << "  Sample threshold: >= " << min_threshold << " (" << n_samples << " cells)\n";
        std::cout << "  Sample spacing: every " << spacing << " cell(s)\n";
        std::cout << "  Peak preservation: " << (preserve_maxima ? "on" : "off") << "\n";
        std::cout << "  Max search radius: " << (max_radius > 0 ? std::to_string(max_radius) + " cells" : std::string("unlimited")) << "\n";
        std::cout << "  Output filename: " << output_name << ".tif\n";
        std::cout << "  Max threads: " << max_threads << "\n";
        std::cout << std::string(70, '=') << "\n\n";

        const auto t_start = std::chrono::high_resolution_clock::now();

        // ===== STEP 1: nearest-sample transform =====
        std::cout << "Computing nearest-sample distance field...\n";
        std::vector<float> dist2;
        std::vector<int32_t> src;
        nearest_sample_transform(is_sample, ncols, nrows, dist2, src);

        // ===== STEP 2: discrete Sibson scatter =====
        // Every cell x hands the value of its nearest sample to all query
        // cells q closer to x than that sample is (|q-x|^2 <= dist2[x]):
        // exactly the cells that would steal x from the Voronoi diagram.
        std::cout << "Interpolating (discrete Sibson)...\n";
        std::vector<double> sum(N, 0.0);
        std::vector<uint32_t> cnt(N, 0);
        const double r2cap = (max_radius > 0)
            ? (double)max_radius * max_radius
            : std::numeric_limits<double>::infinity();

        int rows_done = 0;
#pragma omp parallel for schedule(dynamic, 4)
        for (int y = 0; y < nrows; ++y) {
            for (int x = 0; x < ncols; ++x) {
                const size_t i = (size_t)y * ncols + x;
                if (src[i] < 0) continue;              // unreachable (no samples)
                double d2 = (double)dist2[i];
                if (d2 > r2cap) d2 = r2cap;            // capped reach
                const double v = (double)value[(size_t)src[i]];
                const int rad = (int)std::sqrt(d2);
                const int qy0 = std::max(0, y - rad);
                const int qy1 = std::min(nrows - 1, y + rad);
                for (int qy = qy0; qy <= qy1; ++qy) {
                    const double dy2 = (double)(qy - y) * (qy - y);
                    const int half = (int)std::sqrt(std::max(0.0, d2 - dy2));
                    const int qx0 = std::max(0, x - half);
                    const int qx1 = std::min(ncols - 1, x + half);
                    size_t q = (size_t)qy * ncols + qx0;
                    for (int qx = qx0; qx <= qx1; ++qx, ++q) {
#pragma omp atomic
                        sum[q] += v;
#pragma omp atomic
                        cnt[q]++;
                    }
                }
            }
            // Progress from whichever thread finishes a row: pinning it to
            // thread 0 froze the bar when dynamic scheduling let that thread
            // drain its share early. A critical section is fine at row
            // granularity (thousands of entries per run).
#pragma omp critical
            {
                ++rows_done;
                if (rows_done % 8 == 0 || rows_done == nrows) {
                    const double elapsed = std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - t_start).count();
                    print_progress(rows_done, nrows, elapsed, 40, 0);
                }
            }
        }
        print_progress(nrows, nrows, -1.0, 40, 0);
        std::cout << "\n";

        // ===== STEP 3: finalize & write =====
        std::cout << "Writing interpolated raster...\n";
        std::vector<float> out_vals(N);
        double out_min = std::numeric_limits<double>::max();
        double out_max = std::numeric_limits<double>::lowest();
        double out_sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double v;
            if (cnt[i] > 0)
                v = sum[i] / cnt[i];
            else if (src[i] >= 0)
                v = (double)value[(size_t)src[i]];  // beyond the radius cap
            else
                v = 0.0;
            out_vals[i] = (float)v;
            out_min = std::min(out_min, v);
            out_max = std::max(out_max, v);
            out_sum += v;
        }

        const std::string out_path = interp_join_path(out_dir, output_name + ".tif");
        if (!write_gtiff_raster(out_path, ncols, nrows, gt, wkt, out_vals.data(), GDT_Float32)) {
            return 1;
        }

        const double total_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_start).count();

        save_interp_config({ input_path, out_dir });

        print_green_success("NNI successfully computed!\n");
        std::cout << "\nOutput Summary:\n";
        std::cout << "  Total time: " << std::fixed << std::setprecision(2) << total_time << " sec\n";
        std::cout << "  Sample cells: " << n_samples << "\n";
        std::cout << "  Output min: " << std::setprecision(3) << out_min << "\n";
        std::cout << "  Output max: " << out_max << "\n";
        std::cout << "  Output avg: " << (out_sum / (double)N) << "\n";
        std::cout << "\nOutput Files:\n";
        std::cout << "  - " << out_path << "\n";

        print_question("\nRun another NNI computation? (yes/no)\n"); std::cout << "> ";
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
        saved.input_path = input_path;
        saved.out_dir = out_dir;
    }

    return 0;
}
