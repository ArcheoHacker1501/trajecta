// main_fete_gpu.cu
// GPU implementation of FETE algorithm using Gunrock for SSSP
// VERSION WITH CUDA STREAMS for parallel SSSP execution

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

#include <cuda_runtime.h>
#include "gdal_priv.h"
#include "ogrsf_frmts.h"

// Gunrock headers
#include <gunrock/algorithms/sssp.hxx>
#include <gunrock/cuda/context.hxx>
#include <gunrock/formats/csr.hxx>
#include <gunrock/graph/graph.hxx>
#include "fete_sssp.hxx"

// Forward declarations from main_fete.cu
enum CostFunctionType { TOBLER_WHITE_2015 = 1 };

struct FETEOutput {
    bool success;
    std::string slope_path;
    std::string cost_path;
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

// Offset arrays
static const Off OFFS_8[8] = {
    {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}
};

static const Off OFFS_16[16] = {
    {-1,0},{-1,1},{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},
    {-2,0},{-2,1},{-1,2},{0,2},{1,2},{2,1},{2,0},{2,-1}
};

static const Off OFFS_24[24] = {
    {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1},
    {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1},
    {-2,-2},{-2,0},{-2,2},{0,-2},{0,2},{2,-2},{2,0},{2,2}
};

static const Off OFFS_32[32] = {
    {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1},
    {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1},
    {-2,-2},{-2,0},{-2,2},{0,-2},{0,2},{2,-2},{2,0},{2,2},
    {-3,-1},{-3,1},{-1,-3},{-1,3},{1,-3},{1,3},{3,-1},{3,1}
};

static const Off OFFS_64[64] = {
    {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1},
    {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1},
    {-2,-2},{-2,0},{-2,2},{0,-2},{0,2},{2,-2},{2,0},{2,2},
    {-3,-1},{-3,1},{-1,-3},{-1,3},{1,-3},{1,3},{3,-1},{3,1},
    {-3,-2},{-3,0},{-3,2},{-2,-3},{-2,3},{0,-3},{0,3},{2,-3},{2,3},{3,-2},{3,0},{3,2},
    {-3,-3},{-3,3},{3,-3},{3,3},
    {-4,0},{0,-4},{4,0},{0,4},
    {-4,-1},{-4,1},{-1,-4},{-1,4},{1,-4},{1,4},{4,-1},{4,1}
};

// Helper functions
static inline int idx(int r, int c, int ncols) {
    return r * ncols + c;
}

static inline void idx2coord(int index, int ncols, int& r, int& c) {
    r = index / ncols;
    c = index % ncols;
}

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

static inline void print_progress_gpu(int current, int total, int bar_width = 45) {
    double percentage = (double)current / total;
    int filled = (int)(percentage * bar_width);

    std::cout << "\r[GPU] [" << current << "/" << total << "] ";
    std::cout << std::string(filled, 219);
    std::cout << std::string(bar_width - filled, 177);
    std::cout << " " << (int)(percentage * 100) << "%";
    std::cout.flush();
}

static inline void report_cuda_error(const char* label, cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "\n[CUDA ERROR] " << label << ": " << cudaGetErrorString(err) << "\n";
    }
}

// ========== GPU PROFILING ==========

struct GPUUsageStats {
    size_t memory_allocated_mb;
    size_t memory_transferred_to_gpu_mb;
    size_t memory_transferred_from_gpu_mb;
    double kernel_execution_time_sec;
    double memory_transfer_time_sec;
    double total_gpu_time_sec;
    int kernel_launches;
    int num_streams;
};

class GPUProfiler {
private:
    GPUUsageStats stats;
    cudaEvent_t start_event, stop_event;

public:
    GPUProfiler() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        stats = { 0 };
    }

    ~GPUProfiler() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start_timer() {
        cudaEventRecord(start_event);
    }

    void stop_timer_kernel() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        stats.kernel_execution_time_sec += milliseconds / 1000.0;
        stats.kernel_launches++;
    }

    void stop_timer_transfer() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        stats.memory_transfer_time_sec += milliseconds / 1000.0;
    }

    void record_allocation(size_t bytes) {
        stats.memory_allocated_mb += bytes / (1024 * 1024);
    }

    void record_transfer_to_gpu(size_t bytes) {
        stats.memory_transferred_to_gpu_mb += bytes / (1024 * 1024);
    }

    void record_transfer_from_gpu(size_t bytes) {
        stats.memory_transferred_from_gpu_mb += bytes / (1024 * 1024);
    }

    void set_num_streams(int n) {
        stats.num_streams = n;
    }

    GPUUsageStats get_stats() {
        stats.total_gpu_time_sec = stats.kernel_execution_time_sec + stats.memory_transfer_time_sec;
        return stats;
    }
};

void print_gpu_usage_report(const GPUUsageStats& stats) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "GPU USAGE REPORT\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << "Memory Usage:\n";
    std::cout << "  Total allocated: " << stats.memory_allocated_mb << " MB\n";

    std::cout << "\nMemory Transfers:\n";
    std::cout << "  Host -> GPU: " << stats.memory_transferred_to_gpu_mb << " MB\n";
    std::cout << "  GPU -> Host: " << stats.memory_transferred_from_gpu_mb << " MB\n";
    std::cout << "  Total transferred: "
        << (stats.memory_transferred_to_gpu_mb + stats.memory_transferred_from_gpu_mb)
        << " MB\n";

    std::cout << "\nExecution Breakdown:\n";
    std::cout << "  CUDA Streams used: " << stats.num_streams << "\n";
    std::cout << "  Kernel execution: " << std::fixed << std::setprecision(3)
        << stats.kernel_execution_time_sec << " sec\n";
    std::cout << "  Memory transfers: " << stats.memory_transfer_time_sec << " sec\n";
    std::cout << "  Total GPU time: " << stats.total_gpu_time_sec << " sec\n";
    std::cout << "  SSSP iterations: " << stats.kernel_launches << "\n";

    std::cout << "\nEfficiency Metrics:\n";
    if (stats.memory_transfer_time_sec > 0) {
        std::cout << "  Compute/Transfer ratio: " << std::fixed << std::setprecision(2)
            << (stats.kernel_execution_time_sec / stats.memory_transfer_time_sec) << "x\n";

        if (stats.kernel_execution_time_sec > stats.memory_transfer_time_sec * 2) {
            std::cout << "  Status: COMPUTE-BOUND (good for GPU)\n";
        }
        else if (stats.kernel_execution_time_sec > stats.memory_transfer_time_sec) {
            std::cout << "  Status: BALANCED\n";
        }
        else {
            std::cout << "  Status: MEMORY-BOUND\n";
        }
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
}

// ========== GPU KERNELS ==========

// Kernel per inizializzare dist a infinito e pred a -1
__global__ void init_sssp_arrays_kernel(float* dist, int* pred, int N, float inf_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dist[idx] = inf_val;
        pred[idx] = -1;
    }
}

__global__ void path_trace_and_density_kernel_points(
    const int* pred,
    const int* destinations,
    int num_destinations,
    int source,
    uint32_t* density,
    int buffer_radius,
    int ncols,
    int nrows,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_destinations) return;

    int dest = destinations[idx];
    if (dest == source) return;

    // Check if reachable
    if (pred[dest] == -1) return;

    // Trace path back to source
    int current = dest;
    int max_path_len = N;
    int path_len = 0;

    while (current != -1 && current != source && path_len < max_path_len) {
        atomicAdd(&density[current], 1);

        if (buffer_radius > 0) {
            int r = current / ncols;
            int c = current % ncols;

            for (int dr = -buffer_radius; dr <= buffer_radius; dr++) {
                for (int dc = -buffer_radius; dc <= buffer_radius; dc++) {
                    if (dr == 0 && dc == 0) continue;

                    int nr = r + dr;
                    int nc = c + dc;

                    if (nr >= 0 && nr < nrows && nc >= 0 && nc < ncols) {
                        atomicAdd(&density[nr * ncols + nc], 1);
                    }
                }
            }
        }

        current = pred[current];
        path_len++;
    }

    // Add source if path was found
    if (current == source) {
        atomicAdd(&density[source], 1);

        if (buffer_radius > 0) {
            int r = source / ncols;
            int c = source % ncols;

            for (int dr = -buffer_radius; dr <= buffer_radius; dr++) {
                for (int dc = -buffer_radius; dc <= buffer_radius; dc++) {
                    if (dr == 0 && dc == 0) continue;

                    int nr = r + dr;
                    int nc = c + dc;

                    if (nr >= 0 && nr < nrows && nc >= 0 && nc < ncols) {
                        atomicAdd(&density[nr * ncols + nc], 1);
                    }
                }
            }
        }
    }
}

// ========== STREAM CONFIGURATION ==========

// Struttura per gestire risorse per-stream
struct StreamResources {
    cudaStream_t stream;
    float* d_dist;
    int* d_pred;
    std::shared_ptr<gunrock::gcuda::multi_context_t> context;
};

// ========== MAIN GPU FETE FUNCTION ==========

FETEOutput run_fete_gpu(
    const std::string& dem_path,
    const std::string& pts_path,
    const std::string& out_dir,
    const std::string& slope_filename,
    const std::string& cost_filename,
    const std::string& output_filename,
    int buffer_radius,
    int max_threads,
    int64_t max_ram_mb,
    int num_neighbours,
    bool slope_in_degrees,
    CostFunctionType cost_function,
    int num_streams  // NEW: configurable number of CUDA streams
) {
    FETEOutput output = { false, "", "", "", 0, 0, 0, 0, 0, 0.0, false };
    auto global_start = std::chrono::high_resolution_clock::now();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "GPU MODE ACTIVATED - NVIDIA CUDA (MULTI-STREAM)\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << "Parameters received:\n";
    std::cout << "  DEM: " << dem_path << "\n";
    std::cout << "  Points: " << pts_path << "\n";
    std::cout << "  Neighbours: " << num_neighbours << "-connectivity\n";
    std::cout << "  Buffer radius: " << buffer_radius << " cells\n";
    std::cout << "  CUDA Streams: " << num_streams << " (parallel SSSP)\n";
    std::cout << std::string(70, '=') << "\n\n";

    GPUProfiler profiler;
    profiler.set_num_streams(num_streams);

    const Off* current_offs = OFFS_16;
    int num_offs = 16;

    switch (num_neighbours) {
    case 8:  current_offs = OFFS_8;  num_offs = 8;  break;
    case 16: current_offs = OFFS_16; num_offs = 16; break;
    case 32: current_offs = OFFS_32; num_offs = 32; break;
    case 64: current_offs = OFFS_64; num_offs = 64; break;
    default: current_offs = OFFS_16; num_offs = 16; break;
    }

    // ========== READ DEM ==========

    std::cout << "[CPU] Reading DEM...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    GDALDataset* dem_ds = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
    if (!dem_ds) {
        std::cout << "ERROR: Cannot open DEM\n";
        return output;
    }

    int ncols = dem_ds->GetRasterXSize();
    int nrows = dem_ds->GetRasterYSize();
    int N = nrows * ncols;
    double gt[6];
    dem_ds->GetGeoTransform(gt);
    const char* wkt = dem_ds->GetProjectionRef();

    std::vector<float> dem(N);
    dem_ds->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, ncols, nrows, dem.data(), ncols, nrows, GDT_Float32, 0, 0);

    double res_x = std::abs(gt[1]);
    double res_y = std::abs(gt[5]);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[CPU] DEM loaded: " << nrows << "x" << ncols << " ("
        << std::fixed << std::setprecision(2) << std::chrono::duration<double>(t1 - t0).count() << " sec)\n";

    // ========== SLOPE RASTER ==========

    std::cout << "[CPU] Calculating slope (" << (slope_in_degrees ? "degrees" : "percentage") << ")...\n";
    t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> slope_data(N, 0.0f);
    const float pi_f = 3.14159265f;

#pragma omp parallel for collapse(2) num_threads(max_threads)
    for (int r = 1; r < nrows - 1; ++r) {
        for (int c = 1; c < ncols - 1; ++c) {
            int center = idx(r, c, ncols);
            float dz_dx = (dem[idx(r, c + 1, ncols)] - dem[idx(r, c - 1, ncols)]) / (2.0f * (float)res_x);
            float dz_dy = (dem[idx(r + 1, c, ncols)] - dem[idx(r - 1, c, ncols)]) / (2.0f * (float)res_y);
            float gradient = std::sqrt(dz_dx * dz_dx + dz_dy * dz_dy);

            if (slope_in_degrees) {
                slope_data[center] = std::atan(gradient) * 180.0f / pi_f;
            }
            else {
                slope_data[center] = gradient * 100.0f;
            }
        }
    }

    GDALDriver* gtiff_drv = GetGDALDriverManager()->GetDriverByName("GTiff");
    std::string slope_path = out_dir + "\\" + slope_filename + ".tif";
    GDALDataset* slope_ds = gtiff_drv->Create(slope_path.c_str(), ncols, nrows, 1, GDT_Float32, nullptr);
    slope_ds->SetGeoTransform(gt);
    slope_ds->SetProjection(wkt);
    slope_ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows, slope_data.data(), ncols, nrows, GDT_Float32, 0, 0);
    GDALClose(slope_ds);
    output.slope_path = slope_path;

    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[CPU] Slope raster saved (" << std::fixed << std::setprecision(2)
        << std::chrono::duration<double>(t1 - t0).count() << " sec)\n";

    // ========== COST SURFACE RASTER ==========

    std::cout << "[CPU] Calculating cost surface (" << num_neighbours << "-connectivity)...\n";
    t0 = std::chrono::high_resolution_clock::now();

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

    std::string cost_path = out_dir + "\\" + cost_filename + ".tif";
    GDALDataset* cost_ds = gtiff_drv->Create(cost_path.c_str(), ncols, nrows, 1, GDT_Float32, nullptr);
    cost_ds->SetGeoTransform(gt);
    cost_ds->SetProjection(wkt);
    cost_ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows, cost_surface.data(), ncols, nrows, GDT_Float32, 0, 0);
    GDALClose(cost_ds);
    output.cost_path = cost_path;

    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[CPU] Cost surface saved (" << std::fixed << std::setprecision(2)
        << std::chrono::duration<double>(t1 - t0).count() << " sec)\n";

    // ========== READ POINTS ==========

    std::cout << "[CPU] Reading points...\n";
    t0 = std::chrono::high_resolution_clock::now();

    std::vector<int> point_nodes;
    GDALDataset* pts_ds = (GDALDataset*)GDALOpenEx(pts_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (!pts_ds) {
        std::cout << "ERROR: Cannot open points\n";
        GDALClose(dem_ds);
        return output;
    }

    OGRLayer* layer = pts_ds->GetLayer(0);
    layer->ResetReading();

    OGRFeature* feat;
    while ((feat = layer->GetNextFeature()) != nullptr) {
        OGRGeometry* geom = feat->GetGeometryRef();
        if (geom && wkbFlatten(geom->getGeometryType()) == wkbPoint) {
            OGRPoint* pt = (OGRPoint*)geom;
            int col, row;
            if (world_to_pixel_northup(pt->getX(), pt->getY(), gt, col, row)) {
                if (row >= 0 && row < nrows && col >= 0 && col < ncols) {
                    point_nodes.push_back(idx(row, col, ncols));
                }
            }
        }
        OGRFeature::DestroyFeature(feat);
    }

    int P = (int)point_nodes.size();
    GDALClose(pts_ds);

    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[CPU] Loaded " << P << " points ("
        << std::fixed << std::setprecision(2) << std::chrono::duration<double>(t1 - t0).count() << " sec)\n";

    // ========== BUILD CSR GRAPH ==========

    std::cout << "[CPU] Building CSR graph...\n";
    t0 = std::chrono::high_resolution_clock::now();

    std::vector<int> edge_count(N, 0);

    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            int v = idx(r, c, ncols);
            for (int k = 0; k < num_offs; k++) {
                int nr = r + current_offs[k].dr;
                int nc = c + current_offs[k].dc;
                if (nr >= 0 && nr < nrows && nc >= 0 && nc < ncols) {
                    edge_count[v]++;
                }
            }
        }
    }

    std::vector<int> row_offsets(N + 1);
    row_offsets[0] = 0;
    for (int i = 0; i < N; i++) {
        row_offsets[i + 1] = row_offsets[i] + edge_count[i];
    }

    int total_edges = row_offsets[N];
    std::vector<int> column_indices(total_edges);
    std::vector<float> edge_weights(total_edges);
    std::vector<int> current_pos = row_offsets;

#pragma omp parallel for
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            int v = idx(r, c, ncols);
            float z_from = dem[v];

            for (int k = 0; k < num_offs; k++) {
                int nr = r + current_offs[k].dr;
                int nc = c + current_offs[k].dc;

                if (nr >= 0 && nr < nrows && nc >= 0 && nc < ncols) {
                    int u = idx(nr, nc, ncols);

                    double dh = std::sqrt((current_offs[k].dr * res_y) * (current_offs[k].dr * res_y) +
                        (current_offs[k].dc * res_x) * (current_offs[k].dc * res_x));
                    double dz = dem[u] - z_from;
                    float weight = apply_cost_function(cost_function, dh, dz);

                    int pos;
#pragma omp critical
                    {
                        pos = current_pos[v]++;
                    }

                    column_indices[pos] = u;
                    edge_weights[pos] = weight;
                }
            }
        }
    }

    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[CPU] Graph built: " << N << " vertices, " << total_edges << " edges ("
        << std::fixed << std::setprecision(2) << std::chrono::duration<double>(t1 - t0).count() << " sec)\n";

    // ========== TRANSFER TO GPU ==========

    std::cout << "[GPU] Allocating and transferring...\n";

    using vertex_t = int;
    using edge_t = int;
    using weight_t = float;

    using csr_host_t = gunrock::format::csr_t<gunrock::memory::memory_space_t::host, vertex_t, edge_t, weight_t>;
    using csr_device_t = gunrock::format::csr_t<gunrock::memory::memory_space_t::device, vertex_t, edge_t, weight_t>;

    csr_host_t h_csr(N, N, total_edges);
    std::copy(row_offsets.begin(), row_offsets.end(), h_csr.row_offsets.begin());
    std::copy(column_indices.begin(), column_indices.end(), h_csr.column_indices.begin());
    std::copy(edge_weights.begin(), edge_weights.end(), h_csr.nonzero_values.begin());

    profiler.start_timer();
    csr_device_t d_csr(h_csr);

    // Shared resources
    uint32_t* d_density;
    int* d_points;

    cudaMalloc(&d_density, N * sizeof(uint32_t));
    cudaMalloc(&d_points, P * sizeof(int));
    cudaMemset(d_density, 0, N * sizeof(uint32_t));
    cudaMemcpy(d_points, point_nodes.data(), P * sizeof(int), cudaMemcpyHostToDevice);

    // ========== ALLOCATE PER-STREAM RESOURCES ==========

    std::vector<StreamResources> streams(num_streams);
    
    for (int s = 0; s < num_streams; s++) {
        cudaStreamCreate(&streams[s].stream);
        cudaMalloc(&streams[s].d_dist, N * sizeof(float));
        cudaMalloc(&streams[s].d_pred, N * sizeof(int));
        // Each stream gets its own Gunrock context
        streams[s].context = std::make_shared<gunrock::gcuda::multi_context_t>(0);
    }

    // Record memory allocation
    size_t per_stream_mem = N * sizeof(float) + N * sizeof(int);
    profiler.record_allocation(
        (N + 1) * sizeof(edge_t) +
        total_edges * (sizeof(vertex_t) + sizeof(weight_t)) +
        N * sizeof(uint32_t) +
        P * sizeof(int) +
        num_streams * per_stream_mem  // dist + pred per ogni stream
    );

    profiler.record_transfer_to_gpu(
        (N + 1) * sizeof(edge_t) +
        total_edges * (sizeof(vertex_t) + sizeof(weight_t)) +
        P * sizeof(int)
    );

    profiler.stop_timer_transfer();

    std::cout << "[GPU] Transfer complete\n";
    std::cout << "[GPU] Allocated " << num_streams << " streams with " 
              << (per_stream_mem / (1024 * 1024)) << " MB each\n";

    // ========== RUN GUNROCK SSSP WITH STREAMS ==========

    std::cout << "[GPU] Running SSSP for " << P << " sources using " << num_streams << " streams...\n";

    gunrock::graph::graph_properties_t props;
    props.directed = true;
    props.weighted = true;
    props.symmetric = false;

    auto G = gunrock::graph::build<gunrock::memory::memory_space_t::device>(props, d_csr);

    int threads = 256;
    int blocks_density = (P + threads - 1) / threads;
    int blocks_init = (N + threads - 1) / threads;
    const float dist_inf = std::numeric_limits<float>::max();

    // Debug info per primo SSSP
    const bool debug_sssp = true;
    const int debug_sample = 64;
    bool debug_captured = false;
    int debug_pred_set = 0;
    int debug_dist_finite = 0;
    float debug_min_dist = dist_inf;
    float debug_max_dist = 0.0f;
    int debug_source = -1;
    float debug_source_dist = dist_inf;
    int debug_neighbor = -1;
    float debug_neighbor_dist = dist_inf;
    float debug_neighbor_weight = 0.0f;

    profiler.start_timer();

    // Process sources in batches of num_streams
    int completed = 0;
    
    for (int batch_start = 0; batch_start < P; batch_start += num_streams) {
        int batch_size = std::min(num_streams, P - batch_start);

        // Launch SSSP for each source in this batch (in parallel on different streams)
        for (int s = 0; s < batch_size; s++) {
            int source_idx = batch_start + s;
            int source = point_nodes[source_idx];

            // Initialize arrays for this stream
            init_sssp_arrays_kernel<<<blocks_init, threads, 0, streams[s].stream>>>(
                streams[s].d_dist, streams[s].d_pred, N, dist_inf
            );

            // Debug capture for first source
            if (debug_sssp && source_idx == 0) {
                debug_source = source;
                int row_start = row_offsets[source];
                int row_end = row_offsets[source + 1];
                if (row_start < row_end) {
                    debug_neighbor = column_indices[row_start];
                    debug_neighbor_weight = edge_weights[row_start];
                }
            }
        }

        // Synchronize initialization before SSSP
        for (int s = 0; s < batch_size; s++) {
            cudaStreamSynchronize(streams[s].stream);
        }

        // Launch SSSP for each source in batch
        // NOTE: Gunrock manages its own synchronization internally
        for (int s = 0; s < batch_size; s++) {
            int source_idx = batch_start + s;
            int source = point_nodes[source_idx];

            gunrock::fete_sssp::run(G, source, streams[s].d_dist, streams[s].d_pred, streams[s].context);
        }

        // Wait for all SSSP in this batch to complete
        for (int s = 0; s < batch_size; s++) {
            streams[s].context->get_context(0)->synchronize();
        }

        // Debug capture after first SSSP
        if (debug_sssp && batch_start == 0 && !debug_captured) {
            std::vector<int> pred_sample(debug_sample);
            std::vector<float> dist_sample(debug_sample);
            cudaMemcpy(pred_sample.data(), streams[0].d_pred, debug_sample * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(dist_sample.data(), streams[0].d_dist, debug_sample * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < debug_sample; ++i) {
                if (pred_sample[i] != -1) debug_pred_set++;
                if (dist_sample[i] < dist_inf * 0.5f) {
                    debug_dist_finite++;
                    debug_min_dist = std::min(debug_min_dist, dist_sample[i]);
                    debug_max_dist = std::max(debug_max_dist, dist_sample[i]);
                }
            }

            cudaMemcpy(&debug_source_dist, streams[0].d_dist + debug_source, sizeof(float), cudaMemcpyDeviceToHost);
            if (debug_neighbor >= 0) {
                cudaMemcpy(&debug_neighbor_dist, streams[0].d_dist + debug_neighbor, sizeof(float), cudaMemcpyDeviceToHost);
            }
            debug_captured = true;
        }

        // Launch density kernels for each source in batch
        for (int s = 0; s < batch_size; s++) {
            int source_idx = batch_start + s;
            int source = point_nodes[source_idx];

            path_trace_and_density_kernel_points<<<blocks_density, threads, 0, streams[s].stream>>>(
                streams[s].d_pred, d_points, P, source, d_density, buffer_radius, ncols, nrows, N
            );
        }

        // Wait for density kernels to complete
        for (int s = 0; s < batch_size; s++) {
            cudaStreamSynchronize(streams[s].stream);
        }

        completed += batch_size;
        print_progress_gpu(completed, P);
    }

    profiler.stop_timer_kernel();

    // Print debug info
    if (debug_captured) {
        std::cout << "\n[DEBUG] SSSP sample (first " << debug_sample << " vertices)\n";
        std::cout << "  preds set: " << debug_pred_set << "/" << debug_sample << "\n";
        if (debug_dist_finite > 0) {
            std::cout << "  finite dist: " << debug_dist_finite << "/" << debug_sample
                << " (min/max " << std::fixed << std::setprecision(2) 
                << debug_min_dist << "/" << debug_max_dist << ")\n";
        }
        else {
            std::cout << "  finite dist: 0/" << debug_sample << "\n";
        }

        std::cout << "  source idx: " << debug_source
            << " dist=" << std::fixed << std::setprecision(2) << debug_source_dist << "\n";
        if (debug_neighbor >= 0) {
            std::cout << "  neighbor idx: " << debug_neighbor
                << " w=" << std::fixed << std::setprecision(2) << debug_neighbor_weight
                << " dist=" << debug_neighbor_dist << "\n";
        }

        if (debug_dist_finite == 0) {
            std::cout << "  WARNING: no finite distances detected in sample.\n";
        }
    }

    std::cout << "\n[GPU] SSSP complete\n";

    // ========== TRANSFER BACK ==========

    std::cout << "[GPU->CPU] Transferring results...\n";
    profiler.start_timer();

    std::vector<uint32_t> density(N);
    cudaMemcpy(density.data(), d_density, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    profiler.record_transfer_from_gpu(N * sizeof(uint32_t));
    profiler.stop_timer_transfer();

    // ========== CLEANUP STREAMS ==========

    for (int s = 0; s < num_streams; s++) {
        cudaFree(streams[s].d_dist);
        cudaFree(streams[s].d_pred);
        cudaStreamDestroy(streams[s].stream);
    }
    cudaFree(d_density);
    cudaFree(d_points);

    // ========== SAVE OUTPUTS ==========

    std::cout << "[CPU] Saving outputs...\n";

    std::string density_path = out_dir + "\\" + output_filename + ".tif";
    GDALDataset* density_ds = gtiff_drv->Create(density_path.c_str(), ncols, nrows, 1, GDT_UInt32, nullptr);
    density_ds->SetGeoTransform(gt);
    density_ds->SetProjection(wkt);
    density_ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, ncols, nrows, density.data(), ncols, nrows, GDT_UInt32, 0, 0);
    GDALClose(density_ds);
    output.density_path = density_path;

    // Stats
    uint32_t max_d = 0, min_d = UINT32_MAX;
    uint64_t sum_d = 0;
    int nonzero = 0;

    for (int i = 0; i < N; i++) {
        if (density[i] > 0) {
            max_d = std::max(max_d, density[i]);
            min_d = std::min(min_d, density[i]);
            sum_d += density[i];
            nonzero++;
        }
    }

    output.max_density = max_d;
    output.min_density = (nonzero > 0) ? min_d : 0;
    output.avg_density = (nonzero > 0) ? (uint32_t)(sum_d / nonzero) : 0;
    output.nonzero_cells = nonzero;
    output.total_cells = N;

    GDALClose(dem_ds);

    auto global_end = std::chrono::high_resolution_clock::now();
    output.time_seconds = std::chrono::duration<double>(global_end - global_start).count();
    output.success = true;

    print_gpu_usage_report(profiler.get_stats());

    std::cout << "[CPU] Complete!\n";

    return output;
}
