#pragma once

#include <limits>
#include <chrono>
#include <memory>

#include <gunrock/algorithms/sssp.hxx>
#include <gunrock/cuda/context.hxx>

namespace gunrock {
    namespace fete_sssp {

        /**
         * @brief Wrapper function for Gunrock's SSSP algorithm
         * Designed to work with anisotropic terrain cost calculations (FETE application)
         *
         * @tparam graph_t Graph type
         * @tparam vertex_t Vertex type
         * @tparam edge_t Edge type
         * @tparam weight_t Weight type
         *
         * @param G Input graph
         * @param source Source vertex for SSSP
         * @param dist Output array: shortest distances from source (device pointer)
         * @param pred Output array: predecessors in shortest path tree (device pointer)
         * @param context Optional CUDA context (will create if nullptr)
         *
         * @return Time in milliseconds for SSSP computation
         */
        template <typename graph_t>
        float run(graph_t& G,
            typename graph_t::vertex_type source,
            typename graph_t::weight_type* dist,
            typename graph_t::vertex_type* pred,
            std::shared_ptr<gunrock::gcuda::multi_context_t> context = nullptr) {

            // Create context if not provided
            if (!context) {
                context = std::make_shared<gunrock::gcuda::multi_context_t>(0);
            }

            // Create param and result structures for Gunrock's SSSP
            using vertex_t = typename graph_t::vertex_type;
            using weight_t = typename graph_t::weight_type;

            gunrock::sssp::param_t<vertex_t> param(source);
            gunrock::sssp::result_t<vertex_t, weight_t> result(dist, pred, G.get_number_of_vertices());

            // Time the execution
            auto t0 = std::chrono::high_resolution_clock::now();

            // Call Gunrock's SSSP run function (delegating overload)
            float gpu_time = gunrock::sssp::run(G, param, result, context);

            auto t1 = std::chrono::high_resolution_clock::now();

            // Return total time in milliseconds
            return std::chrono::duration<float, std::milli>(t1 - t0).count();
        }

    }  // namespace fete_sssp
}  // namespace gunrock
