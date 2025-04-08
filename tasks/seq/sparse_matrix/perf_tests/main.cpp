#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sparse_matrix/include/sparse_matrix_seq.hpp"

TEST(sparse_matrix_multiplication_seq, test_pipeline_run) {
    const auto epsilon = 1e-6;
    const auto size = 200;

    auto matrixA = sparse_matrix_multiplication_seq::GenerateRandomMatrix(size * size);
    auto matrixB = sparse_matrix_multiplication_seq::GenerateRandomMatrix(size * size);
    std::vector<double> result(size * size, 0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    task_data_seq->inputs_count = {size, size, size, size};
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    task_data_seq->outputs_count.emplace_back(result.size());

    auto expectedOutput = sparse_matrix_multiplication_seq::MultiplyMatrices(matrixA, size, size,
        matrixB, size, size);
    auto test_task_sequential =
        std::make_shared<sparse_matrix_multiplication_seq::CCSMatrixSeq>(task_data_seq);
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };

    auto perf_results = std::make_shared<ppc::core::PerfResults>();
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
    perf_analyzer->PipelineRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (auto i = 0; i < static_cast<int>(result.size()); i++)
      EXPECT_NEAR(result[i], expectedOutput[i], epsilon);
}

TEST(sparse_matrix_multiplication_seq, test_task_run) {
    const auto epsilon = 1e-6;
    const auto size = 400;

    auto matrixA = sparse_matrix_multiplication_seq::GenerateRandomMatrix(size * size);
    auto matrixB = sparse_matrix_multiplication_seq::GenerateRandomMatrix(size * size);
    std::vector<double> result(size * size, 0);

    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    task_data_seq->inputs_count = {size, size, size, size};
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    task_data_seq->outputs_count.emplace_back(result.size());

    auto expectedOutput = sparse_matrix_multiplication_seq::MultiplyMatrices(matrixA, size, size,
        matrixB, size, size);
    auto test_task_sequential =
        std::make_shared<sparse_matrix_multiplication_seq::CCSMatrixSeq>(task_data_seq);
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&] {
        auto current_time_point = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
        return static_cast<double>(duration) * 1e-9;
    };

    auto perf_results = std::make_shared<ppc::core::PerfResults>();
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_sequential);
    perf_analyzer->TaskRun(perf_attr, perf_results);
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (auto i = 0; i < static_cast<int>(result.size()); i++)
        EXPECT_NEAR(result[i], expectedOutput[i], epsilon);
}