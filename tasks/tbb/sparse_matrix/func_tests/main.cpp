#include <gtest/gtest.h>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "tbb/sparse_matrix/include/sparse_matrix_tbb.hpp"

TEST(sparse_matrix_multiplication_tbb, test_square_matrices) {
  const auto epsilon = 1e-6;

  std::vector<double> matrixA{1, 0, 2, 0, 7, 6, 0, 0, 3};
  std::vector<double> matrixB{0, 3, 10, 1, 0, 0, 4, 0, 0};
  std::vector<double> expectedOutput{8, 3, 10, 31, 0, 0, 12, 0, 0};
  std::vector<double> result(9, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskData->inputs_count = {3, 3, 3, 3};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(result.size());

  sparse_matrix_multiplication_tbb::CCSMatrixTBB multiplicationTask(taskData);
  ASSERT_TRUE(multiplicationTask.Validation()) << "Validation failed!";

  multiplicationTask.PreProcessing();
  multiplicationTask.Run();
  multiplicationTask.PostProcessing();

  for (size_t i = 0; i < result.size(); i++)
    EXPECT_NEAR(result[i], expectedOutput[i], epsilon) << "Mismatch at index " << i;
}

TEST(sparse_matrix_multiplication_tbb, test_rectangular_matrices) {
  const auto epsilon = 1e-6;

  std::vector<double> matrixA{0, 1, 0, 6, 0, 0, 4, 3, 1, 0, 0, 2};
  std::vector<double> matrixB{0.5, 0, 1.5, 0, 0, 8.0, 3.0, 0, 0, 7, 0, 2};
  std::vector<double> expectedOutput{42, 0, 20, 33, 0, 6, 14.5, 0, 5.5};
  std::vector<double> result(9, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskData->inputs_count = {3, 4, 4, 3};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(result.size());

  sparse_matrix_multiplication_tbb::CCSMatrixTBB multiplicationTask(taskData);
  ASSERT_TRUE(multiplicationTask.Validation()) << "Validation failed!";

  multiplicationTask.PreProcessing();
  multiplicationTask.Run();
  multiplicationTask.PostProcessing();

  for (size_t i = 0; i < result.size(); i++)
    EXPECT_NEAR(result[i], expectedOutput[i], epsilon) << "Mismatch at index " << i;
}

TEST(sparse_matrix_multiplication_tbb, test_acceptable_sizes) {
  std::vector<double> matrixA{0, 1, 0, 6, 0, 0, 4, 3, 1, 0, 0, 2};
  std::vector<double> matrixB{0.5, 0, 1.5, 0, 0, 8.0, 3.0, 0, 0, 7, 0, 2};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskData->inputs_count = {4, 3, 4, 3};

  sparse_matrix_multiplication_tbb::CCSMatrixTBB multiplicationTask(taskData);
  ASSERT_FALSE(multiplicationTask.Validation());
}

TEST(sparse_matrix_multiplication_tbb, test_empty_matrices_multiplication) {
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  std::vector<double> result;
  std::vector<double> expectedOutput;

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskData->inputs_count = {0, 0, 0, 0};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(result.size());

  sparse_matrix_multiplication_tbb::CCSMatrixTBB multiplicationTask(taskData);
  ASSERT_TRUE(multiplicationTask.Validation()) << "Validation failed for empty matrices!";

  multiplicationTask.PreProcessing();
  multiplicationTask.Run();
  multiplicationTask.PostProcessing();

  EXPECT_EQ(result, expectedOutput) << "Expected empty result for empty matrices!";
}

TEST(sparse_matrix_multiplication_tbb, test_random_square_matrices_multiplication) {
  const auto epsilon = 1e-6;
  const auto size = 50;

  auto matrixA = sparse_matrix_multiplication_tbb::GenerateRandomMatrix(size * size);
  auto matrixB = sparse_matrix_multiplication_tbb::GenerateRandomMatrix(size * size);
  std::vector<double> result(size * size, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskData->inputs_count = {size, size, size, size};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(result.size());

  auto expectedOutput = sparse_matrix_multiplication_tbb::MultiplyMatrices(matrixA, size, size, matrixB, size, size);

  sparse_matrix_multiplication_tbb::CCSMatrixTBB multiplicationTask(taskData);
  ASSERT_TRUE(multiplicationTask.Validation()) << "Validation failed for random square matrices!";

  multiplicationTask.PreProcessing();
  multiplicationTask.Run();
  multiplicationTask.PostProcessing();

  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_NEAR(result[i], expectedOutput[i], epsilon) << "Mismatch at index " << i;
  }
}

TEST(sparse_matrix_multiplication_tbb, test_random_large_matrices_multiplication) {
  const auto epsilon = 1e-6;

  auto matrixA = sparse_matrix_multiplication_tbb::GenerateRandomMatrix(240);
  auto matrixB = sparse_matrix_multiplication_tbb::GenerateRandomMatrix(240);
  std::vector<double> result(144, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  taskData->inputs_count = {12, 20, 20, 12};
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(result.data()));
  taskData->outputs_count.push_back(result.size());

  auto expectedOutput = sparse_matrix_multiplication_tbb::MultiplyMatrices(matrixA, 12, 20, matrixB, 20, 12);

  sparse_matrix_multiplication_tbb::CCSMatrixTBB multiplicationTask(taskData);
  ASSERT_TRUE(multiplicationTask.Validation()) << "Validation failed for random large matrices!";

  multiplicationTask.PreProcessing();
  multiplicationTask.Run();
  multiplicationTask.PostProcessing();

  for (size_t i = 0; i < result.size(); i++)
    EXPECT_NEAR(result[i], expectedOutput[i], epsilon) << "Mismatch at index " << i;
}

TEST(sparse_matrix_multiplication_tbb, test_matrices_200) {
  const auto size = 200;

  auto matrixA = sparse_matrix_multiplication_tbb::GenerateRandomMatrix(size * size);
  auto matrixB = sparse_matrix_multiplication_tbb::GenerateRandomMatrix(size * size);
  std::vector<double> result(size * size, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  task_data_omp->inputs_count = {size, size, size, size};
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data_omp->outputs_count.emplace_back(result.size());

  sparse_matrix_multiplication_tbb::CCSMatrixTBB multiplicationTask(task_data_omp);
  ASSERT_TRUE(multiplicationTask.Validation()) << "Validation failed for random large matrices!";

  multiplicationTask.PreProcessing();
  clock_t start = clock();
  multiplicationTask.Run();
  clock_t end = clock();
  std::cout << "Time on matrix 200*200 = " << double(end - start);
  multiplicationTask.PostProcessing();
}

TEST(sparse_matrix_multiplication_tbb, test_matrices_300) {
  const auto size = 300;

  auto matrixA = sparse_matrix_multiplication_tbb::GenerateRandomMatrix(size * size);
  auto matrixB = sparse_matrix_multiplication_tbb::GenerateRandomMatrix(size * size);
  std::vector<double> result(size * size, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  task_data_omp->inputs_count = {size, size, size, size};
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data_omp->outputs_count.emplace_back(result.size());

  sparse_matrix_multiplication_tbb::CCSMatrixTBB multiplicationTask(task_data_omp);
  ASSERT_TRUE(multiplicationTask.Validation()) << "Validation failed for random large matrices!";

  multiplicationTask.PreProcessing();
  clock_t start = clock();
  multiplicationTask.Run();
  clock_t end = clock();
  std::cout << "Time on matrix 300*300 = " << double(end - start);
  multiplicationTask.PostProcessing();
}

TEST(sparse_matrix_multiplication_tbb, test_matrices_400) {
  const auto size = 400;

  auto matrixA = sparse_matrix_multiplication_tbb::GenerateRandomMatrix(size * size);
  auto matrixB = sparse_matrix_multiplication_tbb::GenerateRandomMatrix(size * size);
  std::vector<double> result(size * size, 0);

  auto task_data_omp = std::make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
  task_data_omp->inputs_count = {size, size, size, size};
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data_omp->outputs_count.emplace_back(result.size());

  sparse_matrix_multiplication_tbb::CCSMatrixTBB multiplicationTask(task_data_omp);
  ASSERT_TRUE(multiplicationTask.Validation()) << "Validation failed for random large matrices!";

  multiplicationTask.PreProcessing();
  clock_t start = clock();
  multiplicationTask.Run();
  clock_t end = clock();
  std::cout << "Time on matrix 400*400 = " << double(end - start);
  multiplicationTask.PostProcessing();
}
