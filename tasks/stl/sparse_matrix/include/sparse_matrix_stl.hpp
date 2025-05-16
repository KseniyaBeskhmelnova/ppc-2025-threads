#pragma once

#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sparse_matrix_multiplication_stl {

const int threads_count = std::thread::hardware_concurrency();

struct MatrixComponents {
  std::vector<double> values;
  std::vector<int> rows;
  std::vector<int> elementsSum;
  void Resize(size_t values_size, std::optional<size_t> sums_size) noexcept {
    values.resize(values_size);
    rows.resize(values_size);
    if (sums_size.has_value()) {
      elementsSum.resize(sums_size.value());
    }
  }
};

std::vector<double> GenerateRandomMatrix(int dimension);
std::vector<double> MultiplyMatrices(const std::vector<double>& first_matrix, int first_rows, int first_columns,
                                     const std::vector<double>& second_matrix, int second_rows, int second_columns);

class SparseMatrix {
  int rows_count_ = 0;
  int cols_count_ = 0;
  MatrixComponents components_;

  static SparseMatrix ComputeTranspose(const SparseMatrix& matrix);
  static int CountElements(int index, const std::vector<int>& elements_count);

  SparseMatrix MatrixToSparse(int rows_count, int columns_count, const std::vector<double>& values);
  std::vector<double> FromSparseMatrix(const SparseMatrix& matrix);

 public:
  constexpr static double kThreshold = 1e-6;
  SparseMatrix() = default;
  SparseMatrix(int rows, int columns, const std::vector<double>& values, const std::vector<int>& rows_index,
               const std::vector<int>& cumulative_sum) noexcept
      : rows_count_(rows), cols_count_(columns) {
    components_.values = values;
    components_.rows = rows_index;
    components_.elementsSum = cumulative_sum;
  }
  SparseMatrix(int rows_count, int columns_count, MatrixComponents components) noexcept
      : rows_count_(rows_count), cols_count_(columns_count), components_(std::move(components)){};

  const std::vector<double>& GetValues() const noexcept { return components_.values; }
  const std::vector<int>& GetRowIndices() const noexcept { return components_.rows; }
  const std::vector<int>& GetCumulativeElements() const noexcept { return components_.elementsSum; }

  int GetColumnCount() const noexcept { return cols_count_; }
  int GetRowCount() const noexcept { return rows_count_; }

  static double CalculateSum(const SparseMatrix& first_matrix, const SparseMatrix& second_matrix,
                             const std::vector<int>& felements_sum,
                      const std::vector<int>& selements_sum, int i_index, int j_index);
  static std::vector<std::pair<int, int>> StartIndexes(size_t vector_size);
  SparseMatrix operator*(const SparseMatrix& other) const noexcept(false);
};

class CCSMatrixSTL : public ppc::core::Task {
  SparseMatrix first_matrix_;
  SparseMatrix second_matrix_;
  SparseMatrix result_matrix_;

 public:
  explicit CCSMatrixSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace sparse_matrix_multiplication_stl