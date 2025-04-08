#pragma once

#include <iostream>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sparse_matrix_multiplication_seq {

std::vector<double> GenerateRandomMatrix(int dimension);
std::vector<double> MultiplyMatrices(const std::vector<double>& first_matrix, int first_rows, int first_columns,
                                     const std::vector<double>& second_matrix, int second_rows, int second_columns);

class SparseMatrix {
  int rows_count_ = 0;
  int cols_count_ = 0;
  std::vector<double> values_;
  std::vector<int> row_indices_;
  std::vector<int> cumulative_elements_;

  static SparseMatrix ComputeTranspose(const SparseMatrix& matrix);
  static int CountElements(int index, const std::vector<int>& elements_count);

  SparseMatrix MatrixToSparse(int rows_count, int columns_count, const std::vector<double>& values);
  std::vector<double> FromSparseMatrix(const SparseMatrix& matrix);

 public:
  constexpr static double kThreshold = 1e-6;
  SparseMatrix() = default;
  SparseMatrix(int rows, int columns, const std::vector<double>& values, const std::vector<int>& rows_index,
               const std::vector<int>& cumulative_sum) noexcept
      : rows_count_(rows),
        cols_count_(columns),
        values_(values),
        row_indices_(rows_index),
        cumulative_elements_(cumulative_sum) {}

  const std::vector<double>& GetValues() const noexcept { return values_; }
  const std::vector<int>& GetRowIndices() const noexcept { return row_indices_; }
  const std::vector<int>& GetCumulativeElements() const noexcept { return cumulative_elements_; }
  int GetColumnCount() const noexcept { return cols_count_; }
  int GetRowCount() const noexcept { return rows_count_; }

  SparseMatrix operator*(const SparseMatrix& other) const noexcept(false);
};

class CCSMatrixSeq : public ppc::core::Task {
  SparseMatrix first_matrix_;
  SparseMatrix second_matrix_;
  SparseMatrix result_matrix_;

 public:
  explicit CCSMatrixSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace sparse_matrix_multiplication_seq
