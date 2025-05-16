#pragma once

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <optional>
#include <vector>

#include "core/task/include/task.hpp"

namespace sparse_matrix_multiplication_mpi_tbb {

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

class SparseMatrix {
  int rows_count_ = 0;
  int cols_count_ = 0;
  MatrixComponents components_;

  static int CountElements(int index, const std::vector<int>& elements_count);
  static MatrixComponents ComputeLocalMultiplication(const SparseMatrix& first_matrix,
                                                     const SparseMatrix& second_matrix, int start_col, int end_col);
  static SparseMatrix GatherResults(const MatrixComponents& local_result, const SparseMatrix& first_matrix,
                                    const SparseMatrix& second_matrix, boost::mpi::communicator& world,
                                    const std::vector<int>& displacements,
                                    const std::pair<std::vector<int>, std::vector<int>>& sizes);

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
      : rows_count_(rows_count), cols_count_(columns_count), components_(std::move(components)) {}

  const std::vector<double>& GetValues() const noexcept { return components_.values; }
  const std::vector<int>& GetRowIndices() const noexcept { return components_.rows; }
  const std::vector<int>& GetCumulativeElements() const noexcept { return components_.elementsSum; }

  int GetColumnCount() const noexcept { return cols_count_; }
  int GetRowCount() const noexcept { return rows_count_; }

  static MatrixComponents Multiplicate(const SparseMatrix& first_matrix, const SparseMatrix& second_matrix,
                                       int start_col, int end_col, boost::mpi::communicator& world,
                                       const std::vector<int>& displacements);

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& components_.values;
    ar& components_.rows;
    ar& components_.elementsSum;
    ar& rows_count_;
    ar& cols_count_;
  }
};

std::vector<double> GenerateRandomMatrix(int dimension);
std::vector<double> MultiplyMatrices(const std::vector<double>& first_matrix, int first_rows, int first_columns,
                                     const std::vector<double>& second_matrix, int second_rows, int second_columns);
SparseMatrix MatrixToSparse(int rows_count, int columns_count, const std::vector<double>& values);
std::vector<double> FromSparseMatrix(const SparseMatrix& matrix);
static SparseMatrix ComputeTranspose(const SparseMatrix& matrix);

class CCSMatrixMpiTbb : public ppc::core::Task {
  SparseMatrix first_matrix_;
  SparseMatrix second_matrix_;
  SparseMatrix result_matrix_;
  boost::mpi::communicator world_;
  std::vector<int> displacements_;
  MatrixComponents intermediate_data_;
  std::pair<std::vector<int>, std::vector<int>> sizes_;

 public:
  explicit CCSMatrixMpiTbb(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int GetWorldRank() const { return world_.rank(); }
};

}  // namespace sparse_matrix_multiplication_mpi_tbb