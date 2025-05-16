#include "all/sparse_matrix/include/sparse_matrix_all.hpp"

#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/serialization/vector.hpp>
#include <random>

#include "core/util/include/util.hpp"

namespace sparse_matrix_multiplication_mpi_tbb {

std::vector<double> GenerateRandomMatrix(int dimension) {
  std::vector<double> data(dimension);
  std::mt19937 generator(std::random_device{}());

  for (auto& val : data) {
    val = static_cast<double>(generator() % 500);
    if (val > 250.0) val = 0.0;
  }
  return data;
}

std::vector<double> MultiplyMatrices(const std::vector<double>& first_matrix, int first_rows, int first_columns,
                                     const std::vector<double>& second_matrix, int second_rows, int second_columns) {
  if (first_columns != second_rows) throw std::invalid_argument("Matrix dimensions do not match for multiplication");
  std::vector<double> result(first_rows * second_columns, 0.0);
  for (int i = 0; i < first_rows; i++) {
    for (int j = 0; j < second_columns; j++) {
      double sum = 0.0;
      for (int k = 0; k < first_columns; k++)
        sum += first_matrix[i * first_columns + k] * second_matrix[k * second_columns + j];
      result[i * second_columns + j] = sum;
    }
  }
  return result;
}

SparseMatrix ComputeTranspose(const SparseMatrix& matrix) {
  std::vector<double> new_values;
  std::vector<int> new_rows;
  std::vector<int> new_cumulative;
  int max_dim = std::max(matrix.GetRowCount(), matrix.GetColumnCount());
  std::vector<std::vector<double>> grouped_values(max_dim);
  std::vector<std::vector<int>> grouped_indices(max_dim);
  int current_col = 0;
  int count = 0;
  for (size_t i = 0; i < matrix.GetValues().size(); i++) {
    if (count == matrix.GetCumulativeElements()[current_col]) current_col++;
    count++;
    grouped_values[matrix.GetRowIndices()[i]].push_back(matrix.GetValues()[i]);
    grouped_indices[matrix.GetRowIndices()[i]].push_back(current_col);
  }

  for (size_t i = 0; i < grouped_values.size(); i++) {
    for (size_t j = 0; j < grouped_values[i].size(); j++) {
      new_values.push_back(grouped_values[i][j]);
      new_rows.push_back(grouped_indices[i][j]);
    }
    new_cumulative.push_back(new_values.size());
  }
  return SparseMatrix(matrix.GetColumnCount(), matrix.GetRowCount(), new_values, new_rows, new_cumulative);
}

SparseMatrix MatrixToSparse(int rows_count, int columns_count, const std::vector<double>& values) {
  std::vector<double> sparse_values;
  std::vector<int> row_indices;
  std::vector<int> cumulative_elements;

  int count = 0;
  for (int col = 0; col < columns_count; col++) {
    for (int row = 0; row < rows_count; row++) {
      double val = values[row * columns_count + col];
      if (std::abs(val) > SparseMatrix::kThreshold) {
        sparse_values.push_back(val);
        row_indices.push_back(row);
        count++;
      }
    }
    cumulative_elements.push_back(count);
  }
  return SparseMatrix(rows_count, columns_count, sparse_values, row_indices, cumulative_elements);
}

std::vector<double> FromSparseMatrix(const SparseMatrix& matrix) {
  std::vector<double> dense_matrix(matrix.GetRowCount() * matrix.GetColumnCount(), 0.0);
  const auto& values = matrix.GetValues();
  const auto& row_indices = matrix.GetRowIndices();
  const auto& cumulative = matrix.GetCumulativeElements();

  int col = 0;
  int count = 0;
  for (size_t i = 0; i < values.size(); i++) {
    while (count == cumulative[col]) col++;
    count++;
    dense_matrix[row_indices[i] * matrix.GetColumnCount() + col] = values[i];
  }
  return dense_matrix;
}

int SparseMatrix::CountElements(int index, const std::vector<int>& elements_count) {
  if (index == 0) return elements_count[index];
  return elements_count[index] - elements_count[index - 1];
}

MatrixComponents SparseMatrix::ComputeLocalMultiplication(const SparseMatrix& first_matrix,
                                                          const SparseMatrix& second_matrix, int start_col,
                                                          int end_col) {
  const auto& first_sums = first_matrix.GetCumulativeElements();
  const auto& second_sums = second_matrix.GetCumulativeElements();
  int local_cols = end_col - start_col;

  MatrixComponents component;
  if (local_cols <= 0 || first_sums.empty() || second_sums.empty()) {
    component.elementsSum.resize(local_cols, 0);
    return component;
  }

  component.elementsSum.resize(local_cols, 0);
  std::vector<std::vector<double>> local_values(local_cols);
  std::vector<std::vector<int>> local_rows(local_cols);

  int num_threads = std::max(1, ppc::util::GetPPCNumThreads());
  oneapi::tbb::task_arena arena(num_threads);
  arena.execute([&] {
    tbb::parallel_for(
        tbb::blocked_range<int>(0, local_cols, std::max(1, local_cols / num_threads)),
        [&](const tbb::blocked_range<int>& range) {
          for (int local_col = range.begin(); local_col != range.end(); local_col++) {
            int global_col = start_col + local_col;
            if (global_col >= static_cast<int>(second_sums.size()))
              continue;
            for (int row = 0; row < static_cast<int>(first_matrix.GetRowCount()); row++) {
              double sum = 0.0;
              int first_count = CountElements(row, first_sums);
              int second_count = CountElements(global_col, second_sums);
              int first_start = row == 0 ? 0 : first_sums[row - 1];
              int second_start = global_col == 0 ? 0 : second_sums[global_col - 1];

              if (first_start + first_count > static_cast<int>(first_matrix.GetValues().size()) ||
                  second_start + second_count > static_cast<int>(second_matrix.GetValues().size())) {
                std::cerr << "Error: Invalid indices at global_col=" << global_col << ", row=" << row
                          << ", first_start=" << first_start << ", first_count=" << first_count
                          << ", second_start=" << second_start << ", second_count=" << second_count << std::endl;
                continue;
              }

              for (int i = 0; i < first_count; i++)
                for (int j = 0; j < second_count; j++)
                  if (first_matrix.GetRowIndices()[first_start + i] ==
                      second_matrix.GetRowIndices()[second_start + j])
                    sum += first_matrix.GetValues()[first_start + i] * second_matrix.GetValues()[second_start + j];

              if (std::abs(sum) > kThreshold) {
                local_values[local_col].push_back(sum);
                local_rows[local_col].push_back(row);
                component.elementsSum[local_col]++;
              }
            }
          }
        });
  });

  size_t total_values = 0;
  for (int i = 0; i < local_cols; i++) {
    total_values += local_values[i].size();
    component.values.insert(component.values.end(), local_values[i].begin(), local_values[i].end());
    component.rows.insert(component.rows.end(), local_rows[i].begin(), local_rows[i].end());
  }

  for (size_t i = 1; i < component.elementsSum.size(); i++)
    component.elementsSum[i] += component.elementsSum[i - 1];

  return component;
}

SparseMatrix SparseMatrix::GatherResults(const MatrixComponents& local_result, const SparseMatrix& first_matrix,
                                         const SparseMatrix& second_matrix, boost::mpi::communicator& world,
                                         const std::vector<int>& displacements,
                                         const std::pair<std::vector<int>, std::vector<int>>& sizes) {
  if (world.rank() == 0) {
    // Сбор данных от всех процессов
    std::vector<std::vector<double>> all_values(world.size());
    std::vector<std::vector<int>> all_rows(world.size());
    std::vector<std::vector<int>> all_sums(world.size());
    boost::mpi::gather(world, local_result.values, all_values, 0);
    boost::mpi::gather(world, local_result.rows, all_rows, 0);
    boost::mpi::gather(world, local_result.elementsSum, all_sums, 0);

    // Итоговые массивы
    std::vector<double> final_values;
    std::vector<int> final_rows;
    std::vector<int> final_cumulative(second_matrix.GetColumnCount(), 0);

    // Диапазоны столбцов для каждого процесса
    std::vector<std::pair<int, int>> col_ranges(world.size());
    for (int r = 0; r < world.size(); r++) {
      int proc_start = displacements[r];
      int proc_end = (r == world.size() - 1) ? second_matrix.GetColumnCount() : displacements[r + 1];
      col_ranges[r] = {proc_start, proc_end};
    }

    // Сбор данных в порядке столбцов
    for (int col = 0; col < second_matrix.GetColumnCount(); col++) {
      int proc = -1;
      for (int r = 0; r < world.size(); r++) {
        if (col >= col_ranges[r].first && col < col_ranges[r].second) {
          proc = r;
          break;
        }
      }
      if (proc != -1 && !all_sums[proc].empty()) {
        int local_col_idx = col - col_ranges[proc].first;
        int cum_start =
            (local_col_idx == 0) ? 0 : (local_col_idx < all_sums[proc].size() ? all_sums[proc][local_col_idx - 1] : 0);
        int cum_end = local_col_idx < all_sums[proc].size() ? all_sums[proc][local_col_idx] : cum_start;

        final_cumulative[col] = cum_end - cum_start;
        if (cum_end > cum_start && cum_start < all_values[proc].size() && cum_end <= all_values[proc].size()) {
          final_values.insert(final_values.end(), all_values[proc].begin() + cum_start,
                              all_values[proc].begin() + cum_end);
          final_rows.insert(final_rows.end(), all_rows[proc].begin() + cum_start, all_rows[proc].begin() + cum_end);
        }
      }
    }

    // Формирование префиксной суммы для final_cumulative
    for (size_t i = 1; i < final_cumulative.size(); i++) {
      final_cumulative[i] += final_cumulative[i - 1];
    }

    return SparseMatrix(first_matrix.GetRowCount(), second_matrix.GetColumnCount(), final_values, final_rows,
                        final_cumulative);
  } else {
    boost::mpi::gather(world, local_result.values, 0);
    boost::mpi::gather(world, local_result.rows, 0);
    boost::mpi::gather(world, local_result.elementsSum, 0);
    return SparseMatrix();
  }
}

MatrixComponents SparseMatrix::Multiplicate(const SparseMatrix& first_matrix, const SparseMatrix& second_matrix,
                                            int start_col, int end_col, boost::mpi::communicator& world,
                                            const std::vector<int>& displacements) {
  MatrixComponents local_result = ComputeLocalMultiplication(first_matrix, second_matrix, start_col, end_col);

  std::pair<std::vector<int>, std::vector<int>> sizes;
  sizes.first.resize(world.size(), 0);
  sizes.second.resize(world.size(), 0);

  // Сбор размеров локальных данных
  sizes.first[world.rank()] = local_result.values.size();
  sizes.second[world.rank()] = local_result.elementsSum.size();
  boost::mpi::all_gather(world, sizes.first[world.rank()], sizes.first);
  boost::mpi::all_gather(world, sizes.second[world.rank()], sizes.second);

  for (int i = 0; i < world.size(); i++)
    if (sizes.second[i] == 0)
      sizes.second[i] = 1;  // Гарантируем, что elementsSum не пустой

  // Сбор результатов
  SparseMatrix result = GatherResults(local_result, first_matrix, second_matrix, world, displacements, sizes);

  return MatrixComponents{result.GetValues(), result.GetRowIndices(), result.GetCumulativeElements()};
}

bool CCSMatrixMpiTbb::PreProcessingImpl() {
  int f_rows = static_cast<int>(task_data->inputs_count[0]);
  int f_cols = static_cast<int>(task_data->inputs_count[1]);
  int s_rows = static_cast<int>(task_data->inputs_count[2]);
  int s_cols = static_cast<int>(task_data->inputs_count[3]);

  // Формирование входных матриц
  try {
    std::vector<double> f_matrix(reinterpret_cast<double*>(task_data->inputs[0]),
                                 reinterpret_cast<double*>(task_data->inputs[0]) + f_rows * f_cols);
    first_matrix_ = MatrixToSparse(f_rows, f_cols, f_matrix);
    first_matrix_ = ComputeTranspose(first_matrix_);

    std::vector<double> s_matrix(reinterpret_cast<double*>(task_data->inputs[1]),
                                 reinterpret_cast<double*>(task_data->inputs[1]) + s_rows * s_cols);
    second_matrix_ = MatrixToSparse(s_rows, s_cols, s_matrix);
  } catch (const std::exception& e) {
    std::cerr << "Error in matrix creation: " << e.what() << std::endl;
    return false;
  }
  // Распределение столбцов между процессами
  int total_cols = second_matrix_.GetColumnCount();
  int cols_per_process = total_cols / world_.size();
  int remainder = total_cols % world_.size();
  displacements_.resize(world_.size());
  for (int i = 0; i < world_.size(); i++)
    displacements_[i] = i * cols_per_process + std::min(i, remainder);

  return true;
}

bool CCSMatrixMpiTbb::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool CCSMatrixMpiTbb::RunImpl() {
  if (first_matrix_.GetRowCount() == 0 || first_matrix_.GetColumnCount() == 0 || second_matrix_.GetRowCount() == 0 ||
      second_matrix_.GetColumnCount() == 0 || displacements_.empty()) {
    if (world_.rank() == 0)
      result_matrix_ = SparseMatrix(0, 0, {}, {}, {0});
    return true;
  }
  boost::mpi::broadcast(world_, first_matrix_, 0);
  boost::mpi::broadcast(world_, second_matrix_, 0);
  boost::mpi::broadcast(world_, displacements_, 0);

  int start_col = displacements_[world_.rank()];
  int end_col =
      (world_.rank() == world_.size() - 1) ? second_matrix_.GetColumnCount() : displacements_[world_.rank() + 1];

  intermediate_data_ =
      SparseMatrix::Multiplicate(first_matrix_, second_matrix_, start_col, end_col, world_, displacements_);

  if (world_.rank() == 0) {
    result_matrix_ = SparseMatrix(first_matrix_.GetColumnCount(), second_matrix_.GetColumnCount(),
                                  intermediate_data_.values, intermediate_data_.rows, intermediate_data_.elementsSum);
    std::cout << "First matrix non-zeros: " << first_matrix_.GetValues().size() << std::endl;
    std::cout << "Second matrix non-zeros: " << second_matrix_.GetValues().size() << std::endl;
    std::cout << "Result matrix non-zeros: " << result_matrix_.GetValues().size() << std::endl;
  }

  return true;
}

bool CCSMatrixMpiTbb::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto result = FromSparseMatrix(result_matrix_);
    std::copy(result.begin(), result.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}

}  // namespace sparse_matrix_multiplication_mpi_tbb