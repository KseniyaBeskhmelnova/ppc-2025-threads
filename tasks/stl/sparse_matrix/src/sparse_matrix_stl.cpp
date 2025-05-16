#include "stl/sparse_matrix/include/sparse_matrix_stl.hpp"

#include <execution>
#include <random>

namespace sparse_matrix_multiplication_stl {

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

SparseMatrix SparseMatrix::ComputeTranspose(const SparseMatrix& matrix) {
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

double SparseMatrix::CalculateSum(const SparseMatrix& first_matrix, const SparseMatrix& second_matrix,
                                  const std::vector<int>& first_sums, const std::vector<int>& second_sums,
                                  int i, int j) {
  int first_elements_count = CountElements(j, first_sums);
  int second_elements_count = CountElements(i, second_sums);
  int first_start_index = j != 0 ? first_sums[j - 1] : 0;
  int second_start_index = i != 0 ? second_sums[i - 1] : 0;
  double sum = 0.0;
  for (int k = 0; k < first_elements_count; k++)
    for (int r = 0; r < second_elements_count; r++)
      if (first_matrix.GetRowIndices()[first_start_index + k] == second_matrix.GetRowIndices()[second_start_index + r])
        sum += first_matrix.GetValues()[k + first_start_index] * second_matrix.GetValues()[r + second_start_index];
  return sum;
}

std::vector<std::pair<int, int>> SparseMatrix::StartIndexes(size_t vector_size) {
  int n = static_cast<int>(vector_size) % threads_count;
  int count = static_cast<int>(vector_size) / threads_count;
  std::vector<std::pair<int, int>> threads_start_indexes(threads_count);
  for (int i = 0; i < static_cast<int>(threads_start_indexes.size()); ++i) {
    if (i == 0)
      threads_start_indexes[i] = {0, count};
    else
      threads_start_indexes[i] = {threads_start_indexes[i - 1].second, threads_start_indexes[i - 1].second + count};
    if (i < n)
      threads_start_indexes[i].second++;
  }
  return threads_start_indexes;
}

SparseMatrix SparseMatrix::operator*(const SparseMatrix& other) const {
  std::vector<double> result_values;
  std::vector<int> result_rows;
  std::vector<int> result_cumulative(other.GetColumnCount(), 0);

  auto transposed = ComputeTranspose(*this);
  const auto& first_sums = transposed.GetCumulativeElements();
  const auto& second_sums = other.GetCumulativeElements();

  std::vector<std::thread> threads(threads_count);
  std::vector<MatrixComponents> threads_data(threads_count);
  auto function = [&](size_t start, size_t end, size_t index) {
    MatrixComponents component;
    component.elementsSum.resize(end - start);
    for (size_t i = start; i < end; i++) {
      for (size_t j = 0; j < first_sums.size(); j++) {
        double sum = CalculateSum(transposed, other, first_sums, second_sums, static_cast<int>(i), static_cast<int>(j));
        if (sum > kThreshold) {
          component.values.emplace_back(sum);
          component.rows.emplace_back(j);
          component.elementsSum[i - start]++;
        }
      }
    }
    threads_data[index] = std::move(component);
  };
  auto indexes = StartIndexes(second_sums.size());
  for (size_t i = 0; i < threads_data.size(); i++)
    threads[i] = std::thread(function, indexes[i].first, indexes[i].second, i);
  std::ranges::for_each(threads, [&](auto& thread) { thread.join(); });
  MatrixComponents result;
  for (auto& data : threads_data) {
    for (size_t i = 0; i < data.rows.size(); i++) {
      result.rows.emplace_back(data.rows[i]);
      result.values.emplace_back(data.values[i]);
    }
    for (size_t i = 0; i < data.elementsSum.size(); i++)
      result.elementsSum.emplace_back(data.elementsSum[i]);
  }
  for (size_t i = 1; i < result.elementsSum.size(); i++)
    result.elementsSum[i] = result.elementsSum[i] + result.elementsSum[i - 1];
  return {rows_count_, other.GetColumnCount(), result};
}

std::vector<double> GenerateRandomMatrix(int dimension) {
  std::vector<double> data(dimension);
  std::mt19937 generator(std::random_device{}());

  for (auto& val : data) {
    val = static_cast<double>(generator() % 500);
    if (val > 250.0) val = 0.0;
  }
  return data;
}

bool CCSMatrixSTL::PreProcessingImpl() {
  int f_rows = static_cast<int>(task_data->inputs_count[0]);
  int f_cols = static_cast<int>(task_data->inputs_count[1]);
  int s_rows = static_cast<int>(task_data->inputs_count[2]);
  int s_cols = static_cast<int>(task_data->inputs_count[3]);

  if (f_rows == 0 || f_cols == 0 || s_rows == 0 || s_cols == 0) return true;

  std::vector<double> f_matrix(reinterpret_cast<double*>(task_data->inputs[0]),
                               reinterpret_cast<double*>(task_data->inputs[0]) + f_rows * f_cols);
  first_matrix_ = MatrixToSparse(f_rows, f_cols, f_matrix);
  std::vector<double> s_matrix(reinterpret_cast<double*>(task_data->inputs[1]),
                               reinterpret_cast<double*>(task_data->inputs[1]) + s_rows * s_cols);
  second_matrix_ = MatrixToSparse(s_rows, s_cols, s_matrix);
  std::cout << "First matrix non-zeros: " << first_matrix_.GetValues().size() << std::endl;
  std::cout << "Second matrix non-zeros: " << second_matrix_.GetValues().size() << std::endl;
  return true;
}

bool CCSMatrixSTL::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2];
}

bool CCSMatrixSTL::RunImpl() {
  result_matrix_ = first_matrix_ * second_matrix_;
  std::cout << "Result matrix non-zeros: " << result_matrix_.GetValues().size() << std::endl;
  return true;
}

bool CCSMatrixSTL::PostProcessingImpl() {
  auto result = FromSparseMatrix(result_matrix_);
  std::copy(result.begin(), result.end(), reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}
}  // namespace sparse_matrix_multiplication_stl