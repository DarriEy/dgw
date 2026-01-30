/**
 * @file remapper.hpp
 * @brief HRU-to-cell and cell-to-HRU remapping
 */

#pragma once

#include "../core/types.hpp"
#include "../core/mesh.hpp"
#include <stdexcept>

namespace dgw {

/**
 * @brief Handles remapping between SUMMA HRUs and GW mesh cells
 *
 * HRU = Hydrologic Response Unit (SUMMA spatial element)
 * The remapping uses sparse matrices for efficient area-weighted interpolation.
 */
class Remapper {
public:
    Remapper() = default;

    /**
     * @brief Set up remapping from HRU to cell mapping
     *
     * @param hru_to_cell Sparse matrix [n_cells x n_hrus] with area weights
     */
    void set_hru_to_cell_map(const SparseMatrix& hru_to_cell) {
        hru_to_cell_ = hru_to_cell;
        // Build inverse map (cell to HRU)
        cell_to_hru_ = hru_to_cell_.transpose();

        // Normalize columns of cell_to_hru to sum to 1.
        // For RowMajor storage, outerSize() iterates rows, so we must
        // compute column sums manually.
        Vector col_sums = Vector::Zero(cell_to_hru_.cols());
        for (int k = 0; k < cell_to_hru_.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(cell_to_hru_, k); it; ++it) {
                col_sums(it.col()) += it.value();
            }
        }
        for (int k = 0; k < cell_to_hru_.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(cell_to_hru_, k); it; ++it) {
                if (col_sums(it.col()) > 0.0) {
                    it.valueRef() /= col_sums(it.col());
                }
            }
        }
    }

    /**
     * @brief Remap from HRU values to cell values
     *
     * @param hru_values Values at HRUs
     * @return Values at cells (area-weighted)
     */
    Vector hru_to_cell(const Vector& hru_values) const {
        if (hru_to_cell_.cols() == 0) {
            throw std::runtime_error("HRU-to-cell map not set");
        }
        return hru_to_cell_ * hru_values;
    }

    /**
     * @brief Remap from cell values to HRU values
     *
     * @param cell_values Values at cells
     * @return Values at HRUs (area-weighted)
     */
    Vector cell_to_hru(const Vector& cell_values) const {
        if (cell_to_hru_.cols() == 0) {
            throw std::runtime_error("Cell-to-HRU map not set");
        }
        return cell_to_hru_ * cell_values;
    }

    /**
     * @brief Create identity mapping (1-to-1 HRU-cell)
     *
     * @param n Number of HRUs/cells (must be equal)
     */
    void set_identity(Index n) {
        std::vector<SparseTriplet> triplets;
        triplets.reserve(n);
        for (Index i = 0; i < n; ++i) {
            triplets.emplace_back(i, i, 1.0);
        }
        hru_to_cell_.resize(n, n);
        hru_to_cell_.setFromTriplets(triplets.begin(), triplets.end());
        cell_to_hru_ = hru_to_cell_;
    }

    /// Get number of HRUs
    Index n_hrus() const { return hru_to_cell_.cols(); }

    /// Get number of cells
    Index n_cells() const { return hru_to_cell_.rows(); }

    /// Get mapping matrices
    const SparseMatrix& hru_to_cell_map() const { return hru_to_cell_; }
    const SparseMatrix& cell_to_hru_map() const { return cell_to_hru_; }

private:
    SparseMatrix hru_to_cell_;  // [n_cells x n_hrus]
    SparseMatrix cell_to_hru_;  // [n_hrus x n_cells]
};

} // namespace dgw
