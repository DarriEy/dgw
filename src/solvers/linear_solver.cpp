/**
 * @file linear_solver.cpp
 * @brief Linear solver implementations
 */

#include "dgw/solvers/linear_solver.hpp"
#include <stdexcept>
#include <sstream>
#include <cmath>

namespace dgw {

// ============================================================================
// LinearSolverBase
// ============================================================================

void LinearSolverBase::solve(const SparseMatrix& A, const Vector& b, Vector& x) {
    analyze_pattern(A);
    factorize(A);
    solve(b, x);
}

UniquePtr<LinearSolverBase> LinearSolverBase::create(LinearSolver type) {
    switch (type) {
        case LinearSolver::EigenLU:
            return std::make_unique<EigenLUSolver>();
        case LinearSolver::EigenCholesky:
            return std::make_unique<EigenCholeskySolver>();
        case LinearSolver::EigenCG:
            return std::make_unique<EigenCGSolver>();
        case LinearSolver::EigenBiCGSTAB:
            return std::make_unique<EigenBiCGSTABSolver>();
#ifdef DGW_HAS_PETSC
        case LinearSolver::PETScKSP:
            return std::make_unique<PETScKSPSolver>();
        case LinearSolver::PETScDirect:
            return std::make_unique<PETScDirectSolver>();
#endif
        default:
            return std::make_unique<EigenLUSolver>();
    }
}

// ============================================================================
// EigenCGSolver
// ============================================================================

EigenCGSolver::EigenCGSolver(const IterativeSolverConfig& config) {
    solver_.setMaxIterations(config.max_iterations);
    solver_.setTolerance(config.tolerance);
}

std::string EigenCGSolver::info() const {
    std::ostringstream oss;
    oss << "Eigen CG (iters=" << iterations_ << ", error=" << error_ << ")";
    return oss.str();
}

// ============================================================================
// EigenBiCGSTABSolver
// ============================================================================

EigenBiCGSTABSolver::EigenBiCGSTABSolver(const IterativeSolverConfig& config) {
    solver_.setMaxIterations(config.max_iterations);
    solver_.setTolerance(config.tolerance);
}

std::string EigenBiCGSTABSolver::info() const {
    std::ostringstream oss;
    oss << "Eigen BiCGSTAB (iters=" << iterations_ << ", error=" << error_ << ")";
    return oss.str();
}

// ============================================================================
// Solver Utilities
// ============================================================================

namespace solver_utils {

LinearSolver recommend_solver(Index n_unknowns, bool is_symmetric, bool has_petsc) {
#ifdef DGW_HAS_PETSC
    if (has_petsc && n_unknowns > 100000) {
        return is_symmetric ? LinearSolver::PETScKSP : LinearSolver::PETScKSP;
    }
#endif

    if (n_unknowns < 10000) {
        // Direct solver for small problems
        return is_symmetric ? LinearSolver::EigenCholesky : LinearSolver::EigenLU;
    } else {
        // Iterative for larger problems
        return is_symmetric ? LinearSolver::EigenCG : LinearSolver::EigenBiCGSTAB;
    }
}

Size estimate_factorization_memory(const SparseMatrix& A) {
    // Rough estimate: fill-in factor of ~10 for 2D problems
    Size nnz = static_cast<Size>(A.nonZeros());
    Size n = static_cast<Size>(A.rows());
    Size fill_factor = 10;
    return (nnz * fill_factor + n) * sizeof(Real);
}

bool is_spd(const SparseMatrix& A) {
    // Quick check: diagonal must be positive
    for (Index i = 0; i < A.rows(); ++i) {
        if (A.coeff(i, i) <= 0.0) return false;
    }

    // Check symmetry (approximately)
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
            if (std::abs(it.value() - A.coeff(it.col(), it.row())) > 1e-10) {
                return false;
            }
        }
    }

    return true;
}

void diagonal_scale(SparseMatrix& A, Vector& b, Vector& scale) {
    const Index n = A.rows();
    scale.resize(n);

    for (Index i = 0; i < n; ++i) {
        Real diag = A.coeff(i, i);
        scale(i) = 1.0 / std::sqrt(std::abs(diag) + 1e-30);
    }

    // Scale rows and columns
    for (int k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
            it.valueRef() *= scale(it.row()) * scale(it.col());
        }
    }

    // Scale RHS
    b = b.cwiseProduct(scale);
}

void diagonal_unscale(Vector& x, const Vector& scale) {
    x = x.cwiseProduct(scale);
}

} // namespace solver_utils

} // namespace dgw
