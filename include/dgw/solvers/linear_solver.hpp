/**
 * @file linear_solver.hpp
 * @brief Linear solver interfaces with multiple backends
 * 
 * Supports:
 * - Eigen direct solvers (SparseLU, Cholesky)
 * - Eigen iterative solvers (CG, BiCGSTAB)
 * - PETSc (optional, for large-scale)
 * 
 * The solver choice can significantly impact performance:
 * - Direct: Best for small/medium problems, exact solution
 * - Iterative: Required for large problems, need good preconditioner
 */

#pragma once

#include "../core/types.hpp"
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

namespace dgw {

/**
 * @brief Abstract base for linear solvers
 */
class LinearSolverBase {
public:
    virtual ~LinearSolverBase() = default;
    
    /**
     * @brief Analyze sparsity pattern (one-time setup)
     */
    virtual void analyze_pattern(const SparseMatrix& A) = 0;
    
    /**
     * @brief Factorize matrix (call after values change)
     */
    virtual void factorize(const SparseMatrix& A) = 0;
    
    /**
     * @brief Solve A*x = b
     */
    virtual void solve(const Vector& b, Vector& x) = 0;
    
    /**
     * @brief Combined analyze + factorize + solve
     */
    virtual void solve(const SparseMatrix& A, const Vector& b, Vector& x);
    
    /**
     * @brief Check if factorization succeeded
     */
    virtual bool success() const = 0;
    
    /**
     * @brief Get solver info string
     */
    virtual std::string info() const = 0;
    
    /// Factory method
    static UniquePtr<LinearSolverBase> create(LinearSolver type);
};

// ============================================================================
// Eigen Direct Solvers
// ============================================================================

/**
 * @brief Eigen SparseLU solver (general matrices)
 */
class EigenLUSolver : public LinearSolverBase {
public:
    void analyze_pattern(const SparseMatrix& A) override {
        solver_.analyzePattern(A);
    }
    
    void factorize(const SparseMatrix& A) override {
        solver_.factorize(A);
    }
    
    void solve(const Vector& b, Vector& x) override {
        x = solver_.solve(b);
    }
    
    bool success() const override {
        return solver_.info() == Eigen::Success;
    }
    
    std::string info() const override {
        return "Eigen SparseLU";
    }
    
private:
    Eigen::SparseLU<SparseMatrix> solver_;
};

/**
 * @brief Eigen SimplicialLDLT solver (SPD matrices)
 * 
 * Much faster than LU for symmetric positive definite systems.
 * Groundwater Jacobians are typically SPD with proper formulation.
 */
class EigenCholeskySolver : public LinearSolverBase {
public:
    void analyze_pattern(const SparseMatrix& A) override {
        // Convert to column-major for Cholesky
        Eigen::SparseMatrix<Real, Eigen::ColMajor> A_col = A;
        solver_.analyzePattern(A_col);
    }
    
    void factorize(const SparseMatrix& A) override {
        Eigen::SparseMatrix<Real, Eigen::ColMajor> A_col = A;
        solver_.factorize(A_col);
    }
    
    void solve(const Vector& b, Vector& x) override {
        x = solver_.solve(b);
    }
    
    bool success() const override {
        return solver_.info() == Eigen::Success;
    }
    
    std::string info() const override {
        return "Eigen SimplicialLDLT";
    }
    
private:
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Real, Eigen::ColMajor>> solver_;
};

// ============================================================================
// Eigen Iterative Solvers
// ============================================================================

/**
 * @brief Configuration for iterative solvers
 */
struct IterativeSolverConfig {
    Index max_iterations = 1000;
    Real tolerance = 1e-10;
    
    // Preconditioner
    enum class Preconditioner {
        None,
        Diagonal,           // Jacobi
        IncompleteLU,       // ILU(0)
        IncompleteCholesky  // IC(0)
    };
    Preconditioner preconditioner = Preconditioner::IncompleteLU;
};

/**
 * @brief Eigen Conjugate Gradient solver (SPD matrices)
 */
class EigenCGSolver : public LinearSolverBase {
public:
    EigenCGSolver() = default;
    explicit EigenCGSolver(const IterativeSolverConfig& config);
    
    void analyze_pattern(const SparseMatrix& A) override {
        solver_.analyzePattern(A);
    }
    
    void factorize(const SparseMatrix& A) override {
        solver_.factorize(A);
    }
    
    void solve(const Vector& b, Vector& x) override {
        x = solver_.solve(b);
        iterations_ = solver_.iterations();
        error_ = solver_.error();
    }
    
    bool success() const override {
        return solver_.info() == Eigen::Success;
    }
    
    std::string info() const override;
    
    Index iterations() const { return iterations_; }
    Real error() const { return error_; }
    
    void set_max_iterations(Index max_iter) {
        solver_.setMaxIterations(max_iter);
    }
    
    void set_tolerance(Real tol) {
        solver_.setTolerance(tol);
    }
    
private:
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower|Eigen::Upper,
                             Eigen::IncompleteCholesky<Real>> solver_;
    Index iterations_ = 0;
    Real error_ = 0.0;
};

/**
 * @brief Eigen BiCGSTAB solver (non-symmetric matrices)
 */
class EigenBiCGSTABSolver : public LinearSolverBase {
public:
    EigenBiCGSTABSolver() = default;
    explicit EigenBiCGSTABSolver(const IterativeSolverConfig& config);
    
    void analyze_pattern(const SparseMatrix& A) override {
        solver_.analyzePattern(A);
    }
    
    void factorize(const SparseMatrix& A) override {
        solver_.factorize(A);
    }
    
    void solve(const Vector& b, Vector& x) override {
        x = solver_.solve(b);
        iterations_ = solver_.iterations();
        error_ = solver_.error();
    }
    
    bool success() const override {
        return solver_.info() == Eigen::Success;
    }
    
    std::string info() const override;
    
    Index iterations() const { return iterations_; }
    Real error() const { return error_; }
    
private:
    Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<Real>> solver_;
    Index iterations_ = 0;
    Real error_ = 0.0;
};

// ============================================================================
// PETSc Solvers (optional)
// ============================================================================

#ifdef DGW_HAS_PETSC

/**
 * @brief PETSc KSP solver interface
 * 
 * Provides access to PETSc's rich collection of Krylov solvers
 * and preconditioners, essential for large-scale problems.
 */
class PETScKSPSolver : public LinearSolverBase {
public:
    PETScKSPSolver();
    ~PETScKSPSolver();
    
    // No copy (PETSc objects)
    PETScKSPSolver(const PETScKSPSolver&) = delete;
    PETScKSPSolver& operator=(const PETScKSPSolver&) = delete;
    
    void analyze_pattern(const SparseMatrix& A) override;
    void factorize(const SparseMatrix& A) override;
    void solve(const Vector& b, Vector& x) override;
    bool success() const override;
    std::string info() const override;
    
    /**
     * @brief Set KSP type (e.g., "gmres", "cg", "bcgs")
     */
    void set_ksp_type(const std::string& type);
    
    /**
     * @brief Set preconditioner type (e.g., "ilu", "jacobi", "asm", "hypre")
     */
    void set_pc_type(const std::string& type);
    
    /**
     * @brief Set solver options from command line style string
     * 
     * Example: "-ksp_type gmres -pc_type hypre -pc_hypre_type boomeramg"
     */
    void set_options(const std::string& options);
    
    Index iterations() const { return iterations_; }
    
private:
    struct Impl;
    UniquePtr<Impl> impl_;
    Index iterations_ = 0;
    bool success_ = false;
};

/**
 * @brief PETSc direct solver (MUMPS/SuperLU)
 */
class PETScDirectSolver : public LinearSolverBase {
public:
    PETScDirectSolver();
    ~PETScDirectSolver();
    
    void analyze_pattern(const SparseMatrix& A) override;
    void factorize(const SparseMatrix& A) override;
    void solve(const Vector& b, Vector& x) override;
    bool success() const override;
    std::string info() const override;
    
    /**
     * @brief Set package (e.g., "mumps", "superlu_dist")
     */
    void set_package(const std::string& package);
    
private:
    struct Impl;
    UniquePtr<Impl> impl_;
    bool success_ = false;
};

#endif // DGW_HAS_PETSC

// ============================================================================
// Solver Selection Utilities
// ============================================================================

namespace solver_utils {

/**
 * @brief Recommend solver based on problem size and properties
 */
LinearSolver recommend_solver(
    Index n_unknowns,
    bool is_symmetric = true,
    bool has_petsc = false
);

/**
 * @brief Estimate memory for direct factorization
 * 
 * @param A Sparse matrix
 * @return Estimated memory in bytes
 */
Size estimate_factorization_memory(const SparseMatrix& A);

/**
 * @brief Check if matrix is SPD (for solver selection)
 */
bool is_spd(const SparseMatrix& A);

/**
 * @brief Apply diagonal scaling for better conditioning
 */
void diagonal_scale(SparseMatrix& A, Vector& b, Vector& scale);

/**
 * @brief Reverse diagonal scaling on solution
 */
void diagonal_unscale(Vector& x, const Vector& scale);

} // namespace solver_utils

} // namespace dgw
