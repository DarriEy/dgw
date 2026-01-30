/**
 * @file newton.hpp
 * @brief Newton-Raphson solver with Enzyme AD support
 * 
 * Solves nonlinear systems F(x) = 0 using Newton's method:
 *   x_{k+1} = x_k - J⁻¹ F(x_k)
 * 
 * Features:
 * - Line search for globalization
 * - Trust region option
 * - Automatic differentiation via Enzyme for Jacobian
 * - Finite difference Jacobian fallback
 * - Implicit differentiation for gradient computation
 */

#pragma once

#include "../core/types.hpp"
#include "../core/config.hpp"
#include "linear_solver.hpp"
#include <functional>

namespace dgw {

/**
 * @brief Configuration for Newton solver
 */
struct NewtonConfig {
    Index max_iterations = 50;
    Real tolerance = 1e-6;
    Real relative_tolerance = 1e-8;
    
    // Relaxation / damping
    Real initial_relaxation = 1.0;
    Real min_relaxation = 0.01;
    
    // Line search
    bool use_line_search = true;
    Real line_search_alpha = 1e-4;    // Armijo condition parameter
    Real line_search_beta = 0.5;      // Backtracking factor
    Index max_line_search_iters = 10;
    
    // Trust region (alternative to line search)
    bool use_trust_region = false;
    Real initial_trust_radius = 1.0;
    Real max_trust_radius = 100.0;
    
    // Jacobian computation
    enum class JacobianMethod {
        Analytical,         // User-provided Jacobian function
        Enzyme,            // Automatic differentiation via Enzyme
        FiniteDifference,  // Numerical approximation
        BFGS               // Quasi-Newton approximation
    };
    JacobianMethod jacobian_method = JacobianMethod::Analytical;
    Real fd_epsilon = 1e-7;  // For finite difference
    
    // Convergence monitoring
    bool verbose = false;
    ConvergenceCallback callback = nullptr;
};

/**
 * @brief Newton-Raphson solver
 */
class NewtonSolver {
public:
    NewtonSolver();
    explicit NewtonSolver(const NewtonConfig& config);
    explicit NewtonSolver(const SolverConfig& config);
    
    /**
     * @brief Solve F(x) = 0
     * 
     * @param residual_func Function computing F(x)
     * @param jacobian_func Function computing J = ∂F/∂x
     * @param x Initial guess, updated to solution
     * @return Solve result with convergence info
     */
    SolveResult solve(
        const ResidualFunc& residual_func,
        const JacobianFunc& jacobian_func,
        Vector& x
    );
    
    /**
     * @brief Solve with automatic Jacobian (Enzyme or FD)
     */
    SolveResult solve(
        const ResidualFunc& residual_func,
        Vector& x
    );
    
    /**
     * @brief Solve with initial Jacobian sparsity pattern
     * 
     * @param pattern Sparse matrix with correct sparsity (values ignored)
     */
    SolveResult solve(
        const ResidualFunc& residual_func,
        const JacobianFunc& jacobian_func,
        const SparseMatrix& pattern,
        Vector& x
    );
    
    // Configuration access
    NewtonConfig& config() { return config_; }
    const NewtonConfig& config() const { return config_; }
    
    // Linear solver access
    void set_linear_solver(UniquePtr<LinearSolverBase> solver);
    LinearSolverBase& linear_solver() { return *linear_solver_; }
    
    // Statistics from last solve
    Index iterations() const { return last_iterations_; }
    Real final_residual() const { return last_residual_; }
    const std::vector<Real>& residual_history() const { return residual_history_; }
    
    // ========================================================================
    // Enzyme AD Integration
    // ========================================================================
    
    /**
     * @brief Compute gradients using implicit differentiation
     *
     * For F(x_star(theta), theta) = 0, computes dx_star/dtheta using:
     *   (dF/dx)(dx_star/dtheta) = -(dF/dtheta)
     *
     * Or for adjoint (more efficient for many parameters):
     *   (dF/dx)^T lambda = dL/dx
     *   dL/dtheta = -lambda^T (dF/dtheta)
     *
     * @param x Solution point
     * @param adjoint_output dL/dx (adjoint seed from loss)
     * @param dF_dtheta Function computing dF/dtheta
     * @param[out] param_gradients Output: dL/dtheta
     */
    void compute_implicit_gradients(
        const Vector& x,
        const Vector& adjoint_output,
        const JacobianFunc& jacobian_func,
        const std::function<void(const Vector&, Matrix&)>& dF_dtheta,
        Vector& param_gradients
    );
    
#ifdef DGW_HAS_ENZYME
    /**
     * @brief Compute Jacobian using Enzyme
     * 
     * @param residual_func Residual function (will be differentiated)
     * @param x Current point
     * @param pattern Sparsity pattern
     * @param[out] jacobian Output Jacobian
     */
    void compute_jacobian_enzyme(
        const ResidualFunc& residual_func,
        const Vector& x,
        const SparseMatrix& pattern,
        SparseMatrix& jacobian
    );
#endif
    
private:
    NewtonConfig config_;
    UniquePtr<LinearSolverBase> linear_solver_;
    
    // State
    Index last_iterations_ = 0;
    Real last_residual_ = 0.0;
    std::vector<Real> residual_history_;
    
    // Work vectors
    Vector residual_;
    Vector delta_x_;
    SparseMatrix jacobian_;
    bool pattern_pre_analyzed_ = false;
    
    // Line search
    Real line_search(
        const ResidualFunc& residual_func,
        const Vector& x,
        const Vector& direction,
        const Vector& current_residual
    );
    
    // Trust region
    Real trust_region_step(
        const ResidualFunc& residual_func,
        const JacobianFunc& jacobian_func,
        Vector& x,
        Real trust_radius
    );
    
    // Finite difference Jacobian
    void compute_jacobian_fd(
        const ResidualFunc& residual_func,
        const Vector& x,
        SparseMatrix& jacobian
    );
    
    // Convergence check
    bool check_convergence(const Vector& residual, const Vector& delta_x);
};

// ============================================================================
// Picard (Fixed Point) Iteration
// ============================================================================

/**
 * @brief Picard iteration solver (simpler than Newton, may converge slower)
 * 
 * Solves A(x)*x = b(x) by iterating:
 *   A(x_k) * x_{k+1} = b(x_k)
 */
class PicardSolver {
public:
    PicardSolver() = default;
    explicit PicardSolver(const SolverConfig& config);
    
    /**
     * @brief Solve using Picard iteration
     * 
     * @param system_func Function computing (A, b) from current x
     * @param x Initial guess, updated to solution
     */
    SolveResult solve(
        const std::function<void(const Vector&, SparseMatrix&, Vector&)>& system_func,
        Vector& x
    );
    
    // Configuration
    Index max_iterations = 100;
    Real tolerance = 1e-6;
    Real relaxation = 1.0;  // Under-relaxation (0 < ω ≤ 1)
    bool verbose = false;
    
private:
    UniquePtr<LinearSolverBase> linear_solver_;
};

// ============================================================================
// Hybrid Newton-Picard
// ============================================================================

/**
 * @brief Hybrid solver: Picard for initial iterations, Newton near solution
 * 
 * Picard is more robust far from solution, Newton converges faster near solution.
 */
class HybridSolver {
public:
    HybridSolver() = default;
    
    SolveResult solve(
        const ResidualFunc& residual_func,
        const JacobianFunc& jacobian_func,
        const std::function<void(const Vector&, SparseMatrix&, Vector&)>& picard_system,
        Vector& x
    );
    
    // Switch from Picard to Newton when residual < threshold
    Real switch_threshold = 1e-2;
    Index max_picard_iterations = 10;
    
private:
    PicardSolver picard_;
    NewtonSolver newton_;
};

} // namespace dgw
