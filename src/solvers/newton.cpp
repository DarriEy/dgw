/**
 * @file newton.cpp
 * @brief Newton-Raphson solver implementation
 */

#include "dgw/solvers/newton.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <chrono>

namespace dgw {

// ============================================================================
// NewtonSolver
// ============================================================================

NewtonSolver::NewtonSolver() {
    linear_solver_ = std::make_unique<EigenLUSolver>();
}

NewtonSolver::NewtonSolver(const NewtonConfig& config)
    : config_(config) {
    linear_solver_ = std::make_unique<EigenLUSolver>();
}

NewtonSolver::NewtonSolver(const SolverConfig& config) {
    config_.max_iterations = config.max_newton_iterations;
    config_.tolerance = config.newton_tolerance;
    config_.initial_relaxation = config.newton_relaxation;
    config_.use_line_search = config.use_line_search;

    switch (config.nonlinear_solver) {
        case NonlinearSolver::Newton:
            config_.use_line_search = false;
            config_.use_trust_region = false;
            break;
        case NonlinearSolver::NewtonLineSearch:
            config_.use_line_search = true;
            break;
        case NonlinearSolver::TrustRegion:
            config_.use_trust_region = true;
            config_.use_line_search = false;
            break;
        default:
            break;
    }

    linear_solver_ = LinearSolverBase::create(config.linear_solver);
}

SolveResult NewtonSolver::solve(
    const ResidualFunc& residual_func,
    const JacobianFunc& jacobian_func,
    Vector& x
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    const Index n = x.size();

    residual_.resize(n);
    delta_x_.resize(n);
    residual_history_.clear();

    // Initial residual
    residual_func(x, residual_);
    Real norm0 = residual_.norm();
    Real norm = norm0;
    residual_history_.push_back(norm);

    if (config_.verbose) {
        std::cerr << "Newton iter 0: ||F|| = " << norm << "\n";
    }

    if (norm < config_.tolerance) {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_time).count();
        last_iterations_ = 0;
        last_residual_ = norm;
        return {true, 0, norm, ms, "Already converged"};
    }

    bool pattern_analyzed = pattern_pre_analyzed_;
    pattern_pre_analyzed_ = false;  // Reset for next call

    for (Index iter = 0; iter < config_.max_iterations; ++iter) {
        // Compute Jacobian
        jacobian_func(x, jacobian_);

        // Analyze sparsity pattern on first iteration
        if (!pattern_analyzed) {
            linear_solver_->analyze_pattern(jacobian_);
            pattern_analyzed = true;
        }

        // Factorize
        linear_solver_->factorize(jacobian_);
        if (!linear_solver_->success()) {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start_time).count();
            last_iterations_ = iter + 1;
            last_residual_ = norm;
            return {false, iter + 1, norm, ms, "Linear solver factorization failed"};
        }

        // Solve J * delta_x = -F
        Vector neg_residual = -residual_;
        linear_solver_->solve(neg_residual, delta_x_);

        // Apply update with line search or damping
        Real alpha = config_.initial_relaxation;

        if (config_.use_line_search) {
            alpha = line_search(residual_func, x, delta_x_, residual_);
        }

        x += alpha * delta_x_;

        // Compute new residual
        residual_func(x, residual_);
        norm = residual_.norm();
        residual_history_.push_back(norm);

        if (config_.verbose) {
            std::cerr << "Newton iter " << (iter + 1) << ": ||F|| = " << norm
                      << " (alpha=" << alpha << ")\n";
        }

        if (config_.callback) {
            if (!config_.callback(iter + 1, norm)) {
                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end - start_time).count();
                last_iterations_ = iter + 1;
                last_residual_ = norm;
                return {false, iter + 1, norm, ms, "Cancelled by callback"};
            }
        }

        // Check convergence
        if (check_convergence(residual_, delta_x_)) {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start_time).count();
            last_iterations_ = iter + 1;
            last_residual_ = norm;
            return {true, iter + 1, norm, ms, "Converged"};
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start_time).count();
    last_iterations_ = config_.max_iterations;
    last_residual_ = norm;
    return {false, config_.max_iterations, norm, ms, "Maximum iterations reached"};
}

SolveResult NewtonSolver::solve(
    const ResidualFunc& residual_func,
    Vector& x
) {
    // Use finite difference Jacobian
    JacobianFunc fd_jacobian = [this, &residual_func](const Vector& x_val, SparseMatrix& jac) {
        compute_jacobian_fd(residual_func, x_val, jac);
    };
    return solve(residual_func, fd_jacobian, x);
}

SolveResult NewtonSolver::solve(
    const ResidualFunc& residual_func,
    const JacobianFunc& jacobian_func,
    const SparseMatrix& pattern,
    Vector& x
) {
    // Pre-analyze pattern and mark as analyzed
    jacobian_ = pattern;
    linear_solver_->analyze_pattern(jacobian_);
    pattern_pre_analyzed_ = true;
    return solve(residual_func, jacobian_func, x);
}

void NewtonSolver::set_linear_solver(UniquePtr<LinearSolverBase> solver) {
    linear_solver_ = std::move(solver);
}

void NewtonSolver::compute_implicit_gradients(
    const Vector& x,
    const Vector& adjoint_output,
    const JacobianFunc& jacobian_func,
    const std::function<void(const Vector&, Matrix&)>& dF_dtheta,
    Vector& param_gradients
) {
    const Index n = x.size();

    // Compute Jacobian at solution point
    SparseMatrix J;
    jacobian_func(x, J);

    // Solve adjoint system: J^T * lambda = adjoint_output
    // (Using transpose)
    SparseMatrix JT = J.transpose();
    linear_solver_->analyze_pattern(JT);
    linear_solver_->factorize(JT);

    Vector lambda(n);
    linear_solver_->solve(adjoint_output, lambda);

    // Compute dF/dtheta
    Matrix dF_dp;
    dF_dtheta(x, dF_dp);

    // param_gradients = -lambda^T * dF/dtheta
    param_gradients = -(dF_dp.transpose() * lambda);
}

Real NewtonSolver::line_search(
    const ResidualFunc& residual_func,
    const Vector& x,
    const Vector& direction,
    const Vector& current_residual
) {
    Real alpha = 1.0;
    Real f0 = 0.5 * current_residual.squaredNorm();
    Real slope = -2.0 * f0;  // Approximate directional derivative

    Vector x_trial = x;
    Vector r_trial(x.size());

    for (Index k = 0; k < config_.max_line_search_iters; ++k) {
        x_trial = x + alpha * direction;
        residual_func(x_trial, r_trial);
        Real f_trial = 0.5 * r_trial.squaredNorm();

        // Armijo condition
        if (f_trial <= f0 + config_.line_search_alpha * alpha * slope) {
            return alpha;
        }

        alpha *= config_.line_search_beta;

        if (alpha < config_.min_relaxation) {
            return config_.min_relaxation;
        }
    }

    return alpha;
}

Real NewtonSolver::trust_region_step(
    const ResidualFunc& residual_func,
    const JacobianFunc& jacobian_func,
    Vector& x,
    Real trust_radius
) {
    // Simple dogleg trust region
    // For now, just clamp the Newton step to trust radius
    if (delta_x_.norm() > trust_radius) {
        delta_x_ *= trust_radius / delta_x_.norm();
    }
    return delta_x_.norm();
}

void NewtonSolver::compute_jacobian_fd(
    const ResidualFunc& residual_func,
    const Vector& x,
    SparseMatrix& jacobian
) {
    const Index n = x.size();
    Vector F0(n), F_pert(n);
    residual_func(x, F0);

    std::vector<SparseTriplet> triplets;
    triplets.reserve(n * 7);

    Vector x_pert = x;
    for (Index j = 0; j < n; ++j) {
        Real eps = config_.fd_epsilon * std::max(std::abs(x(j)), 1.0);
        x_pert(j) += eps;
        residual_func(x_pert, F_pert);
        x_pert(j) = x(j);

        Vector col = (F_pert - F0) / eps;
        for (Index i = 0; i < n; ++i) {
            if (std::abs(col(i)) > 1e-15) {
                triplets.emplace_back(i, j, col(i));
            }
        }
    }

    jacobian.resize(n, n);
    jacobian.setFromTriplets(triplets.begin(), triplets.end());
}

bool NewtonSolver::check_convergence(const Vector& residual, const Vector& delta_x) {
    Real abs_norm = residual.norm();
    // Relative convergence: step size relative to solution magnitude
    Real x_scale = std::max(delta_x.norm(), 1e-30);
    Real rel_norm = delta_x.norm() / (residual.norm() + x_scale + 1e-30);

    return abs_norm < config_.tolerance ||
           rel_norm < config_.relative_tolerance;
}

// ============================================================================
// PicardSolver
// ============================================================================

PicardSolver::PicardSolver(const SolverConfig& config) {
    max_iterations = config.max_newton_iterations;
    tolerance = config.newton_tolerance;
    linear_solver_ = LinearSolverBase::create(config.linear_solver);
}

SolveResult PicardSolver::solve(
    const std::function<void(const Vector&, SparseMatrix&, Vector&)>& system_func,
    Vector& x
) {
    const Index n = x.size();
    SparseMatrix A;
    Vector b(n);
    Vector x_new(n);

    linear_solver_ = std::make_unique<EigenLUSolver>();

    bool pattern_analyzed = false;

    for (Index iter = 0; iter < max_iterations; ++iter) {
        system_func(x, A, b);

        if (!pattern_analyzed) {
            linear_solver_->analyze_pattern(A);
            pattern_analyzed = true;
        }

        linear_solver_->factorize(A);
        if (!linear_solver_->success()) {
            return {false, iter + 1, 0.0, 0.0, "Picard: linear solver failed"};
        }

        linear_solver_->solve(b, x_new);

        // Under-relaxation
        Real diff = (x_new - x).norm();
        x = (1.0 - relaxation) * x + relaxation * x_new;

        if (diff < tolerance) {
            return {true, iter + 1, diff, 0.0, "Picard converged"};
        }
    }

    return {false, max_iterations, 0.0, 0.0, "Picard: max iterations"};
}

// ============================================================================
// HybridSolver
// ============================================================================

SolveResult HybridSolver::solve(
    const ResidualFunc& residual_func,
    const JacobianFunc& jacobian_func,
    const std::function<void(const Vector&, SparseMatrix&, Vector&)>& picard_system,
    Vector& x
) {
    // Start with Picard
    Vector residual(x.size());
    residual_func(x, residual);

    if (residual.norm() > switch_threshold) {
        picard_.max_iterations = max_picard_iterations;
        auto result = picard_.solve(picard_system, x);
        if (result.converged) return result;
    }

    // Switch to Newton
    return newton_.solve(residual_func, jacobian_func, x);
}

} // namespace dgw
