/**
 * @file time_stepping.hpp
 * @brief Adaptive time stepping controller
 */

#pragma once

#include "../core/types.hpp"
#include "../core/config.hpp"
#include <algorithm>
#include <cmath>

namespace dgw {

/**
 * @brief Adaptive time stepping controller
 *
 * Controls time step size based on Newton solver convergence behavior.
 */
class TimeStepper {
public:
    TimeStepper() = default;

    explicit TimeStepper(const SolverConfig& config)
        : dt_(config.dt_initial)
        , dt_min_(config.dt_min)
        , dt_max_(config.dt_max)
        , growth_factor_(config.dt_growth_factor)
        , reduction_factor_(config.dt_reduction_factor)
        , method_(config.time_stepping)
        , error_tolerance_(config.error_tolerance)
    {}

    /// Get current time step
    Real current_dt() const { return dt_; }

    /// Set current time step
    void set_dt(Real dt) { dt_ = std::clamp(dt, dt_min_, dt_max_); }

    /// Compute next time step after successful solve
    Real compute_next_dt(Index newton_iters, Index max_iters, Real mass_balance_error) {
        Real ratio = 1.0;

        switch (method_) {
            case TimeSteppingMethod::BackwardEuler:
            case TimeSteppingMethod::CrankNicolson:
            case TimeSteppingMethod::BDF2:
                // Heuristic based on Newton iterations
                if (newton_iters <= 3) {
                    ratio = growth_factor_;
                } else if (newton_iters <= max_iters / 2) {
                    ratio = 1.0;
                } else {
                    ratio = std::max(0.5, 1.0 - 0.1 * (newton_iters - max_iters / 2));
                }
                break;

            case TimeSteppingMethod::Adaptive: {
                // Error-controlled adaptive stepping
                if (mass_balance_error < error_tolerance_ * 0.1) {
                    ratio = growth_factor_;
                } else if (mass_balance_error < error_tolerance_) {
                    // Scale by safety factor * (tol/error)^(1/p)
                    Real safety = 0.9;
                    Real order = 1.0;  // First-order for backward Euler
                    ratio = safety * std::pow(error_tolerance_ / (mass_balance_error + 1e-30), 1.0 / order);
                    ratio = std::min(ratio, growth_factor_);
                } else {
                    ratio = reduction_factor_;
                }
                break;
            }
        }

        dt_ = std::clamp(dt_ * ratio, dt_min_, dt_max_);
        return dt_;
    }

    /// Reduce time step after failed solve
    Real reduce_dt() {
        dt_ = std::max(dt_ * reduction_factor_, dt_min_);
        return dt_;
    }

    /// Adjust dt to hit target time exactly
    Real adjust_for_target(Real current_time, Real target_time) {
        Real remaining = target_time - current_time;
        if (remaining <= 0.0) return 0.0;

        if (dt_ > remaining) {
            return remaining;  // Don't overshoot
        }

        // If we're close to the target, adjust to hit it
        if (remaining < 1.5 * dt_) {
            return remaining;
        }

        return dt_;
    }

    /// Check if we should output at this step
    bool should_output(Real current_time, Real output_interval) const {
        if (output_interval <= 0.0) return false;
        Real n = std::floor(current_time / output_interval);
        Real next_output = (n + 1.0) * output_interval;
        return (current_time + dt_ >= next_output - 1e-10);
    }

    /// Get time stepping method
    TimeSteppingMethod method() const { return method_; }

    /// Get implicit weight (theta) for time discretization
    Real implicit_weight() const {
        switch (method_) {
            case TimeSteppingMethod::BackwardEuler:
                return 1.0;  // Fully implicit
            case TimeSteppingMethod::CrankNicolson:
                return 0.5;  // Central
            case TimeSteppingMethod::BDF2:
                // BDF2 requires different time discretization:
                // (3*y_{n+1} - 4*y_n + y_{n-1}) / (2*dt) = f(y_{n+1})
                // The implicit_weight is not sufficient to express BDF2.
                // The physics modules must check for BDF2 and use the correct formula.
                // Returning 1.0 here as a fallback (backward Euler behavior).
                return 1.0;
            case TimeSteppingMethod::Adaptive:
                return 1.0;  // Default to backward Euler
            default:
                return 1.0;
        }
    }

private:
    Real dt_ = 3600.0;
    Real dt_min_ = 60.0;
    Real dt_max_ = 86400.0;
    Real growth_factor_ = 1.5;
    Real reduction_factor_ = 0.5;
    TimeSteppingMethod method_ = TimeSteppingMethod::BackwardEuler;
    Real error_tolerance_ = 0.01;
};

} // namespace dgw
