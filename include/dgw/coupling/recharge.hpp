/**
 * @file recharge.hpp
 * @brief Recharge handling and vadose zone delay
 */

#pragma once

#include "../core/types.hpp"
#include "../core/mesh.hpp"
#include "../core/parameters.hpp"
#include <cmath>
#include <algorithm>

namespace dgw {

/**
 * @brief Handles recharge from land surface with optional vadose zone delay
 */
class RechargeHandler {
public:
    RechargeHandler() = default;

    explicit RechargeHandler(VadoseMethod method) : method_(method) {}

    /// Set vadose zone method
    void set_method(VadoseMethod method) { method_ = method; }

    /// Set parameters
    void set_parameters(const RechargeParameters& params) { params_ = params; }

    /**
     * @brief Compute effective recharge at water table
     *
     * @param surface_recharge Recharge rate at surface [m/s]
     * @param water_table_depth Depth to water table [m]
     * @param mesh Mesh
     * @param dt Time step [s]
     * @return Effective recharge at water table [m/s]
     */
    Vector compute_recharge(
        const Vector& surface_recharge,
        const Vector& water_table_depth,
        const Mesh& mesh,
        Real dt
    ) {
        switch (method_) {
            case VadoseMethod::Direct:
                return compute_direct(surface_recharge);

            case VadoseMethod::ExponentialLag:
                return compute_exponential_lag(surface_recharge, water_table_depth, dt);

            case VadoseMethod::KinematicWave:
                return compute_kinematic_wave(surface_recharge, water_table_depth, dt);

            default:
                return compute_direct(surface_recharge);
        }
    }

    /// Get stored vadose zone water
    const Vector& vadose_storage() const { return vadose_storage_; }

private:
    VadoseMethod method_ = VadoseMethod::Direct;
    RechargeParameters params_;

    // Vadose zone storage (for lag methods)
    Vector vadose_storage_;
    Vector pending_recharge_;

    Vector compute_direct(const Vector& surface_recharge) {
        return surface_recharge;
    }

    Vector compute_exponential_lag(
        const Vector& surface_recharge,
        const Vector& water_table_depth,
        Real dt
    ) {
        const Index n = surface_recharge.size();

        // Initialize storage if needed
        if (vadose_storage_.size() != n) {
            vadose_storage_.resize(n);
            vadose_storage_.setZero();
        }

        Vector effective_recharge(n);

        for (Index i = 0; i < n; ++i) {
            // Compute lag time based on depth to water table
            Real lag_coef = (params_.lag_coefficient.size() > 0) ?
                params_.lag_coefficient(i) : 1000.0;  // Default: 1000 s/m
            Real tau = std::clamp(
                lag_coef * water_table_depth(i),
                params_.min_lag,
                params_.max_lag
            );

            // Exponential decay: dS/dt = R_surface - S/tau
            Real decay = std::exp(-dt / tau);
            vadose_storage_(i) = vadose_storage_(i) * decay
                               + surface_recharge(i) * tau * (1.0 - decay);

            // Effective recharge is drainage from vadose zone
            effective_recharge(i) = vadose_storage_(i) / tau;
        }

        return effective_recharge;
    }

    Vector compute_kinematic_wave(
        const Vector& surface_recharge,
        const Vector& water_table_depth,
        Real dt
    ) {
        const Index n = surface_recharge.size();

        if (vadose_storage_.size() != n) {
            vadose_storage_.resize(n);
            vadose_storage_.setZero();
        }

        Vector effective_recharge(n);

        for (Index i = 0; i < n; ++i) {
            Real K_unsat = (params_.unsat_K.size() > 0) ?
                params_.unsat_K(i) : 1e-6;
            Real theta = (params_.unsat_theta.size() > 0) ?
                params_.unsat_theta(i) : 0.3;
            Real depth = water_table_depth(i);

            // Kinematic wave velocity
            Real velocity = K_unsat / theta;

            // Travel time through vadose zone
            Real travel_time = (depth > 0.0) ? depth / velocity : 0.0;

            if (travel_time <= dt) {
                // Recharge arrives within this timestep
                effective_recharge(i) = surface_recharge(i);
            } else {
                // Partial arrival - simple linear delay
                // Fraction that arrives directly
                Real fraction = dt / travel_time;
                effective_recharge(i) = surface_recharge(i) * fraction;

                // Remainder goes to vadose storage
                vadose_storage_(i) += surface_recharge(i) * dt * (1.0 - fraction);

                // Drain existing vadose storage
                Real drain = std::min(vadose_storage_(i), vadose_storage_(i) * dt / travel_time);
                vadose_storage_(i) -= drain;
                effective_recharge(i) += drain / dt;
            }
        }

        return effective_recharge;
    }
};

} // namespace dgw
