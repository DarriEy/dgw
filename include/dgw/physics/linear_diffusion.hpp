/**
 * @file linear_diffusion.hpp
 * @brief Linear diffusion equation solver (simplest groundwater model)
 *
 * Solves: Sy * dh/dt = T * nabla^2(h) + R
 *
 * This is the simplest case: constant transmissivity T = K * b,
 * making the equation linear. Suitable for:
 * - Preliminary analysis
 * - Quick screening models
 * - Educational purposes
 */

#pragma once

#include "physics_base.hpp"

namespace dgw {

class LinearDiffusion : public PhysicsBase {
public:
    LinearDiffusion() = default;
    explicit LinearDiffusion(const PhysicsDecisions& decisions);

    GoverningEquation type() const override { return GoverningEquation::LinearDiffusion; }
    std::string name() const override { return "LinearDiffusion"; }

    void compute_residual(
        const State& state, const Parameters& params,
        const Mesh& mesh, Real dt, Vector& residual
    ) const override;

    void compute_jacobian(
        const State& state, const Parameters& params,
        const Mesh& mesh, Real dt, SparseMatrix& jacobian
    ) const override;

    void compute_fluxes(
        const State& state, const Parameters& params,
        const Mesh& mesh, Vector& face_fluxes
    ) const override;

    void initialize_state(
        const Mesh& mesh, const Parameters& params,
        const Config& config, State& state
    ) const override;

    SparseMatrix allocate_jacobian(const Mesh& mesh) const override;

    void set_recharge(const Vector& recharge_rate) override;
    void set_stream_stage(const Vector& stream_stage) override;
    void set_pumping(const Vector& pumping) override;

    void apply_boundary_conditions(
        const Mesh& mesh, const Parameters& params,
        Vector& residual, SparseMatrix& jacobian
    ) const override;

    Vector water_table_depth(const State& state, const Mesh& mesh) const override;
    Vector stream_exchange(const State& state, const Parameters& params, const Mesh& mesh) const override;
    Real total_storage(const State& state, const Parameters& params, const Mesh& mesh) const override;
    Real mass_balance_error(const State& state, const Parameters& params, const Mesh& mesh, Real dt) const override;

private:
    PhysicsDecisions decisions_;
};

} // namespace dgw
