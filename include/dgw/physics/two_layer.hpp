/**
 * @file two_layer.hpp
 * @brief Two-layer groundwater solver (unconfined + confined)
 * 
 * Solves coupled system based on PCR-GLOBWB/GLOBGM architecture:
 * 
 * Layer 1 (Unconfined - Boussinesq):
 *   Sy * ∂h₁/∂t = ∇·(K₁(h₁-z_bot1)∇h₁) + R - Q_river - Q_leak
 * 
 * Layer 2 (Confined - Linear):
 *   Ss·b₂ * ∂h₂/∂t = ∇·(T₂∇h₂) + Q_leak - Q_pump
 * 
 * Leakage between layers:
 *   Q_leak = K_conf/b_conf * (h₁ - h₂)
 * 
 * This is appropriate for global-scale modeling where:
 * - Top layer responds to climate (recharge, rivers)
 * - Bottom layer provides long-term storage, deep pumping
 * - Confining layer controls vertical exchange
 */

#pragma once

#include "physics_base.hpp"
#include "boussinesq.hpp"
#include "confined.hpp"

namespace dgw {

/**
 * @brief Two-layer coupled groundwater solver
 */
class TwoLayerSolver : public PhysicsBase {
public:
    TwoLayerSolver() = default;
    explicit TwoLayerSolver(const PhysicsDecisions& decisions);
    
    GoverningEquation type() const override { return GoverningEquation::TwoLayer; }
    std::string name() const override { return "TwoLayer"; }
    
    // Core computation
    void compute_residual(
        const State& state,
        const Parameters& params,
        const Mesh& mesh,
        Real dt,
        Vector& residual
    ) const override;
    
    void compute_jacobian(
        const State& state,
        const Parameters& params,
        const Mesh& mesh,
        Real dt,
        SparseMatrix& jacobian
    ) const override;
    
    void compute_fluxes(
        const State& state,
        const Parameters& params,
        const Mesh& mesh,
        Vector& face_fluxes
    ) const override;
    
    // Initialization
    void initialize_state(
        const Mesh& mesh,
        const Parameters& params,
        const Config& config,
        State& state
    ) const override;
    
    SparseMatrix allocate_jacobian(const Mesh& mesh) const override;
    
    // Source terms
    void set_recharge(const Vector& recharge_rate) override;
    void set_stream_stage(const Vector& stream_stage) override;
    void set_pumping(const Vector& pumping) override;
    
    /**
     * @brief Set pumping for specific layer
     * @param layer 0 = unconfined, 1 = confined
     * @param pumping Pumping rate [m³/s]
     */
    void set_pumping_layer(Index layer, const Vector& pumping);
    
    void apply_boundary_conditions(
        const Mesh& mesh,
        const Parameters& params,
        Vector& residual,
        SparseMatrix& jacobian
    ) const override;
    
    // Outputs
    Vector water_table_depth(
        const State& state,
        const Mesh& mesh
    ) const override;
    
    /**
     * @brief Get confined layer head
     */
    Vector confined_head(const State& state) const;
    
    /**
     * @brief Get leakage between layers [m³/s]
     * Positive = downward (layer 1 → layer 2)
     */
    Vector leakage(
        const State& state,
        const Parameters& params,
        const Mesh& mesh
    ) const;
    
    Vector stream_exchange(
        const State& state,
        const Parameters& params,
        const Mesh& mesh
    ) const override;
    
    Real total_storage(
        const State& state,
        const Parameters& params,
        const Mesh& mesh
    ) const override;
    
    /**
     * @brief Storage by layer
     */
    std::pair<Real, Real> storage_by_layer(
        const State& state,
        const Parameters& params,
        const Mesh& mesh
    ) const;
    
    Real mass_balance_error(
        const State& state,
        const Parameters& params,
        const Mesh& mesh,
        Real dt
    ) const override;
    
    // Two-layer specific options
    
    /**
     * @brief Set coupling method
     * 
     * @param monolithic If true, solve both layers simultaneously.
     *                   If false, iterate between layers (may need multiple iterations).
     */
    void set_monolithic(bool monolithic) { monolithic_ = monolithic; }
    
    /**
     * @brief Set coupling iteration parameters
     */
    void set_coupling_options(Index max_iters, Real tolerance) {
        max_coupling_iters_ = max_iters;
        coupling_tolerance_ = tolerance;
    }
    
private:
    PhysicsDecisions decisions_;
    
    // Sub-solvers for each layer (used in sequential mode)
    UniquePtr<BoussinesqSolver> layer1_solver_;
    UniquePtr<ConfinedSolver> layer2_solver_;
    
    // Coupling options
    bool monolithic_ = true;          // Solve simultaneously
    Index max_coupling_iters_ = 10;   // For sequential coupling
    Real coupling_tolerance_ = 1e-4;
    
    // Pumping by layer
    Vector pumping_layer1_;
    Vector pumping_layer2_;
    
    // Helper: compute leakance L = K_conf / b_conf
    Vector compute_leakance(const ParametersTwoLayer& params) const;
    
    // Helper: compute leakage flux
    Vector compute_leakage(
        const Vector& h1,
        const Vector& h2,
        const Vector& leakance,
        const Mesh& mesh
    ) const;
    
    // Monolithic system assembly
    void assemble_monolithic_residual(
        const StateTwoLayer& state,
        const ParametersTwoLayer& params,
        const Mesh& mesh,
        Real dt,
        Vector& residual
    ) const;
    
    void assemble_monolithic_jacobian(
        const StateTwoLayer& state,
        const ParametersTwoLayer& params,
        const Mesh& mesh,
        Real dt,
        SparseMatrix& jacobian
    ) const;
    
    // Sequential coupling iteration
    SolveResult solve_sequential(
        StateTwoLayer& state,
        const ParametersTwoLayer& params,
        const Mesh& mesh,
        Real dt
    ) const;
};

// ============================================================================
// Enzyme-compatible kernels for two-layer system
// ============================================================================

namespace two_layer_kernels {

/**
 * @brief Compute leakage between layers
 * 
 * Q_leak = leakance * (h1 - h2) * cell_area
 */
inline void compute_leakage(
    const Real* __restrict__ h1,
    const Real* __restrict__ h2,
    const Real* __restrict__ leakance,
    const Real* __restrict__ cell_area,
    Index n_cells,
    Real* __restrict__ leakage_out
) {
    for (Index i = 0; i < n_cells; ++i) {
        leakage_out[i] = leakance[i] * (h1[i] - h2[i]) * cell_area[i];
    }
}

/**
 * @brief Monolithic residual for two-layer system
 * 
 * State vector layout: [h1_0, h1_1, ..., h1_n, h2_0, h2_1, ..., h2_n]
 * Residual layout:     [F1_0, F1_1, ..., F1_n, F2_0, F2_1, ..., F2_n]
 */
void monolithic_residual(
    // State
    const Real* __restrict__ h,             // [2*n_cells]: h1 then h2
    const Real* __restrict__ h_old,
    // Layer 1 parameters
    const Real* __restrict__ K1,
    const Real* __restrict__ Sy,
    const Real* __restrict__ z_surface,
    const Real* __restrict__ z_bottom1,
    // Layer 2 parameters  
    const Real* __restrict__ T2,            // Transmissivity (constant)
    const Real* __restrict__ S2,            // Storage coefficient
    // Leakage
    const Real* __restrict__ leakance,
    // Source terms
    const Real* __restrict__ recharge,
    const Real* __restrict__ stream_exchange,
    const Real* __restrict__ pumping1,
    const Real* __restrict__ pumping2,
    // Mesh
    const Index* __restrict__ cell_neighbors,
    const Index* __restrict__ cell_neighbors_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Real* __restrict__ cell_area,
    Index n_cells,
    Real dt,
    // Output
    Real* __restrict__ residual            // [2*n_cells]
);

/**
 * @brief Monolithic Jacobian for two-layer system
 * 
 * Block structure:
 *   [J11  J12]   where J11 = ∂F1/∂h1, J12 = ∂F1/∂h2
 *   [J21  J22]         J21 = ∂F2/∂h1, J22 = ∂F2/∂h2
 * 
 * J12 and J21 only have diagonal entries (leakage coupling).
 */
void monolithic_jacobian(
    // State
    const Real* __restrict__ h,
    // Layer 1 parameters
    const Real* __restrict__ K1,
    const Real* __restrict__ Sy,
    const Real* __restrict__ z_bottom1,
    // Layer 2 parameters
    const Real* __restrict__ T2,
    const Real* __restrict__ S2,
    // Leakage
    const Real* __restrict__ leakance,
    // Mesh
    const Index* __restrict__ cell_neighbors,
    const Index* __restrict__ cell_neighbors_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Real* __restrict__ cell_area,
    Index n_cells,
    Real dt,
    // Output (triplet format)
    Real* __restrict__ jac_values,
    Index* __restrict__ jac_rows,
    Index* __restrict__ jac_cols,
    Index& nnz
);

/**
 * @brief Convert monolithic state to layer states
 */
inline void unpack_state(
    const Real* __restrict__ h_mono,
    Index n_cells,
    Real* __restrict__ h1,
    Real* __restrict__ h2
) {
    for (Index i = 0; i < n_cells; ++i) {
        h1[i] = h_mono[i];
        h2[i] = h_mono[n_cells + i];
    }
}

/**
 * @brief Convert layer states to monolithic
 */
inline void pack_state(
    const Real* __restrict__ h1,
    const Real* __restrict__ h2,
    Index n_cells,
    Real* __restrict__ h_mono
) {
    for (Index i = 0; i < n_cells; ++i) {
        h_mono[i] = h1[i];
        h_mono[n_cells + i] = h2[i];
    }
}

} // namespace two_layer_kernels

} // namespace dgw
