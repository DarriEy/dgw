/**
 * @file confined.hpp
 * @brief Confined aquifer solver (linear diffusion equation)
 * 
 * Solves the 2D linear confined aquifer equation:
 * 
 *   S * ∂h/∂t = ∇·(T∇h) + Q
 * 
 * where:
 *   h = hydraulic head [m]
 *   S = storage coefficient = Ss * b [-]
 *   T = transmissivity = K * b [m²/s]
 *   b = aquifer thickness [m]
 *   Q = source/sink terms [m/s]
 * 
 * This is linear because T is constant (aquifer is fully saturated
 * with constant thickness), making it much simpler than Boussinesq.
 */

#pragma once

#include "physics_base.hpp"

namespace dgw {

/**
 * @brief Linear confined aquifer solver
 */
class ConfinedSolver : public PhysicsBase {
public:
    ConfinedSolver() = default;
    explicit ConfinedSolver(const PhysicsDecisions& decisions);
    
    GoverningEquation type() const override { return GoverningEquation::Confined; }
    std::string name() const override { return "Confined"; }
    
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
    
    void apply_boundary_conditions(
        const State& state,
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
     * @brief Get potentiometric surface (head relative to datum)
     */
    Vector potentiometric_surface(const State& state) const;
    
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
    
    Real mass_balance_error(
        const State& state,
        const Parameters& params,
        const Mesh& mesh,
        Real dt
    ) const override;
    
    /**
     * @brief Check if system matrix can be precomputed
     * 
     * For linear problems with constant transmissivity, the
     * Jacobian only depends on dt, not on state. This allows
     * factorization to be reused across Newton iterations.
     */
    bool is_linear() const { return true; }
    
    /**
     * @brief Precompute system matrix for given dt
     * 
     * A = S/dt * I - ∇·(T∇)
     */
    void precompute_system_matrix(
        const Parameters& params,
        const Mesh& mesh,
        Real dt,
        SparseMatrix& A
    ) const;
    
private:
    PhysicsDecisions decisions_;
    
    // For linear problems, we can cache the system matrix
    mutable bool matrix_cached_ = false;
    mutable Real cached_dt_ = 0.0;
    mutable SparseMatrix cached_matrix_;
};

// ============================================================================
// Enzyme-compatible kernels for confined flow
// ============================================================================

namespace confined_kernels {

/**
 * @brief Compute residual for confined flow (linear)
 */
void residual_kernel(
    const Real* __restrict__ head,
    const Real* __restrict__ head_old,
    const Real* __restrict__ T,             // Transmissivity [m²/s]
    const Real* __restrict__ S,             // Storage coefficient [-]
    const Real* __restrict__ source,        // Source term [m/s]
    const Index* __restrict__ cell_neighbors_data,
    const Index* __restrict__ cell_neighbors_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Real* __restrict__ cell_area,
    Index n_cells,
    Real dt,
    Real* __restrict__ residual
);

/**
 * @brief Assemble system matrix for confined flow
 * 
 * Since the system is linear, A is independent of h.
 */
void assemble_matrix(
    const Real* __restrict__ T,
    const Real* __restrict__ S,
    const Index* __restrict__ cell_neighbors_data,
    const Index* __restrict__ cell_neighbors_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Real* __restrict__ cell_area,
    Index n_cells,
    Real dt,
    Real* __restrict__ matrix_values,
    Index* __restrict__ matrix_rows,
    Index* __restrict__ matrix_cols,
    Index& nnz
);

/**
 * @brief Inter-cell transmissivity (harmonic mean)
 */
inline Real intercell_T(Real T_i, Real T_j) {
    return 2.0 * T_i * T_j / (T_i + T_j + 1e-30);
}

} // namespace confined_kernels

} // namespace dgw
