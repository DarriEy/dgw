/**
 * @file boussinesq.hpp
 * @brief Boussinesq equation solver for unconfined aquifers
 * 
 * Solves the 2D nonlinear Boussinesq equation:
 * 
 *   Sy * ∂h/∂t = ∇·(K * (h - z_bot) * ∇h) + R - Q_stream - Q_pump
 * 
 * where:
 *   h = hydraulic head [m]
 *   Sy = specific yield [-]
 *   K = hydraulic conductivity [m/s]
 *   z_bot = aquifer bottom elevation [m]
 *   R = recharge rate [m/s]
 *   Q_stream = stream-aquifer exchange [m/s]
 *   Q_pump = pumping [m/s]
 * 
 * The nonlinearity arises from transmissivity T = K*(h - z_bot).
 */

#pragma once

#include "physics_base.hpp"

namespace dgw {

/**
 * @brief Boussinesq equation solver
 */
class BoussinesqSolver : public PhysicsBase {
public:
    BoussinesqSolver() = default;
    explicit BoussinesqSolver(const PhysicsDecisions& decisions);
    
    GoverningEquation type() const override { return GoverningEquation::Boussinesq; }
    std::string name() const override { return "Boussinesq"; }
    
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
    
    // Boussinesq-specific methods
    
    /**
     * @brief Compute transmissivity T = K * saturated_thickness
     * 
     * @param h Head at cell
     * @param K Conductivity at cell
     * @param z_bot Bottom elevation at cell
     * @param method Transmissivity method
     * @return Transmissivity [m²/s]
     */
    static Real compute_transmissivity(
        Real h, Real K, Real z_bot,
        TransmissivityMethod method = TransmissivityMethod::Standard
    );
    
    /**
     * @brief Compute inter-cell transmissivity
     * 
     * Uses harmonic mean of cell transmissivities, weighted upstream
     * for stability in steep gradients.
     */
    static Real intercell_transmissivity(
        Real T_i, Real T_j, Real h_i, Real h_j,
        TransmissivityMethod method = TransmissivityMethod::Standard
    );
    
    /**
     * @brief Compute ∂T/∂h for Jacobian
     */
    static Real transmissivity_derivative(
        Real h, Real K, Real z_bot,
        TransmissivityMethod method = TransmissivityMethod::Standard
    );
    
private:
    PhysicsDecisions decisions_;
    
    // Smoothing parameters for transmissivity
    Real smoothing_eps_ = 0.01;  // Smoothing zone [m]
    
    // Compute Darcy flux at face (internal helper)
    Real compute_face_flux(
        Index face_id,
        const Vector& head,
        const Parameters2D& params,
        const Mesh& mesh
    ) const;
    
    // Compute contribution to Jacobian from face flux
    void add_face_jacobian_contribution(
        Index face_id,
        const Vector& head,
        const Parameters2D& params,
        const Mesh& mesh,
        std::vector<SparseTriplet>& triplets
    ) const;
};

// ============================================================================
// Enzyme-compatible standalone functions for AD
// ============================================================================

namespace boussinesq_kernels {

/**
 * @brief Compute residual (Enzyme-friendly pure function)
 * 
 * This function is designed for efficient AD through Enzyme.
 * No hidden state, all inputs/outputs explicit.
 */
void residual_kernel(
    const Real* __restrict__ head,
    const Real* __restrict__ head_old,
    const Real* __restrict__ K,
    const Real* __restrict__ Sy,
    const Real* __restrict__ z_surface,
    const Real* __restrict__ z_bottom,
    const Real* __restrict__ recharge,
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
 * @brief Compute Jacobian entries (Enzyme-friendly)
 */
void jacobian_kernel(
    const Real* __restrict__ head,
    const Real* __restrict__ K,
    const Real* __restrict__ Sy,
    const Real* __restrict__ z_bottom,
    const Index* __restrict__ cell_neighbors_data,
    const Index* __restrict__ cell_neighbors_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Real* __restrict__ cell_area,
    Index n_cells,
    Real dt,
    Real* __restrict__ jacobian_data,
    Index* __restrict__ jacobian_row,
    Index* __restrict__ jacobian_col
);

/**
 * @brief Smooth transmissivity function for numerical stability
 * 
 * Uses smooth approximation near zero:
 *   T = K * smooth_max(h - z_bot, 0)
 * 
 * where smooth_max uses a polynomial transition.
 */
inline Real smooth_transmissivity(Real h, Real K, Real z_bot, Real eps = 0.01) {
    Real b = h - z_bot;  // Saturated thickness
    if (b > eps) {
        return K * b;
    } else if (b < 0) {
        return 0.0;
    } else {
        // C1 cubic Hermite: f(0)=0, f'(0)=0, f(eps)=eps, f'(eps)=1
        // f(t) = eps * t^2 * (2 - t) where t = b/eps
        Real t = b / eps;
        Real smooth_b = eps * t * t * (2.0 - t);
        return K * smooth_b;
    }
}

/**
 * @brief Derivative of smooth transmissivity w.r.t. head
 */
inline Real smooth_transmissivity_dh(Real h, Real K, Real z_bot, Real eps = 0.01) {
    Real b = h - z_bot;
    if (b > eps) {
        return K;
    } else if (b < 0) {
        return 0.0;
    } else {
        // Derivative of eps * t^2 * (2-t) w.r.t. h = t*(4-3t) where t=b/eps
        Real t = b / eps;
        return K * t * (4.0 - 3.0 * t);
    }
}

/**
 * @brief Harmonic mean for inter-cell conductivity
 */
inline Real harmonic_mean(Real a, Real b) {
    if (a <= 0.0 || b <= 0.0) return 0.0;
    return 2.0 * a * b / (a + b);
}

/**
 * @brief Upstream-weighted inter-cell transmissivity
 * 
 * Provides numerical stability for steep gradients.
 */
inline Real upstream_transmissivity(Real T_i, Real T_j, Real h_i, Real h_j) {
    // Weight toward upstream cell (higher head)
    if (h_i > h_j) {
        return 0.7 * T_i + 0.3 * T_j;  // Weighted toward i
    } else {
        return 0.3 * T_i + 0.7 * T_j;  // Weighted toward j
    }
}

} // namespace boussinesq_kernels

} // namespace dgw
