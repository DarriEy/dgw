/**
 * @file richards_3d.hpp
 * @brief 3D Richards equation solver for variably-saturated flow
 * 
 * Solves the 3D Richards equation:
 * 
 *   ∂θ/∂t = ∇·(K(ψ) ∇(ψ + z)) - S
 * 
 * Rewritten in mixed form (ψ-based):
 * 
 *   C(ψ) ∂ψ/∂t = ∇·(K(ψ) ∇(ψ + z)) - S
 * 
 * where:
 *   ψ = pressure head [m] (negative in unsaturated zone)
 *   θ = volumetric water content [-]
 *   K(ψ) = hydraulic conductivity [m/s]
 *   C(ψ) = specific moisture capacity = dθ/dψ [1/m]
 *   z = elevation [m]
 *   S = sink term (root water uptake, etc.) [1/s]
 * 
 * Constitutive relations:
 *   θ(ψ) = water retention curve (van Genuchten, Brooks-Corey, etc.)
 *   K(ψ) = K_sat * Kr(Se(ψ)) (Mualem or other model)
 * 
 * This is the most general groundwater physics option, appropriate for:
 *   - Vadose zone flow
 *   - Coupled saturated/unsaturated
 *   - Capillary rise
 *   - Infiltration dynamics
 */

#pragma once

#include "physics_base.hpp"
#include "water_retention.hpp"

namespace dgw {

/**
 * @brief 3D Richards equation solver
 */
class Richards3DSolver : public PhysicsBase {
public:
    Richards3DSolver() = default;
    explicit Richards3DSolver(const PhysicsDecisions& decisions);
    
    GoverningEquation type() const override { return GoverningEquation::Richards3D; }
    std::string name() const override { return "Richards3D"; }
    
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
     * @brief Set evapotranspiration sink
     * @param et_rate ET rate per cell [m/s]
     * @param root_distribution Vertical distribution of roots [-]
     */
    void set_evapotranspiration(const Vector& et_rate, const Vector& root_distribution);
    
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
     * @brief Find water table elevation (where ψ = 0)
     */
    Vector water_table_elevation(
        const State& state,
        const Mesh& mesh
    ) const;
    
    /**
     * @brief Get saturation field
     */
    Vector saturation(const State& state, const Parameters& params) const;
    
    /**
     * @brief Get water content field
     */
    Vector water_content(const State& state, const Parameters& params) const;
    
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
    
    // Richards-specific methods
    
    /**
     * @brief Update constitutive relations from current ψ
     * 
     * Computes θ(ψ), K(ψ), C(ψ) for all cells.
     * These are cached for efficiency during Newton iteration.
     */
    void update_constitutive(
        const StateRichards3D& state,
        const WaterRetentionParams& retention
    );
    
    /**
     * @brief Get cached hydraulic conductivity
     */
    const Vector& cached_K() const { return K_cache_; }
    
    /**
     * @brief Get cached moisture capacity
     */
    const Vector& cached_C() const { return C_cache_; }
    
    /**
     * @brief Get cached water content
     */
    const Vector& cached_theta() const { return theta_cache_; }
    
    // Numerical options
    
    /**
     * @brief Set mass lumping option
     * 
     * Mass lumping improves stability but reduces accuracy.
     * Recommended for highly nonlinear problems.
     */
    void set_mass_lumping(bool enable) { use_mass_lumping_ = enable; }
    
    /**
     * @brief Set modified Picard iteration
     * 
     * Uses (θ^{n+1,m} - θ^n)/dt instead of C(ψ)*(ψ^{n+1,m} - ψ^n)/dt
     * for better mass conservation.
     */
    void set_modified_picard(bool enable) { use_modified_picard_ = enable; }
    
    /**
     * @brief Set upstream weighting for K
     * 
     * Improves stability for infiltration fronts.
     */
    void set_upstream_weighting(Real weight) { upstream_weight_ = weight; }
    
private:
    PhysicsDecisions decisions_;
    
    // Cached constitutive relation values
    mutable Vector K_cache_;      // K(ψ) at each cell
    mutable Vector C_cache_;      // C(ψ) = dθ/dψ at each cell
    mutable Vector theta_cache_;  // θ(ψ) at each cell
    mutable Vector Kr_cache_;     // Kr(Se) relative conductivity
    
    // Numerical options
    bool use_mass_lumping_ = false;
    bool use_modified_picard_ = true;
    Real upstream_weight_ = 0.5;  // 0 = centered, 1 = full upstream
    
    // ET parameters
    Vector et_rate_;
    Vector root_distribution_;
    
    // Helper: compute inter-cell conductivity
    Real intercell_K(
        Index cell_i, Index cell_j,
        const Vector& psi,
        const WaterRetentionParams& retention,
        const Mesh3D& mesh
    ) const;
    
    // Helper: compute gravity-driven flux component
    Real gravity_flux(
        Index cell_i, Index cell_j,
        Real K_face,
        const Mesh3D& mesh
    ) const;
    
    // Helper: compute sink term (ET + pumping)
    Real compute_sink(Index cell, const Mesh3D& mesh) const;
};

// ============================================================================
// Enzyme-compatible kernels for 3D Richards
// ============================================================================

namespace richards_kernels {

/**
 * @brief Van Genuchten water content θ(ψ)
 */
inline Real van_genuchten_theta(
    Real psi,
    Real theta_r,
    Real theta_s,
    Real alpha,
    Real n
) {
    if (psi >= 0.0) {
        return theta_s;  // Saturated
    }
    Real m = 1.0 - 1.0/n;
    Real Se = std::pow(1.0 + std::pow(alpha * std::abs(psi), n), -m);
    return theta_r + (theta_s - theta_r) * Se;
}

/**
 * @brief Van Genuchten effective saturation Se(ψ)
 */
inline Real van_genuchten_Se(Real psi, Real alpha, Real n) {
    if (psi >= 0.0) {
        return 1.0;
    }
    Real m = 1.0 - 1.0/n;
    return std::pow(1.0 + std::pow(alpha * std::abs(psi), n), -m);
}

/**
 * @brief Van Genuchten specific moisture capacity C(ψ) = dθ/dψ
 */
inline Real van_genuchten_C(
    Real psi,
    Real theta_r,
    Real theta_s,
    Real alpha,
    Real n
) {
    if (psi >= 0.0) {
        return 0.0;  // Saturated, no change in θ with ψ
    }
    Real m = 1.0 - 1.0/n;
    Real abs_psi = std::abs(psi);
    Real alpha_psi_n = std::pow(alpha * abs_psi, n);
    Real Se = std::pow(1.0 + alpha_psi_n, -m);
    
    // C = -(θs - θr) * m * n * α * (α|ψ|)^(n-1) * (1 + (α|ψ|)^n)^(-m-1)
    Real C = (theta_s - theta_r) * m * n * alpha 
           * std::pow(alpha * abs_psi, n - 1.0)
           * std::pow(1.0 + alpha_psi_n, -m - 1.0);
    return C;
}

/**
 * @brief Mualem relative conductivity Kr(Se)
 */
inline Real mualem_Kr(Real Se, Real m, Real l = 0.5) {
    if (Se >= 1.0) return 1.0;
    if (Se <= 0.0) return 0.0;
    
    Real term = 1.0 - std::pow(1.0 - std::pow(Se, 1.0/m), m);
    return std::pow(Se, l) * term * term;
}

/**
 * @brief Van Genuchten-Mualem hydraulic conductivity K(ψ)
 */
inline Real van_genuchten_K(
    Real psi,
    Real K_sat,
    Real alpha,
    Real n,
    Real l = 0.5
) {
    if (psi >= 0.0) {
        return K_sat;
    }
    Real m = 1.0 - 1.0/n;
    Real Se = van_genuchten_Se(psi, alpha, n);
    Real Kr = mualem_Kr(Se, m, l);
    return K_sat * Kr;
}

/**
 * @brief Brooks-Corey water content θ(ψ)
 */
inline Real brooks_corey_theta(
    Real psi,
    Real theta_r,
    Real theta_s,
    Real psi_b,  // Bubbling pressure (positive)
    Real lambda  // Pore size distribution index
) {
    if (psi >= -psi_b) {
        return theta_s;
    }
    Real Se = std::pow(psi_b / std::abs(psi), lambda);
    return theta_r + (theta_s - theta_r) * Se;
}

/**
 * @brief Brooks-Corey specific moisture capacity
 */
inline Real brooks_corey_C(
    Real psi,
    Real theta_r,
    Real theta_s,
    Real psi_b,
    Real lambda
) {
    if (psi >= -psi_b) {
        return 0.0;
    }
    Real abs_psi = std::abs(psi);
    Real Se = std::pow(psi_b / abs_psi, lambda);
    // C = (θs - θr) * λ * Se / |ψ|
    return (theta_s - theta_r) * lambda * Se / abs_psi;
}

/**
 * @brief Brooks-Corey relative conductivity
 */
inline Real brooks_corey_Kr(Real Se, Real lambda) {
    if (Se >= 1.0) return 1.0;
    if (Se <= 0.0) return 0.0;
    return std::pow(Se, (2.0 + 3.0 * lambda) / lambda);
}

/**
 * @brief Residual kernel for Richards equation
 */
void residual_kernel_3d(
    const Real* __restrict__ psi,           // Pressure head
    const Real* __restrict__ psi_old,       // Previous timestep
    const Real* __restrict__ theta,         // Current water content
    const Real* __restrict__ theta_old,     // Previous water content
    const Real* __restrict__ K,             // Current conductivity
    const Real* __restrict__ z,             // Cell elevations
    const Real* __restrict__ cell_volume,
    const Index* __restrict__ cell_faces,   // CSR: faces for each cell
    const Index* __restrict__ cell_faces_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Index* __restrict__ face_cells,   // [n_faces, 2]: left/right cells
    const Real* __restrict__ source,        // Source/sink terms
    Index n_cells,
    Real dt,
    bool use_modified_picard,
    Real* __restrict__ residual
);

/**
 * @brief Jacobian kernel for Richards equation
 */
void jacobian_kernel_3d(
    const Real* __restrict__ psi,
    const Real* __restrict__ K,
    const Real* __restrict__ C,             // Moisture capacity
    const Real* __restrict__ dK_dpsi,       // ∂K/∂ψ
    const Real* __restrict__ z,
    const Real* __restrict__ cell_volume,
    const Index* __restrict__ cell_faces,
    const Index* __restrict__ cell_faces_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Index* __restrict__ face_cells,
    Index n_cells,
    Real dt,
    bool use_mass_lumping,
    Real* __restrict__ jac_values,
    Index* __restrict__ jac_rows,
    Index* __restrict__ jac_cols,
    Index& nnz
);

/**
 * @brief Inter-cell conductivity with upstream weighting
 * 
 * For infiltration fronts, upstream weighting improves stability.
 */
inline Real intercell_K_upstream(
    Real K_i, Real K_j,
    Real psi_i, Real psi_j,
    Real z_i, Real z_j,
    Real weight = 0.5
) {
    // Total head gradient determines upstream cell
    Real h_i = psi_i + z_i;
    Real h_j = psi_j + z_j;
    
    // Harmonic mean as base
    Real K_harmonic = 2.0 * K_i * K_j / (K_i + K_j + 1e-30);
    
    // Upstream weighting
    Real K_upstream = (h_i > h_j) ? K_i : K_j;
    
    return (1.0 - weight) * K_harmonic + weight * K_upstream;
}

} // namespace richards_kernels

// ============================================================================
// Boundary Condition Helpers for Richards
// ============================================================================

namespace richards_bc {

/**
 * @brief Flux boundary condition (Neumann)
 * 
 * Applied at surface for recharge/infiltration.
 */
struct FluxBC {
    Real flux;  // [m/s], positive = into domain
};

/**
 * @brief Pressure head boundary condition (Dirichlet)
 * 
 * Can be used for:
 * - Fixed water table (ψ = 0 at z = z_wt)
 * - Ponded infiltration (ψ = ponding_depth)
 * - Seepage face (ψ = 0 where saturated)
 */
struct HeadBC {
    Real psi;  // [m]
};

/**
 * @brief Free drainage (unit gradient)
 * 
 * Applied at bottom boundary:
 *   ∂h/∂z = 1  →  flux = -K(ψ)
 */
struct FreeDrainageBC {};

/**
 * @brief Seepage face boundary condition
 * 
 * Two-part condition:
 *   - If ψ < 0: no-flow (unsaturated)
 *   - If ψ ≥ 0: ψ = 0 (water exits at atmospheric pressure)
 */
struct SeepageFaceBC {};

/**
 * @brief Atmospheric boundary condition
 * 
 * Switches between flux-limited and head-limited:
 *   - If soil can accept full potential flux: flux BC
 *   - If flux would require ψ > 0: head BC with ψ = 0
 *   - If flux would require ψ < ψ_min: head BC with ψ = ψ_min
 */
struct AtmosphericBC {
    Real potential_flux;  // Potential evaporation/infiltration [m/s]
    Real psi_min;         // Minimum ψ for evaporation (wilting point) [m]
};

using BoundaryCondition = std::variant<FluxBC, HeadBC, FreeDrainageBC, 
                                        SeepageFaceBC, AtmosphericBC>;

} // namespace richards_bc

} // namespace dgw
