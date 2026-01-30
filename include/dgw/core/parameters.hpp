/**
 * @file parameters.hpp
 * @brief Model parameters for dGW
 * 
 * Contains all spatially-distributed and scalar parameters:
 * - Hydraulic conductivity (K)
 * - Storage parameters (Sy, Ss)
 * - Water retention parameters (for Richards)
 * - Boundary condition parameters
 * - Stream-aquifer coupling parameters
 */

#pragma once

#include "types.hpp"
#include "mesh.hpp"

namespace dgw {

/**
 * @brief Parameters for 2D saturated flow (Boussinesq or linear diffusion)
 */
struct Parameters2D {
    // Hydraulic properties (per cell)
    Vector K;                       ///< Hydraulic conductivity [m/s]
    Vector Sy;                      ///< Specific yield [-] (unconfined storage)
    Vector Ss;                      ///< Specific storage [1/m] (confined)
    
    // Geometry (could also be in mesh, but useful here for sensitivity)
    Vector z_surface;               ///< Land surface elevation [m]
    Vector z_bottom;                ///< Aquifer bottom elevation [m]
    
    // Anisotropy (optional)
    Vector K_vertical;              ///< Vertical K if different from horizontal
    Real anisotropy_ratio = 1.0;    ///< Kv/Kh if uniform
    
    // Transmissivity limits (for numerical stability)
    Real T_min = 1e-10;             ///< Minimum transmissivity [m²/s]
    Real T_max = 1e6;               ///< Maximum transmissivity [m²/s]
    
    /// Initialize uniform parameters
    void initialize_uniform(Index n_cells, Real K_val, Real Sy_val);
    
    /// Initialize from spatial data
    void initialize(const Vector& K_in, const Vector& Sy_in,
                   const Vector& z_surf, const Vector& z_bot);
    
    /// Compute transmissivity T = K * saturated_thickness
    Vector transmissivity(const Vector& head) const;
    
    /// Compute saturated thickness
    Vector saturated_thickness(const Vector& head) const;
};

/**
 * @brief Parameters for two-layer system
 */
struct ParametersTwoLayer {
    // Layer 1: Unconfined
    Vector K1;                      ///< Layer 1 hydraulic conductivity [m/s]
    Vector Sy;                      ///< Specific yield [-]
    Vector z_surface;               ///< Land surface [m]
    Vector z_bottom_1;              ///< Layer 1 bottom (= confining layer top) [m]
    
    // Confining layer
    Vector K_confining;             ///< Vertical K of confining layer [m/s]
    Vector thickness_confining;     ///< Confining layer thickness [m]
    
    // Layer 2: Confined
    Vector K2;                      ///< Layer 2 hydraulic conductivity [m/s]
    Vector Ss2;                     ///< Specific storage [1/m]
    Vector thickness_2;             ///< Layer 2 thickness [m]
    Vector z_bottom_2;              ///< Layer 2 bottom [m]
    
    /// Compute leakance (conductance for vertical flow)
    Vector leakance() const;
    
    /// Compute layer 1 transmissivity (nonlinear)
    Vector T1(const Vector& h1) const;
    
    /// Compute layer 2 transmissivity (constant)
    Vector T2() const;
};

/**
 * @brief Parameters for multi-layer system
 */
struct ParametersMultiLayer {
    Index n_layers;
    
    // Per-layer properties [n_layers][n_cells]
    std::vector<Vector> K;              ///< Hydraulic conductivity
    std::vector<Vector> storage;        ///< Sy (unconfined) or Ss*b (confined)
    std::vector<Vector> thickness;      ///< Layer thickness
    std::vector<bool> is_confined;      ///< True if layer is confined
    
    // Inter-layer properties [n_layers-1][n_cells]
    std::vector<Vector> vertical_K;     ///< Vertical K between layers
    std::vector<Vector> aquitard_thickness; ///< Confining layer thickness
    
    /// Get leakance between layer k and k+1
    Vector leakance(Index k) const;
};

/**
 * @brief Water retention curve parameters (for Richards equation)
 */
struct WaterRetentionParams {
    // van Genuchten parameters (most common)
    Vector theta_r;                 ///< Residual water content [-]
    Vector theta_s;                 ///< Saturated water content [-]
    Vector alpha;                   ///< Air entry parameter [1/m]
    Vector n_vg;                    ///< Pore size distribution [-]
    Vector m_vg;                    ///< = 1 - 1/n (derived)
    
    // Mualem parameters for K(Se)
    Vector K_sat;                   ///< Saturated hydraulic conductivity [m/s]
    Vector l_mualem;                ///< Pore connectivity parameter [-]
    
    // Brooks-Corey alternative
    Vector psi_b;                   ///< Bubbling pressure [m]
    Vector lambda_bc;               ///< Pore size distribution [-]
    
    RetentionModel model = RetentionModel::VanGenuchten;
    
    /// Initialize van Genuchten parameters
    void initialize_van_genuchten(
        const Vector& theta_r_in, const Vector& theta_s_in,
        const Vector& alpha_in, const Vector& n_in,
        const Vector& K_sat_in, Real l = 0.5);
    
    /// Initialize Brooks-Corey parameters
    void initialize_brooks_corey(
        const Vector& theta_r_in, const Vector& theta_s_in,
        const Vector& psi_b_in, const Vector& lambda_in,
        const Vector& K_sat_in);
    
    /// Initialize from soil texture class
    void initialize_from_texture(const std::vector<std::string>& texture_class);
    
    // Constitutive relations
    
    /// Water content θ(ψ)
    Vector water_content(const Vector& psi) const;
    
    /// Specific moisture capacity C(ψ) = dθ/dψ
    Vector moisture_capacity(const Vector& psi) const;
    
    /// Effective saturation Se(ψ)
    Vector effective_saturation(const Vector& psi) const;
    
    /// Relative hydraulic conductivity Kr(Se)
    Vector relative_conductivity(const Vector& Se) const;
    
    /// Hydraulic conductivity K(ψ) = K_sat * Kr(Se(ψ))
    Vector hydraulic_conductivity(const Vector& psi) const;
};

/**
 * @brief Stream-aquifer exchange parameters
 */
struct StreamParameters {
    // Per-river-segment properties
    Vector streambed_K;             ///< Streambed conductivity [m/s]
    Vector streambed_thickness;     ///< Streambed thickness [m]
    Vector stream_width;            ///< Stream width [m]
    Vector stream_length;           ///< Segment length in cell [m]
    Vector streambed_elevation;     ///< Streambed bottom [m]
    
    // Clogging layer (optional)
    Vector clogging_K;              ///< Clogging layer K [m/s]
    Vector clogging_thickness;      ///< Clogging layer thickness [m]
    bool has_clogging = false;
    
    /// Compute conductance C = K*W*L/b
    Vector conductance() const;
    
    /// Compute conductance with clogging layer
    Vector conductance_with_clogging() const;
};

/**
 * @brief Recharge and vadose zone parameters
 */
struct RechargeParameters {
    VadoseMethod method = VadoseMethod::Direct;
    
    // Exponential lag parameters
    Vector lag_coefficient;         ///< τ = coef * depth_to_wt [s/m]
    Real min_lag = 3600.0;          ///< Minimum lag time [s] (1 hour)
    Real max_lag = 86400.0 * 365;   ///< Maximum lag time [s] (1 year)
    
    // Kinematic wave parameters (if using that method)
    Vector unsat_K;                 ///< Unsaturated zone K [m/s]
    Vector unsat_theta;             ///< Unsaturated zone porosity [-]
};

/**
 * @brief Boundary condition parameters
 */
struct BoundaryParameters {
    // General head boundary
    Vector ghb_head;                ///< External head [m]
    Vector ghb_conductance;         ///< Conductance [m²/s]
    
    // Fixed flux boundary
    Vector flux_values;             ///< Specified flux [m³/s]
    
    // Drain parameters
    Vector drain_elevation;         ///< Drain elevation [m]
    Vector drain_conductance;       ///< Drain conductance [m²/s]
    
    // Seepage face
    std::vector<Index> seepage_faces; ///< Face indices for seepage BC
};

/**
 * @brief Complete parameter set (variant-based like State)
 */
class Parameters {
public:
    Parameters() = default;
    explicit Parameters(GoverningEquation physics) : physics_(physics) {}
    
    GoverningEquation physics() const { return physics_; }
    
    // Accessors for specific parameter types
    Parameters2D& as_2d() { return params_2d_; }
    const Parameters2D& as_2d() const { return params_2d_; }
    
    ParametersTwoLayer& as_two_layer() { return params_two_layer_; }
    const ParametersTwoLayer& as_two_layer() const { return params_two_layer_; }
    
    ParametersMultiLayer& as_multi_layer() { return params_multi_layer_; }
    const ParametersMultiLayer& as_multi_layer() const { return params_multi_layer_; }
    
    WaterRetentionParams& retention() { return retention_params_; }
    const WaterRetentionParams& retention() const { return retention_params_; }
    
    StreamParameters& stream() { return stream_params_; }
    const StreamParameters& stream() const { return stream_params_; }
    
    RechargeParameters& recharge() { return recharge_params_; }
    const RechargeParameters& recharge() const { return recharge_params_; }
    
    BoundaryParameters& boundary() { return boundary_params_; }
    const BoundaryParameters& boundary() const { return boundary_params_; }
    
    /// Pack trainable parameters into flat vector (for optimization)
    Vector pack_trainable() const;
    
    /// Unpack trainable parameters from flat vector
    void unpack_trainable(const Vector& packed);
    
    /// Get number of trainable parameters
    Index n_trainable() const;
    
    /// Get parameter names (for reporting)
    std::vector<std::string> trainable_names() const;
    
    /// Load parameters from file
    void load(const std::string& filename, const Mesh& mesh);
    
    /// Save parameters to file
    void save(const std::string& filename) const;
    
private:
    GoverningEquation physics_ = GoverningEquation::Boussinesq;
    
    Parameters2D params_2d_;
    ParametersTwoLayer params_two_layer_;
    ParametersMultiLayer params_multi_layer_;
    WaterRetentionParams retention_params_;
    StreamParameters stream_params_;
    RechargeParameters recharge_params_;
    BoundaryParameters boundary_params_;
    
    // Mask for which parameters are trainable
    std::vector<bool> trainable_mask_;
};

// ============================================================================
// Default Parameter Values (from literature)
// ============================================================================

namespace default_params {

// Hydraulic conductivity [m/s] by texture class
constexpr Real K_clay = 1e-9;
constexpr Real K_silt = 1e-7;
constexpr Real K_sand = 1e-4;
constexpr Real K_gravel = 1e-2;

// Specific yield [-] by texture class
constexpr Real Sy_clay = 0.03;
constexpr Real Sy_silt = 0.08;
constexpr Real Sy_sand = 0.23;
constexpr Real Sy_gravel = 0.25;

// Specific storage [1/m] (confined aquifers)
constexpr Real Ss_soft_clay = 1e-2;
constexpr Real Ss_stiff_clay = 1e-3;
constexpr Real Ss_sand = 1e-4;
constexpr Real Ss_rock = 1e-6;

// Van Genuchten parameters by texture (Carsel & Parrish, 1988)
// K_sat values from Rawls et al. (1982) in m/s
struct VGParams {
    Real theta_r, theta_s, alpha, n, K_sat;
};

constexpr VGParams vg_sand              = {0.045, 0.43, 14.5, 2.68, 8.25e-5};
constexpr VGParams vg_loamy_sand        = {0.057, 0.41, 12.4, 2.28, 4.05e-5};
constexpr VGParams vg_sandy_loam        = {0.065, 0.41,  7.5, 1.89, 1.23e-5};
constexpr VGParams vg_loam              = {0.078, 0.43,  3.6, 1.56, 2.89e-6};
constexpr VGParams vg_silt              = {0.034, 0.46,  1.6, 1.37, 6.94e-7};
constexpr VGParams vg_silt_loam         = {0.067, 0.45,  2.0, 1.41, 1.25e-6};
constexpr VGParams vg_sandy_clay_loam   = {0.100, 0.39,  5.9, 1.48, 3.64e-6};
constexpr VGParams vg_clay_loam         = {0.095, 0.41,  1.9, 1.31, 7.22e-7};
constexpr VGParams vg_silty_clay_loam   = {0.089, 0.43,  1.0, 1.23, 1.94e-7};
constexpr VGParams vg_sandy_clay        = {0.100, 0.38,  2.7, 1.23, 3.33e-7};
constexpr VGParams vg_silty_clay        = {0.070, 0.36,  0.5, 1.09, 5.56e-8};
constexpr VGParams vg_clay              = {0.068, 0.38,  0.8, 1.09, 1.67e-7};

} // namespace default_params

} // namespace dgw
