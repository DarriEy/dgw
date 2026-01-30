/**
 * @file water_retention.hpp
 * @brief Soil water retention and hydraulic conductivity functions
 * 
 * Implements constitutive relations for variably-saturated flow:
 * - van Genuchten-Mualem model
 * - Brooks-Corey model
 * - Clapp-Hornberger model
 * 
 * These are critical for Richards equation and affect numerical behavior.
 */

#pragma once

#include "../core/types.hpp"
#include <cmath>

namespace dgw {

/**
 * @brief Van Genuchten-Mualem water retention model
 * 
 * θ(ψ) = θr + (θs - θr) * [1 + (α|ψ|)^n]^(-m)
 * K(Se) = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2
 * 
 * Parameters:
 *   θr = residual water content
 *   θs = saturated water content  
 *   α = air entry parameter [1/m]
 *   n = pore size distribution index
 *   m = 1 - 1/n (Mualem constraint)
 *   l = pore connectivity parameter (typically 0.5)
 */
class VanGenuchten {
public:
    VanGenuchten() = default;
    
    VanGenuchten(Real theta_r, Real theta_s, Real alpha, Real n, 
                 Real K_sat, Real l = 0.5)
        : theta_r_(theta_r), theta_s_(theta_s), alpha_(alpha), n_(n),
          m_(1.0 - 1.0/n), K_sat_(K_sat), l_(l) {}
    
    /// Effective saturation Se(ψ) ∈ [0, 1]
    Real effective_saturation(Real psi) const {
        if (psi >= 0.0) return 1.0;
        Real alpha_psi_n = std::pow(alpha_ * std::abs(psi), n_);
        return std::pow(1.0 + alpha_psi_n, -m_);
    }
    
    /// Water content θ(ψ)
    Real water_content(Real psi) const {
        Real Se = effective_saturation(psi);
        return theta_r_ + (theta_s_ - theta_r_) * Se;
    }
    
    /// Specific moisture capacity C(ψ) = dθ/dψ
    Real moisture_capacity(Real psi) const {
        if (psi >= 0.0) return 0.0;
        
        Real abs_psi = std::abs(psi);
        Real alpha_psi = alpha_ * abs_psi;
        Real alpha_psi_n = std::pow(alpha_psi, n_);
        Real denom = 1.0 + alpha_psi_n;
        
        // C = (θs - θr) * α * m * n * (α|ψ|)^(n-1) * (1 + (α|ψ|)^n)^(-m-1)
        Real C = (theta_s_ - theta_r_) * alpha_ * m_ * n_ 
               * std::pow(alpha_psi, n_ - 1.0)
               * std::pow(denom, -m_ - 1.0);
        return C;
    }
    
    /// Relative hydraulic conductivity Kr(Se) ∈ [0, 1]
    Real relative_conductivity(Real Se) const {
        if (Se >= 1.0) return 1.0;
        if (Se <= 0.0) return 0.0;
        
        // Kr = Se^l * [1 - (1 - Se^(1/m))^m]^2
        Real Se_1_m = std::pow(Se, 1.0/m_);
        Real term = 1.0 - std::pow(1.0 - Se_1_m, m_);
        return std::pow(Se, l_) * term * term;
    }
    
    /// Hydraulic conductivity K(ψ) = Ks * Kr(Se(ψ))
    Real hydraulic_conductivity(Real psi) const {
        Real Se = effective_saturation(psi);
        return K_sat_ * relative_conductivity(Se);
    }
    
    /// Derivative dK/dψ (for Jacobian)
    Real dK_dpsi(Real psi) const {
        if (psi >= 0.0) return 0.0;
        
        Real Se = effective_saturation(psi);
        Real dSe_dpsi = moisture_capacity(psi) / (theta_s_ - theta_r_);
        Real dKr_dSe = relative_conductivity_derivative(Se);
        
        return K_sat_ * dKr_dSe * dSe_dpsi;
    }
    
    /// Derivative dKr/dSe
    Real relative_conductivity_derivative(Real Se) const {
        if (Se >= 1.0 || Se <= 0.0) return 0.0;
        
        Real Se_1_m = std::pow(Se, 1.0/m_);
        Real one_minus = 1.0 - Se_1_m;
        Real one_minus_m = std::pow(one_minus, m_);
        Real term = 1.0 - one_minus_m;
        
        // dKr/dSe = l*Se^(l-1)*term^2 + Se^l * 2*term * d(term)/dSe
        // where d(term)/dSe = m * (1-Se^(1/m))^(m-1) * (1/m) * Se^(1/m-1)
        //                   = (1-Se^(1/m))^(m-1) * Se^(1/m-1)
        
        Real dterm_dSe = std::pow(one_minus, m_ - 1.0) * std::pow(Se, 1.0/m_ - 1.0);
        
        return l_ * std::pow(Se, l_ - 1.0) * term * term
             + std::pow(Se, l_) * 2.0 * term * dterm_dSe;
    }
    
    // Vectorized versions for efficiency
    void water_content_vec(const Real* psi, Index n, Real* theta) const {
        #pragma omp simd
        for (Index i = 0; i < n; ++i) {
            theta[i] = water_content(psi[i]);
        }
    }
    
    void hydraulic_conductivity_vec(const Real* psi, Index n, Real* K) const {
        #pragma omp simd
        for (Index i = 0; i < n; ++i) {
            K[i] = hydraulic_conductivity(psi[i]);
        }
    }
    
    void moisture_capacity_vec(const Real* psi, Index n, Real* C) const {
        #pragma omp simd
        for (Index i = 0; i < n; ++i) {
            C[i] = moisture_capacity(psi[i]);
        }
    }
    
    // Accessors
    Real theta_r() const { return theta_r_; }
    Real theta_s() const { return theta_s_; }
    Real alpha() const { return alpha_; }
    Real n() const { return n_; }
    Real m() const { return m_; }
    Real K_sat() const { return K_sat_; }
    Real l() const { return l_; }
    
private:
    Real theta_r_ = 0.0;   // Residual water content
    Real theta_s_ = 0.4;   // Saturated water content
    Real alpha_ = 1.0;     // Air entry [1/m]
    Real n_ = 2.0;         // Pore size distribution
    Real m_ = 0.5;         // = 1 - 1/n
    Real K_sat_ = 1e-5;    // Saturated K [m/s]
    Real l_ = 0.5;         // Pore connectivity
};

/**
 * @brief Brooks-Corey water retention model
 * 
 * Se = (ψb/|ψ|)^λ  for |ψ| > ψb
 * Se = 1           for |ψ| ≤ ψb
 * Kr = Se^((2+3λ)/λ)
 * 
 * Parameters:
 *   ψb = bubbling pressure (air entry value) [m]
 *   λ = pore size distribution index
 */
class BrooksCorey {
public:
    BrooksCorey() = default;
    
    BrooksCorey(Real theta_r, Real theta_s, Real psi_b, Real lambda,
                Real K_sat)
        : theta_r_(theta_r), theta_s_(theta_s), psi_b_(psi_b),
          lambda_(lambda), K_sat_(K_sat) {}
    
    Real effective_saturation(Real psi) const {
        if (psi >= -psi_b_) return 1.0;
        return std::pow(psi_b_ / std::abs(psi), lambda_);
    }
    
    Real water_content(Real psi) const {
        Real Se = effective_saturation(psi);
        return theta_r_ + (theta_s_ - theta_r_) * Se;
    }
    
    Real moisture_capacity(Real psi) const {
        if (psi >= -psi_b_) return 0.0;
        
        Real abs_psi = std::abs(psi);
        Real Se = std::pow(psi_b_ / abs_psi, lambda_);
        
        // C = (θs - θr) * λ * Se / |ψ|
        return (theta_s_ - theta_r_) * lambda_ * Se / abs_psi;
    }
    
    Real relative_conductivity(Real Se) const {
        if (Se >= 1.0) return 1.0;
        if (Se <= 0.0) return 0.0;
        return std::pow(Se, (2.0 + 3.0 * lambda_) / lambda_);
    }
    
    Real hydraulic_conductivity(Real psi) const {
        Real Se = effective_saturation(psi);
        return K_sat_ * relative_conductivity(Se);
    }
    
    Real dK_dpsi(Real psi) const {
        if (psi >= -psi_b_) return 0.0;
        
        Real Se = effective_saturation(psi);
        Real dSe_dpsi = moisture_capacity(psi) / (theta_s_ - theta_r_);
        Real dKr_dSe = (2.0 + 3.0 * lambda_) / lambda_ 
                     * std::pow(Se, (2.0 + 2.0 * lambda_) / lambda_);
        
        return K_sat_ * dKr_dSe * dSe_dpsi;
    }
    
private:
    Real theta_r_ = 0.0;
    Real theta_s_ = 0.4;
    Real psi_b_ = 0.1;     // Bubbling pressure [m]
    Real lambda_ = 0.5;    // Pore size distribution
    Real K_sat_ = 1e-5;
};

/**
 * @brief Clapp-Hornberger model (simplified, used in many land surface models)
 * 
 * θ = θs * (ψ/ψs)^(-1/b)
 * K = Ks * (θ/θs)^(2b+3)
 */
class ClappHornberger {
public:
    ClappHornberger() = default;
    
    ClappHornberger(Real theta_s, Real psi_s, Real b, Real K_sat)
        : theta_s_(theta_s), psi_s_(psi_s), b_(b), K_sat_(K_sat) {}
    
    Real water_content(Real psi) const {
        if (psi >= psi_s_) return theta_s_;
        return theta_s_ * std::pow(psi / psi_s_, -1.0 / b_);
    }
    
    Real hydraulic_conductivity_from_theta(Real theta) const {
        Real Se = theta / theta_s_;
        return K_sat_ * std::pow(Se, 2.0 * b_ + 3.0);
    }
    
    Real hydraulic_conductivity(Real psi) const {
        Real theta = water_content(psi);
        return hydraulic_conductivity_from_theta(theta);
    }
    
private:
    Real theta_s_ = 0.4;   // Saturated water content
    Real psi_s_ = -0.1;    // Saturation matric potential [m]
    Real b_ = 5.0;         // Clapp-Hornberger exponent
    Real K_sat_ = 1e-5;    // Saturated K [m/s]
};

/**
 * @brief Regularized van Genuchten for numerical stability
 * 
 * Near saturation, standard van Genuchten has C(ψ) → 0, causing
 * numerical issues. This version adds regularization.
 */
class VanGenuchtenRegularized : public VanGenuchten {
public:
    using VanGenuchten::VanGenuchten;
    
    /// Regularized moisture capacity (non-zero even when saturated)
    Real moisture_capacity_regularized(Real psi) const {
        Real C_std = VanGenuchten::moisture_capacity(psi);
        
        // Add small regularization based on storage coefficient
        // This represents elastic storage in saturated zone
        Real C_sat = Ss_ * (theta_s() - theta_r());
        
        // Smooth blend near saturation
        Real Se = effective_saturation(psi);
        Real blend = smoothstep(Se, 0.99, 1.0);
        
        return C_std + blend * C_sat;
    }
    
    /// Set specific storage for regularization
    void set_specific_storage(Real Ss) { Ss_ = Ss; }
    
private:
    Real Ss_ = 1e-4;  // Specific storage for regularization
    
    // Smooth step function
    static Real smoothstep(Real x, Real edge0, Real edge1) {
        Real t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }
};

/**
 * @brief Pedotransfer functions for estimating retention parameters from texture
 */
namespace pedotransfer {

/**
 * @brief Estimate van Genuchten parameters from soil texture (Carsel & Parrish 1988)
 */
struct TextureClass {
    const char* name;
    Real theta_r;
    Real theta_s;
    Real alpha;    // [1/cm] in original, converted to [1/m]
    Real n;
    Real K_sat;    // [cm/day] in original, converted to [m/s]
};

// Standard USDA texture classes (parameters from Carsel & Parrish 1988)
// α converted from 1/cm to 1/m (×100), Ks from cm/day to m/s (÷8640000)
constexpr TextureClass SAND         = {"Sand",         0.045, 0.43, 14.5,  2.68, 8.25e-5};
constexpr TextureClass LOAMY_SAND   = {"Loamy Sand",   0.057, 0.41, 12.4,  2.28, 4.05e-5};
constexpr TextureClass SANDY_LOAM   = {"Sandy Loam",   0.065, 0.41,  7.5,  1.89, 1.23e-5};
constexpr TextureClass LOAM         = {"Loam",         0.078, 0.43,  3.6,  1.56, 2.89e-6};
constexpr TextureClass SILT         = {"Silt",         0.034, 0.46,  1.6,  1.37, 6.94e-7};
constexpr TextureClass SILT_LOAM    = {"Silt Loam",    0.067, 0.45,  2.0,  1.41, 1.25e-6};
constexpr TextureClass SANDY_CLAY_LOAM = {"Sandy Clay Loam", 0.100, 0.39, 5.9, 1.48, 3.64e-6};
constexpr TextureClass CLAY_LOAM    = {"Clay Loam",    0.095, 0.41,  1.9,  1.31, 7.22e-7};
constexpr TextureClass SILTY_CLAY_LOAM = {"Silty Clay Loam", 0.089, 0.43, 1.0, 1.23, 1.94e-7};
constexpr TextureClass SANDY_CLAY   = {"Sandy Clay",   0.100, 0.38,  2.7,  1.23, 3.33e-7};
constexpr TextureClass SILTY_CLAY   = {"Silty Clay",   0.070, 0.36,  0.5,  1.09, 5.56e-8};
constexpr TextureClass CLAY         = {"Clay",         0.068, 0.38,  0.8,  1.09, 5.56e-7};

/**
 * @brief Get van Genuchten model for texture class
 */
inline VanGenuchten from_texture(const TextureClass& tex) {
    return VanGenuchten(tex.theta_r, tex.theta_s, tex.alpha, tex.n, tex.K_sat);
}

/**
 * @brief Estimate parameters from sand/silt/clay fractions using Rosetta-like approach
 * 
 * @param sand Sand fraction [0-1]
 * @param silt Silt fraction [0-1]  
 * @param clay Clay fraction [0-1]
 */
VanGenuchten from_texture_fractions(Real sand, Real silt, Real clay);

} // namespace pedotransfer

// ============================================================================
// Free-function interface for scalar van Genuchten computations
// Used by gradient tests and Enzyme-friendly kernels
// ============================================================================

namespace retention {

/// Water content theta(psi) for van Genuchten
inline Real vg_theta(Real psi, Real theta_r, Real theta_s, Real alpha, Real n) {
    VanGenuchten vg(theta_r, theta_s, alpha, n, 0.0);
    return vg.water_content(psi);
}

/// Moisture capacity C(psi) = dtheta/dpsi
inline Real vg_C(Real psi, Real theta_r, Real theta_s, Real alpha, Real n) {
    VanGenuchten vg(theta_r, theta_s, alpha, n, 0.0);
    return vg.moisture_capacity(psi);
}

/// Hydraulic conductivity K(psi)
inline Real vg_K(Real psi, Real K_sat, Real alpha, Real n) {
    VanGenuchten vg(0.0, 1.0, alpha, n, K_sat);
    return vg.hydraulic_conductivity(psi);
}

/// Derivative dK/dpsi
inline Real vg_dK_dpsi(Real psi, Real K_sat, Real alpha, Real n) {
    VanGenuchten vg(0.0, 1.0, alpha, n, K_sat);
    return vg.dK_dpsi(psi);
}

} // namespace retention

} // namespace dgw
