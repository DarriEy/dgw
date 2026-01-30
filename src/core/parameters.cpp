/**
 * @file parameters.cpp
 * @brief Model parameters implementations
 */

#include "dgw/core/parameters.hpp"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace dgw {

// ============================================================================
// Parameters2D
// ============================================================================

void Parameters2D::initialize_uniform(Index n_cells, Real K_val, Real Sy_val) {
    K.resize(n_cells);
    K.setConstant(K_val);
    Sy.resize(n_cells);
    Sy.setConstant(Sy_val);
    Ss.resize(n_cells);
    Ss.setConstant(1e-4);

    z_surface.resize(n_cells);
    z_surface.setZero();
    z_bottom.resize(n_cells);
    z_bottom.setZero();
}

void Parameters2D::initialize(const Vector& K_in, const Vector& Sy_in,
                              const Vector& z_surf, const Vector& z_bot) {
    K = K_in;
    Sy = Sy_in;
    z_surface = z_surf;
    z_bottom = z_bot;

    Ss.resize(K.size());
    Ss.setConstant(1e-4);
}

Vector Parameters2D::transmissivity(const Vector& head) const {
    const Index n = K.size();
    Vector T(n);
    for (Index i = 0; i < n; ++i) {
        Real sat_thick = std::max(head(i) - z_bottom(i), 0.0);
        T(i) = std::clamp(K(i) * sat_thick, T_min, T_max);
    }
    return T;
}

Vector Parameters2D::saturated_thickness(const Vector& head) const {
    const Index n = K.size();
    Vector b(n);
    for (Index i = 0; i < n; ++i) {
        b(i) = std::max(head(i) - z_bottom(i), 0.0);
    }
    return b;
}

// ============================================================================
// ParametersTwoLayer
// ============================================================================

Vector ParametersTwoLayer::leakance() const {
    const Index n = K_confining.size();
    Vector L(n);
    for (Index i = 0; i < n; ++i) {
        L(i) = K_confining(i) / (thickness_confining(i) + 1e-30);
    }
    return L;
}

Vector ParametersTwoLayer::T1(const Vector& h1) const {
    const Index n = K1.size();
    Vector T(n);
    for (Index i = 0; i < n; ++i) {
        Real sat_thick = std::max(h1(i) - z_bottom_1(i), 0.0);
        T(i) = K1(i) * sat_thick;
    }
    return T;
}

Vector ParametersTwoLayer::T2() const {
    const Index n = K2.size();
    Vector T(n);
    for (Index i = 0; i < n; ++i) {
        T(i) = K2(i) * thickness_2(i);
    }
    return T;
}

// ============================================================================
// ParametersMultiLayer
// ============================================================================

Vector ParametersMultiLayer::leakance(Index k) const {
    const Index n = vertical_K[k].size();
    Vector L(n);
    for (Index i = 0; i < n; ++i) {
        L(i) = vertical_K[k](i) / (aquitard_thickness[k](i) + 1e-30);
    }
    return L;
}

// ============================================================================
// WaterRetentionParams
// ============================================================================

void WaterRetentionParams::initialize_van_genuchten(
    const Vector& theta_r_in, const Vector& theta_s_in,
    const Vector& alpha_in, const Vector& n_in,
    const Vector& K_sat_in, Real l
) {
    model = RetentionModel::VanGenuchten;
    theta_r = theta_r_in;
    theta_s = theta_s_in;
    alpha = alpha_in;
    n_vg = n_in;
    K_sat = K_sat_in;

    // Compute m = 1 - 1/n
    m_vg.resize(n_vg.size());
    for (Index i = 0; i < n_vg.size(); ++i) {
        m_vg(i) = 1.0 - 1.0 / n_vg(i);
    }

    l_mualem.resize(K_sat.size());
    l_mualem.setConstant(l);
}

void WaterRetentionParams::initialize_brooks_corey(
    const Vector& theta_r_in, const Vector& theta_s_in,
    const Vector& psi_b_in, const Vector& lambda_in,
    const Vector& K_sat_in
) {
    model = RetentionModel::BrooksCorey;
    theta_r = theta_r_in;
    theta_s = theta_s_in;
    psi_b = psi_b_in;
    lambda_bc = lambda_in;
    K_sat = K_sat_in;
}

void WaterRetentionParams::initialize_from_texture(const std::vector<std::string>& texture_class) {
    const Index n = static_cast<Index>(texture_class.size());
    model = RetentionModel::VanGenuchten;

    theta_r.resize(n);
    theta_s.resize(n);
    alpha.resize(n);
    n_vg.resize(n);
    m_vg.resize(n);
    K_sat.resize(n);
    l_mualem.resize(n);
    l_mualem.setConstant(0.5);

    for (Index i = 0; i < n; ++i) {
        const auto& tex = texture_class[i];
        default_params::VGParams vg;

        if (tex == "sand") vg = default_params::vg_sand;
        else if (tex == "loamy_sand") vg = default_params::vg_loamy_sand;
        else if (tex == "sandy_loam") vg = default_params::vg_sandy_loam;
        else if (tex == "loam") vg = default_params::vg_loam;
        else if (tex == "silt") vg = default_params::vg_silt;
        else if (tex == "silt_loam") vg = default_params::vg_silt_loam;
        else if (tex == "sandy_clay_loam") vg = default_params::vg_sandy_clay_loam;
        else if (tex == "clay_loam") vg = default_params::vg_clay_loam;
        else if (tex == "silty_clay_loam") vg = default_params::vg_silty_clay_loam;
        else if (tex == "sandy_clay") vg = default_params::vg_sandy_clay;
        else if (tex == "silty_clay") vg = default_params::vg_silty_clay;
        else if (tex == "clay") vg = default_params::vg_clay;
        else vg = default_params::vg_loam;  // Default

        theta_r(i) = vg.theta_r;
        theta_s(i) = vg.theta_s;
        alpha(i) = vg.alpha;
        n_vg(i) = vg.n;
        m_vg(i) = 1.0 - 1.0 / vg.n;
        K_sat(i) = vg.K_sat;
    }
}

Vector WaterRetentionParams::water_content(const Vector& psi) const {
    const Index n = psi.size();
    Vector theta(n);

    for (Index i = 0; i < n; ++i) {
        if (model == RetentionModel::VanGenuchten) {
            if (psi(i) >= 0.0) {
                theta(i) = theta_s(i);
            } else {
                Real alpha_psi_n = std::pow(alpha(i) * std::abs(psi(i)), n_vg(i));
                Real Se = std::pow(1.0 + alpha_psi_n, -m_vg(i));
                theta(i) = theta_r(i) + (theta_s(i) - theta_r(i)) * Se;
            }
        } else if (model == RetentionModel::BrooksCorey) {
            if (psi(i) >= -psi_b(i)) {
                theta(i) = theta_s(i);
            } else {
                Real Se = std::pow(psi_b(i) / std::abs(psi(i)), lambda_bc(i));
                theta(i) = theta_r(i) + (theta_s(i) - theta_r(i)) * Se;
            }
        }
    }
    return theta;
}

Vector WaterRetentionParams::moisture_capacity(const Vector& psi) const {
    const Index n = psi.size();
    Vector C(n);

    for (Index i = 0; i < n; ++i) {
        if (model == RetentionModel::VanGenuchten) {
            if (psi(i) >= 0.0) {
                C(i) = 0.0;
            } else {
                Real abs_psi = std::abs(psi(i));
                Real alpha_psi_n = std::pow(alpha(i) * abs_psi, n_vg(i));
                C(i) = (theta_s(i) - theta_r(i)) * alpha(i) * m_vg(i) * n_vg(i)
                       * std::pow(alpha(i) * abs_psi, n_vg(i) - 1.0)
                       * std::pow(1.0 + alpha_psi_n, -m_vg(i) - 1.0);
            }
        } else if (model == RetentionModel::BrooksCorey) {
            if (psi(i) >= -psi_b(i)) {
                C(i) = 0.0;
            } else {
                Real abs_psi = std::abs(psi(i));
                Real Se = std::pow(psi_b(i) / abs_psi, lambda_bc(i));
                C(i) = (theta_s(i) - theta_r(i)) * lambda_bc(i) * Se / abs_psi;
            }
        }
    }
    return C;
}

Vector WaterRetentionParams::effective_saturation(const Vector& psi) const {
    const Index n = psi.size();
    Vector Se(n);

    for (Index i = 0; i < n; ++i) {
        if (model == RetentionModel::VanGenuchten) {
            if (psi(i) >= 0.0) {
                Se(i) = 1.0;
            } else {
                Real alpha_psi_n = std::pow(alpha(i) * std::abs(psi(i)), n_vg(i));
                Se(i) = std::pow(1.0 + alpha_psi_n, -m_vg(i));
            }
        } else if (model == RetentionModel::BrooksCorey) {
            if (psi(i) >= -psi_b(i)) {
                Se(i) = 1.0;
            } else {
                Se(i) = std::pow(psi_b(i) / std::abs(psi(i)), lambda_bc(i));
            }
        }
    }
    return Se;
}

Vector WaterRetentionParams::relative_conductivity(const Vector& Se) const {
    const Index n = Se.size();
    Vector Kr(n);

    for (Index i = 0; i < n; ++i) {
        if (Se(i) >= 1.0) {
            Kr(i) = 1.0;
        } else if (Se(i) <= 0.0) {
            Kr(i) = 0.0;
        } else if (model == RetentionModel::VanGenuchten) {
            Real Se_1_m = std::pow(Se(i), 1.0 / m_vg(i));
            Real term = 1.0 - std::pow(1.0 - Se_1_m, m_vg(i));
            Kr(i) = std::pow(Se(i), l_mualem(i)) * term * term;
        } else if (model == RetentionModel::BrooksCorey) {
            Kr(i) = std::pow(Se(i), (2.0 + 3.0 * lambda_bc(i)) / lambda_bc(i));
        }
    }
    return Kr;
}

Vector WaterRetentionParams::hydraulic_conductivity(const Vector& psi) const {
    Vector Se = effective_saturation(psi);
    Vector Kr = relative_conductivity(Se);

    const Index n = psi.size();
    Vector K(n);
    for (Index i = 0; i < n; ++i) {
        K(i) = K_sat(i) * Kr(i);
    }
    return K;
}

// ============================================================================
// StreamParameters
// ============================================================================

Vector StreamParameters::conductance() const {
    const Index n = streambed_K.size();
    Vector C(n);
    for (Index i = 0; i < n; ++i) {
        C(i) = streambed_K(i) * stream_width(i) * stream_length(i)
               / (streambed_thickness(i) + 1e-30);
    }
    return C;
}

Vector StreamParameters::conductance_with_clogging() const {
    const Index n = streambed_K.size();
    Vector C(n);
    for (Index i = 0; i < n; ++i) {
        Real R_bed = streambed_thickness(i) / (streambed_K(i) + 1e-30);
        Real R_clog = has_clogging ?
            clogging_thickness(i) / (clogging_K(i) + 1e-30) : 0.0;
        Real area = stream_width(i) * stream_length(i);
        C(i) = area / (R_bed + R_clog + 1e-30);
    }
    return C;
}

// ============================================================================
// Parameters (variant wrapper)
// ============================================================================

Vector Parameters::pack_trainable() const {
    // Pack K and Sy as trainable parameters for 2D physics
    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined: {
            const auto& p = params_2d_;
            Index n = p.K.size();
            Vector packed(2 * n);
            packed.head(n) = p.K.array().log().matrix();  // Log-transform K
            packed.tail(n) = p.Sy;
            return packed;
        }
        case GoverningEquation::TwoLayer: {
            const auto& p = params_two_layer_;
            Index n = p.K1.size();
            Vector packed(3 * n);
            packed.segment(0, n) = p.K1.array().log().matrix();
            packed.segment(n, n) = p.K2.array().log().matrix();
            packed.segment(2 * n, n) = p.Sy;
            return packed;
        }
        default:
            return Vector();
    }
}

void Parameters::unpack_trainable(const Vector& packed) {
    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined: {
            auto& p = params_2d_;
            Index n = p.K.size();
            p.K = packed.head(n).array().exp().matrix();
            p.Sy = packed.tail(n);
            break;
        }
        case GoverningEquation::TwoLayer: {
            auto& p = params_two_layer_;
            Index n = p.K1.size();
            p.K1 = packed.segment(0, n).array().exp().matrix();
            p.K2 = packed.segment(n, n).array().exp().matrix();
            p.Sy = packed.segment(2 * n, n);
            break;
        }
        default:
            break;
    }
}

Index Parameters::n_trainable() const {
    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined:
            return 2 * params_2d_.K.size();
        case GoverningEquation::TwoLayer:
            return 3 * params_two_layer_.K1.size();
        default:
            return 0;
    }
}

std::vector<std::string> Parameters::trainable_names() const {
    std::vector<std::string> names;
    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined:
            names.push_back("log_K");
            names.push_back("Sy");
            break;
        case GoverningEquation::TwoLayer:
            names.push_back("log_K1");
            names.push_back("log_K2");
            names.push_back("Sy");
            break;
        default:
            break;
    }
    return names;
}

void Parameters::load(const std::string& filename, const Mesh& mesh) {
    // Simple CSV loader: first line is header, subsequent lines are values
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open parameter file: " + filename);
    }

    std::string line;
    std::getline(file, line);  // Header

    const Index n = mesh.n_cells();

    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined: {
            auto& p = params_2d_;
            p.K.resize(n);
            p.Sy.resize(n);
            p.Ss.resize(n);
            p.z_surface.resize(n);
            p.z_bottom.resize(n);

            for (Index i = 0; i < n && std::getline(file, line); ++i) {
                std::istringstream iss(line);
                char comma;
                iss >> p.K(i) >> comma >> p.Sy(i) >> comma
                    >> p.Ss(i) >> comma >> p.z_surface(i) >> comma >> p.z_bottom(i);
            }
            break;
        }
        case GoverningEquation::TwoLayer: {
            auto& p = params_two_layer_;
            p.K1.resize(n);
            p.K2.resize(n);
            p.Sy.resize(n);
            p.Ss2.resize(n);
            p.K_confining.resize(n);
            p.thickness_confining.resize(n);

            for (Index i = 0; i < n && std::getline(file, line); ++i) {
                std::istringstream iss(line);
                char comma;
                iss >> p.K1(i) >> comma >> p.K2(i) >> comma
                    >> p.Sy(i) >> comma >> p.Ss2(i) >> comma
                    >> p.K_confining(i) >> comma >> p.thickness_confining(i);
            }
            break;
        }
        default:
            break;
    }
}

void Parameters::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open parameter file for writing: " + filename);
    }

    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined: {
            const auto& p = params_2d_;
            file << "K,Sy,Ss,z_surface,z_bottom\n";
            for (Index i = 0; i < p.K.size(); ++i) {
                file << p.K(i) << "," << p.Sy(i) << ","
                     << p.Ss(i) << "," << p.z_surface(i) << "," << p.z_bottom(i) << "\n";
            }
            break;
        }
        case GoverningEquation::TwoLayer: {
            const auto& p = params_two_layer_;
            file << "K1,K2,Sy,Ss2,K_confining,thickness_confining\n";
            for (Index i = 0; i < p.K1.size(); ++i) {
                file << p.K1(i) << "," << p.K2(i) << ","
                     << p.Sy(i) << "," << p.Ss2(i) << ","
                     << p.K_confining(i) << "," << p.thickness_confining(i) << "\n";
            }
            break;
        }
        default:
            break;
    }
}

} // namespace dgw
