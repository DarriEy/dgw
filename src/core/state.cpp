/**
 * @file state.cpp
 * @brief Model state implementations
 */

#include "dgw/core/state.hpp"
#include "dgw/core/config.hpp"
#include <stdexcept>
#include <cmath>

namespace dgw {

// ============================================================================
// State2D
// ============================================================================

void State2D::initialize(const Mesh& mesh, Real initial_head) {
    const Index n = mesh.n_cells();
    head.resize(n);
    head.setConstant(initial_head);
    head_old = head;

    vadose_storage.resize(n);
    vadose_storage.setZero();

    recharge_flux.resize(n);
    recharge_flux.setZero();
    stream_exchange.resize(n);
    stream_exchange.setZero();
    boundary_flux.resize(n);
    boundary_flux.setZero();
    cell_storage_change.resize(n);
    cell_storage_change.setZero();

    time = 0.0;
    dt = 0.0;
}

void State2D::initialize(const Mesh& mesh, const Vector& initial_head) {
    const Index n = mesh.n_cells();
    if (initial_head.size() != n) {
        throw std::invalid_argument("initial_head size mismatch with mesh");
    }
    head = initial_head;
    head_old = head;

    vadose_storage.resize(n);
    vadose_storage.setZero();
    recharge_flux.resize(n);
    recharge_flux.setZero();
    stream_exchange.resize(n);
    stream_exchange.setZero();
    boundary_flux.resize(n);
    boundary_flux.setZero();
    cell_storage_change.resize(n);
    cell_storage_change.setZero();

    time = 0.0;
    dt = 0.0;
}

Real State2D::total_storage(const Mesh& mesh, const Vector& Sy) const {
    Real storage = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        Real A = mesh.cell_volume(i);
        Real sat_thick = std::max(head(i) - mesh.cell(i).z_bottom, 0.0);
        storage += Sy(i) * A * sat_thick;
    }
    return storage;
}

Real State2D::mass_balance_error() const {
    // Sum storage change vs net fluxes
    Real dS = cell_storage_change.sum();
    Real net_flux = recharge_flux.sum() - stream_exchange.sum() + boundary_flux.sum();
    return std::abs(dS - net_flux) / (std::abs(net_flux) + 1e-10);
}

// ============================================================================
// StateTwoLayer
// ============================================================================

void StateTwoLayer::initialize(const Mesh& mesh, Real h1_init, Real h2_init) {
    const Index n = mesh.n_cells();

    h1.resize(n);
    h1.setConstant(h1_init);
    h1_old = h1;

    h2.resize(n);
    h2.setConstant(h2_init);
    h2_old = h2;

    vadose_storage.resize(n);
    vadose_storage.setZero();
    leakage.resize(n);
    leakage.setZero();
    recharge.resize(n);
    recharge.setZero();
    pumping_1.resize(n);
    pumping_1.setZero();
    pumping_2.resize(n);
    pumping_2.setZero();
    stream_exchange.resize(n);
    stream_exchange.setZero();

    time = 0.0;
    dt = 0.0;
}

void StateTwoLayer::initialize(const Mesh& mesh, const Vector& h1_init, const Vector& h2_init) {
    const Index n = mesh.n_cells();

    h1 = h1_init;
    h1_old = h1;
    h2 = h2_init;
    h2_old = h2;

    vadose_storage.resize(n);
    vadose_storage.setZero();
    leakage.resize(n);
    leakage.setZero();
    recharge.resize(n);
    recharge.setZero();
    pumping_1.resize(n);
    pumping_1.setZero();
    pumping_2.resize(n);
    pumping_2.setZero();
    stream_exchange.resize(n);
    stream_exchange.setZero();

    time = 0.0;
    dt = 0.0;
}

void StateTwoLayer::advance_time(Real new_dt) {
    h1_old = h1;
    h2_old = h2;
    dt = new_dt;
    time += new_dt;
}

// ============================================================================
// StateMultiLayer
// ============================================================================

void StateMultiLayer::initialize(const MeshLayered& mesh, const std::vector<Real>& initial_heads) {
    n_layers = mesh.n_layers();
    Index n_cols = mesh.base_mesh().n_cells();

    if (static_cast<Index>(initial_heads.size()) != n_layers) {
        throw std::invalid_argument(
            "initial_heads size (" + std::to_string(initial_heads.size()) +
            ") must match number of layers (" + std::to_string(n_layers) + ")");
    }

    head.resize(n_layers);
    head_old.resize(n_layers);
    for (Index k = 0; k < n_layers; ++k) {
        head[k].resize(n_cols);
        head[k].setConstant(initial_heads[k]);
        head_old[k] = head[k];
    }

    vertical_leakage.resize(n_layers - 1);
    for (Index k = 0; k < n_layers - 1; ++k) {
        vertical_leakage[k].resize(n_cols);
        vertical_leakage[k].setZero();
    }

    recharge.resize(n_cols);
    recharge.setZero();

    pumping.resize(n_layers);
    for (Index k = 0; k < n_layers; ++k) {
        pumping[k].resize(n_cols);
        pumping[k].setZero();
    }

    stream_exchange.resize(n_cols);
    stream_exchange.setZero();

    time = 0.0;
    dt = 0.0;
}

void StateMultiLayer::advance_time(Real new_dt) {
    for (Index k = 0; k < n_layers; ++k) {
        head_old[k] = head[k];
    }
    dt = new_dt;
    time += new_dt;
}

// ============================================================================
// StateRichards3D
// ============================================================================

void StateRichards3D::initialize(const Mesh3D& mesh, Real initial_psi) {
    const Index n = mesh.n_cells();
    pressure_head.resize(n);
    pressure_head.setConstant(initial_psi);
    pressure_head_old = pressure_head;

    water_content.resize(n);
    water_content.setZero();
    hydraulic_conductivity.resize(n);
    hydraulic_conductivity.setZero();
    specific_moisture_capacity.resize(n);
    specific_moisture_capacity.setZero();
    saturation.resize(n);
    saturation.setZero();

    recharge.resize(n);
    recharge.setZero();
    evapotranspiration.resize(n);
    evapotranspiration.setZero();
    bottom_flux.resize(n);
    bottom_flux.setZero();
    lateral_flux.resize(n);
    lateral_flux.setZero();

    time = 0.0;
    dt = 0.0;
}

void StateRichards3D::initialize(const Mesh3D& mesh, const Vector& initial_psi) {
    const Index n = mesh.n_cells();
    if (initial_psi.size() != n) {
        throw std::invalid_argument(
            "initial_psi size (" + std::to_string(initial_psi.size()) +
            ") must match mesh n_cells (" + std::to_string(n) + ")");
    }
    pressure_head = initial_psi;
    pressure_head_old = pressure_head;

    water_content.resize(n);
    water_content.setZero();
    hydraulic_conductivity.resize(n);
    hydraulic_conductivity.setZero();
    specific_moisture_capacity.resize(n);
    specific_moisture_capacity.setZero();
    saturation.resize(n);
    saturation.setZero();

    recharge.resize(n);
    recharge.setZero();
    evapotranspiration.resize(n);
    evapotranspiration.setZero();
    bottom_flux.resize(n);
    bottom_flux.setZero();
    lateral_flux.resize(n);
    lateral_flux.setZero();

    time = 0.0;
    dt = 0.0;
}

void StateRichards3D::initialize_hydrostatic(const Mesh3D& mesh, Real water_table_elevation) {
    const Index n = mesh.n_cells();
    pressure_head.resize(n);
    for (Index i = 0; i < n; ++i) {
        Real z = mesh.cell_centroid(i).z();
        pressure_head(i) = water_table_elevation - z;
    }
    pressure_head_old = pressure_head;

    water_content.resize(n);
    water_content.setZero();
    hydraulic_conductivity.resize(n);
    hydraulic_conductivity.setZero();
    specific_moisture_capacity.resize(n);
    specific_moisture_capacity.setZero();
    saturation.resize(n);
    saturation.setZero();

    recharge.resize(n);
    recharge.setZero();
    evapotranspiration.resize(n);
    evapotranspiration.setZero();
    bottom_flux.resize(n);
    bottom_flux.setZero();
    lateral_flux.resize(n);
    lateral_flux.setZero();

    time = 0.0;
    dt = 0.0;
}

void StateRichards3D::advance_time(Real new_dt) {
    pressure_head_old = pressure_head;
    dt = new_dt;
    time += new_dt;
}

void StateRichards3D::update_constitutive(
    const Vector& theta_r, const Vector& theta_s,
    const Vector& alpha, const Vector& n_vg,
    const Vector& K_sat, RetentionModel model
) {
    const Index n = pressure_head.size();
    water_content.resize(n);
    hydraulic_conductivity.resize(n);
    specific_moisture_capacity.resize(n);
    saturation.resize(n);

    for (Index i = 0; i < n; ++i) {
        Real psi = pressure_head(i);

        if (model == RetentionModel::VanGenuchten) {
            Real m = 1.0 - 1.0 / n_vg(i);

            // Effective saturation
            if (psi >= 0.0) {
                saturation(i) = 1.0;
                water_content(i) = theta_s(i);
                hydraulic_conductivity(i) = K_sat(i);
                specific_moisture_capacity(i) = 0.0;
            } else {
                Real alpha_psi_n = std::pow(alpha(i) * std::abs(psi), n_vg(i));
                Real Se = std::pow(1.0 + alpha_psi_n, -m);
                saturation(i) = Se;
                water_content(i) = theta_r(i) + (theta_s(i) - theta_r(i)) * Se;

                // Mualem conductivity
                Real Se_1_m = std::pow(Se, 1.0 / m);
                Real term = 1.0 - std::pow(1.0 - Se_1_m, m);
                Real Kr = std::sqrt(Se) * term * term;
                hydraulic_conductivity(i) = K_sat(i) * Kr;

                // Moisture capacity
                specific_moisture_capacity(i) =
                    (theta_s(i) - theta_r(i)) * alpha(i) * m * n_vg(i)
                    * std::pow(alpha(i) * std::abs(psi), n_vg(i) - 1.0)
                    * std::pow(1.0 + alpha_psi_n, -m - 1.0);
            }
        } else if (model == RetentionModel::BrooksCorey) {
            Real psi_b = alpha(i);  // Reuse alpha as bubbling pressure
            Real lambda = n_vg(i);  // Reuse n_vg as lambda

            if (psi >= -psi_b) {
                saturation(i) = 1.0;
                water_content(i) = theta_s(i);
                hydraulic_conductivity(i) = K_sat(i);
                specific_moisture_capacity(i) = 0.0;
            } else {
                Real Se = std::pow(psi_b / std::abs(psi), lambda);
                saturation(i) = Se;
                water_content(i) = theta_r(i) + (theta_s(i) - theta_r(i)) * Se;
                Real Kr = std::pow(Se, (2.0 + 3.0 * lambda) / lambda);
                hydraulic_conductivity(i) = K_sat(i) * Kr;
                specific_moisture_capacity(i) =
                    (theta_s(i) - theta_r(i)) * lambda * Se / std::abs(psi);
            }
        } else if (model == RetentionModel::ClappHornberger) {
            // Clapp-Hornberger model: uses same parameter slots as Brooks-Corey
            // alpha -> psi_s (air entry pressure), n_vg -> b (pore size index)
            Real psi_s = alpha(i);
            Real b = n_vg(i);

            if (psi >= -psi_s) {
                saturation(i) = 1.0;
                water_content(i) = theta_s(i);
                hydraulic_conductivity(i) = K_sat(i);
                specific_moisture_capacity(i) = 0.0;
            } else {
                Real Se = std::pow(psi_s / std::abs(psi), 1.0 / b);
                saturation(i) = Se;
                water_content(i) = theta_r(i) + (theta_s(i) - theta_r(i)) * Se;
                Real Kr = std::pow(Se, 2.0 * b + 3.0);
                hydraulic_conductivity(i) = K_sat(i) * Kr;
                specific_moisture_capacity(i) =
                    (theta_s(i) - theta_r(i)) * Se / (b * std::abs(psi));
            }
        } else if (model == RetentionModel::Tabulated) {
            // Tabulated model not yet implemented - fall back to saturated
            saturation(i) = 1.0;
            water_content(i) = theta_s(i);
            hydraulic_conductivity(i) = K_sat(i);
            specific_moisture_capacity(i) = 0.0;
        }
    }
}

Real StateRichards3D::total_water_storage(const Mesh3D& mesh) const {
    Real total = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        total += water_content(i) * mesh.cell_volume(i);
    }
    return total;
}

// ============================================================================
// Combined State
// ============================================================================

void State::initialize(const Mesh& mesh, const Config& config) {
    physics_ = config.physics.governing_equation;

    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq: {
            data_ = State2D{};
            auto& s = as_2d();
            s.initialize(mesh, 0.0);
            break;
        }
        case GoverningEquation::Confined: {
            data_ = State2D{};
            auto& s = as_2d();
            s.initialize(mesh, 0.0);
            break;
        }
        case GoverningEquation::TwoLayer: {
            data_ = StateTwoLayer{};
            auto& s = as_two_layer();
            s.initialize(mesh, 0.0, 0.0);
            break;
        }
        case GoverningEquation::MultiLayer: {
            data_ = StateMultiLayer{};
            // MultiLayer requires MeshLayered, defer full init
            break;
        }
        case GoverningEquation::Richards3D: {
            data_ = StateRichards3D{};
            // Richards requires Mesh3D, defer full init
            break;
        }
    }
}

Real State::time() const {
    return std::visit([](const auto& s) { return s.time; }, data_);
}

Vector& State::primary_state() {
    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined:
            return as_2d().head;
        case GoverningEquation::TwoLayer:
            return as_two_layer().h1;  // Return layer 1 head
        case GoverningEquation::MultiLayer:
            return as_multi_layer().head[0];  // Return top layer
        case GoverningEquation::Richards3D:
            return as_richards().pressure_head;
        default:
            return as_2d().head;
    }
}

const Vector& State::primary_state() const {
    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined:
            return as_2d().head;
        case GoverningEquation::TwoLayer:
            return as_two_layer().h1;
        case GoverningEquation::MultiLayer:
            return as_multi_layer().head[0];
        case GoverningEquation::Richards3D:
            return as_richards().pressure_head;
        default:
            return as_2d().head;
    }
}

Vector State::pack() const {
    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined: {
            return as_2d().head;
        }
        case GoverningEquation::TwoLayer: {
            const auto& s = as_two_layer();
            Vector packed(s.h1.size() + s.h2.size());
            packed.head(s.h1.size()) = s.h1;
            packed.tail(s.h2.size()) = s.h2;
            return packed;
        }
        case GoverningEquation::MultiLayer: {
            const auto& s = as_multi_layer();
            Index total = 0;
            for (const auto& h : s.head) total += h.size();
            Vector packed(total);
            Index offset = 0;
            for (const auto& h : s.head) {
                packed.segment(offset, h.size()) = h;
                offset += h.size();
            }
            return packed;
        }
        case GoverningEquation::Richards3D: {
            return as_richards().pressure_head;
        }
        default:
            return Vector();
    }
}

void State::unpack(const Vector& packed) {
    switch (physics_) {
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined: {
            as_2d().head = packed;
            break;
        }
        case GoverningEquation::TwoLayer: {
            auto& s = as_two_layer();
            Index n = s.h1.size();
            s.h1 = packed.head(n);
            s.h2 = packed.tail(packed.size() - n);
            break;
        }
        case GoverningEquation::MultiLayer: {
            auto& s = as_multi_layer();
            Index offset = 0;
            for (auto& h : s.head) {
                h = packed.segment(offset, h.size());
                offset += h.size();
            }
            break;
        }
        case GoverningEquation::Richards3D: {
            as_richards().pressure_head = packed;
            break;
        }
        default:
            break;
    }
}

} // namespace dgw
