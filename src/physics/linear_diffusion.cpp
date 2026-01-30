/**
 * @file linear_diffusion.cpp
 * @brief Linear diffusion equation solver implementation
 */

#include "dgw/physics/linear_diffusion.hpp"
#include <cmath>

namespace dgw {

LinearDiffusion::LinearDiffusion(const PhysicsDecisions& decisions)
    : decisions_(decisions) {}

void LinearDiffusion::compute_residual(
    const State& state, const Parameters& params,
    const Mesh& mesh, Real dt, Vector& residual
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();
    const Index n = mesh.n_cells();

    residual.resize(n);
    residual.setZero();

    // Storage: Sy * A * (h - h_old) / dt
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        residual(i) = p.Sy(i) * A_i * (s.head(i) - s.head_old(i)) / dt;
    }

    // Flux: constant T = K * (z_surface - z_bottom)
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);

        Real b_i = p.z_surface(i) - p.z_bottom(i);
        Real b_j = p.z_surface(j) - p.z_bottom(j);
        Real T_i = p.K(i) * std::max(b_i, 0.0);
        Real T_j = p.K(j) * std::max(b_j, 0.0);
        Real T_ij = (T_i + T_j > 0.0) ? 2.0 * T_i * T_j / (T_i + T_j) : 0.0;

        Real W = face.area;
        Real L = face.distance;
        Real flux = T_ij * W * (s.head(j) - s.head(i)) / L;

        residual(i) -= flux;
        residual(j) += flux;
    }

    // Sources
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        if (recharge_.size() > 0) residual(i) -= recharge_(i) * A_i;
        if (pumping_.size() > 0) residual(i) += pumping_(i);
    }
}

void LinearDiffusion::compute_jacobian(
    const State& state, const Parameters& params,
    const Mesh& mesh, Real dt, SparseMatrix& jacobian
) const {
    const auto& p = params.as_2d();
    const Index n = mesh.n_cells();

    std::vector<SparseTriplet> triplets;
    triplets.reserve(n * 7);

    // Diagonal: storage
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        triplets.emplace_back(i, i, p.Sy(i) * A_i / dt);
    }

    // Off-diagonal: constant T fluxes
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);
        Real b_i = p.z_surface(i) - p.z_bottom(i);
        Real b_j = p.z_surface(j) - p.z_bottom(j);
        Real T_i = p.K(i) * std::max(b_i, 0.0);
        Real T_j = p.K(j) * std::max(b_j, 0.0);
        Real T_ij = (T_i + T_j > 0.0) ? 2.0 * T_i * T_j / (T_i + T_j) : 0.0;

        Real coef = T_ij * face.area / face.distance;

        // dF_i/dh_i += coef, dF_i/dh_j -= coef
        triplets.emplace_back(i, i, coef);
        triplets.emplace_back(i, j, -coef);
        triplets.emplace_back(j, j, coef);
        triplets.emplace_back(j, i, -coef);
    }

    jacobian.resize(n, n);
    jacobian.setFromTriplets(triplets.begin(), triplets.end());
}

void LinearDiffusion::compute_fluxes(
    const State& state, const Parameters& params,
    const Mesh& mesh, Vector& face_fluxes
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();

    face_fluxes.resize(mesh.n_faces());
    face_fluxes.setZero();

    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);
        Real b_i = p.z_surface(i) - p.z_bottom(i);
        Real b_j = p.z_surface(j) - p.z_bottom(j);
        Real T_i = p.K(i) * std::max(b_i, 0.0);
        Real T_j = p.K(j) * std::max(b_j, 0.0);
        Real T_ij = (T_i + T_j > 0.0) ? 2.0 * T_i * T_j / (T_i + T_j) : 0.0;

        face_fluxes(f) = T_ij * face.area * (s.head(j) - s.head(i)) / face.distance;
    }
}

void LinearDiffusion::initialize_state(
    const Mesh& mesh, const Parameters& params,
    const Config& config, State& state
) const {
    auto& s = state.as_2d();
    const Index n = mesh.n_cells();

    s.head.resize(n);
    s.head_old.resize(n);
    s.vadose_storage.resize(n);
    s.vadose_storage.setZero();

    for (Index i = 0; i < n; ++i) {
        s.head(i) = mesh.cell(i).z_surface;
    }
    s.head_old = s.head;
    s.time = config.time.start_time;
}

SparseMatrix LinearDiffusion::allocate_jacobian(const Mesh& mesh) const {
    const Index n = mesh.n_cells();
    std::vector<SparseTriplet> triplets;
    triplets.reserve(n * 7);

    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 0.0);
        for (Index j : mesh.cell_neighbors(i)) {
            triplets.emplace_back(i, j, 0.0);
        }
    }

    SparseMatrix J(n, n);
    J.setFromTriplets(triplets.begin(), triplets.end());
    return J;
}

void LinearDiffusion::set_recharge(const Vector& r) { recharge_ = r; }
void LinearDiffusion::set_stream_stage(const Vector& s) { stream_stage_ = s; }
void LinearDiffusion::set_pumping(const Vector& p) { pumping_ = p; }

void LinearDiffusion::apply_boundary_conditions(
    const State& state,
    const Mesh& mesh, const Parameters& params,
    Vector& residual, SparseMatrix& jacobian
) const {
    const auto& s = state.as_2d();

    for (Index f : mesh.boundary_faces()) {
        const Face& face = mesh.face(f);
        Index cell = (face.cell_left >= 0) ? face.cell_left : face.cell_right;

        switch (face.bc_type) {
            case BoundaryType::NoFlow:
                break;
            case BoundaryType::FixedHead: {
                Real h_bc = face.bc_value;
                residual(cell) = s.head(cell) - h_bc;
                for (SparseMatrix::InnerIterator it(jacobian, cell); it; ++it) {
                    it.valueRef() = (it.col() == cell) ? 1.0 : 0.0;
                }
                break;
            }
            case BoundaryType::FixedFlux:
                residual(cell) -= face.bc_value;
                break;
            default:
                break;
        }
    }
}

Vector LinearDiffusion::water_table_depth(const State& state, const Mesh& mesh) const {
    const auto& s = state.as_2d();
    Vector depth(mesh.n_cells());
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        depth(i) = std::max(mesh.cell(i).z_surface - s.head(i), 0.0);
    }
    return depth;
}

Vector LinearDiffusion::stream_exchange(
    const State& state, const Parameters& params, const Mesh& mesh
) const {
    Vector exchange(mesh.n_cells());
    exchange.setZero();
    return exchange;
}

Real LinearDiffusion::total_storage(
    const State& state, const Parameters& params, const Mesh& mesh
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();
    Real storage = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        storage += p.Sy(i) * mesh.cell_volume(i) * std::max(s.head(i) - p.z_bottom(i), 0.0);
    }
    return storage;
}

Real LinearDiffusion::mass_balance_error(
    const State& state, const Parameters& params, const Mesh& mesh, Real dt
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();

    Real dS = 0.0;
    Real Q_in = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        Real A = mesh.cell_volume(i);
        dS += p.Sy(i) * A * (s.head(i) - s.head_old(i));
        if (recharge_.size() > 0) Q_in += recharge_(i) * A * dt;
    }
    if (pumping_.size() > 0) Q_in -= pumping_.sum() * dt;

    return std::abs(dS - Q_in) / (std::abs(Q_in) + 1e-10);
}

} // namespace dgw
