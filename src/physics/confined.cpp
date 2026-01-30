/**
 * @file confined.cpp
 * @brief Confined aquifer solver implementation
 */

#include "dgw/physics/confined.hpp"
#include <cmath>

namespace dgw {

ConfinedSolver::ConfinedSolver(const PhysicsDecisions& decisions)
    : decisions_(decisions) {}

void ConfinedSolver::compute_residual(
    const State& state, const Parameters& params,
    const Mesh& mesh, Real dt, Vector& residual
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();
    const Index n = mesh.n_cells();

    residual.resize(n);
    residual.setZero();

    // Storage: S * A * (h - h_old) / dt where S = Ss * b
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        Real b = p.z_surface(i) - p.z_bottom(i);
        Real S = p.Ss(i) * b;
        residual(i) = S * A_i * (s.head(i) - s.head_old(i)) / dt;
    }

    // Flux terms with constant transmissivity T = K * b
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);
        Real b_i = p.z_surface(i) - p.z_bottom(i);
        Real b_j = p.z_surface(j) - p.z_bottom(j);
        Real T_i = p.K(i) * b_i;
        Real T_j = p.K(j) * b_j;
        Real T_ij = confined_kernels::intercell_T(T_i, T_j);

        Real W = face.area;
        Real L = face.distance;
        Real flux = T_ij * W * (s.head(j) - s.head(i)) / L;

        residual(i) -= flux;
        residual(j) += flux;
    }

    // Source terms
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        if (recharge_.size() > 0) residual(i) -= recharge_(i) * A_i;
        if (pumping_.size() > 0) residual(i) += pumping_(i);
    }

    // Stream exchange
    if (stream_stage_.size() > 0) {
        for (Index cell_id : mesh.river_cells()) {
            Real h_gw = s.head(cell_id);
            Real h_stream = stream_stage_(cell_id);
            Real C = params.stream().conductance()(cell_id);
            residual(cell_id) += C * (h_gw - h_stream);
        }
    }
}

void ConfinedSolver::compute_jacobian(
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
        Real b = p.z_surface(i) - p.z_bottom(i);
        Real S = p.Ss(i) * b;
        triplets.emplace_back(i, i, S * A_i / dt);
    }

    // Off-diagonal: constant T
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);
        Real b_i = p.z_surface(i) - p.z_bottom(i);
        Real b_j = p.z_surface(j) - p.z_bottom(j);
        Real T_i = p.K(i) * b_i;
        Real T_j = p.K(j) * b_j;
        Real T_ij = confined_kernels::intercell_T(T_i, T_j);

        Real coef = T_ij * face.area / face.distance;

        triplets.emplace_back(i, i, coef);
        triplets.emplace_back(i, j, -coef);
        triplets.emplace_back(j, j, coef);
        triplets.emplace_back(j, i, -coef);
    }

    // Stream exchange diagonal
    if (stream_stage_.size() > 0) {
        for (Index cell_id : mesh.river_cells()) {
            Real C = params.stream().conductance()(cell_id);
            triplets.emplace_back(cell_id, cell_id, C);
        }
    }

    jacobian.resize(n, n);
    jacobian.setFromTriplets(triplets.begin(), triplets.end());
}

void ConfinedSolver::compute_fluxes(
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
        Real T_ij = confined_kernels::intercell_T(p.K(i) * b_i, p.K(j) * b_j);

        face_fluxes(f) = T_ij * face.area * (s.head(j) - s.head(i)) / face.distance;
    }
}

void ConfinedSolver::initialize_state(
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

SparseMatrix ConfinedSolver::allocate_jacobian(const Mesh& mesh) const {
    const Index n = mesh.n_cells();
    std::vector<SparseTriplet> triplets;
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

void ConfinedSolver::set_recharge(const Vector& r) { recharge_ = r; }
void ConfinedSolver::set_stream_stage(const Vector& s) { stream_stage_ = s; }
void ConfinedSolver::set_pumping(const Vector& p) { pumping_ = p; }

void ConfinedSolver::apply_boundary_conditions(
    const State& state,
    const Mesh& mesh, const Parameters& params,
    Vector& residual, SparseMatrix& jacobian
) const {
    const auto& s = state.as_2d();

    for (Index f : mesh.boundary_faces()) {
        const Face& face = mesh.face(f);
        Index cell = (face.cell_left >= 0) ? face.cell_left : face.cell_right;
        switch (face.bc_type) {
            case BoundaryType::NoFlow: break;
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
            default: break;
        }
    }
}

Vector ConfinedSolver::water_table_depth(const State& state, const Mesh& mesh) const {
    // For confined aquifer, "water table depth" is distance from surface
    // to the potentiometric surface
    const auto& s = state.as_2d();
    Vector depth(mesh.n_cells());
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        depth(i) = std::max(mesh.cell(i).z_surface - s.head(i), 0.0);
    }
    return depth;
}

Vector ConfinedSolver::potentiometric_surface(const State& state) const {
    return state.as_2d().head;
}

Vector ConfinedSolver::stream_exchange(
    const State& state, const Parameters& params, const Mesh& mesh
) const {
    const auto& s = state.as_2d();
    Vector exchange(mesh.n_cells());
    exchange.setZero();

    if (stream_stage_.size() == 0) return exchange;

    for (Index cell_id : mesh.river_cells()) {
        Real C = params.stream().conductance()(cell_id);
        exchange(cell_id) = C * (s.head(cell_id) - stream_stage_(cell_id));
    }
    return exchange;
}

Real ConfinedSolver::total_storage(
    const State& state, const Parameters& params, const Mesh& mesh
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();
    Real storage = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        Real b = p.z_surface(i) - p.z_bottom(i);
        storage += p.Ss(i) * b * mesh.cell_volume(i) * s.head(i);
    }
    return storage;
}

Real ConfinedSolver::mass_balance_error(
    const State& state, const Parameters& params, const Mesh& mesh, Real dt
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();

    Real dS = 0.0;
    Real Q_in = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        Real A = mesh.cell_volume(i);
        Real b = p.z_surface(i) - p.z_bottom(i);
        dS += p.Ss(i) * b * A * (s.head(i) - s.head_old(i));
        if (recharge_.size() > 0) Q_in += recharge_(i) * A * dt;
    }
    if (pumping_.size() > 0) Q_in -= pumping_.sum() * dt;

    Vector exch = stream_exchange(state, params, mesh);
    Q_in -= exch.sum() * dt;

    return std::abs(dS - Q_in) / (std::abs(Q_in) + 1e-10);
}

void ConfinedSolver::precompute_system_matrix(
    const Parameters& params, const Mesh& mesh, Real dt, SparseMatrix& A
) const {
    const auto& p = params.as_2d();
    const Index n = mesh.n_cells();

    std::vector<SparseTriplet> triplets;
    triplets.reserve(n * 7);

    for (Index i = 0; i < n; ++i) {
        Real area = mesh.cell_volume(i);
        Real b = p.z_surface(i) - p.z_bottom(i);
        triplets.emplace_back(i, i, p.Ss(i) * b * area / dt);
    }

    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);
        Real b_i = p.z_surface(i) - p.z_bottom(i);
        Real b_j = p.z_surface(j) - p.z_bottom(j);
        Real T_ij = confined_kernels::intercell_T(p.K(i) * b_i, p.K(j) * b_j);
        Real coef = T_ij * face.area / face.distance;

        triplets.emplace_back(i, i, coef);
        triplets.emplace_back(i, j, -coef);
        triplets.emplace_back(j, j, coef);
        triplets.emplace_back(j, i, -coef);
    }

    A.resize(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());

    cached_matrix_ = A;
    cached_dt_ = dt;
    matrix_cached_ = true;
}

// Kernel implementations
namespace confined_kernels {

void residual_kernel(
    const Real* __restrict__ head,
    const Real* __restrict__ head_old,
    const Real* __restrict__ T,
    const Real* __restrict__ S,
    const Real* __restrict__ source,
    const Index* __restrict__ cell_neighbors_data,
    const Index* __restrict__ cell_neighbors_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Real* __restrict__ cell_area,
    Index n_cells,
    Real dt,
    Real* __restrict__ residual
) {
    for (Index i = 0; i < n_cells; ++i) {
        residual[i] = S[i] * cell_area[i] * (head[i] - head_old[i]) / dt;
        if (source) residual[i] -= source[i] * cell_area[i];
    }

    for (Index i = 0; i < n_cells; ++i) {
        Index start = cell_neighbors_ptr[i];
        Index end = cell_neighbors_ptr[i + 1];

        for (Index k = start; k < end; ++k) {
            Index j = cell_neighbors_data[k];
            if (j <= i) continue;

            Real T_ij = intercell_T(T[i], T[j]);
            Real W = face_area[k];
            Real L = face_distance[k];
            Real flux = T_ij * W * (head[j] - head[i]) / L;

            residual[i] -= flux;
            residual[j] += flux;
        }
    }
}

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
) {
    nnz = 0;

    for (Index i = 0; i < n_cells; ++i) {
        // Diagonal: storage
        Real diag = S[i] * cell_area[i] / dt;

        Index start = cell_neighbors_ptr[i];
        Index end = cell_neighbors_ptr[i + 1];

        for (Index k = start; k < end; ++k) {
            Index j = cell_neighbors_data[k];
            Real T_ij = intercell_T(T[i], T[j]);
            Real coef = T_ij * face_area[k] / face_distance[k];

            diag += coef;

            matrix_values[nnz] = -coef;
            matrix_rows[nnz] = i;
            matrix_cols[nnz] = j;
            nnz++;
        }

        matrix_values[nnz] = diag;
        matrix_rows[nnz] = i;
        matrix_cols[nnz] = i;
        nnz++;
    }
}

} // namespace confined_kernels

} // namespace dgw
