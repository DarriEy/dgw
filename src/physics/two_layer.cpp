/**
 * @file two_layer.cpp
 * @brief Two-layer coupled groundwater solver implementation
 */

#include "dgw/physics/two_layer.hpp"
#include <cmath>

namespace dgw {

TwoLayerSolver::TwoLayerSolver(const PhysicsDecisions& decisions)
    : decisions_(decisions) {
    layer1_solver_ = std::make_unique<BoussinesqSolver>(decisions);
    layer2_solver_ = std::make_unique<ConfinedSolver>(decisions);
}

void TwoLayerSolver::compute_residual(
    const State& state, const Parameters& params,
    const Mesh& mesh, Real dt, Vector& residual
) const {
    const auto& s = state.as_two_layer();
    const auto& p = params.as_two_layer();
    const Index n = mesh.n_cells();

    residual.resize(2 * n);
    residual.setZero();

    assemble_monolithic_residual(s, p, mesh, dt, residual);
}

void TwoLayerSolver::compute_jacobian(
    const State& state, const Parameters& params,
    const Mesh& mesh, Real dt, SparseMatrix& jacobian
) const {
    const auto& s = state.as_two_layer();
    const auto& p = params.as_two_layer();
    const Index n = mesh.n_cells();

    assemble_monolithic_jacobian(s, p, mesh, dt, jacobian);
}

void TwoLayerSolver::compute_fluxes(
    const State& state, const Parameters& params,
    const Mesh& mesh, Vector& face_fluxes
) const {
    const auto& s = state.as_two_layer();
    const auto& p = params.as_two_layer();

    face_fluxes.resize(mesh.n_faces());
    face_fluxes.setZero();

    // Layer 1 fluxes (Boussinesq)
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);

        Real T_i = p.K1(i) * std::max(s.h1(i) - p.z_bottom_1(i), 0.0);
        Real T_j = p.K1(j) * std::max(s.h1(j) - p.z_bottom_1(j), 0.0);
        Real T_ij = (T_i + T_j > 0.0) ? 2.0 * T_i * T_j / (T_i + T_j) : 0.0;

        face_fluxes(f) = T_ij * face.area * (s.h1(j) - s.h1(i)) / face.distance;
    }
}

void TwoLayerSolver::initialize_state(
    const Mesh& mesh, const Parameters& params,
    const Config& config, State& state
) const {
    auto& s = state.as_two_layer();
    const Index n = mesh.n_cells();

    s.h1.resize(n);
    s.h1_old.resize(n);
    s.h2.resize(n);
    s.h2_old.resize(n);
    s.vadose_storage.resize(n);
    s.vadose_storage.setZero();
    s.leakage.resize(n);
    s.leakage.setZero();
    s.recharge.resize(n);
    s.recharge.setZero();
    s.pumping_1.resize(n);
    s.pumping_1.setZero();
    s.pumping_2.resize(n);
    s.pumping_2.setZero();
    s.stream_exchange.resize(n);
    s.stream_exchange.setZero();

    for (Index i = 0; i < n; ++i) {
        s.h1(i) = mesh.cell(i).z_surface;
        s.h2(i) = mesh.cell(i).z_surface;
    }
    s.h1_old = s.h1;
    s.h2_old = s.h2;
    s.time = config.time.start_time;
}

SparseMatrix TwoLayerSolver::allocate_jacobian(const Mesh& mesh) const {
    const Index n = mesh.n_cells();
    const Index dim = 2 * n;

    std::vector<SparseTriplet> triplets;
    triplets.reserve(dim * 8);

    for (Index i = 0; i < n; ++i) {
        // Layer 1 block
        triplets.emplace_back(i, i, 0.0);
        for (Index j : mesh.cell_neighbors(i)) {
            triplets.emplace_back(i, j, 0.0);
        }
        // Coupling: layer 1 to layer 2
        triplets.emplace_back(i, n + i, 0.0);

        // Layer 2 block
        triplets.emplace_back(n + i, n + i, 0.0);
        for (Index j : mesh.cell_neighbors(i)) {
            triplets.emplace_back(n + i, n + j, 0.0);
        }
        // Coupling: layer 2 to layer 1
        triplets.emplace_back(n + i, i, 0.0);
    }

    SparseMatrix J(dim, dim);
    J.setFromTriplets(triplets.begin(), triplets.end());
    return J;
}

void TwoLayerSolver::set_recharge(const Vector& r) { recharge_ = r; }
void TwoLayerSolver::set_stream_stage(const Vector& s) { stream_stage_ = s; }
void TwoLayerSolver::set_pumping(const Vector& p) {
    pumping_ = p;
    pumping_layer1_ = p;  // Default: pumping from layer 1
}

void TwoLayerSolver::set_pumping_layer(Index layer, const Vector& pumping) {
    if (layer == 0) pumping_layer1_ = pumping;
    else pumping_layer2_ = pumping;
}

void TwoLayerSolver::apply_boundary_conditions(
    const State& state,
    const Mesh& mesh, const Parameters& params,
    Vector& residual, SparseMatrix& jacobian
) const {
    const auto& s = state.as_two_layer();
    const Index n = mesh.n_cells();

    for (Index f : mesh.boundary_faces()) {
        const Face& face = mesh.face(f);
        Index cell = (face.cell_left >= 0) ? face.cell_left : face.cell_right;

        switch (face.bc_type) {
            case BoundaryType::NoFlow: break;
            case BoundaryType::FixedHead: {
                Real h_bc = face.bc_value;
                // Apply to both layers: replace equation with h - h_bc = 0
                residual(cell) = s.h1(cell) - h_bc;
                residual(n + cell) = s.h2(cell) - h_bc;
                // Zero out Jacobian rows and set diagonal to 1
                for (SparseMatrix::InnerIterator it(jacobian, cell); it; ++it) {
                    it.valueRef() = (it.row() == cell) ? 1.0 : 0.0;
                }
                for (SparseMatrix::InnerIterator it(jacobian, n + cell); it; ++it) {
                    it.valueRef() = (it.row() == static_cast<SparseMatrix::StorageIndex>(n + cell)) ? 1.0 : 0.0;
                }
                break;
            }
            default: break;
        }
    }
}

Vector TwoLayerSolver::water_table_depth(const State& state, const Mesh& mesh) const {
    const auto& s = state.as_two_layer();
    Vector depth(mesh.n_cells());
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        depth(i) = std::max(mesh.cell(i).z_surface - s.h1(i), 0.0);
    }
    return depth;
}

Vector TwoLayerSolver::confined_head(const State& state) const {
    return state.as_two_layer().h2;
}

Vector TwoLayerSolver::leakage(
    const State& state, const Parameters& params, const Mesh& mesh
) const {
    const auto& s = state.as_two_layer();
    const auto& p = params.as_two_layer();
    Vector leak = compute_leakance(p);

    const Index n = mesh.n_cells();
    Vector result(n);
    for (Index i = 0; i < n; ++i) {
        result(i) = leak(i) * (s.h1(i) - s.h2(i)) * mesh.cell_volume(i);
    }
    return result;
}

Vector TwoLayerSolver::stream_exchange(
    const State& state, const Parameters& params, const Mesh& mesh
) const {
    const auto& s = state.as_two_layer();
    Vector exchange(mesh.n_cells());
    exchange.setZero();

    if (stream_stage_.size() == 0) return exchange;

    for (Index cell_id : mesh.river_cells()) {
        Real C = params.stream().conductance()(cell_id);
        exchange(cell_id) = C * (s.h1(cell_id) - stream_stage_(cell_id));
    }
    return exchange;
}

Real TwoLayerSolver::total_storage(
    const State& state, const Parameters& params, const Mesh& mesh
) const {
    auto [s1, s2] = storage_by_layer(state, params, mesh);
    return s1 + s2;
}

std::pair<Real, Real> TwoLayerSolver::storage_by_layer(
    const State& state, const Parameters& params, const Mesh& mesh
) const {
    const auto& s = state.as_two_layer();
    const auto& p = params.as_two_layer();

    Real storage1 = 0.0, storage2 = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        Real A = mesh.cell_volume(i);
        Real sat1 = std::max(s.h1(i) - p.z_bottom_1(i), 0.0);
        storage1 += p.Sy(i) * A * sat1;
        storage2 += p.Ss2(i) * p.thickness_2(i) * A * s.h2(i);
    }
    return {storage1, storage2};
}

Real TwoLayerSolver::mass_balance_error(
    const State& state, const Parameters& params, const Mesh& mesh, Real dt
) const {
    const auto& s = state.as_two_layer();
    const auto& p = params.as_two_layer();

    Real dS = 0.0;
    Real Q_in = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        Real A = mesh.cell_volume(i);
        Real sat1 = std::max(s.h1(i) - p.z_bottom_1(i), 0.0);
        Real sat1_old = std::max(s.h1_old(i) - p.z_bottom_1(i), 0.0);
        dS += p.Sy(i) * A * (sat1 - sat1_old);
        dS += p.Ss2(i) * p.thickness_2(i) * A * (s.h2(i) - s.h2_old(i));
        if (recharge_.size() > 0) Q_in += recharge_(i) * A * dt;
    }
    if (pumping_layer1_.size() > 0) Q_in -= pumping_layer1_.sum() * dt;
    if (pumping_layer2_.size() > 0) Q_in -= pumping_layer2_.sum() * dt;

    Vector exch = stream_exchange(state, params, mesh);
    Q_in -= exch.sum() * dt;

    return std::abs(dS - Q_in) / (std::abs(Q_in) + 1e-10);
}

// Private helpers

Vector TwoLayerSolver::compute_leakance(const ParametersTwoLayer& params) const {
    return params.leakance();
}

Vector TwoLayerSolver::compute_leakage(
    const Vector& h1, const Vector& h2,
    const Vector& leakance, const Mesh& mesh
) const {
    const Index n = h1.size();
    Vector leak(n);
    for (Index i = 0; i < n; ++i) {
        leak(i) = leakance(i) * (h1(i) - h2(i)) * mesh.cell_volume(i);
    }
    return leak;
}

void TwoLayerSolver::assemble_monolithic_residual(
    const StateTwoLayer& state, const ParametersTwoLayer& params,
    const Mesh& mesh, Real dt, Vector& residual
) const {
    const Index n = mesh.n_cells();
    Vector leakance_vec = compute_leakance(params);

    // Layer 1: Boussinesq
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        Real Sy = params.Sy(i);

        // Storage
        residual(i) = Sy * A_i * (state.h1(i) - state.h1_old(i)) / dt;

        // Leakage (positive = downward, leaving layer 1)
        Real leak = leakance_vec(i) * (state.h1(i) - state.h2(i)) * A_i;
        residual(i) += leak;

        // Recharge
        if (recharge_.size() > 0) residual(i) -= recharge_(i) * A_i;
        if (pumping_layer1_.size() > 0) residual(i) += pumping_layer1_(i);
    }

    // Layer 1 fluxes
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);
        Real T_i = params.K1(i) * std::max(state.h1(i) - params.z_bottom_1(i), 0.0);
        Real T_j = params.K1(j) * std::max(state.h1(j) - params.z_bottom_1(j), 0.0);
        Real T_ij = (T_i + T_j > 0.0) ? 2.0 * T_i * T_j / (T_i + T_j) : 0.0;
        Real flux = T_ij * face.area * (state.h1(j) - state.h1(i)) / face.distance;

        residual(i) -= flux;
        residual(j) += flux;
    }

    // Layer 2: Confined
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        Real S2 = params.Ss2(i) * params.thickness_2(i);

        // Storage
        residual(n + i) = S2 * A_i * (state.h2(i) - state.h2_old(i)) / dt;

        // Leakage (negative = receiving from layer 1)
        Real leak = leakance_vec(i) * (state.h1(i) - state.h2(i)) * A_i;
        residual(n + i) -= leak;

        if (pumping_layer2_.size() > 0) residual(n + i) += pumping_layer2_(i);
    }

    // Layer 2 fluxes
    Vector T2 = params.T2();
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);
        Real T_ij = confined_kernels::intercell_T(T2(i), T2(j));
        Real flux = T_ij * face.area * (state.h2(j) - state.h2(i)) / face.distance;

        residual(n + i) -= flux;
        residual(n + j) += flux;
    }
}

void TwoLayerSolver::assemble_monolithic_jacobian(
    const StateTwoLayer& state, const ParametersTwoLayer& params,
    const Mesh& mesh, Real dt, SparseMatrix& jacobian
) const {
    const Index n = mesh.n_cells();
    const Index dim = 2 * n;
    Vector leakance_vec = compute_leakance(params);

    std::vector<SparseTriplet> triplets;
    triplets.reserve(dim * 8);

    // Layer 1 diagonal (storage + leakage)
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        Real diag = params.Sy(i) * A_i / dt + leakance_vec(i) * A_i;
        triplets.emplace_back(i, i, diag);

        // Coupling: dF1_i/dh2_i = -leakance * A
        triplets.emplace_back(i, n + i, -leakance_vec(i) * A_i);
    }

    // Layer 1 off-diagonal (Boussinesq fluxes)
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);
        Real T_i = params.K1(i) * std::max(state.h1(i) - params.z_bottom_1(i), 0.0);
        Real T_j = params.K1(j) * std::max(state.h1(j) - params.z_bottom_1(j), 0.0);
        Real T_ij = (T_i + T_j > 0.0) ? 2.0 * T_i * T_j / (T_i + T_j) : 0.0;
        Real coef = T_ij * face.area / face.distance;

        triplets.emplace_back(i, i, coef);
        triplets.emplace_back(i, j, -coef);
        triplets.emplace_back(j, j, coef);
        triplets.emplace_back(j, i, -coef);
    }

    // Layer 2 diagonal (storage + leakage)
    Vector T2 = params.T2();
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        Real S2 = params.Ss2(i) * params.thickness_2(i);
        Real diag = S2 * A_i / dt + leakance_vec(i) * A_i;
        triplets.emplace_back(n + i, n + i, diag);

        // Coupling: dF2_i/dh1_i = -leakance * A
        triplets.emplace_back(n + i, i, -leakance_vec(i) * A_i);
    }

    // Layer 2 off-diagonal
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;

        const Face& face = mesh.face(f);
        Real T_ij = confined_kernels::intercell_T(T2(i), T2(j));
        Real coef = T_ij * face.area / face.distance;

        triplets.emplace_back(n + i, n + i, coef);
        triplets.emplace_back(n + i, n + j, -coef);
        triplets.emplace_back(n + j, n + j, coef);
        triplets.emplace_back(n + j, n + i, -coef);
    }

    jacobian.resize(dim, dim);
    jacobian.setFromTriplets(triplets.begin(), triplets.end());
}

SolveResult TwoLayerSolver::solve_sequential(
    StateTwoLayer& state, const ParametersTwoLayer& params,
    const Mesh& mesh, Real dt
) const {
    // Sequential coupling iteration
    // Not used in monolithic mode, provided for compatibility
    return {true, 1, 0.0, 0.0, "Sequential not implemented - use monolithic"};
}

// Kernel implementations
namespace two_layer_kernels {

void monolithic_residual(
    const Real* __restrict__ h,
    const Real* __restrict__ h_old,
    const Real* __restrict__ K1,
    const Real* __restrict__ Sy,
    const Real* __restrict__ z_surface,
    const Real* __restrict__ z_bottom1,
    const Real* __restrict__ T2,
    const Real* __restrict__ S2,
    const Real* __restrict__ leakance,
    const Real* __restrict__ recharge,
    const Real* __restrict__ stream_exchange,
    const Real* __restrict__ pumping1,
    const Real* __restrict__ pumping2,
    const Index* __restrict__ cell_neighbors,
    const Index* __restrict__ cell_neighbors_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Real* __restrict__ cell_area,
    Index n_cells,
    Real dt,
    Real* __restrict__ residual
) {
    const Real* h1 = h;
    const Real* h2 = h + n_cells;
    const Real* h1_old = h_old;
    const Real* h2_old = h_old + n_cells;
    Real* F1 = residual;
    Real* F2 = residual + n_cells;

    // Layer 1 storage + sources
    for (Index i = 0; i < n_cells; ++i) {
        F1[i] = Sy[i] * cell_area[i] * (h1[i] - h1_old[i]) / dt;
        F1[i] += leakance[i] * (h1[i] - h2[i]) * cell_area[i];
        if (recharge) F1[i] -= recharge[i] * cell_area[i];
        if (pumping1) F1[i] += pumping1[i];
    }

    // Layer 2 storage + sources
    for (Index i = 0; i < n_cells; ++i) {
        F2[i] = S2[i] * cell_area[i] * (h2[i] - h2_old[i]) / dt;
        F2[i] -= leakance[i] * (h1[i] - h2[i]) * cell_area[i];
        if (pumping2) F2[i] += pumping2[i];
    }
}

void monolithic_jacobian(
    const Real* __restrict__ h,
    const Real* __restrict__ K1,
    const Real* __restrict__ Sy,
    const Real* __restrict__ z_bottom1,
    const Real* __restrict__ T2,
    const Real* __restrict__ S2,
    const Real* __restrict__ leakance,
    const Index* __restrict__ cell_neighbors,
    const Index* __restrict__ cell_neighbors_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Real* __restrict__ cell_area,
    Index n_cells,
    Real dt,
    Real* __restrict__ jac_values,
    Index* __restrict__ jac_rows,
    Index* __restrict__ jac_cols,
    Index& nnz
) {
    nnz = 0;

    // Diagonal entries for both layers
    for (Index i = 0; i < n_cells; ++i) {
        // Layer 1 diagonal
        Real diag1 = Sy[i] * cell_area[i] / dt + leakance[i] * cell_area[i];
        jac_values[nnz] = diag1;
        jac_rows[nnz] = i;
        jac_cols[nnz] = i;
        nnz++;

        // Layer 1 -> Layer 2 coupling
        jac_values[nnz] = -leakance[i] * cell_area[i];
        jac_rows[nnz] = i;
        jac_cols[nnz] = n_cells + i;
        nnz++;

        // Layer 2 diagonal
        Real diag2 = S2[i] * cell_area[i] / dt + leakance[i] * cell_area[i];
        jac_values[nnz] = diag2;
        jac_rows[nnz] = n_cells + i;
        jac_cols[nnz] = n_cells + i;
        nnz++;

        // Layer 2 -> Layer 1 coupling
        jac_values[nnz] = -leakance[i] * cell_area[i];
        jac_rows[nnz] = n_cells + i;
        jac_cols[nnz] = i;
        nnz++;
    }
}

} // namespace two_layer_kernels

} // namespace dgw
