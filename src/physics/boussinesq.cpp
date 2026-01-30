/**
 * @file boussinesq.cpp
 * @brief Boussinesq equation solver implementation
 */

#include "dgw/physics/boussinesq.hpp"
#include "dgw/core/mesh.hpp"
#include "dgw/core/state.hpp"
#include "dgw/core/parameters.hpp"
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dgw {

BoussinesqSolver::BoussinesqSolver(const PhysicsDecisions& decisions)
    : decisions_(decisions) {
    if (decisions.transmissivity == TransmissivityMethod::Smoothed) {
        smoothing_eps_ = 0.01;  // 1 cm smoothing zone
    }
}

void BoussinesqSolver::compute_residual(
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    Real dt,
    Vector& residual
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();
    const Index n = mesh.n_cells();
    
    residual.resize(n);
    residual.setZero();
    
    // Storage term: Sy * A * (h - h_old) / dt
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);  // In 2D, "volume" is area
        residual(i) = p.Sy(i) * A_i * (s.head(i) - s.head_old(i)) / dt;
    }
    
    // Flux terms: -Σ Q_ij for each cell
    // Q_ij = T_ij * W_ij * (h_j - h_i) / L_ij
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;
        
        const Face& face = mesh.face(f);
        
        // Compute transmissivities
        Real T_i = compute_transmissivity(s.head(i), p.K(i), p.z_bottom(i),
                                          decisions_.transmissivity);
        Real T_j = compute_transmissivity(s.head(j), p.K(j), p.z_bottom(j),
                                          decisions_.transmissivity);
        
        // Inter-cell transmissivity
        Real T_ij = intercell_transmissivity(T_i, T_j, s.head(i), s.head(j),
                                             decisions_.transmissivity);
        
        // Darcy flux: Q = -T * W * dh/dL
        Real W_ij = face.area;  // Face "area" is length in 2D
        Real L_ij = face.distance;
        Real flux = T_ij * W_ij * (s.head(j) - s.head(i)) / L_ij;
        
        // Add to residuals (flux leaves i, enters j)
        #pragma omp atomic
        residual(i) -= flux;
        #pragma omp atomic
        residual(j) += flux;
    }
    
    // Source terms
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        
        // Recharge (positive = into aquifer)
        if (recharge_.size() > 0) {
            residual(i) -= recharge_(i) * A_i;
        }
        
        // Pumping (positive = out of aquifer)
        if (pumping_.size() > 0) {
            residual(i) += pumping_(i);  // Already in m³/s
        }
    }
    
    // Stream-aquifer exchange
    if (stream_stage_.size() > 0) {
        for (Index cell_id : mesh.river_cells()) {
            const RiverSegment* seg = mesh.river_segment(cell_id);
            if (!seg) continue;
            
            Real h_gw = s.head(cell_id);
            Real h_stream = stream_stage_(cell_id);
            
            // Conductance
            Real C = params.stream().conductance()(cell_id);
            
            // Exchange flux (positive = gaining stream = water leaving aquifer)
            Real Q_stream = C * (h_gw - h_stream);
            
            residual(cell_id) += Q_stream;
        }
    }
}

void BoussinesqSolver::compute_jacobian(
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    Real dt,
    SparseMatrix& jacobian
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();
    const Index n = mesh.n_cells();
    
    std::vector<SparseTriplet> triplets;
    triplets.reserve(n * 7);  // Estimate: each cell has ~6 neighbors + diagonal
    
    // Diagonal terms from storage
    for (Index i = 0; i < n; ++i) {
        Real A_i = mesh.cell_volume(i);
        Real diag = p.Sy(i) * A_i / dt;
        triplets.emplace_back(i, i, diag);
    }
    
    // Off-diagonal terms from fluxes
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;
        
        const Face& face = mesh.face(f);
        Real W_ij = face.area;
        Real L_ij = face.distance;
        
        // Transmissivities
        Real h_i = s.head(i), h_j = s.head(j);
        Real T_i = compute_transmissivity(h_i, p.K(i), p.z_bottom(i),
                                          decisions_.transmissivity);
        Real T_j = compute_transmissivity(h_j, p.K(j), p.z_bottom(j),
                                          decisions_.transmissivity);
        Real T_ij = intercell_transmissivity(T_i, T_j, h_i, h_j,
                                             decisions_.transmissivity);
        
        // Derivatives of transmissivity
        Real dT_i = transmissivity_derivative(h_i, p.K(i), p.z_bottom(i),
                                              decisions_.transmissivity);
        Real dT_j = transmissivity_derivative(h_j, p.K(j), p.z_bottom(j),
                                              decisions_.transmissivity);
        
        // Flux: Q = T_ij * W/L * (h_j - h_i)
        // ∂Q/∂h_i = -T_ij * W/L + (h_j - h_i) * W/L * ∂T_ij/∂h_i
        // ∂Q/∂h_j = +T_ij * W/L + (h_j - h_i) * W/L * ∂T_ij/∂h_j
        
        Real coef = W_ij / L_ij;
        Real dh = h_j - h_i;
        
        // Harmonic mean: T_ij = 2*T_i*T_j/(T_i+T_j)
        // ∂T_ij/∂h_i = 2*T_j^2/(T_i+T_j)^2 * dT_i
        // ∂T_ij/∂h_j = 2*T_i^2/(T_i+T_j)^2 * dT_j
        Real denom_T = T_i + T_j + 1e-30;
        Real dTij_dhi = 2.0 * T_j * T_j / (denom_T * denom_T) * dT_i;
        Real dTij_dhj = 2.0 * T_i * T_i / (denom_T * denom_T) * dT_j;

        Real dQ_dhi = -T_ij * coef + dh * coef * dTij_dhi;
        Real dQ_dhj = +T_ij * coef + dh * coef * dTij_dhj;
        
        // Contribution to cell i residual
        // F_i -= Q, so ∂F_i/∂h_i -= ∂Q/∂h_i, ∂F_i/∂h_j -= ∂Q/∂h_j
        triplets.emplace_back(i, i, -dQ_dhi);
        triplets.emplace_back(i, j, -dQ_dhj);
        
        // Contribution to cell j residual
        // F_j += Q, so ∂F_j/∂h_i += ∂Q/∂h_i, ∂F_j/∂h_j += ∂Q/∂h_j
        triplets.emplace_back(j, i, +dQ_dhi);
        triplets.emplace_back(j, j, +dQ_dhj);
    }
    
    // Stream-aquifer exchange contribution to diagonal
    if (stream_stage_.size() > 0) {
        for (Index cell_id : mesh.river_cells()) {
            Real C = params.stream().conductance()(cell_id);
            // Q_stream = C * (h_gw - h_stream)
            // ∂Q/∂h_gw = C
            triplets.emplace_back(cell_id, cell_id, C);
        }
    }
    
    // Build sparse matrix
    jacobian.resize(n, n);
    jacobian.setFromTriplets(triplets.begin(), triplets.end());
}

void BoussinesqSolver::compute_fluxes(
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    Vector& face_fluxes
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
        
        Real T_i = compute_transmissivity(s.head(i), p.K(i), p.z_bottom(i),
                                          decisions_.transmissivity);
        Real T_j = compute_transmissivity(s.head(j), p.K(j), p.z_bottom(j),
                                          decisions_.transmissivity);
        Real T_ij = intercell_transmissivity(T_i, T_j, s.head(i), s.head(j),
                                             decisions_.transmissivity);
        
        Real W_ij = face.area;
        Real L_ij = face.distance;
        
        // Positive flux = flow from i to j
        face_fluxes(f) = T_ij * W_ij * (s.head(j) - s.head(i)) / L_ij;
    }
}

void BoussinesqSolver::initialize_state(
    const Mesh& mesh,
    const Parameters& params,
    const Config& config,
    State& state
) const {
    auto& s = state.as_2d();
    const Index n = mesh.n_cells();
    
    s.head.resize(n);
    s.head_old.resize(n);
    s.vadose_storage.resize(n);
    s.vadose_storage.setZero();
    
    // Default: set head to surface elevation (fully saturated)
    for (Index i = 0; i < n; ++i) {
        s.head(i) = mesh.cell(i).z_surface;
    }
    s.head_old = s.head;
    
    s.time = config.time.start_time;
}

SparseMatrix BoussinesqSolver::allocate_jacobian(const Mesh& mesh) const {
    const Index n = mesh.n_cells();
    
    // Count non-zeros: diagonal + 2 entries per interior face
    Index nnz = n;  // Diagonal
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (!mesh.is_boundary_face(f)) {
            nnz += 4;  // Each face contributes to 4 Jacobian entries
        }
    }
    
    SparseMatrix J(n, n);
    J.reserve(nnz);
    
    // Build sparsity pattern
    std::vector<SparseTriplet> triplets;
    triplets.reserve(nnz);
    
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 0.0);
        for (Index j : mesh.cell_neighbors(i)) {
            triplets.emplace_back(i, j, 0.0);
        }
    }
    
    J.setFromTriplets(triplets.begin(), triplets.end());
    return J;
}

void BoussinesqSolver::set_recharge(const Vector& recharge_rate) {
    recharge_ = recharge_rate;
}

void BoussinesqSolver::set_stream_stage(const Vector& stream_stage) {
    stream_stage_ = stream_stage;
}

void BoussinesqSolver::set_pumping(const Vector& pumping) {
    pumping_ = pumping;
}

void BoussinesqSolver::apply_boundary_conditions(
    const State& state,
    const Mesh& mesh,
    const Parameters& params,
    Vector& residual,
    SparseMatrix& jacobian
) const {
    const auto& s = state.as_2d();

    // Apply BCs at boundary faces
    for (Index f : mesh.boundary_faces()) {
        const Face& face = mesh.face(f);
        Index cell = (face.cell_left >= 0) ? face.cell_left : face.cell_right;

        switch (face.bc_type) {
            case BoundaryType::NoFlow:
                // Nothing to do - no flux contribution
                break;

            case BoundaryType::FixedHead: {
                // Strong enforcement: replace equation with h - h_bc = 0
                Real h_bc = face.bc_value;
                residual(cell) = s.head(cell) - h_bc;
                // Zero out row and set diagonal to 1
                for (SparseMatrix::InnerIterator it(jacobian, cell); it; ++it) {
                    it.valueRef() = (it.col() == cell) ? 1.0 : 0.0;
                }
                break;
            }

            case BoundaryType::FixedFlux: {
                // Add specified flux to residual
                Real q_bc = face.bc_value;  // [m³/s]
                residual(cell) -= q_bc;
                break;
            }

            case BoundaryType::GeneralHead: {
                // Q = C * (h_external - h): positive = inflow
                Real h_ext = face.bc_value;
                Real C = params.boundary().ghb_conductance(f);
                residual(cell) -= C * (h_ext - s.head(cell));
                // Jacobian: d(-C*(h_ext - h))/dh = C
                jacobian.coeffRef(cell, cell) += C;
                break;
            }

            default:
                break;
        }
    }
}

Vector BoussinesqSolver::water_table_depth(
    const State& state,
    const Mesh& mesh
) const {
    const auto& s = state.as_2d();
    Vector depth(mesh.n_cells());
    
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        Real z_surf = mesh.cell(i).z_surface;
        depth(i) = z_surf - s.head(i);
        if (depth(i) < 0.0) depth(i) = 0.0;  // Above surface = 0 depth
    }
    
    return depth;
}

Vector BoussinesqSolver::stream_exchange(
    const State& state,
    const Parameters& params,
    const Mesh& mesh
) const {
    const auto& s = state.as_2d();
    Vector exchange(mesh.n_cells());
    exchange.setZero();
    
    if (stream_stage_.size() == 0) return exchange;
    
    for (Index cell_id : mesh.river_cells()) {
        Real h_gw = s.head(cell_id);
        Real h_stream = stream_stage_(cell_id);
        Real C = params.stream().conductance()(cell_id);
        
        exchange(cell_id) = C * (h_gw - h_stream);
    }
    
    return exchange;
}

Real BoussinesqSolver::total_storage(
    const State& state,
    const Parameters& params,
    const Mesh& mesh
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();
    
    Real storage = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        Real A_i = mesh.cell_volume(i);
        Real sat_thickness = std::max(s.head(i) - p.z_bottom(i), 0.0);
        storage += p.Sy(i) * A_i * sat_thickness;
    }
    
    return storage;
}

Real BoussinesqSolver::mass_balance_error(
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    Real dt
) const {
    const auto& s = state.as_2d();
    const auto& p = params.as_2d();
    
    // Storage change
    Real dS = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        Real A_i = mesh.cell_volume(i);
        dS += p.Sy(i) * A_i * (s.head(i) - s.head_old(i));
    }
    
    // Total inflows/outflows
    Real Q_in = 0.0;
    
    // Recharge
    if (recharge_.size() > 0) {
        for (Index i = 0; i < mesh.n_cells(); ++i) {
            Q_in += recharge_(i) * mesh.cell_volume(i) * dt;
        }
    }
    
    // Pumping (negative)
    if (pumping_.size() > 0) {
        Q_in -= pumping_.sum() * dt;
    }
    
    // Stream exchange
    Vector exch = stream_exchange(state, params, mesh);
    Q_in -= exch.sum() * dt;
    
    // Mass balance error
    return std::abs(dS - Q_in) / (std::abs(Q_in) + 1e-10);
}

// Static helper methods

Real BoussinesqSolver::compute_transmissivity(
    Real h, Real K, Real z_bot,
    TransmissivityMethod method
) {
    Real b = h - z_bot;  // Saturated thickness
    
    switch (method) {
        case TransmissivityMethod::Standard:
            return K * std::max(b, 0.0);
            
        case TransmissivityMethod::Smoothed:
            return boussinesq_kernels::smooth_transmissivity(h, K, z_bot);
            
        case TransmissivityMethod::Upstream:
            // Same as standard for single cell
            return K * std::max(b, 0.0);
            
        default:
            return K * std::max(b, 0.0);
    }
}

Real BoussinesqSolver::intercell_transmissivity(
    Real T_i, Real T_j, Real h_i, Real h_j,
    TransmissivityMethod method
) {
    switch (method) {
        case TransmissivityMethod::Standard:
        case TransmissivityMethod::Smoothed:
            return boussinesq_kernels::harmonic_mean(T_i, T_j);
            
        case TransmissivityMethod::Upstream:
            return boussinesq_kernels::upstream_transmissivity(T_i, T_j, h_i, h_j);
            
        case TransmissivityMethod::Harmonic:
            return boussinesq_kernels::harmonic_mean(T_i, T_j);
            
        default:
            return boussinesq_kernels::harmonic_mean(T_i, T_j);
    }
}

Real BoussinesqSolver::transmissivity_derivative(
    Real h, Real K, Real z_bot,
    TransmissivityMethod method
) {
    Real b = h - z_bot;
    
    switch (method) {
        case TransmissivityMethod::Standard:
            return (b > 0.0) ? K : 0.0;
            
        case TransmissivityMethod::Smoothed:
            return boussinesq_kernels::smooth_transmissivity_dh(h, K, z_bot);
            
        default:
            return (b > 0.0) ? K : 0.0;
    }
}

// ============================================================================
// Enzyme-compatible kernel implementations
// ============================================================================

namespace boussinesq_kernels {

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
) {
    // Storage term
    #pragma omp parallel for
    for (Index i = 0; i < n_cells; ++i) {
        residual[i] = Sy[i] * cell_area[i] * (head[i] - head_old[i]) / dt;
        
        // Recharge
        if (recharge) {
            residual[i] -= recharge[i] * cell_area[i];
        }
    }
    
    // Flux terms (loop over cells, then neighbors)
    #pragma omp parallel for
    for (Index i = 0; i < n_cells; ++i) {
        Index start = cell_neighbors_ptr[i];
        Index end = cell_neighbors_ptr[i + 1];
        
        for (Index k = start; k < end; ++k) {
            Index j = cell_neighbors_data[k];
            if (j <= i) continue;  // Only process each face once
            
            // Transmissivities
            Real T_i = smooth_transmissivity(head[i], K[i], z_bottom[i]);
            Real T_j = smooth_transmissivity(head[j], K[j], z_bottom[j]);
            Real T_ij = harmonic_mean(T_i, T_j);
            
            // Face properties (simplified: assume stored at neighbor index)
            Real W = face_area[k];
            Real L = face_distance[k];
            
            // Darcy flux
            Real flux = T_ij * W * (head[j] - head[i]) / L;
            
            // Update residuals
            #pragma omp atomic
            residual[i] -= flux;
            #pragma omp atomic
            residual[j] += flux;
        }
    }
}

} // namespace boussinesq_kernels

} // namespace dgw
