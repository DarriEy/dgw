/**
 * @file richards_3d.cpp
 * @brief 3D Richards equation solver implementation
 * 
 * Implements the mixed-form Richards equation for variably-saturated flow.
 * This is the most general and computationally expensive physics option.
 */

#include "dgw/physics/richards_3d.hpp"
#include "dgw/physics/water_retention.hpp"
#include "dgw/core/mesh.hpp"
#include "dgw/core/state.hpp"
#include "dgw/core/parameters.hpp"
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dgw {

Richards3DSolver::Richards3DSolver(const PhysicsDecisions& decisions)
    : decisions_(decisions) {}

void Richards3DSolver::compute_residual(
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    Real dt,
    Vector& residual
) const {
    const auto& s = state.as_richards();
    const auto& retention = params.retention();
    const Mesh3D& mesh3d = dynamic_cast<const Mesh3D&>(mesh);
    const Index n = mesh.n_cells();
    
    residual.resize(n);
    residual.setZero();
    
    // Update constitutive relations from current pressure head
    const_cast<Richards3DSolver*>(this)->update_constitutive(s, retention);
    
    // Storage term: (θ^{n+1} - θ^n) * V / dt  (modified Picard form)
    // Or: C(ψ) * (ψ^{n+1} - ψ^n) * V / dt  (standard form)
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Real V_i = mesh.cell_volume(i);
        
        if (use_modified_picard_) {
            // Modified Picard: better mass conservation
            // Uses actual water content change, not linearized
            Real theta_new = theta_cache_(i);
            Real theta_old = s.water_content(i);  // Stored from previous step
            residual(i) = (theta_new - theta_old) * V_i / dt;
        } else {
            // Standard form
            Real C_i = C_cache_(i);
            Real dpsi = s.pressure_head(i) - s.pressure_head_old(i);
            residual(i) = C_i * dpsi * V_i / dt;
        }
    }
    
    // Flux terms: -Σ Q_ij for each cell
    // Q_ij = -K_ij * A_ij * (∂h/∂n) = -K_ij * A_ij * ((ψ_j + z_j) - (ψ_i + z_i)) / L_ij
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;
        
        const Face& face = mesh.face(f);
        
        // Total heads
        Real h_i = s.pressure_head(i) + mesh.cell_centroid(i).z();
        Real h_j = s.pressure_head(j) + mesh.cell_centroid(j).z();
        
        // Inter-cell conductivity
        Real K_ij = intercell_K(i, j, s.pressure_head, retention, mesh3d);
        
        // Darcy flux: Q = -K * A * dh/dL
        Real A_ij = face.area;
        Real L_ij = face.distance;
        Real flux = -K_ij * A_ij * (h_j - h_i) / L_ij;
        
        // Add to residuals (divergence)
        // ∂θ/∂t = div(K∇h) - S → residual = storage - div(K∇h) + S = 0
        // flux = -K*A*(h_j - h_i)/L is outward Darcy flux from i (q·n·A)
        // div(q)·V for cell i = +flux, and div(K∇h) = -div(q)
        // So: F_i = storage + flux + sink = 0
        #pragma omp atomic
        residual(i) += flux;   // +flux because F = storage + div(q)
        #pragma omp atomic
        residual(j) -= flux;   // opposite sign for neighbor
    }
    
    // Source/sink terms
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Real V_i = mesh.cell_volume(i);
        Real sink = compute_sink(i, mesh3d);
        residual(i) += sink * V_i;
    }
    
    // Top boundary: recharge (adds water, reduces residual)
    if (recharge_.size() > 0) {
        for (Index i : mesh3d.surface_cells()) {
            Real layer_thickness = mesh3d.layers()[0].z_top - mesh3d.layers()[0].z_bottom;
            Real A_top = (layer_thickness > 0.0) ? mesh.cell_volume(i) / layer_thickness : mesh.cell_volume(i);
            residual(i) -= recharge_(i) * A_top;  // Recharge is a source, subtract from F
        }
    }
}

void Richards3DSolver::compute_jacobian(
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    Real dt,
    SparseMatrix& jacobian
) const {
    const auto& s = state.as_richards();
    const auto& retention = params.retention();
    const Mesh3D& mesh3d = dynamic_cast<const Mesh3D&>(mesh);
    const Index n = mesh.n_cells();
    
    // Ensure constitutive relations are up to date
    const_cast<Richards3DSolver*>(this)->update_constitutive(s, retention);
    
    std::vector<SparseTriplet> triplets;
    triplets.reserve(n * 10);  // Estimate: 3D has more neighbors
    
    // Storage term contribution to diagonal
    for (Index i = 0; i < n; ++i) {
        Real V_i = mesh.cell_volume(i);
        Real diag_storage;
        
        if (use_modified_picard_) {
            // ∂(θ-θold)/∂ψ = ∂θ/∂ψ = C(ψ)
            diag_storage = C_cache_(i) * V_i / dt;
        } else {
            // ∂(C*(ψ-ψold))/∂ψ = C + (ψ-ψold)*∂C/∂ψ ≈ C
            diag_storage = C_cache_(i) * V_i / dt;
        }
        
        if (use_mass_lumping_) {
            // Mass lumping: all storage on diagonal
            triplets.emplace_back(i, i, diag_storage);
        } else {
            triplets.emplace_back(i, i, diag_storage);
        }
    }
    
    // Flux term contributions
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) continue;
        
        auto [i, j] = mesh.face_cells(f);
        if (i < 0 || j < 0) continue;
        
        const Face& face = mesh.face(f);
        Real A_ij = face.area;
        Real L_ij = face.distance;
        
        // Conductivities and their derivatives
        Real K_i = K_cache_(i);
        Real K_j = K_cache_(j);
        Real K_ij = intercell_K(i, j, s.pressure_head, retention, mesh3d);
        
        // Heads
        Real psi_i = s.pressure_head(i);
        Real psi_j = s.pressure_head(j);
        Real z_i = mesh.cell_centroid(i).z();
        Real z_j = mesh.cell_centroid(j).z();
        Real h_i = psi_i + z_i;
        Real h_j = psi_j + z_j;
        Real dh = h_j - h_i;
        
        // Base conductance
        Real cond = K_ij * A_ij / L_ij;
        
        // ∂K_ij/∂ψ_i and ∂K_ij/∂ψ_j (for upstream weighting)
        Real dK_dpsi_i = 0.0, dK_dpsi_j = 0.0;
        
        // Simplified: use chain rule with harmonic mean
        // K_ij = 2*K_i*K_j/(K_i+K_j)
        // ∂K_ij/∂K_i = 2*K_j^2/(K_i+K_j)^2
        Real denom = K_i + K_j + 1e-30;
        Real dKij_dKi = 2.0 * K_j * K_j / (denom * denom);
        Real dKij_dKj = 2.0 * K_i * K_i / (denom * denom);
        
        // dK/dψ from retention curve
        // Use van Genuchten derivative (stored or computed)
        VanGenuchten vg(retention.theta_r(i), retention.theta_s(i),
                        retention.alpha(i), retention.n_vg(i),
                        retention.K_sat(i), retention.l_mualem(i));
        dK_dpsi_i = vg.dK_dpsi(psi_i) * dKij_dKi;
        
        VanGenuchten vg_j(retention.theta_r(j), retention.theta_s(j),
                         retention.alpha(j), retention.n_vg(j),
                         retention.K_sat(j), retention.l_mualem(j));
        dK_dpsi_j = vg_j.dK_dpsi(psi_j) * dKij_dKj;
        
        // Flux: Q = -K_ij * A/L * (h_j - h_i)
        // ∂Q/∂ψ_i = -∂K_ij/∂ψ_i * A/L * dh - K_ij * A/L * (-1)
        //         = -dK_dpsi_i * A/L * dh + cond
        // ∂Q/∂ψ_j = -dK_dpsi_j * A/L * dh - cond
        
        Real dQ_dpsi_i = -dK_dpsi_i * A_ij / L_ij * dh + cond;
        Real dQ_dpsi_j = -dK_dpsi_j * A_ij / L_ij * dh - cond;
        
        // Contribution to residual i: F_i += flux (outward Darcy flux)
        // ∂F_i/∂ψ_i += ∂flux/∂ψ_i
        // ∂F_i/∂ψ_j += ∂flux/∂ψ_j
        triplets.emplace_back(i, i, +dQ_dpsi_i);
        triplets.emplace_back(i, j, +dQ_dpsi_j);

        // Contribution to residual j: F_j -= flux (opposite sign)
        triplets.emplace_back(j, i, -dQ_dpsi_i);
        triplets.emplace_back(j, j, -dQ_dpsi_j);
    }
    
    // Assemble
    jacobian.resize(n, n);
    jacobian.setFromTriplets(triplets.begin(), triplets.end());
}

void Richards3DSolver::compute_fluxes(
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    Vector& face_fluxes
) const {
    const auto& s = state.as_richards();
    const auto& retention = params.retention();
    const Mesh3D& mesh3d = dynamic_cast<const Mesh3D&>(mesh);
    
    face_fluxes.resize(mesh.n_faces());
    
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (mesh.is_boundary_face(f)) {
            face_fluxes(f) = 0.0;
            continue;
        }
        
        auto [i, j] = mesh.face_cells(f);
        const Face& face = mesh.face(f);
        
        Real h_i = s.pressure_head(i) + mesh.cell_centroid(i).z();
        Real h_j = s.pressure_head(j) + mesh.cell_centroid(j).z();
        
        Real K_ij = intercell_K(i, j, s.pressure_head, retention, mesh3d);
        
        // Negative sign matches residual convention: flux = -K * A * dh/dL
        face_fluxes(f) = -K_ij * face.area * (h_j - h_i) / face.distance;
    }
}

void Richards3DSolver::initialize_state(
    const Mesh& mesh,
    const Parameters& params,
    const Config& config,
    State& state
) const {
    auto& s = state.as_richards();
    const Mesh3D& mesh3d = dynamic_cast<const Mesh3D&>(mesh);
    const Index n = mesh.n_cells();
    
    s.pressure_head.resize(n);
    s.pressure_head_old.resize(n);
    s.water_content.resize(n);
    s.hydraulic_conductivity.resize(n);
    s.specific_moisture_capacity.resize(n);
    s.saturation.resize(n);
    s.recharge.resize(n);
    s.evapotranspiration.resize(n);
    s.bottom_flux.resize(n);
    s.lateral_flux.resize(n);
    
    // Initialize with hydrostatic equilibrium
    // Find approximate water table elevation
    Real z_wt = 0.0;  // Default water table at z=0
    
    // Set pressure head: ψ = z_wt - z (hydrostatic)
    for (Index i = 0; i < n; ++i) {
        Real z = mesh.cell_centroid(i).z();
        s.pressure_head(i) = z_wt - z;
    }
    
    s.pressure_head_old = s.pressure_head;
    
    // Initialize constitutive relations
    const auto& retention = params.retention();
    const_cast<Richards3DSolver*>(this)->update_constitutive(s, retention);
    
    s.time = config.time.start_time;
}

SparseMatrix Richards3DSolver::allocate_jacobian(const Mesh& mesh) const {
    const Index n = mesh.n_cells();
    
    // Estimate NNZ
    Index nnz = n;  // Diagonal
    for (Index f = 0; f < mesh.n_faces(); ++f) {
        if (!mesh.is_boundary_face(f)) {
            nnz += 4;
        }
    }
    
    SparseMatrix J(n, n);
    J.reserve(nnz);
    
    std::vector<SparseTriplet> triplets;
    for (Index i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, 0.0);
        for (Index j : mesh.cell_neighbors(i)) {
            triplets.emplace_back(i, j, 0.0);
        }
    }
    
    J.setFromTriplets(triplets.begin(), triplets.end());
    return J;
}

void Richards3DSolver::set_recharge(const Vector& recharge_rate) {
    recharge_ = recharge_rate;
}

void Richards3DSolver::set_stream_stage(const Vector& stream_stage) {
    stream_stage_ = stream_stage;
}

void Richards3DSolver::set_pumping(const Vector& pumping) {
    pumping_ = pumping;
}

void Richards3DSolver::set_evapotranspiration(
    const Vector& et_rate,
    const Vector& root_distribution
) {
    et_rate_ = et_rate;
    root_distribution_ = root_distribution;
}

void Richards3DSolver::apply_boundary_conditions(
    const State& state,
    const Mesh& mesh,
    const Parameters& params,
    Vector& residual,
    SparseMatrix& jacobian
) const {
    // Boundary conditions are handled in compute_residual for Richards
    // This is a placeholder for more complex BC handling
}

Vector Richards3DSolver::water_table_depth(
    const State& state,
    const Mesh& mesh
) const {
    const auto& s = state.as_richards();
    const Mesh3D& mesh3d = dynamic_cast<const Mesh3D&>(mesh);
    
    // Find water table in each column (where pressure_head = 0)
    const Index n_layers = mesh3d.n_layers() > 0 ? mesh3d.n_layers() : 1;
    const Index n_cols = mesh3d.n_cells() / n_layers;
    Vector depth(n_cols);
    depth.setZero();

    // Simplified: return surface cell depth per column
    for (Index col = 0; col < n_cols; ++col) {
        Real z_surface = mesh3d.cell(col * n_layers).z_surface;
        
        // Find depth where ψ = 0 in this column
        bool found = false;
        for (Index layer = 0; layer < mesh3d.n_layers(); ++layer) {
            Index cell = col * mesh3d.n_layers() + layer;
            if (s.pressure_head(cell) >= 0.0) {
                Real z = mesh.cell_centroid(cell).z();
                depth(col) = z_surface - z;
                found = true;
                break;
            }
        }
        
        if (!found) {
            // Water table below domain
            depth(col) = z_surface - mesh3d.min_coords().z();
        }
    }
    
    return depth;
}

Vector Richards3DSolver::water_table_elevation(
    const State& state,
    const Mesh& mesh
) const {
    const auto& s = state.as_richards();
    const Mesh3D& mesh3d = dynamic_cast<const Mesh3D&>(mesh);
    
    const Index n_layers_e = mesh3d.n_layers() > 0 ? mesh3d.n_layers() : 1;
    const Index n_cols_e = mesh3d.n_cells() / n_layers_e;
    Vector elevation(n_cols_e);
    elevation.setConstant(mesh3d.min_coords().z());  // Default: bottom of domain

    for (Index col = 0; col < n_cols_e; ++col) {
        // Linear interpolation to find z where ψ = 0
        for (Index layer = 0; layer < mesh3d.n_layers() - 1; ++layer) {
            Index cell_top = col * mesh3d.n_layers() + layer;
            Index cell_bot = col * mesh3d.n_layers() + layer + 1;
            
            Real psi_top = s.pressure_head(cell_top);
            Real psi_bot = s.pressure_head(cell_bot);
            
            if (psi_top >= 0.0 && psi_bot < 0.0) {
                // Water table is between these cells
                Real z_top = mesh.cell_centroid(cell_top).z();
                Real z_bot = mesh.cell_centroid(cell_bot).z();
                
                // Linear interpolation
                Real t = psi_top / (psi_top - psi_bot);
                elevation(col) = z_top + t * (z_bot - z_top);
                break;
            } else if (psi_top >= 0.0) {
                elevation(col) = mesh.cell_centroid(cell_top).z();
            }
        }
    }
    
    return elevation;
}

Vector Richards3DSolver::saturation(const State& state, const Parameters& params) const {
    return state.as_richards().saturation;
}

Vector Richards3DSolver::water_content(const State& state, const Parameters& params) const {
    return state.as_richards().water_content;
}

Vector Richards3DSolver::stream_exchange(
    const State& state,
    const Parameters& params,
    const Mesh& mesh
) const {
    // 3D Richards typically doesn't have 2D river cells
    // Return zero for now
    Vector exchange(mesh.n_cells());
    exchange.setZero();
    return exchange;
}

Real Richards3DSolver::total_storage(
    const State& state,
    const Parameters& params,
    const Mesh& mesh
) const {
    const auto& s = state.as_richards();
    
    Real storage = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        storage += s.water_content(i) * mesh.cell_volume(i);
    }
    
    return storage;
}

Real Richards3DSolver::mass_balance_error(
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    Real dt
) const {
    const auto& s = state.as_richards();
    
    // Storage change
    Real dS = 0.0;
    for (Index i = 0; i < mesh.n_cells(); ++i) {
        Real V = mesh.cell_volume(i);
        dS += (theta_cache_(i) - s.water_content(i)) * V;
    }
    
    // Compute total flux
    Real Q_in = 0.0;
    
    // Recharge
    if (recharge_.size() > 0) {
        const Mesh3D& mesh3d = dynamic_cast<const Mesh3D&>(mesh);
        for (Index i : mesh3d.surface_cells()) {
            Real A = mesh.cell_volume(i);  // Approximate
            Q_in += recharge_(i) * A * dt;
        }
    }
    
    // ET
    if (et_rate_.size() > 0) {
        for (Index i = 0; i < mesh.n_cells(); ++i) {
            Q_in -= et_rate_(i) * mesh.cell_volume(i) * dt;
        }
    }
    
    return std::abs(dS - Q_in) / (std::abs(Q_in) + 1e-10);
}

void Richards3DSolver::update_constitutive(
    const StateRichards3D& state,
    const WaterRetentionParams& retention
) {
    const Index n = state.pressure_head.size();
    
    K_cache_.resize(n);
    C_cache_.resize(n);
    theta_cache_.resize(n);
    Kr_cache_.resize(n);
    
    #pragma omp parallel for
    for (Index i = 0; i < n; ++i) {
        Real psi = state.pressure_head(i);
        
        // Create van Genuchten model for this cell
        VanGenuchten vg(
            retention.theta_r(i),
            retention.theta_s(i),
            retention.alpha(i),
            retention.n_vg(i),
            retention.K_sat(i),
            (retention.l_mualem.size() > 0) ? retention.l_mualem(i) : 0.5
        );
        
        theta_cache_(i) = vg.water_content(psi);
        K_cache_(i) = vg.hydraulic_conductivity(psi);
        C_cache_(i) = vg.moisture_capacity(psi);
        Kr_cache_(i) = vg.relative_conductivity(vg.effective_saturation(psi));
    }
}

Real Richards3DSolver::intercell_K(
    Index cell_i, Index cell_j,
    const Vector& psi,
    const WaterRetentionParams& retention,
    const Mesh3D& mesh
) const {
    Real K_i = K_cache_(cell_i);
    Real K_j = K_cache_(cell_j);
    
    Real z_i = mesh.cell_centroid(cell_i).z();
    Real z_j = mesh.cell_centroid(cell_j).z();
    
    return richards_kernels::intercell_K_upstream(
        K_i, K_j, psi(cell_i), psi(cell_j), z_i, z_j, upstream_weight_);
}

Real Richards3DSolver::gravity_flux(
    Index cell_i, Index cell_j,
    Real K_face,
    const Mesh3D& mesh
) const {
    Real z_i = mesh.cell_centroid(cell_i).z();
    Real z_j = mesh.cell_centroid(cell_j).z();
    
    // Gravity component: -K * dz/dL
    Real L = mesh.intercell_distance(cell_i, cell_j);
    return -K_face * (z_j - z_i) / L;
}

Real Richards3DSolver::compute_sink(Index cell, const Mesh3D& mesh) const {
    Real sink = 0.0;
    
    // Pumping
    if (pumping_.size() > 0) {
        sink += pumping_(cell) / mesh.cell_volume(cell);
    }
    
    // Evapotranspiration (distributed by root distribution)
    if (et_rate_.size() > 0 && root_distribution_.size() > 0) {
        sink += et_rate_(cell) * root_distribution_(cell);
    }
    
    return sink;
}

// ============================================================================
// Kernel implementations
// ============================================================================

namespace richards_kernels {

void residual_kernel_3d(
    const Real* __restrict__ psi,
    const Real* __restrict__ psi_old,
    const Real* __restrict__ theta,
    const Real* __restrict__ theta_old,
    const Real* __restrict__ K,
    const Real* __restrict__ z,
    const Real* __restrict__ cell_volume,
    const Index* __restrict__ cell_faces,
    const Index* __restrict__ cell_faces_ptr,
    const Real* __restrict__ face_area,
    const Real* __restrict__ face_distance,
    const Index* __restrict__ face_cells,
    const Real* __restrict__ source,
    Index n_cells,
    Real dt,
    bool use_modified_picard,
    Real* __restrict__ residual
) {
    // Storage term
    #pragma omp parallel for
    for (Index i = 0; i < n_cells; ++i) {
        if (use_modified_picard) {
            residual[i] = (theta[i] - theta_old[i]) * cell_volume[i] / dt;
        } else {
            // Would need C(ψ) here
            residual[i] = (theta[i] - theta_old[i]) * cell_volume[i] / dt;
        }
        
        // Source/sink term (positive sink = extraction, adds to residual)
        if (source) {
            residual[i] += source[i] * cell_volume[i];
        }
    }
    
    // Flux terms
    #pragma omp parallel for
    for (Index i = 0; i < n_cells; ++i) {
        Index f_start = cell_faces_ptr[i];
        Index f_end = cell_faces_ptr[i + 1];
        
        for (Index f_idx = f_start; f_idx < f_end; ++f_idx) {
            Index f = cell_faces[f_idx];
            Index i_face = face_cells[2*f];
            Index j_face = face_cells[2*f + 1];
            
            if (j_face < 0) continue;  // Boundary face
            if (i != i_face && i != j_face) continue;  // Safety check
            
            Index other = (i == i_face) ? j_face : i_face;
            
            // Total heads
            Real h_i = psi[i] + z[i];
            Real h_other = psi[other] + z[other];
            
            // Inter-cell K (harmonic mean)
            Real K_face = 2.0 * K[i] * K[other] / (K[i] + K[other] + 1e-30);
            
            // Darcy flux
            Real A = face_area[f];
            Real L = face_distance[f];
            Real flux = -K_face * A * (h_other - h_i) / L;
            
            // flux = -K*A*(h_other - h_i)/L is outward Darcy flux from cell i
            // F_i += outward_flux (consistent with class method sign convention)
            residual[i] += flux;
        }
    }
}

} // namespace richards_kernels

} // namespace dgw
