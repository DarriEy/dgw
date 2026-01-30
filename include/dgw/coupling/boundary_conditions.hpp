/**
 * @file boundary_conditions.hpp
 * @brief Boundary condition application for dGW
 */

#pragma once

#include "../core/types.hpp"
#include "../core/mesh.hpp"
#include "../core/parameters.hpp"
#include <cmath>
#include <algorithm>

namespace dgw {

/**
 * @brief Manages and applies boundary conditions to the system
 */
class BoundaryConditions {
public:
    BoundaryConditions() = default;

    /**
     * @brief Apply all boundary conditions to residual vector
     *
     * @param mesh Mesh
     * @param params Parameters (includes BoundaryParameters)
     * @param head Current head values
     * @param residual Residual vector (modified in place)
     */
    void apply_to_residual(
        const Mesh& mesh,
        const Parameters& params,
        const Vector& head,
        Vector& residual
    ) const {
        const auto& bp = params.boundary();
        auto bfaces = mesh.boundary_faces();

        for (Index f : bfaces) {
            const Face& face = mesh.face(f);
            Index cell = (face.cell_left >= 0) ? face.cell_left : face.cell_right;
            if (cell < 0) continue;

            switch (face.bc_type) {
                case BoundaryType::NoFlow:
                    // Nothing to add (natural BC)
                    break;

                case BoundaryType::FixedHead:
                    apply_fixed_head_residual(cell, face.bc_value, head, residual);
                    break;

                case BoundaryType::FixedFlux:
                    apply_fixed_flux_residual(cell, face, bp, residual);
                    break;

                case BoundaryType::GeneralHead:
                    apply_ghb_residual(cell, head, bp, residual);
                    break;

                case BoundaryType::Drain:
                    apply_drain_residual(cell, head, bp, residual);
                    break;

                case BoundaryType::SeepageFace:
                    apply_seepage_residual(cell, face, head, residual);
                    break;

                default:
                    break;
            }
        }
    }

    /**
     * @brief Apply all boundary conditions to Jacobian
     *
     * @param mesh Mesh
     * @param params Parameters
     * @param head Current head values
     * @param triplets Jacobian triplets (appended)
     */
    void apply_to_jacobian(
        const Mesh& mesh,
        const Parameters& params,
        const Vector& head,
        std::vector<SparseTriplet>& triplets
    ) const {
        const auto& bp = params.boundary();
        auto bfaces = mesh.boundary_faces();

        for (Index f : bfaces) {
            const Face& face = mesh.face(f);
            Index cell = (face.cell_left >= 0) ? face.cell_left : face.cell_right;
            if (cell < 0) continue;

            switch (face.bc_type) {
                case BoundaryType::NoFlow:
                    break;

                case BoundaryType::FixedHead:
                    apply_fixed_head_jacobian(cell, triplets);
                    break;

                case BoundaryType::FixedFlux:
                    // Fixed flux doesn't depend on head
                    break;

                case BoundaryType::GeneralHead:
                    apply_ghb_jacobian(cell, bp, triplets);
                    break;

                case BoundaryType::Drain:
                    apply_drain_jacobian(cell, head, bp, triplets);
                    break;

                case BoundaryType::SeepageFace:
                    apply_seepage_jacobian(cell, face, head, triplets);
                    break;

                default:
                    break;
            }
        }
    }

    /**
     * @brief Apply fixed head BCs by modifying residual and Jacobian directly
     *
     * This zeroes the residual row and sets J(i,i)=1 for Dirichlet cells,
     * effectively replacing the equation with h_i = h_specified.
     */
    static void enforce_dirichlet(
        const Mesh& mesh,
        const Vector& head,
        Vector& residual,
        SparseMatrix& jacobian
    ) {
        auto bfaces = mesh.boundary_faces();

        for (Index f : bfaces) {
            const Face& face = mesh.face(f);
            if (face.bc_type != BoundaryType::FixedHead) continue;

            Index cell = (face.cell_left >= 0) ? face.cell_left : face.cell_right;
            if (cell < 0) continue;

            // Replace residual equation with: h - h_bc = 0
            residual(cell) = head(cell) - face.bc_value;

            // Zero out Jacobian row and set diagonal to 1
            // For column-major storage, InnerIterator(j, k) iterates column k.
            // To zero row 'cell', we must iterate all columns and zero entries in that row.
            for (Index k = 0; k < jacobian.outerSize(); ++k) {
                for (SparseMatrix::InnerIterator it(jacobian, k); it; ++it) {
                    if (it.row() == cell) {
                        it.valueRef() = (it.col() == cell) ? 1.0 : 0.0;
                    }
                }
            }
        }
    }

private:
    // Fixed Head (Dirichlet)
    static void apply_fixed_head_residual(
        Index cell, Real h_bc, const Vector& head, Vector& residual
    ) {
        constexpr Real penalty = 1e20;
        residual(cell) += penalty * (head(cell) - h_bc);
    }

    static void apply_fixed_head_jacobian(
        Index cell, std::vector<SparseTriplet>& triplets
    ) {
        constexpr Real penalty = 1e20;
        triplets.emplace_back(cell, cell, penalty);
    }

    // Fixed Flux (Neumann)
    static void apply_fixed_flux_residual(
        Index cell, const Face& face,
        const BoundaryParameters& bp,
        Vector& residual
    ) {
        if (bp.flux_values.size() > cell) {
            residual(cell) -= bp.flux_values(cell);
        } else {
            residual(cell) -= face.bc_value * face.area;
        }
    }

    // General Head Boundary (Robin / mixed)
    static void apply_ghb_residual(
        Index cell, const Vector& head,
        const BoundaryParameters& bp,
        Vector& residual
    ) {
        if (bp.ghb_head.size() <= cell || bp.ghb_conductance.size() <= cell) return;
        Real C = bp.ghb_conductance(cell);
        Real h_ext = bp.ghb_head(cell);
        residual(cell) -= C * (h_ext - head(cell));
    }

    static void apply_ghb_jacobian(
        Index cell, const BoundaryParameters& bp,
        std::vector<SparseTriplet>& triplets
    ) {
        if (bp.ghb_conductance.size() <= cell) return;
        Real C = bp.ghb_conductance(cell);
        triplets.emplace_back(cell, cell, C);
    }

    // Drain
    static void apply_drain_residual(
        Index cell, const Vector& head,
        const BoundaryParameters& bp,
        Vector& residual
    ) {
        if (bp.drain_elevation.size() <= cell || bp.drain_conductance.size() <= cell) return;
        Real z_drain = bp.drain_elevation(cell);
        Real C_drain = bp.drain_conductance(cell);
        if (head(cell) > z_drain) {
            Real Q = C_drain * (head(cell) - z_drain);
            residual(cell) += Q;
        }
    }

    static void apply_drain_jacobian(
        Index cell, const Vector& head,
        const BoundaryParameters& bp,
        std::vector<SparseTriplet>& triplets
    ) {
        if (bp.drain_elevation.size() <= cell || bp.drain_conductance.size() <= cell) return;
        Real z_drain = bp.drain_elevation(cell);
        Real C_drain = bp.drain_conductance(cell);
        if (head(cell) > z_drain) {
            triplets.emplace_back(cell, cell, C_drain);
        }
    }

    // Seepage Face
    static void apply_seepage_residual(
        Index cell, const Face& face, const Vector& head, Vector& residual
    ) {
        Real z_surface = face.centroid.z();
        if (head(cell) > z_surface) {
            constexpr Real penalty = 1e15;
            residual(cell) += penalty * (head(cell) - z_surface);
        }
    }

    static void apply_seepage_jacobian(
        Index cell, const Face& face, const Vector& head,
        std::vector<SparseTriplet>& triplets
    ) {
        Real z_surface = face.centroid.z();
        if (head(cell) > z_surface) {
            constexpr Real penalty = 1e15;
            triplets.emplace_back(cell, cell, penalty);
        }
    }
};

} // namespace dgw
