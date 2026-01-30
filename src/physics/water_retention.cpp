/**
 * @file water_retention.cpp
 * @brief Soil water retention function implementations
 */

#include "dgw/physics/water_retention.hpp"
#include "dgw/physics/physics_base.hpp"
#include "dgw/physics/boussinesq.hpp"
#include "dgw/physics/confined.hpp"
#include "dgw/physics/two_layer.hpp"
#include "dgw/physics/richards_3d.hpp"
#include "dgw/physics/linear_diffusion.hpp"
#include "dgw/core/config.hpp"
#include <cmath>
#include <stdexcept>

namespace dgw {

namespace pedotransfer {

VanGenuchten from_texture_fractions(Real sand, Real silt, Real clay) {
    // Simple nearest-texture approach (Rosetta-like)
    // Uses USDA texture triangle classification

    // Normalize
    Real total = sand + silt + clay;
    if (total <= 0.0) {
        return from_texture(LOAM);  // Default
    }
    sand /= total;
    silt /= total;
    clay /= total;

    // Simplified USDA classification
    if (clay >= 0.40) {
        if (silt >= 0.40) return from_texture(SILTY_CLAY);
        if (sand >= 0.45) return from_texture(SANDY_CLAY);
        return from_texture(CLAY);
    }
    if (clay >= 0.27) {
        if (sand >= 0.20 && sand < 0.45) return from_texture(CLAY_LOAM);
        if (silt >= 0.40) return from_texture(SILTY_CLAY_LOAM);
        return from_texture(SANDY_CLAY_LOAM);
    }
    if (silt >= 0.80) return from_texture(SILT);
    if (silt >= 0.50) {
        if (clay < 0.12) return from_texture(SILT);
        return from_texture(SILT_LOAM);
    }
    if (sand >= 0.85) return from_texture(SAND);
    if (sand >= 0.70) return from_texture(LOAMY_SAND);
    if (sand >= 0.52) return from_texture(SANDY_LOAM);
    return from_texture(LOAM);
}

} // namespace pedotransfer

// PhysicsBase default implementation for parameter gradients
void PhysicsBase::compute_parameter_gradients(
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    const Vector& adjoint_state,
    Parameters& param_gradients
) const {
    // Default: no gradient computation (finite differences fallback)
    // Enzyme-enabled builds will override this
}

// Physics factory functions
UniquePtr<PhysicsBase> create_physics(const Config& config) {
    return create_physics(config.physics.governing_equation);
}

UniquePtr<PhysicsBase> create_physics(GoverningEquation type) {
    PhysicsDecisions decisions;
    decisions.governing_equation = type;

    switch (type) {
        case GoverningEquation::LinearDiffusion:
            return std::make_unique<LinearDiffusion>(decisions);
        case GoverningEquation::Boussinesq:
            return std::make_unique<BoussinesqSolver>(decisions);
        case GoverningEquation::Confined:
            return std::make_unique<ConfinedSolver>(decisions);
        case GoverningEquation::TwoLayer:
            return std::make_unique<TwoLayerSolver>(decisions);
        case GoverningEquation::Richards3D:
            return std::make_unique<Richards3DSolver>(decisions);
        case GoverningEquation::MultiLayer:
            throw std::runtime_error(
                "MultiLayer physics solver is not yet implemented. "
                "Use TwoLayer for two-layer systems or Richards3D for full 3D.");
        default:
            throw std::runtime_error("Unknown physics type");
    }
}

} // namespace dgw
