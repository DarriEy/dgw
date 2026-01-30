/**
 * @file physics_base.hpp
 * @brief Abstract base class for all physics modules
 * 
 * Defines the interface that all physics implementations must follow.
 * This enables SUMMA-style swappable physics at runtime.
 */

#pragma once

#include "../core/types.hpp"
#include "../core/mesh.hpp"
#include "../core/state.hpp"
#include "../core/parameters.hpp"
#include "../core/config.hpp"

namespace dgw {

/**
 * @brief Abstract base class for physics modules
 * 
 * All physics implementations (LinearDiffusion, Boussinesq, Richards, etc.)
 * inherit from this class and implement the virtual methods.
 */
class PhysicsBase {
public:
    virtual ~PhysicsBase() = default;
    
    /// Physics type identifier
    virtual GoverningEquation type() const = 0;
    
    /// Human-readable name
    virtual std::string name() const = 0;
    
    // ========================================================================
    // Core Computation Methods
    // ========================================================================
    
    /**
     * @brief Compute residual F(h) for the governing equation
     * 
     * For time stepping: F(h) = M*(h - h_old)/dt - A*h - b = 0
     * where M is mass matrix, A is stiffness matrix, b is source terms.
     * 
     * @param state Current state
     * @param params Parameters
     * @param mesh Mesh
     * @param dt Timestep
     * @param[out] residual Output residual vector
     */
    virtual void compute_residual(
        const State& state,
        const Parameters& params,
        const Mesh& mesh,
        Real dt,
        Vector& residual
    ) const = 0;
    
    /**
     * @brief Compute Jacobian J = ∂F/∂h
     * 
     * @param state Current state
     * @param params Parameters
     * @param mesh Mesh
     * @param dt Timestep
     * @param[out] jacobian Output sparse Jacobian matrix
     */
    virtual void compute_jacobian(
        const State& state,
        const Parameters& params,
        const Mesh& mesh,
        Real dt,
        SparseMatrix& jacobian
    ) const = 0;
    
    /**
     * @brief Compute fluxes at cell faces
     * 
     * @param state Current state
     * @param params Parameters
     * @param mesh Mesh
     * @param[out] face_fluxes Output flux at each face [m³/s]
     */
    virtual void compute_fluxes(
        const State& state,
        const Parameters& params,
        const Mesh& mesh,
        Vector& face_fluxes
    ) const = 0;
    
    // ========================================================================
    // Initialization
    // ========================================================================
    
    /**
     * @brief Initialize state for this physics
     * 
     * @param mesh Mesh
     * @param params Parameters
     * @param config Configuration
     * @param[out] state Initialized state
     */
    virtual void initialize_state(
        const Mesh& mesh,
        const Parameters& params,
        const Config& config,
        State& state
    ) const = 0;
    
    /**
     * @brief Allocate Jacobian sparsity pattern
     * 
     * @param mesh Mesh
     * @return Jacobian with correct sparsity pattern (values = 0)
     */
    virtual SparseMatrix allocate_jacobian(const Mesh& mesh) const = 0;
    
    // ========================================================================
    // Source Terms and Boundary Conditions
    // ========================================================================
    
    /**
     * @brief Set recharge (from land surface model)
     * 
     * @param recharge_rate Recharge rate per cell [m/s]
     */
    virtual void set_recharge(const Vector& recharge_rate) = 0;
    
    /**
     * @brief Set stream stages (from routing model)
     * 
     * @param stream_stage Stage at each river cell [m]
     */
    virtual void set_stream_stage(const Vector& stream_stage) = 0;
    
    /**
     * @brief Set pumping rates
     * 
     * @param pumping Pumping rate at each cell [m³/s] (positive = extraction)
     */
    virtual void set_pumping(const Vector& pumping) = 0;
    
    /**
     * @brief Apply boundary conditions to residual and Jacobian
     */
    virtual void apply_boundary_conditions(
        const Mesh& mesh,
        const Parameters& params,
        Vector& residual,
        SparseMatrix& jacobian
    ) const = 0;
    
    // ========================================================================
    // Outputs
    // ========================================================================
    
    /**
     * @brief Compute water table depth below surface
     */
    virtual Vector water_table_depth(
        const State& state,
        const Mesh& mesh
    ) const = 0;
    
    /**
     * @brief Compute stream-aquifer exchange flux
     * 
     * @return Exchange flux at each river cell [m³/s], positive = gaining stream
     */
    virtual Vector stream_exchange(
        const State& state,
        const Parameters& params,
        const Mesh& mesh
    ) const = 0;
    
    /**
     * @brief Compute total storage
     * 
     * @return Total water volume in system [m³]
     */
    virtual Real total_storage(
        const State& state,
        const Parameters& params,
        const Mesh& mesh
    ) const = 0;
    
    /**
     * @brief Compute mass balance error
     */
    virtual Real mass_balance_error(
        const State& state,
        const Parameters& params,
        const Mesh& mesh,
        Real dt
    ) const = 0;
    
    // ========================================================================
    // Automatic Differentiation Support
    // ========================================================================
    
    /**
     * @brief Compute gradient of output w.r.t. parameters using Enzyme
     * 
     * For adjoint sensitivity: given ∂L/∂h, compute ∂L/∂θ
     * 
     * @param state Current state
     * @param params Parameters
     * @param mesh Mesh
     * @param adjoint_state Adjoint (∂L/∂h)
     * @param[out] param_gradients Gradients w.r.t. parameters
     */
    virtual void compute_parameter_gradients(
        const State& state,
        const Parameters& params,
        const Mesh& mesh,
        const Vector& adjoint_state,
        Parameters& param_gradients
    ) const;
    
    /**
     * @brief Check if gradients are available
     */
    virtual bool supports_gradients() const { return true; }
    
protected:
    // Cached source terms
    Vector recharge_;
    Vector stream_stage_;
    Vector pumping_;
};

/**
 * @brief Factory function to create physics module from config
 */
UniquePtr<PhysicsBase> create_physics(const Config& config);

/**
 * @brief Factory function to create physics module by type
 */
UniquePtr<PhysicsBase> create_physics(GoverningEquation type);

} // namespace dgw
