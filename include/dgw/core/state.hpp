/**
 * @file state.hpp
 * @brief Model state for dGW
 * 
 * Contains all prognostic (time-evolving) variables:
 * - Head/pressure for each cell/layer
 * - Vadose zone storage (if applicable)
 * - Mass balance tracking
 */

#pragma once

#include "types.hpp"
#include "mesh.hpp"

namespace dgw {

/**
 * @brief State for 2D single-layer groundwater model
 */
struct State2D {
    Vector head;                    ///< Hydraulic head [m]
    Vector head_old;                ///< Head at previous timestep
    Vector vadose_storage;          ///< Water in vadose zone [m]
    
    // Fluxes (for output/mass balance)
    Vector recharge_flux;           ///< Recharge to water table [m/s]
    Vector stream_exchange;         ///< Stream-aquifer exchange [m³/s]
    Vector boundary_flux;           ///< Flux at boundaries [m³/s]
    Vector cell_storage_change;     ///< ∂(Sy*h)/∂t * Volume
    
    // Time tracking
    Real time = 0.0;                ///< Current simulation time [s]
    Real dt = 0.0;                  ///< Current timestep [s]
    
    /// Initialize state for given mesh
    void initialize(const Mesh& mesh, Real initial_head);
    
    /// Initialize from spatially variable initial condition
    void initialize(const Mesh& mesh, const Vector& initial_head);
    
    /// Copy current state to old state
    void advance_time(Real new_dt) {
        head_old = head;
        dt = new_dt;
        time += new_dt;
    }
    
    /// Compute total storage [m³]
    Real total_storage(const Mesh& mesh, const Vector& Sy) const;
    
    /// Compute mass balance error
    Real mass_balance_error() const;
};

/**
 * @brief State for two-layer groundwater model
 */
struct StateTwoLayer {
    // Layer 1: Unconfined
    Vector h1;                      ///< Water table elevation [m]
    Vector h1_old;                  ///< Previous timestep
    Vector vadose_storage;          ///< Vadose zone storage [m]
    
    // Layer 2: Confined
    Vector h2;                      ///< Confined aquifer head [m]
    Vector h2_old;                  ///< Previous timestep
    
    // Inter-layer exchange
    Vector leakage;                 ///< Leakage from layer 1 to 2 [m³/s]
    
    // External fluxes
    Vector recharge;                ///< Recharge to layer 1 [m/s]
    Vector pumping_1;               ///< Pumping from layer 1 [m³/s]
    Vector pumping_2;               ///< Pumping from layer 2 [m³/s]
    Vector stream_exchange;         ///< River exchange (layer 1 only) [m³/s]
    
    Real time = 0.0;
    Real dt = 0.0;
    
    void initialize(const Mesh& mesh, Real h1_init, Real h2_init);
    void initialize(const Mesh& mesh, const Vector& h1_init, const Vector& h2_init);
    void advance_time(Real new_dt);
};

/**
 * @brief State for multi-layer groundwater model
 */
struct StateMultiLayer {
    /// Head for each layer [n_layers][n_cells]
    std::vector<Vector> head;
    std::vector<Vector> head_old;
    
    /// Vertical leakage between layers [n_layers-1][n_cells]
    std::vector<Vector> vertical_leakage;
    
    Vector recharge;                ///< Top boundary recharge
    std::vector<Vector> pumping;    ///< Pumping per layer
    Vector stream_exchange;         ///< River exchange (top layer)
    
    Real time = 0.0;
    Real dt = 0.0;
    Index n_layers = 0;
    
    void initialize(const MeshLayered& mesh, const std::vector<Real>& initial_heads);
    void advance_time(Real new_dt);
    
    /// Get head at (column, layer)
    Real& head_at(Index col, Index layer) { return head[layer](col); }
    const Real& head_at(Index col, Index layer) const { return head[layer](col); }
};

/**
 * @brief State for 3D Richards equation
 * 
 * Uses pressure head ψ as primary variable:
 *   h = ψ + z  (total head)
 *   ψ < 0 unsaturated, ψ ≥ 0 saturated
 */
struct StateRichards3D {
    Vector pressure_head;           ///< Pressure head ψ [m] (negative in unsaturated)
    Vector pressure_head_old;       ///< Previous timestep
    
    // Derived quantities (updated each iteration)
    Vector water_content;           ///< θ(ψ) volumetric water content [-]
    Vector hydraulic_conductivity;  ///< K(ψ) [m/s]
    Vector specific_moisture_capacity; ///< C(ψ) = dθ/dψ [1/m]
    Vector saturation;              ///< Se = (θ - θr)/(θs - θr) [-]
    
    // Fluxes
    Vector recharge;                ///< Surface recharge [m/s]
    Vector evapotranspiration;      ///< ET sink [m/s] (root zone)
    Vector bottom_flux;             ///< Bottom boundary flux [m/s]
    Vector lateral_flux;            ///< Lateral boundary flux [m³/s]
    
    Real time = 0.0;
    Real dt = 0.0;
    
    void initialize(const Mesh3D& mesh, Real initial_psi);
    void initialize(const Mesh3D& mesh, const Vector& initial_psi);
    void initialize_hydrostatic(const Mesh3D& mesh, Real water_table_elevation);
    void advance_time(Real new_dt);
    
    /// Update derived quantities from current pressure head
    void update_constitutive(const Vector& theta_r, const Vector& theta_s,
                             const Vector& alpha, const Vector& n_vg,
                             const Vector& K_sat, RetentionModel model);
    
    /// Get total head h = ψ + z at cell i
    Real total_head(Index i, const Mesh3D& mesh) const {
        return pressure_head(i) + mesh.cell_centroid(i).z();
    }
    
    /// Check if cell is saturated
    bool is_saturated(Index i) const {
        return pressure_head(i) >= 0.0;
    }
    
    /// Compute total water storage [m³]
    Real total_water_storage(const Mesh3D& mesh) const;
};

/**
 * @brief Combined state that holds whichever physics is active
 */
class State {
public:
    State() = default;
    
    /// Construct for specific physics type
    explicit State(GoverningEquation physics) : physics_(physics) {}
    
    GoverningEquation physics() const { return physics_; }
    
    // Type-safe accessors
    State2D& as_2d() { return std::get<State2D>(data_); }
    const State2D& as_2d() const { return std::get<State2D>(data_); }
    
    StateTwoLayer& as_two_layer() { return std::get<StateTwoLayer>(data_); }
    const StateTwoLayer& as_two_layer() const { return std::get<StateTwoLayer>(data_); }
    
    StateMultiLayer& as_multi_layer() { return std::get<StateMultiLayer>(data_); }
    const StateMultiLayer& as_multi_layer() const { return std::get<StateMultiLayer>(data_); }
    
    StateRichards3D& as_richards() { return std::get<StateRichards3D>(data_); }
    const StateRichards3D& as_richards() const { return std::get<StateRichards3D>(data_); }
    
    /// Initialize based on physics type and mesh
    void initialize(const Mesh& mesh, const Config& config);
    
    /// Get current time
    Real time() const;
    
    /// Get primary state vector (for solver interface)
    Vector& primary_state();
    const Vector& primary_state() const;
    
    /// Pack all state into flat vector (for checkpointing)
    Vector pack() const;
    
    /// Unpack state from flat vector
    void unpack(const Vector& packed);
    
private:
    GoverningEquation physics_ = GoverningEquation::Boussinesq;
    std::variant<State2D, StateTwoLayer, StateMultiLayer, StateRichards3D> data_;
};

} // namespace dgw
