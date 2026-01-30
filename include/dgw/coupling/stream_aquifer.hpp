/**
 * @file stream_aquifer.hpp
 * @brief Stream-aquifer exchange calculations
 * 
 * Implements various formulations for groundwater-surface water interaction:
 * - Simple conductance (MODFLOW RIV/SFR style)
 * - Conductance with clogging layer
 * - Disconnected stream (losing reach in arid regions)
 * - Dynamic streambed evolution
 */

#pragma once

#include "../core/types.hpp"
#include "../core/mesh.hpp"
#include "../core/parameters.hpp"

namespace dgw {

/**
 * @brief Stream-aquifer exchange calculator
 */
class StreamAquiferExchange {
public:
    StreamAquiferExchange() = default;
    explicit StreamAquiferExchange(StreamExchangeMethod method);
    
    /**
     * @brief Set exchange method
     */
    void set_method(StreamExchangeMethod method) { method_ = method; }
    StreamExchangeMethod method() const { return method_; }
    
    /**
     * @brief Compute exchange flux for all river cells
     * 
     * @param h_gw Groundwater head at river cells [m]
     * @param h_stream Stream stage at river cells [m]
     * @param params Stream parameters
     * @param mesh Mesh with river cell information
     * @return Exchange flux [m³/s], positive = gaining (GW → stream)
     */
    Vector compute_exchange(
        const Vector& h_gw,
        const Vector& h_stream,
        const StreamParameters& params,
        const Mesh& mesh
    ) const;
    
    /**
     * @brief Compute exchange for single cell
     */
    Real compute_exchange_cell(
        Index cell,
        Real h_gw,
        Real h_stream,
        const StreamParameters& params,
        const Mesh& mesh
    ) const;
    
    /**
     * @brief Compute derivative of exchange w.r.t. groundwater head
     * 
     * Needed for Jacobian assembly.
     */
    Real dQ_dh_gw(
        Index cell,
        Real h_gw,
        Real h_stream,
        const StreamParameters& params,
        const Mesh& mesh
    ) const;
    
    /**
     * @brief Compute derivative of exchange w.r.t. stream stage
     */
    Real dQ_dh_stream(
        Index cell,
        Real h_gw,
        Real h_stream,
        const StreamParameters& params,
        const Mesh& mesh
    ) const;
    
    /**
     * @brief Add stream exchange contribution to residual
     */
    void add_to_residual(
        const Vector& h_gw,
        const Vector& h_stream,
        const StreamParameters& params,
        const Mesh& mesh,
        Vector& residual
    ) const;
    
    /**
     * @brief Add stream exchange contribution to Jacobian
     */
    void add_to_jacobian(
        const Vector& h_gw,
        const Vector& h_stream,
        const StreamParameters& params,
        const Mesh& mesh,
        std::vector<SparseTriplet>& triplets
    ) const;
    
private:
    StreamExchangeMethod method_ = StreamExchangeMethod::Conductance;
    
    // Method-specific implementations
    Real exchange_conductance(
        Real h_gw, Real h_stream,
        Real conductance
    ) const;
    
    Real exchange_conductance_clogging(
        Real h_gw, Real h_stream,
        Real streambed_K, Real streambed_thick,
        Real clogging_K, Real clogging_thick,
        Real length, Real width
    ) const;
    
    Real exchange_kinematic_losing(
        Real h_gw, Real h_stream,
        Real streambed_elev,
        Real conductance
    ) const;
    
    Real exchange_saturated_unsaturated(
        Real h_gw, Real h_stream,
        Real streambed_elev,
        Real streambed_thick,
        Real conductance
    ) const;
};

// ============================================================================
// Enzyme-compatible kernels
// ============================================================================

namespace stream_kernels {

/**
 * @brief Simple conductance exchange
 * 
 * Q = C * (h_stream - h_gw)
 * 
 * Positive Q = gaining stream (GW flows into stream)
 */
inline Real conductance_exchange(
    Real h_gw,
    Real h_stream,
    Real conductance
) {
    return conductance * (h_stream - h_gw);
}

/**
 * @brief Derivative of conductance exchange w.r.t. h_gw
 */
inline Real conductance_dQ_dhgw(Real conductance) {
    return -conductance;
}

/**
 * @brief Exchange with clogging layer
 * 
 * Total resistance = streambed resistance + clogging layer resistance
 * C_eff = 1 / (1/C_bed + 1/C_clog)
 */
inline Real clogging_exchange(
    Real h_gw,
    Real h_stream,
    Real C_bed,
    Real C_clog
) {
    Real C_eff = 1.0 / (1.0/C_bed + 1.0/C_clog);
    return C_eff * (h_stream - h_gw);
}

/**
 * @brief Disconnected (kinematic losing) stream
 * 
 * For arid regions where stream is perched above water table.
 * When h_gw < streambed_elev, the stream is disconnected and
 * flux depends only on (h_stream - streambed_elev).
 */
inline Real kinematic_losing_exchange(
    Real h_gw,
    Real h_stream,
    Real streambed_elev,
    Real conductance
) {
    if (h_gw >= streambed_elev) {
        // Connected: normal conductance
        return conductance * (h_stream - h_gw);
    } else {
        // Disconnected: fixed driving head
        // Stream always loses at rate controlled by streambed
        return conductance * (h_stream - streambed_elev);
    }
}

/**
 * @brief Derivative of kinematic losing exchange
 */
inline Real kinematic_losing_dQ_dhgw(
    Real h_gw,
    Real streambed_elev,
    Real conductance
) {
    if (h_gw >= streambed_elev) {
        return -conductance;
    } else {
        return 0.0;  // No dependence on h_gw when disconnected
    }
}

/**
 * @brief Smooth saturated-unsaturated transition
 * 
 * Uses smooth transition between connected and disconnected states
 * to avoid numerical issues from discontinuous derivative.
 */
inline Real smooth_transition_exchange(
    Real h_gw,
    Real h_stream,
    Real streambed_elev,
    Real streambed_thick,
    Real conductance,
    Real transition_zone = 0.1  // Width of smooth transition [m]
) {
    Real z_bot = streambed_elev - streambed_thick;
    
    if (h_gw >= streambed_elev) {
        // Fully connected
        return conductance * (h_stream - h_gw);
    } else if (h_gw <= z_bot) {
        // Fully disconnected
        return conductance * (h_stream - streambed_elev);
    } else {
        // Transition zone: smooth interpolation
        Real t = (h_gw - z_bot) / streambed_thick;
        // Smooth step: 3t² - 2t³
        Real blend = t * t * (3.0 - 2.0 * t);
        
        Real Q_connected = conductance * (h_stream - h_gw);
        Real Q_disconnected = conductance * (h_stream - streambed_elev);
        
        return blend * Q_connected + (1.0 - blend) * Q_disconnected;
    }
}

/**
 * @brief Compute streambed conductance
 * 
 * C = K_bed * L * W / b_bed
 * 
 * @param K_bed Streambed hydraulic conductivity [m/s]
 * @param length Stream length in cell [m]
 * @param width Stream width [m]
 * @param thickness Streambed thickness [m]
 */
inline Real compute_conductance(
    Real K_bed,
    Real length,
    Real width,
    Real thickness
) {
    return K_bed * length * width / thickness;
}

} // namespace stream_kernels

// ============================================================================
// River Routing Coupling Interface
// ============================================================================

/**
 * @brief Interface for coupling with river routing model (e.g., dRoute)
 */
class RoutingCoupler {
public:
    virtual ~RoutingCoupler() = default;
    
    /**
     * @brief Receive stream stages from routing model
     */
    virtual void receive_stage(const Vector& stage) = 0;
    
    /**
     * @brief Send exchange fluxes to routing model
     */
    virtual Vector send_exchange() const = 0;
    
    /**
     * @brief Get river cell mapping (routing reach → GW cell)
     */
    virtual const std::vector<Index>& reach_to_cell_map() const = 0;
};

/**
 * @brief Simple routing coupler with direct mapping
 */
class SimpleRoutingCoupler : public RoutingCoupler {
public:
    SimpleRoutingCoupler(const Mesh& gw_mesh);
    
    void receive_stage(const Vector& stage) override;
    Vector send_exchange() const override;
    const std::vector<Index>& reach_to_cell_map() const override {
        return reach_to_cell_;
    }
    
    void set_exchange(const Vector& exchange) { exchange_ = exchange; }
    const Vector& stage() const { return stage_; }
    
private:
    std::vector<Index> reach_to_cell_;
    Vector stage_;
    Vector exchange_;
};

} // namespace dgw
