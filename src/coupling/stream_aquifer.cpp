/**
 * @file stream_aquifer.cpp
 * @brief Stream-aquifer exchange implementation
 */

#include "dgw/coupling/stream_aquifer.hpp"
#include <cmath>

namespace dgw {

StreamAquiferExchange::StreamAquiferExchange(StreamExchangeMethod method)
    : method_(method) {}

Vector StreamAquiferExchange::compute_exchange(
    const Vector& h_gw, const Vector& h_stream,
    const StreamParameters& params, const Mesh& mesh
) const {
    auto river_cells = mesh.river_cells();
    Vector exchange(mesh.n_cells());
    exchange.setZero();

    for (Index cell_id : river_cells) {
        exchange(cell_id) = compute_exchange_cell(
            cell_id, h_gw(cell_id), h_stream(cell_id), params, mesh);
    }
    return exchange;
}

Real StreamAquiferExchange::compute_exchange_cell(
    Index cell, Real h_gw, Real h_stream,
    const StreamParameters& params, const Mesh& mesh
) const {
    const RiverSegment* seg = mesh.river_segment(cell);
    if (!seg) return 0.0;

    Real C = params.conductance()(cell);

    switch (method_) {
        case StreamExchangeMethod::Conductance:
            return exchange_conductance(h_gw, h_stream, C);

        case StreamExchangeMethod::ConductanceClogging:
            return exchange_conductance_clogging(
                h_gw, h_stream,
                params.streambed_K(cell), params.streambed_thickness(cell),
                params.has_clogging ? params.clogging_K(cell) : params.streambed_K(cell),
                params.has_clogging ? params.clogging_thickness(cell) : 0.0,
                seg->length, seg->width);

        case StreamExchangeMethod::KinematicLosing:
            return exchange_kinematic_losing(h_gw, h_stream, seg->streambed_elevation, C);

        case StreamExchangeMethod::SaturatedUnsaturated:
            return exchange_saturated_unsaturated(
                h_gw, h_stream, seg->streambed_elevation,
                seg->streambed_thickness, C);

        default:
            return exchange_conductance(h_gw, h_stream, C);
    }
}

Real StreamAquiferExchange::dQ_dh_gw(
    Index cell, Real h_gw, Real h_stream,
    const StreamParameters& params, const Mesh& mesh
) const {
    const RiverSegment* seg = mesh.river_segment(cell);
    if (!seg) return 0.0;

    Real C = params.conductance()(cell);

    switch (method_) {
        case StreamExchangeMethod::Conductance:
        case StreamExchangeMethod::ConductanceClogging:
            return -C;

        case StreamExchangeMethod::KinematicLosing:
            return stream_kernels::kinematic_losing_dQ_dhgw(
                h_gw, seg->streambed_elevation, C);

        case StreamExchangeMethod::SaturatedUnsaturated: {
            // Numerical derivative
            Real eps = 1e-6;
            Real Q1 = compute_exchange_cell(cell, h_gw + eps, h_stream, params, mesh);
            Real Q0 = compute_exchange_cell(cell, h_gw, h_stream, params, mesh);
            return (Q1 - Q0) / eps;
        }

        default:
            return -C;
    }
}

Real StreamAquiferExchange::dQ_dh_stream(
    Index cell, Real h_gw, Real h_stream,
    const StreamParameters& params, const Mesh& mesh
) const {
    Real C = params.conductance()(cell);
    return C;  // Q = C * (h_stream - h_gw), so dQ/dh_stream = C
}

void StreamAquiferExchange::add_to_residual(
    const Vector& h_gw, const Vector& h_stream,
    const StreamParameters& params, const Mesh& mesh,
    Vector& residual
) const {
    for (Index cell_id : mesh.river_cells()) {
        Real Q = compute_exchange_cell(
            cell_id, h_gw(cell_id), h_stream(cell_id), params, mesh);
        // Exchange leaves aquifer (positive = gaining stream)
        residual(cell_id) += Q;
    }
}

void StreamAquiferExchange::add_to_jacobian(
    const Vector& h_gw, const Vector& h_stream,
    const StreamParameters& params, const Mesh& mesh,
    std::vector<SparseTriplet>& triplets
) const {
    for (Index cell_id : mesh.river_cells()) {
        Real dQ = dQ_dh_gw(cell_id, h_gw(cell_id), h_stream(cell_id), params, mesh);
        triplets.emplace_back(cell_id, cell_id, -dQ);  // Note sign
    }
}

// Private method implementations

Real StreamAquiferExchange::exchange_conductance(
    Real h_gw, Real h_stream, Real conductance
) const {
    return stream_kernels::conductance_exchange(h_gw, h_stream, conductance);
}

Real StreamAquiferExchange::exchange_conductance_clogging(
    Real h_gw, Real h_stream,
    Real streambed_K, Real streambed_thick,
    Real clogging_K, Real clogging_thick,
    Real length, Real width
) const {
    Real C_bed = streambed_K * length * width / (streambed_thick + 1e-30);
    Real C_clog = clogging_K * length * width / (clogging_thick + 1e-30);
    return stream_kernels::clogging_exchange(h_gw, h_stream, C_bed, C_clog);
}

Real StreamAquiferExchange::exchange_kinematic_losing(
    Real h_gw, Real h_stream, Real streambed_elev, Real conductance
) const {
    return stream_kernels::kinematic_losing_exchange(
        h_gw, h_stream, streambed_elev, conductance);
}

Real StreamAquiferExchange::exchange_saturated_unsaturated(
    Real h_gw, Real h_stream, Real streambed_elev,
    Real streambed_thick, Real conductance
) const {
    return stream_kernels::smooth_transition_exchange(
        h_gw, h_stream, streambed_elev, streambed_thick, conductance);
}

// SimpleRoutingCoupler

SimpleRoutingCoupler::SimpleRoutingCoupler(const Mesh& gw_mesh) {
    auto river = gw_mesh.river_cells();
    reach_to_cell_.assign(river.begin(), river.end());
    stage_.resize(static_cast<Index>(river.size()));
    stage_.setZero();
    exchange_.resize(static_cast<Index>(river.size()));
    exchange_.setZero();
}

void SimpleRoutingCoupler::receive_stage(const Vector& stage) {
    stage_ = stage;
}

Vector SimpleRoutingCoupler::send_exchange() const {
    return exchange_;
}

} // namespace dgw
