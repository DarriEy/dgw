/**
 * @file config.cpp
 * @brief Configuration parsing and validation
 */

#include "dgw/core/config.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace dgw {

// ============================================================================
// PhysicsDecisions
// ============================================================================

void PhysicsDecisions::print(std::ostream& os) const {
    os << "Physics Decisions:\n";
    os << "  Governing equation: " << config_io::to_string(governing_equation) << "\n";
    os << "  Transmissivity:     " << config_io::to_string(transmissivity) << "\n";
    os << "  Stream exchange:    " << config_io::to_string(stream_exchange) << "\n";
    os << "  Vadose method:      " << config_io::to_string(vadose) << "\n";
    os << "  Retention model:    " << config_io::to_string(retention) << "\n";
    if (governing_equation == GoverningEquation::MultiLayer) {
        os << "  Number of layers:   " << n_layers << "\n";
    }
}

bool PhysicsDecisions::validate() const {
    // Richards3D requires specific retention model
    if (governing_equation == GoverningEquation::Richards3D) {
        if (retention == RetentionModel::Tabulated) {
            // Tabulated not yet supported
            return false;
        }
    }

    // MultiLayer needs > 1 layer
    if (governing_equation == GoverningEquation::MultiLayer && n_layers < 2) {
        return false;
    }

    return true;
}

// ============================================================================
// Config
// ============================================================================

Config Config::from_file(const std::filesystem::path& filepath) {
    Config config;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filepath.string());
    }

    // Simple key-value parser (no YAML library dependency)
    std::string line;
    std::string current_section;

    while (std::getline(file, line)) {
        // Strip comments
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        // Trim whitespace
        auto start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        // Section headers (lines ending with ':' and no value)
        auto colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);

            // Trim value
            auto val_start = value.find_first_not_of(" \t");
            if (val_start == std::string::npos) {
                // Section header
                current_section = key;
                continue;
            }
            value = value.substr(val_start);
            auto val_end = value.find_last_not_of(" \t\r\n");
            if (val_end != std::string::npos) {
                value = value.substr(0, val_end + 1);
            }

            // Trim key
            auto key_end = key.find_last_not_of(" \t");
            if (key_end != std::string::npos) {
                key = key.substr(0, key_end + 1);
            }

            // Parse based on section
            if (current_section == "physics") {
                if (key == "governing_equation")
                    config.physics.governing_equation = config_io::governing_equation_from_string(value);
                else if (key == "transmissivity")
                    config.physics.transmissivity = config_io::transmissivity_from_string(value);
                else if (key == "stream_exchange")
                    config.physics.stream_exchange = config_io::stream_exchange_from_string(value);
                else if (key == "vadose")
                    config.physics.vadose = config_io::vadose_from_string(value);
                else if (key == "retention")
                    config.physics.retention = config_io::retention_from_string(value);
                else if (key == "n_layers")
                    config.physics.n_layers = std::stoi(value);
            } else if (current_section == "solver") {
                if (key == "nonlinear_solver")
                    config.solver.nonlinear_solver = config_io::nonlinear_solver_from_string(value);
                else if (key == "max_newton_iterations")
                    config.solver.max_newton_iterations = std::stoi(value);
                else if (key == "newton_tolerance")
                    config.solver.newton_tolerance = std::stod(value);
                else if (key == "linear_solver")
                    config.solver.linear_solver = config_io::linear_solver_from_string(value);
                else if (key == "time_stepping")
                    config.solver.time_stepping = config_io::time_stepping_from_string(value);
                else if (key == "dt_initial")
                    config.solver.dt_initial = std::stod(value);
                else if (key == "dt_min")
                    config.solver.dt_min = std::stod(value);
                else if (key == "dt_max")
                    config.solver.dt_max = std::stod(value);
            } else if (current_section == "time") {
                if (key == "start_time")
                    config.time.start_time = std::stod(value);
                else if (key == "end_time")
                    config.time.end_time = std::stod(value);
                else if (key == "output_interval")
                    config.time.output_interval = std::stod(value);
            } else if (current_section == "output") {
                if (key == "output_dir")
                    config.output.output_dir = value;
                else if (key == "output_prefix")
                    config.output.output_prefix = value;
            } else if (current_section == "coupling") {
                if (key == "coupled_to_land_surface")
                    config.coupling.coupled_to_land_surface = (value == "true");
                else if (key == "coupled_to_routing")
                    config.coupling.coupled_to_routing = (value == "true");
                else if (key == "coupling_dt")
                    config.coupling.coupling_dt = std::stod(value);
            } else if (current_section == "files" || current_section.empty()) {
                if (key == "mesh_file")
                    config.mesh_file = value;
                else if (key == "parameters_file")
                    config.parameters_file = value;
                else if (key == "initial_conditions_file")
                    config.initial_conditions_file = value;
                else if (key == "forcing_file")
                    config.forcing_file = value;
            }
        }
    }

    return config;
}

void Config::to_file(const std::filesystem::path& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot write config file: " + filepath.string());
    }

    file << "# dGW Configuration File\n\n";

    file << "files:\n";
    file << "  mesh_file: " << mesh_file.string() << "\n";
    file << "  parameters_file: " << parameters_file.string() << "\n";
    file << "  initial_conditions_file: " << initial_conditions_file.string() << "\n";
    file << "  forcing_file: " << forcing_file.string() << "\n\n";

    file << "physics:\n";
    file << "  governing_equation: " << config_io::to_string(physics.governing_equation) << "\n";
    file << "  transmissivity: " << config_io::to_string(physics.transmissivity) << "\n";
    file << "  stream_exchange: " << config_io::to_string(physics.stream_exchange) << "\n";
    file << "  vadose: " << config_io::to_string(physics.vadose) << "\n";
    file << "  retention: " << config_io::to_string(physics.retention) << "\n";
    file << "  n_layers: " << physics.n_layers << "\n\n";

    file << "solver:\n";
    file << "  nonlinear_solver: " << config_io::to_string(solver.nonlinear_solver) << "\n";
    file << "  max_newton_iterations: " << solver.max_newton_iterations << "\n";
    file << "  newton_tolerance: " << solver.newton_tolerance << "\n";
    file << "  linear_solver: " << config_io::to_string(solver.linear_solver) << "\n";
    file << "  time_stepping: " << config_io::to_string(solver.time_stepping) << "\n";
    file << "  dt_initial: " << solver.dt_initial << "\n";
    file << "  dt_min: " << solver.dt_min << "\n";
    file << "  dt_max: " << solver.dt_max << "\n\n";

    file << "time:\n";
    file << "  start_time: " << time.start_time << "\n";
    file << "  end_time: " << time.end_time << "\n";
    file << "  output_interval: " << time.output_interval << "\n\n";

    file << "output:\n";
    file << "  output_dir: " << output.output_dir.string() << "\n";
    file << "  output_prefix: " << output.output_prefix << "\n";
}

bool Config::validate() const {
    if (!physics.validate()) return false;

    // Check solver compatibility
    if (solver.dt_min >= solver.dt_max) return false;
    if (solver.dt_initial <= 0) return false;
    if (time.end_time <= time.start_time) return false;

    return true;
}

void Config::print_summary(std::ostream& os) const {
    os << "=== dGW Configuration ===\n";
    physics.print(os);
    os << "Solver:\n";
    os << "  Nonlinear: " << config_io::to_string(solver.nonlinear_solver) << "\n";
    os << "  Linear:    " << config_io::to_string(solver.linear_solver) << "\n";
    os << "  dt:        " << solver.dt_initial << " [" << solver.dt_min << ", " << solver.dt_max << "]\n";
    os << "Time:\n";
    os << "  Start:     " << time.start_time << "\n";
    os << "  End:       " << time.end_time << "\n";
    os << "========================\n";
}

void Config::validate_physics() const {
    physics.validate();
}

void Config::validate_solver() const {
    if (solver.max_newton_iterations < 1) {
        throw std::invalid_argument("max_newton_iterations must be >= 1");
    }
}

void Config::validate_paths() const {
    // Only validate if paths are set
    if (!mesh_file.empty() && !std::filesystem::exists(mesh_file)) {
        throw std::runtime_error("Mesh file not found: " + mesh_file.string());
    }
}

// ============================================================================
// config_io helpers
// ============================================================================

namespace config_io {

PhysicsDecisions parse_physics(const std::string& yaml_str) {
    PhysicsDecisions decisions;
    // Simplified parsing - delegates to from_file
    return decisions;
}

SolverConfig parse_solver(const std::string& yaml_str) {
    SolverConfig config;
    return config;
}

std::string to_string(GoverningEquation eq) {
    switch (eq) {
        case GoverningEquation::LinearDiffusion: return "LinearDiffusion";
        case GoverningEquation::Boussinesq: return "Boussinesq";
        case GoverningEquation::Confined: return "Confined";
        case GoverningEquation::TwoLayer: return "TwoLayer";
        case GoverningEquation::MultiLayer: return "MultiLayer";
        case GoverningEquation::Richards3D: return "Richards3D";
        default: return "Unknown";
    }
}

std::string to_string(TransmissivityMethod tm) {
    switch (tm) {
        case TransmissivityMethod::Standard: return "Standard";
        case TransmissivityMethod::Smoothed: return "Smoothed";
        case TransmissivityMethod::Harmonic: return "Harmonic";
        case TransmissivityMethod::Upstream: return "Upstream";
        default: return "Unknown";
    }
}

std::string to_string(StreamExchangeMethod se) {
    switch (se) {
        case StreamExchangeMethod::Conductance: return "Conductance";
        case StreamExchangeMethod::ConductanceClogging: return "ConductanceClogging";
        case StreamExchangeMethod::KinematicLosing: return "KinematicLosing";
        case StreamExchangeMethod::SaturatedUnsaturated: return "SaturatedUnsaturated";
        default: return "Unknown";
    }
}

std::string to_string(VadoseMethod vm) {
    switch (vm) {
        case VadoseMethod::Direct: return "Direct";
        case VadoseMethod::ExponentialLag: return "ExponentialLag";
        case VadoseMethod::KinematicWave: return "KinematicWave";
        case VadoseMethod::FullRichards: return "FullRichards";
        default: return "Unknown";
    }
}

std::string to_string(RetentionModel rm) {
    switch (rm) {
        case RetentionModel::VanGenuchten: return "VanGenuchten";
        case RetentionModel::BrooksCorey: return "BrooksCorey";
        case RetentionModel::ClappHornberger: return "ClappHornberger";
        case RetentionModel::Tabulated: return "Tabulated";
        default: return "Unknown";
    }
}

std::string to_string(LinearSolver ls) {
    switch (ls) {
        case LinearSolver::EigenLU: return "EigenLU";
        case LinearSolver::EigenCholesky: return "EigenCholesky";
        case LinearSolver::EigenCG: return "EigenCG";
        case LinearSolver::EigenBiCGSTAB: return "EigenBiCGSTAB";
#ifdef DGW_HAS_PETSC
        case LinearSolver::PETScKSP: return "PETScKSP";
        case LinearSolver::PETScDirect: return "PETScDirect";
#endif
        default: return "Unknown";
    }
}

std::string to_string(NonlinearSolver ns) {
    switch (ns) {
        case NonlinearSolver::Newton: return "Newton";
        case NonlinearSolver::Picard: return "Picard";
        case NonlinearSolver::NewtonLineSearch: return "NewtonLineSearch";
        case NonlinearSolver::TrustRegion: return "TrustRegion";
        default: return "Unknown";
    }
}

std::string to_string(TimeSteppingMethod ts) {
    switch (ts) {
        case TimeSteppingMethod::BackwardEuler: return "BackwardEuler";
        case TimeSteppingMethod::CrankNicolson: return "CrankNicolson";
        case TimeSteppingMethod::BDF2: return "BDF2";
        case TimeSteppingMethod::Adaptive: return "Adaptive";
        default: return "Unknown";
    }
}

// String to enum converters

GoverningEquation governing_equation_from_string(const std::string& s) {
    if (s == "LinearDiffusion") return GoverningEquation::LinearDiffusion;
    if (s == "Boussinesq") return GoverningEquation::Boussinesq;
    if (s == "Confined") return GoverningEquation::Confined;
    if (s == "TwoLayer") return GoverningEquation::TwoLayer;
    if (s == "MultiLayer") return GoverningEquation::MultiLayer;
    if (s == "Richards3D") return GoverningEquation::Richards3D;
    throw std::invalid_argument("Unknown governing equation: " + s);
}

TransmissivityMethod transmissivity_from_string(const std::string& s) {
    if (s == "Standard") return TransmissivityMethod::Standard;
    if (s == "Smoothed") return TransmissivityMethod::Smoothed;
    if (s == "Harmonic") return TransmissivityMethod::Harmonic;
    if (s == "Upstream") return TransmissivityMethod::Upstream;
    throw std::invalid_argument("Unknown transmissivity method: " + s);
}

StreamExchangeMethod stream_exchange_from_string(const std::string& s) {
    if (s == "Conductance") return StreamExchangeMethod::Conductance;
    if (s == "ConductanceClogging") return StreamExchangeMethod::ConductanceClogging;
    if (s == "KinematicLosing") return StreamExchangeMethod::KinematicLosing;
    if (s == "SaturatedUnsaturated") return StreamExchangeMethod::SaturatedUnsaturated;
    throw std::invalid_argument("Unknown stream exchange method: " + s);
}

VadoseMethod vadose_from_string(const std::string& s) {
    if (s == "Direct") return VadoseMethod::Direct;
    if (s == "ExponentialLag") return VadoseMethod::ExponentialLag;
    if (s == "KinematicWave") return VadoseMethod::KinematicWave;
    if (s == "FullRichards") return VadoseMethod::FullRichards;
    throw std::invalid_argument("Unknown vadose method: " + s);
}

RetentionModel retention_from_string(const std::string& s) {
    if (s == "VanGenuchten") return RetentionModel::VanGenuchten;
    if (s == "BrooksCorey") return RetentionModel::BrooksCorey;
    if (s == "ClappHornberger") return RetentionModel::ClappHornberger;
    if (s == "Tabulated") return RetentionModel::Tabulated;
    throw std::invalid_argument("Unknown retention model: " + s);
}

LinearSolver linear_solver_from_string(const std::string& s) {
    if (s == "EigenLU") return LinearSolver::EigenLU;
    if (s == "EigenCholesky") return LinearSolver::EigenCholesky;
    if (s == "EigenCG") return LinearSolver::EigenCG;
    if (s == "EigenBiCGSTAB") return LinearSolver::EigenBiCGSTAB;
#ifdef DGW_HAS_PETSC
    if (s == "PETScKSP") return LinearSolver::PETScKSP;
    if (s == "PETScDirect") return LinearSolver::PETScDirect;
#endif
    throw std::invalid_argument("Unknown linear solver: " + s);
}

NonlinearSolver nonlinear_solver_from_string(const std::string& s) {
    if (s == "Newton") return NonlinearSolver::Newton;
    if (s == "Picard") return NonlinearSolver::Picard;
    if (s == "NewtonLineSearch") return NonlinearSolver::NewtonLineSearch;
    if (s == "TrustRegion") return NonlinearSolver::TrustRegion;
    throw std::invalid_argument("Unknown nonlinear solver: " + s);
}

TimeSteppingMethod time_stepping_from_string(const std::string& s) {
    if (s == "BackwardEuler") return TimeSteppingMethod::BackwardEuler;
    if (s == "CrankNicolson") return TimeSteppingMethod::CrankNicolson;
    if (s == "BDF2") return TimeSteppingMethod::BDF2;
    if (s == "Adaptive") return TimeSteppingMethod::Adaptive;
    throw std::invalid_argument("Unknown time stepping method: " + s);
}

} // namespace config_io

} // namespace dgw
