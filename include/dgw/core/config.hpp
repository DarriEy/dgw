/**
 * @file config.hpp
 * @brief Configuration and decisions for dGW (SUMMA-style)
 * 
 * Defines the runtime configuration including:
 * - Physics choices (governing equation, numerics)
 * - Solver options
 * - I/O settings
 * - Coupling options
 */

#pragma once

#include "types.hpp"
#include <string>
#include <filesystem>

namespace dgw {

/**
 * @brief Physics decisions (analogous to SUMMA decisions file)
 */
struct PhysicsDecisions {
    // Governing equation
    GoverningEquation governing_equation = GoverningEquation::Boussinesq;
    
    // Transmissivity formulation
    TransmissivityMethod transmissivity = TransmissivityMethod::Standard;
    
    // Storage coefficient behavior
    StorageMethod storage = StorageMethod::Constant;
    
    // Stream-aquifer exchange
    StreamExchangeMethod stream_exchange = StreamExchangeMethod::Conductance;
    
    // Vadose zone treatment
    VadoseMethod vadose = VadoseMethod::Direct;
    
    // Water retention (for Richards)
    RetentionModel retention = RetentionModel::VanGenuchten;
    
    // Number of layers (for MultiLayer)
    Index n_layers = 1;
    
    /// Print decisions to stream
    void print(std::ostream& os) const;
    
    /// Validate decisions consistency
    bool validate() const;
};

/**
 * @brief Solver configuration
 */
struct SolverConfig {
    // Nonlinear solver
    NonlinearSolver nonlinear_solver = NonlinearSolver::Newton;
    Index max_newton_iterations = 50;
    Real newton_tolerance = 1e-6;
    Real newton_relaxation = 1.0;  // Under-relaxation factor
    bool use_line_search = true;
    
    // Linear solver
    LinearSolver linear_solver = LinearSolver::EigenCholesky;
    Index max_linear_iterations = 1000;  // For iterative solvers
    Real linear_tolerance = 1e-10;
    
    // Time stepping
    TimeSteppingMethod time_stepping = TimeSteppingMethod::BackwardEuler;
    Real dt_initial = 3600.0;       // [s] Initial timestep (1 hour)
    Real dt_min = 60.0;             // [s] Minimum timestep (1 minute)
    Real dt_max = 86400.0;          // [s] Maximum timestep (1 day)
    Real dt_growth_factor = 1.5;    // Max increase per step
    Real dt_reduction_factor = 0.5; // Reduction on failure
    
    // Adaptive time stepping error control
    Real error_tolerance = 0.01;    // Relative error for adaptive stepping
    
    // Coupling iteration (for two-layer)
    Index max_coupling_iterations = 10;
    Real coupling_tolerance = 1e-4;
};

/**
 * @brief Time control configuration
 */
struct TimeConfig {
    Real start_time = 0.0;          ///< Simulation start [s]
    Real end_time = 86400.0;        ///< Simulation end [s]
    Real output_interval = 3600.0;  ///< Output frequency [s]
    
    // Calendar time (optional)
    std::string start_datetime;     ///< ISO format: "2020-01-01T00:00:00"
    std::string time_units;         ///< e.g., "seconds since 2020-01-01"
};

/**
 * @brief Output configuration
 */
struct OutputConfig {
    std::filesystem::path output_dir = "./output";
    std::string output_prefix = "dgw";
    
    // What to output
    bool output_head = true;
    bool output_water_table_depth = true;
    bool output_saturation = true;      // For Richards
    bool output_fluxes = true;
    bool output_mass_balance = true;
    bool output_velocities = false;
    bool output_gradients = false;      // AD sensitivities
    
    // Format
    enum class Format { NetCDF, VTK, CSV } format = Format::NetCDF;
    
    // Compression
    bool compress = true;
    int compression_level = 4;
};

/**
 * @brief Coupling configuration (for use with SUMMA, dRoute, etc.)
 */
struct CouplingConfig {
    bool coupled_to_land_surface = false;
    bool coupled_to_routing = false;
    
    // Remapping
    std::filesystem::path hru_mesh_file;
    std::filesystem::path river_mesh_file;
    
    // Coupling timestep (may differ from internal timestep)
    Real coupling_dt = 3600.0;      // [s]
    
    // Feedback options
    bool provide_water_table = true;    // To land surface model
    bool provide_river_exchange = true; // To routing model
    bool receive_recharge = true;       // From land surface model
    bool receive_river_stage = true;    // From routing model
};

/**
 * @brief Parallel execution configuration
 */
struct ParallelConfig {
    int num_threads = 0;            ///< 0 = use all available
    bool use_gpu = false;
    int gpu_device = 0;
    
    // Domain decomposition (for MPI)
    bool use_mpi = false;
    int mpi_rank = 0;
    int mpi_size = 1;
};

/**
 * @brief Complete configuration
 */
class Config {
public:
    Config() = default;
    
    // Load from YAML file
    static Config from_file(const std::filesystem::path& filepath);
    
    // Save to YAML file
    void to_file(const std::filesystem::path& filepath) const;
    
    // Validate configuration
    bool validate() const;
    
    // Sub-configurations
    PhysicsDecisions physics;
    SolverConfig solver;
    TimeConfig time;
    OutputConfig output;
    CouplingConfig coupling;
    ParallelConfig parallel;
    
    // File paths
    std::filesystem::path mesh_file;
    std::filesystem::path parameters_file;
    std::filesystem::path initial_conditions_file;
    std::filesystem::path forcing_file;
    
    // Print summary
    void print_summary(std::ostream& os) const;
    
private:
    void validate_physics() const;
    void validate_solver() const;
    void validate_paths() const;
};

// ============================================================================
// YAML Parser Helpers
// ============================================================================

namespace config_io {

/// Parse physics decisions from YAML node
PhysicsDecisions parse_physics(const std::string& yaml_str);

/// Parse solver config from YAML node
SolverConfig parse_solver(const std::string& yaml_str);

/// Convert enum to string
std::string to_string(GoverningEquation eq);
std::string to_string(TransmissivityMethod tm);
std::string to_string(StreamExchangeMethod se);
std::string to_string(VadoseMethod vm);
std::string to_string(RetentionModel rm);
std::string to_string(LinearSolver ls);
std::string to_string(NonlinearSolver ns);
std::string to_string(TimeSteppingMethod ts);

/// Convert string to enum
GoverningEquation governing_equation_from_string(const std::string& s);
TransmissivityMethod transmissivity_from_string(const std::string& s);
StreamExchangeMethod stream_exchange_from_string(const std::string& s);
VadoseMethod vadose_from_string(const std::string& s);
RetentionModel retention_from_string(const std::string& s);
LinearSolver linear_solver_from_string(const std::string& s);
NonlinearSolver nonlinear_solver_from_string(const std::string& s);
TimeSteppingMethod time_stepping_from_string(const std::string& s);

} // namespace config_io

} // namespace dgw
