/**
 * @file dgw.hpp
 * @brief Main dGW model class
 * 
 * This is the primary interface for the differentiable groundwater model.
 * It orchestrates mesh, physics, solvers, and provides a clean API for:
 * - Standalone simulation
 * - Coupled simulation (with SUMMA, dRoute)
 * - Gradient computation via Enzyme AD
 * - BMI compliance for NextGen
 */

#pragma once

// Core includes
#include "core/types.hpp"
#include "core/mesh.hpp"
#include "core/state.hpp"
#include "core/parameters.hpp"
#include "core/config.hpp"

// Physics includes
#include "physics/physics_base.hpp"
#include "physics/boussinesq.hpp"
#include "physics/confined.hpp"
#include "physics/linear_diffusion.hpp"
#include "physics/richards_3d.hpp"
#include "physics/two_layer.hpp"

// Solver includes
#include "solvers/newton.hpp"
#include "solvers/linear_solver.hpp"

// Coupling includes
#include "coupling/stream_aquifer.hpp"

// BMI includes
#include "bmi/bmi.hpp"

namespace dgw {

/**
 * @brief Main differentiable groundwater model
 * 
 * Example usage:
 * @code
 * // Create model from config
 * auto model = DGW::from_config("config.yaml");
 * 
 * // Or build programmatically
 * DGW model;
 * model.set_mesh(mesh);
 * model.set_physics(GoverningEquation::Boussinesq);
 * model.set_parameters(params);
 * model.initialize();
 * 
 * // Run simulation
 * while (model.time() < end_time) {
 *     model.step();
 * }
 * 
 * // Get gradients
 * auto grads = model.compute_gradients(loss_gradient);
 * @endcode
 */
class DGW {
public:
    DGW();
    ~DGW();
    
    // No copy (has unique_ptr members)
    DGW(const DGW&) = delete;
    DGW& operator=(const DGW&) = delete;
    
    // Move OK
    DGW(DGW&&) noexcept;
    DGW& operator=(DGW&&) noexcept;
    
    // ========================================================================
    // Factory Methods
    // ========================================================================
    
    /**
     * @brief Create model from configuration file
     */
    static DGW from_config(const std::string& config_file);
    
    /**
     * @brief Create model from config object
     */
    static DGW from_config(const Config& config);
    
    // ========================================================================
    // Setup Methods
    // ========================================================================
    
    /**
     * @brief Set mesh
     */
    void set_mesh(Ptr<Mesh> mesh);
    
    /**
     * @brief Set physics type
     */
    void set_physics(GoverningEquation physics);
    
    /**
     * @brief Set parameters
     */
    void set_parameters(const Parameters& params);
    
    /**
     * @brief Set configuration
     */
    void set_config(const Config& config);
    
    /**
     * @brief Initialize model (call after setup)
     */
    void initialize();
    
    /**
     * @brief Initialize from existing state (for restart)
     */
    void initialize(const State& initial_state);
    
    // ========================================================================
    // Time Stepping
    // ========================================================================
    
    /**
     * @brief Advance one time step
     * 
     * Uses adaptive time stepping if configured.
     * 
     * @return Step result with convergence info
     */
    StepResult step();
    
    /**
     * @brief Advance with specified time step
     */
    StepResult step(Real dt);
    
    /**
     * @brief Advance until target time
     */
    void step_until(Real target_time);
    
    /**
     * @brief Run simulation to end time
     */
    void run();
    
    // ========================================================================
    // Coupling Interface (for SUMMA, dRoute, etc.)
    // ========================================================================
    
    /**
     * @brief Set recharge from land surface model
     * 
     * @param recharge Recharge rate [m/s] at each cell
     */
    void set_recharge(const Vector& recharge);
    
    /**
     * @brief Set recharge remapped from HRUs
     * 
     * Uses internal HRU→cell mapping matrix.
     * 
     * @param hru_recharge Recharge rate [m/s] at each HRU
     */
    void set_recharge_from_hrus(const Vector& hru_recharge);
    
    /**
     * @brief Set stream stage from routing model
     * 
     * @param stage Stream stage [m] at each river cell
     */
    void set_stream_stage(const Vector& stage);
    
    /**
     * @brief Set pumping rates
     * 
     * @param pumping Pumping rate [m³/s] at each cell (positive = extraction)
     */
    void set_pumping(const Vector& pumping);
    
    /**
     * @brief Get water table depth (for land surface feedback)
     * 
     * @return Depth to water table [m] at each cell
     */
    Vector get_water_table_depth() const;
    
    /**
     * @brief Get water table depth remapped to HRUs
     */
    Vector get_water_table_depth_hrus() const;
    
    /**
     * @brief Get stream-aquifer exchange (for routing feedback)
     * 
     * @return Exchange flux [m³/s] at each river cell, positive = gaining
     */
    Vector get_stream_exchange() const;
    
    /**
     * @brief Get capillary rise flux (for land surface feedback)
     * 
     * @return Upward flux [m/s] at each cell
     */
    Vector get_capillary_rise() const;
    
    // ========================================================================
    // State Access
    // ========================================================================
    
    /**
     * @brief Get current state
     */
    const State& state() const { return state_; }
    State& state() { return state_; }
    
    /**
     * @brief Get head field
     */
    const Vector& head() const;
    
    /**
     * @brief Get current time
     */
    Real time() const { return state_.time(); }
    
    /**
     * @brief Get mesh
     */
    const Mesh& mesh() const { return *mesh_; }
    
    /**
     * @brief Get parameters
     */
    const Parameters& parameters() const { return params_; }
    Parameters& parameters() { return params_; }
    
    /**
     * @brief Get configuration
     */
    const Config& config() const { return config_; }
    
    // ========================================================================
    // Outputs
    // ========================================================================
    
    /**
     * @brief Compute total storage [m³]
     */
    Real total_storage() const;
    
    /**
     * @brief Compute mass balance error
     */
    Real mass_balance_error() const;
    
    /**
     * @brief Get fluxes at cell faces
     */
    Vector face_fluxes() const;
    
    /**
     * @brief Get Darcy velocities at cell centers
     */
    Matrix cell_velocities() const;
    
    /**
     * @brief Write output to file
     */
    void write_output(const std::string& filename) const;
    
    /**
     * @brief Write output to NetCDF
     */
    void write_netcdf(const std::string& filename) const;
    
    // ========================================================================
    // Automatic Differentiation (Enzyme)
    // ========================================================================
    
    /**
     * @brief Compute gradients of loss w.r.t. parameters
     * 
     * Uses adjoint method via Enzyme:
     * Given ∂L/∂h (gradient of loss w.r.t. final state),
     * computes ∂L/∂θ (gradient w.r.t. parameters).
     * 
     * @param loss_gradient ∂L/∂h at final state
     * @return Parameter gradients ∂L/∂θ
     */
    Parameters compute_gradients(const Vector& loss_gradient) const;
    
    /**
     * @brief Compute gradients w.r.t. specific parameters
     * 
     * @param loss_gradient ∂L/∂h
     * @param param_names Which parameters to differentiate
     * @return Map of parameter name to gradient vector
     */
    std::unordered_map<std::string, Vector> compute_gradients(
        const Vector& loss_gradient,
        const std::vector<std::string>& param_names
    ) const;
    
    /**
     * @brief Run forward pass and store checkpoints for adjoint
     * 
     * Call this before compute_gradients if running multiple steps.
     */
    void forward_with_checkpoints();
    
    /**
     * @brief Run adjoint pass using stored checkpoints
     */
    Parameters adjoint_pass(const Vector& final_loss_gradient) const;
    
    /**
     * @brief Check gradients via finite differences
     * 
     * For verification/debugging.
     */
    Parameters check_gradients_fd(
        const Vector& loss_gradient,
        Real epsilon = 1e-7
    ) const;
    
    /**
     * @brief Get Jacobian ∂h_new/∂h_old for one timestep
     */
    SparseMatrix get_state_jacobian() const;
    
    /**
     * @brief Get ∂h_new/∂θ for one timestep
     */
    Matrix get_parameter_jacobian() const;
    
    // ========================================================================
    // Advanced
    // ========================================================================
    
    /**
     * @brief Get physics module
     */
    PhysicsBase& physics() { return *physics_; }
    const PhysicsBase& physics() const { return *physics_; }
    
    /**
     * @brief Get Newton solver
     */
    NewtonSolver& solver() { return solver_; }
    const NewtonSolver& solver() const { return solver_; }
    
    /**
     * @brief Get stream-aquifer exchange calculator
     */
    StreamAquiferExchange& stream_exchange_calc() { return stream_exchange_; }
    
    /**
     * @brief Set output callback
     */
    void set_output_callback(OutputCallback callback) {
        output_callback_ = callback;
    }
    
private:
    // Components
    Ptr<Mesh> mesh_;
    UniquePtr<PhysicsBase> physics_;
    NewtonSolver solver_;
    StreamAquiferExchange stream_exchange_;
    
    // State
    State state_;
    Parameters params_;
    Config config_;
    
    // Source terms (from coupling)
    Vector recharge_;
    Vector stream_stage_;
    Vector pumping_;
    
    // Remapping (HRU ↔ cell)
    SparseMatrix hru_to_cell_;
    SparseMatrix cell_to_hru_;
    
    // Checkpoints for adjoint
    std::vector<Vector> state_checkpoints_;
    std::vector<Real> time_checkpoints_;
    
    // Callbacks
    OutputCallback output_callback_;
    
    // Helpers
    void setup_physics();
    void setup_solver();
    void apply_source_terms();
    Real compute_adaptive_dt() const;
};

// ============================================================================
// BMI Implementation
// ============================================================================

/**
 * @brief BMI-compliant wrapper for DGW
 * 
 * Provides NextGen compatibility.
 */
class DGW_BMI : public bmi::Bmi {
public:
    DGW_BMI();
    ~DGW_BMI() override;
    
    // Model control
    void Initialize(std::string config_file) override;
    void Update() override;
    void UpdateUntil(double time) override;
    void Finalize() override;
    
    // Model info
    std::string GetComponentName() override { return "dGW"; }
    int GetInputItemCount() override;
    int GetOutputItemCount() override;
    std::vector<std::string> GetInputVarNames() override;
    std::vector<std::string> GetOutputVarNames() override;
    
    // Variable info
    int GetVarGrid(std::string name) override;
    std::string GetVarType(std::string name) override;
    std::string GetVarUnits(std::string name) override;
    int GetVarItemsize(std::string name) override;
    int GetVarNbytes(std::string name) override;
    std::string GetVarLocation(std::string name) override;
    
    // Time
    double GetCurrentTime() override;
    double GetStartTime() override;
    double GetEndTime() override;
    double GetTimeStep() override;
    std::string GetTimeUnits() override { return "s"; }
    
    // Getters/setters
    void GetValue(std::string name, void *dest) override;
    void *GetValuePtr(std::string name) override;
    void GetValueAtIndices(std::string name, void *dest, int *inds, int count) override;
    void SetValue(std::string name, void *src) override;
    void SetValueAtIndices(std::string name, int *inds, int count, void *src) override;
    
    // Grid info
    int GetGridRank(int grid) override;
    int GetGridSize(int grid) override;
    std::string GetGridType(int grid) override;
    void GetGridShape(int grid, int *shape) override;
    void GetGridSpacing(int grid, double *spacing) override;
    void GetGridOrigin(int grid, double *origin) override;
    void GetGridX(int grid, double *x) override;
    void GetGridY(int grid, double *y) override;
    void GetGridZ(int grid, double *z) override;
    int GetGridNodeCount(int grid) override;
    int GetGridEdgeCount(int grid) override;
    int GetGridFaceCount(int grid) override;
    void GetGridEdgeNodes(int grid, int *edge_nodes) override;
    void GetGridFaceEdges(int grid, int *face_edges) override;
    void GetGridFaceNodes(int grid, int *face_nodes) override;
    void GetGridNodesPerFace(int grid, int *nodes_per_face) override;
    
    // ========================================================================
    // AD Extension (beyond standard BMI)
    // ========================================================================
    
    /**
     * @brief Get gradient of output w.r.t. input
     * 
     * Extension for differentiable models.
     */
    void GetGradient(std::string output_var, std::string input_var, void *dest);
    
    /**
     * @brief Run adjoint with given seed
     */
    void RunAdjoint(std::string output_var, void *adjoint_seed);
    
    /**
     * @brief Get adjoint result for parameter
     */
    void GetAdjointResult(std::string param_name, void *dest);
    
private:
    UniquePtr<DGW> model_;
    
    // Variable metadata
    struct VarInfo {
        std::string units;
        std::string type;
        std::string location;
        int grid;
        int itemsize;
    };
    std::unordered_map<std::string, VarInfo> input_vars_;
    std::unordered_map<std::string, VarInfo> output_vars_;
    
    void setup_variables();
};

} // namespace dgw
