/**
 * @file dgw_bmi.hpp
 * @brief dGW implementation of BMI interface
 * 
 * This is the main entry point for using dGW with NextGen
 * or any other BMI-compliant framework.
 */

#pragma once

#include "bmi.hpp"
#include "../core/types.hpp"
#include "../core/mesh.hpp"
#include "../core/state.hpp"
#include "../core/parameters.hpp"
#include "../core/config.hpp"
#include "../physics/physics_base.hpp"
#include "../solvers/newton.hpp"

namespace dgw {

/**
 * @brief dGW model with BMI interface
 * 
 * Implements standard BMI plus extensions for:
 * - Automatic differentiation (gradient access)
 * - Adjoint computation
 * - Parameter sensitivity
 */
class DGW_NextGen : public bmi::Bmi {
public:
    DGW_NextGen();
    ~DGW_NextGen() override;
    
    // ========================================================================
    // BMI Model Control
    // ========================================================================
    
    void Initialize(std::string config_file) override;
    void Update() override;
    void UpdateUntil(double time) override;
    void Finalize() override;
    
    // ========================================================================
    // BMI Model Information
    // ========================================================================
    
    std::string GetComponentName() override { return "dGW"; }
    int GetInputItemCount() override;
    int GetOutputItemCount() override;
    std::vector<std::string> GetInputVarNames() override;
    std::vector<std::string> GetOutputVarNames() override;
    
    // ========================================================================
    // BMI Variable Information
    // ========================================================================
    
    int GetVarGrid(std::string name) override;
    std::string GetVarType(std::string name) override;
    std::string GetVarUnits(std::string name) override;
    int GetVarItemsize(std::string name) override;
    int GetVarNbytes(std::string name) override;
    std::string GetVarLocation(std::string name) override;
    
    // ========================================================================
    // BMI Time Functions
    // ========================================================================
    
    double GetCurrentTime() override;
    double GetStartTime() override;
    double GetEndTime() override;
    double GetTimeStep() override;
    std::string GetTimeUnits() override { return "s"; }
    
    // ========================================================================
    // BMI Variable Access
    // ========================================================================
    
    void GetValue(std::string name, void *dest) override;
    void *GetValuePtr(std::string name) override;
    void GetValueAtIndices(std::string name, void *dest, int *inds, int count) override;
    void SetValue(std::string name, void *src) override;
    void SetValueAtIndices(std::string name, int *inds, int count, void *src) override;
    
    // ========================================================================
    // BMI Grid Information
    // ========================================================================
    
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
    // dGW Extensions (beyond standard BMI)
    // ========================================================================
    
    /**
     * @brief Get gradient of output variable w.r.t. input variable
     * 
     * Uses Enzyme AD to compute ∂output/∂input.
     * 
     * @param output_name Output variable name (e.g., "head")
     * @param input_name Input variable name (e.g., "hydraulic_conductivity")
     * @param dest Destination array [n_output × n_input] or sparse
     */
    void GetGradient(std::string output_name, std::string input_name, void *dest);
    
    /**
     * @brief Run adjoint computation
     * 
     * Given ∂L/∂output (adjoint seed), computes ∂L/∂parameters.
     * 
     * @param output_name Output variable that loss depends on
     * @param adjoint_seed Array of ∂L/∂output values
     */
    void RunAdjoint(std::string output_name, void *adjoint_seed);
    
    /**
     * @brief Get parameter gradients after adjoint run
     * 
     * @param param_name Parameter name
     * @param dest Destination for ∂L/∂param
     */
    void GetParameterGradient(std::string param_name, void *dest);
    
    /**
     * @brief Check if AD is available
     */
    bool SupportsGradients() const;
    
    /**
     * @brief Get mass balance error from last timestep
     */
    double GetMassBalanceError() const;
    
    /**
     * @brief Get total storage [m³]
     */
    double GetTotalStorage() const;
    
    // ========================================================================
    // Direct Access (for C++ users, bypassing BMI string interface)
    // ========================================================================
    
    const Mesh& mesh() const { return *mesh_; }
    const State& state() const { return state_; }
    State& state() { return state_; }
    const Parameters& parameters() const { return params_; }
    Parameters& parameters() { return params_; }
    const Config& config() const { return config_; }
    
    /**
     * @brief Step with explicit dt (ignores config timestep)
     */
    StepResult step(Real dt);
    
    /**
     * @brief Set recharge directly (for coupling)
     */
    void set_recharge(const Vector& recharge);
    
    /**
     * @brief Set river stage directly (for coupling)
     */
    void set_river_stage(const Vector& stage);
    
    /**
     * @brief Get water table depth
     */
    Vector get_water_table_depth() const;
    
    /**
     * @brief Get stream-aquifer exchange
     */
    Vector get_stream_exchange() const;
    
private:
    // Configuration
    Config config_;
    
    // Mesh
    Ptr<Mesh> mesh_;
    
    // State and parameters
    State state_;
    Parameters params_;
    
    // Physics module (polymorphic based on config)
    UniquePtr<PhysicsBase> physics_;
    
    // Solver
    UniquePtr<NewtonSolver> newton_solver_;
    UniquePtr<LinearSolverBase> linear_solver_;
    
    // Jacobian (reused across timesteps)
    SparseMatrix jacobian_;
    bool jacobian_allocated_ = false;
    
    // Gradient storage (for adjoint)
    Vector adjoint_state_;
    Parameters param_gradients_;
    bool gradients_computed_ = false;
    
    // Time tracking
    Real current_time_ = 0.0;
    Real start_time_ = 0.0;
    Real end_time_ = 0.0;
    Real dt_ = 0.0;
    
    // Statistics
    Real last_mass_balance_error_ = 0.0;
    Index total_newton_iters_ = 0;
    Index total_timesteps_ = 0;
    
    // Variable registry
    struct VarInfo {
        std::string units;
        std::string location;  // "node", "face", "cell"
        bool is_input;
        bool is_output;
        std::function<void*(DGW_NextGen&)> get_ptr;
        std::function<Index(const DGW_NextGen&)> get_size;
    };
    std::unordered_map<std::string, VarInfo> var_registry_;
    
    void register_variables();
    void setup_solvers();
    void allocate_jacobian();
};

} // namespace dgw

// ============================================================================
// Factory Functions (for dynamic loading)
// ============================================================================

/**
 * @brief Factory functions for NextGen's dynamic library loading.
 *
 * Note: extern "C" must be at namespace scope (outside dgw namespace).
 */
extern "C" {
    dgw::bmi::Bmi* create_bmi();
    void destroy_bmi(dgw::bmi::Bmi* model);
    // DGW_NextGen factory functions not yet available (class not implemented):
    // dgw::DGW_NextGen* create_dgw();
    // void destroy_dgw(dgw::DGW_NextGen* model);
}
