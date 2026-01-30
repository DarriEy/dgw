/**
 * @file dgw_bmi.cpp
 * @brief Implementation of the DGW standalone model and DGW_BMI wrapper
 *
 * Implements:
 * - dgw::DGW      (standalone model from dgw.hpp)
 * - dgw::DGW_BMI  (BMI wrapper from dgw.hpp)
 * - Factory functions for dynamic loading
 *
 * Note: dgw_bmi.hpp defines an alternative DGW class inheriting directly
 * from bmi::Bmi. That interface is not implemented here to avoid the name
 * conflict; use DGW_BMI instead for BMI-compliant access.
 */

#include "dgw/dgw.hpp"
#include "dgw/physics/physics_base.hpp"
#include "dgw/solvers/linear_solver.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace dgw {

// ============================================================================
// DGW standalone model (from dgw.hpp)
// ============================================================================

DGW::DGW() = default;
DGW::~DGW() = default;
DGW::DGW(DGW&&) noexcept = default;
DGW& DGW::operator=(DGW&&) noexcept = default;

DGW DGW::from_config(const std::string& config_file) {
    Config config = Config::from_file(config_file);
    return from_config(config);
}

DGW DGW::from_config(const Config& config) {
    DGW model;
    model.set_config(config);

    if (!config.mesh_file.empty()) {
        model.set_mesh(Mesh::from_file(config.mesh_file.string()));
    }

    model.set_physics(config.physics.governing_equation);
    model.initialize();
    return model;
}

void DGW::set_mesh(Ptr<Mesh> mesh) {
    mesh_ = mesh;
}

void DGW::set_physics(GoverningEquation physics) {
    config_.physics.governing_equation = physics;
    setup_physics();
}

void DGW::set_parameters(const Parameters& params) {
    params_ = params;
}

void DGW::set_config(const Config& config) {
    config_ = config;
}

void DGW::initialize() {
    if (!mesh_) {
        throw std::runtime_error("Mesh not set before initialize()");
    }
    if (!physics_) {
        setup_physics();
    }

    state_ = State(config_.physics.governing_equation);
    state_.initialize(*mesh_, config_);
    setup_solver();
}

void DGW::initialize(const State& initial_state) {
    state_ = initial_state;
    if (!physics_) {
        setup_physics();
    }
    setup_solver();
}

void DGW::setup_physics() {
    physics_ = create_physics(config_.physics.governing_equation);
}

void DGW::setup_solver() {
    solver_ = NewtonSolver(config_.solver);
}

StepResult DGW::step() {
    return step(config_.solver.dt_initial);
}

StepResult DGW::step(Real dt) {
    StepResult result;

    apply_source_terms();

    Vector& primary = state_.primary_state();
    Vector h_old = primary;

    // Residual and Jacobian closures for Newton solver
    auto residual_func = [&](const Vector& h, Vector& res) {
        primary = h;
        physics_->compute_residual(state_, params_, *mesh_, dt, res);
    };

    auto jacobian_func = [&](const Vector& h, SparseMatrix& jac) {
        primary = h;
        physics_->compute_jacobian(state_, params_, *mesh_, dt, jac);
    };

    SparseMatrix jac = physics_->allocate_jacobian(*mesh_);

    SolveResult solve_result = solver_.solve(
        residual_func, jacobian_func, jac, primary);

    result.success = solve_result.converged;
    result.dt_actual = dt;
    result.newton_iters = solve_result.iterations;

    if (result.success) {
        switch (config_.physics.governing_equation) {
            case GoverningEquation::Boussinesq:
            case GoverningEquation::LinearDiffusion:
            case GoverningEquation::Confined:
                state_.as_2d().advance_time(dt);
                break;
            case GoverningEquation::TwoLayer:
                state_.as_two_layer().advance_time(dt);
                break;
            default:
                break;
        }
        result.dt_next = std::min(
            dt * config_.solver.dt_growth_factor,
            config_.solver.dt_max);
    } else {
        primary = h_old;
        result.dt_next = dt * config_.solver.dt_reduction_factor;
    }

    if (output_callback_) {
        output_callback_(state_.time(), state_);
    }

    return result;
}

void DGW::step_until(Real target_time) {
    Real dt = config_.solver.dt_initial;
    while (state_.time() < target_time - constants::EPSILON) {
        Real remaining = target_time - state_.time();
        Real step_dt = std::min(dt, remaining);
        auto result = step(step_dt);
        if (result.success) {
            dt = result.dt_next;
        } else {
            dt = result.dt_next;
            auto retry = step(dt);
            if (!retry.success) {
                throw std::runtime_error("Time stepping failed after retry");
            }
            dt = retry.dt_next;
        }
    }
}

void DGW::run() {
    step_until(config_.time.end_time);
}

void DGW::apply_source_terms() {
    if (recharge_.size() > 0) {
        physics_->set_recharge(recharge_);
    }
    if (stream_stage_.size() > 0) {
        physics_->set_stream_stage(stream_stage_);
    }
    if (pumping_.size() > 0) {
        physics_->set_pumping(pumping_);
    }
}

void DGW::set_recharge(const Vector& recharge) {
    recharge_ = recharge;
}

void DGW::set_recharge_from_hrus(const Vector& hru_recharge) {
    if (hru_to_cell_.cols() > 0) {
        recharge_ = hru_to_cell_ * hru_recharge;
    } else {
        recharge_ = hru_recharge;
    }
}

void DGW::set_stream_stage(const Vector& stage) {
    stream_stage_ = stage;
}

void DGW::set_pumping(const Vector& pumping) {
    pumping_ = pumping;
}

Vector DGW::get_water_table_depth() const {
    return physics_->water_table_depth(state_, *mesh_);
}

Vector DGW::get_water_table_depth_hrus() const {
    Vector wtd = get_water_table_depth();
    if (cell_to_hru_.cols() > 0) {
        return cell_to_hru_ * wtd;
    }
    return wtd;
}

Vector DGW::get_stream_exchange() const {
    return physics_->stream_exchange(state_, params_, *mesh_);
}

Vector DGW::get_capillary_rise() const {
    Vector cap(mesh_->n_cells());
    cap.setZero();
    return cap;
}

const Vector& DGW::head() const {
    return state_.primary_state();
}

Real DGW::total_storage() const {
    return physics_->total_storage(state_, params_, *mesh_);
}

Real DGW::mass_balance_error() const {
    return physics_->mass_balance_error(state_, params_, *mesh_,
        config_.solver.dt_initial);
}

Vector DGW::face_fluxes() const {
    Vector fluxes;
    physics_->compute_fluxes(state_, params_, *mesh_, fluxes);
    return fluxes;
}

Matrix DGW::cell_velocities() const {
    Index n = mesh_->n_cells();
    Matrix vel(n, 2);
    vel.setZero();
    return vel;
}

void DGW::write_output(const std::string& /*filename*/) const {
    // Placeholder
}

void DGW::write_netcdf(const std::string& /*filename*/) const {
#ifdef DGW_HAS_NETCDF
    // NetCDF output would go here
#endif
}

Parameters DGW::compute_gradients(const Vector& loss_gradient) const {
    Parameters grads(config_.physics.governing_equation);
    physics_->compute_parameter_gradients(state_, params_, *mesh_,
        loss_gradient, grads);
    return grads;
}

std::unordered_map<std::string, Vector> DGW::compute_gradients(
    const Vector& loss_gradient,
    const std::vector<std::string>& /*param_names*/
) const {
    std::unordered_map<std::string, Vector> result;
    Parameters grads = compute_gradients(loss_gradient);
    result["K"] = grads.as_2d().K;
    result["Sy"] = grads.as_2d().Sy;
    return result;
}

void DGW::forward_with_checkpoints() {
    state_checkpoints_.clear();
    time_checkpoints_.clear();
    state_checkpoints_.push_back(state_.primary_state());
    time_checkpoints_.push_back(state_.time());
}

Parameters DGW::adjoint_pass(const Vector& final_loss_gradient) const {
    return compute_gradients(final_loss_gradient);
}

Parameters DGW::check_gradients_fd(
    const Vector& /*loss_gradient*/, Real /*epsilon*/
) const {
    Parameters grads(config_.physics.governing_equation);
    // Full FD verification would require non-const model copy
    return grads;
}

SparseMatrix DGW::get_state_jacobian() const {
    SparseMatrix jac = physics_->allocate_jacobian(*mesh_);
    physics_->compute_jacobian(state_, params_, *mesh_,
        config_.solver.dt_initial, jac);
    return jac;
}

Matrix DGW::get_parameter_jacobian() const {
    Index n_state = state_.primary_state().size();
    Index n_param = params_.n_trainable();
    Matrix jac(n_state, n_param);
    jac.setZero();
    return jac;
}

Real DGW::compute_adaptive_dt() const {
    return config_.solver.dt_initial;
}

// ============================================================================
// DGW_BMI (BMI wrapper from dgw.hpp)
// ============================================================================

DGW_BMI::DGW_BMI() : model_(std::make_unique<DGW>()) {}
DGW_BMI::~DGW_BMI() = default;

void DGW_BMI::Initialize(std::string config_file) {
    *model_ = DGW::from_config(config_file);
    setup_variables();
}

void DGW_BMI::Update() {
    model_->step();
}

void DGW_BMI::UpdateUntil(double time) {
    model_->step_until(static_cast<Real>(time));
}

void DGW_BMI::Finalize() {
    // Cleanup handled by destructors
}

int DGW_BMI::GetInputItemCount() {
    return static_cast<int>(input_vars_.size());
}

int DGW_BMI::GetOutputItemCount() {
    return static_cast<int>(output_vars_.size());
}

std::vector<std::string> DGW_BMI::GetInputVarNames() {
    std::vector<std::string> names;
    names.reserve(input_vars_.size());
    for (const auto& [name, info] : input_vars_) {
        names.push_back(name);
    }
    return names;
}

std::vector<std::string> DGW_BMI::GetOutputVarNames() {
    std::vector<std::string> names;
    names.reserve(output_vars_.size());
    for (const auto& [name, info] : output_vars_) {
        names.push_back(name);
    }
    return names;
}

int DGW_BMI::GetVarGrid(std::string /*name*/) {
    return 0;
}

std::string DGW_BMI::GetVarType(std::string /*name*/) {
    return "double";
}

std::string DGW_BMI::GetVarUnits(std::string name) {
    if (auto it = output_vars_.find(name); it != output_vars_.end())
        return it->second.units;
    if (auto it = input_vars_.find(name); it != input_vars_.end())
        return it->second.units;
    return "";
}

int DGW_BMI::GetVarItemsize(std::string /*name*/) {
    return sizeof(double);
}

int DGW_BMI::GetVarNbytes(std::string name) {
    return GetVarItemsize(name) * GetGridSize(GetVarGrid(name));
}

std::string DGW_BMI::GetVarLocation(std::string name) {
    if (auto it = output_vars_.find(name); it != output_vars_.end())
        return it->second.location;
    if (auto it = input_vars_.find(name); it != input_vars_.end())
        return it->second.location;
    return "node";
}

double DGW_BMI::GetCurrentTime() {
    return static_cast<double>(model_->time());
}

double DGW_BMI::GetStartTime() {
    return static_cast<double>(model_->config().time.start_time);
}

double DGW_BMI::GetEndTime() {
    return static_cast<double>(model_->config().time.end_time);
}

double DGW_BMI::GetTimeStep() {
    return static_cast<double>(model_->config().solver.dt_initial);
}

void DGW_BMI::GetValue(std::string name, void* dest) {
    auto* d = static_cast<double*>(dest);
    if (name == "head") {
        const auto& h = model_->head();
        std::memcpy(d, h.data(), static_cast<size_t>(h.size()) * sizeof(double));
    } else if (name == "water_table_depth") {
        Vector wtd = model_->get_water_table_depth();
        std::memcpy(d, wtd.data(), static_cast<size_t>(wtd.size()) * sizeof(double));
    } else if (name == "stream_exchange") {
        Vector ex = model_->get_stream_exchange();
        std::memcpy(d, ex.data(), static_cast<size_t>(ex.size()) * sizeof(double));
    } else {
        throw std::runtime_error("Unknown output variable: " + name);
    }
}

void* DGW_BMI::GetValuePtr(std::string name) {
    if (name == "head") {
        return const_cast<double*>(model_->head().data());
    }
    throw std::runtime_error("GetValuePtr not available for: " + name);
}

void DGW_BMI::GetValueAtIndices(
    std::string name, void* dest, int* inds, int count
) {
    auto* d = static_cast<double*>(dest);
    if (name == "head") {
        const auto& h = model_->head();
        for (int i = 0; i < count; ++i) {
            d[i] = h(inds[i]);
        }
    } else {
        throw std::runtime_error("GetValueAtIndices not available for: " + name);
    }
}

void DGW_BMI::SetValue(std::string name, void* src) {
    auto* s = static_cast<double*>(src);
    Index n = model_->mesh().n_cells();

    if (name == "recharge") {
        Vector r = Eigen::Map<Vector>(s, n);
        model_->set_recharge(r);
    } else if (name == "stream_stage") {
        Vector stage = Eigen::Map<Vector>(s, n);
        model_->set_stream_stage(stage);
    } else if (name == "pumping") {
        Vector pump = Eigen::Map<Vector>(s, n);
        model_->set_pumping(pump);
    } else {
        throw std::runtime_error("Cannot set variable: " + name);
    }
}

void DGW_BMI::SetValueAtIndices(
    std::string name, int* inds, int count, void* src
) {
    auto* s = static_cast<double*>(src);
    if (name == "recharge") {
        Vector r = Vector::Zero(model_->mesh().n_cells());
        for (int i = 0; i < count; ++i) {
            r(inds[i]) = s[i];
        }
        model_->set_recharge(r);
    } else {
        throw std::runtime_error("SetValueAtIndices not available for: " + name);
    }
}

int DGW_BMI::GetGridRank(int /*grid*/) { return 2; }

int DGW_BMI::GetGridSize(int /*grid*/) {
    return static_cast<int>(model_->mesh().n_cells());
}

std::string DGW_BMI::GetGridType(int /*grid*/) { return "unstructured"; }

void DGW_BMI::GetGridShape(int /*grid*/, int* /*shape*/) {}
void DGW_BMI::GetGridSpacing(int /*grid*/, double* /*spacing*/) {}
void DGW_BMI::GetGridOrigin(int /*grid*/, double* /*origin*/) {}

void DGW_BMI::GetGridX(int /*grid*/, double* x) {
    Index n = model_->mesh().n_cells();
    for (Index i = 0; i < n; ++i) {
        x[i] = model_->mesh().cell_centroid(i).x();
    }
}

void DGW_BMI::GetGridY(int /*grid*/, double* y) {
    Index n = model_->mesh().n_cells();
    for (Index i = 0; i < n; ++i) {
        y[i] = model_->mesh().cell_centroid(i).y();
    }
}

void DGW_BMI::GetGridZ(int /*grid*/, double* z) {
    Index n = model_->mesh().n_cells();
    for (Index i = 0; i < n; ++i) {
        z[i] = model_->mesh().cell_centroid(i).z();
    }
}

int DGW_BMI::GetGridNodeCount(int /*grid*/) {
    return static_cast<int>(model_->mesh().n_nodes());
}

int DGW_BMI::GetGridEdgeCount(int /*grid*/) {
    return static_cast<int>(model_->mesh().n_faces());
}

int DGW_BMI::GetGridFaceCount(int /*grid*/) {
    return static_cast<int>(model_->mesh().n_cells());
}

void DGW_BMI::GetGridEdgeNodes(int /*grid*/, int* /*edge_nodes*/) {}
void DGW_BMI::GetGridFaceEdges(int /*grid*/, int* /*face_edges*/) {}
void DGW_BMI::GetGridFaceNodes(int /*grid*/, int* /*face_nodes*/) {}
void DGW_BMI::GetGridNodesPerFace(int /*grid*/, int* /*nodes_per_face*/) {}

// AD extension stubs
void DGW_BMI::GetGradient(
    std::string /*output_var*/, std::string /*input_var*/, void* /*dest*/
) {}

void DGW_BMI::RunAdjoint(std::string /*output_var*/, void* /*adjoint_seed*/) {}

void DGW_BMI::GetAdjointResult(std::string /*param_name*/, void* /*dest*/) {}

void DGW_BMI::setup_variables() {
    input_vars_["recharge"]     = {"m s-1",   "double", "node", 0, sizeof(double)};
    input_vars_["stream_stage"] = {"m",       "double", "node", 0, sizeof(double)};
    input_vars_["pumping"]      = {"m3 s-1",  "double", "node", 0, sizeof(double)};

    output_vars_["head"]              = {"m",      "double", "node", 0, sizeof(double)};
    output_vars_["water_table_depth"] = {"m",      "double", "node", 0, sizeof(double)};
    output_vars_["stream_exchange"]   = {"m3 s-1", "double", "node", 0, sizeof(double)};
}

} // namespace dgw

// ============================================================================
// Factory functions (extern "C" for dynamic loading by NextGen)
// ============================================================================

extern "C" {

dgw::bmi::Bmi* create_bmi() {
    return new dgw::DGW_BMI();
}

void destroy_bmi(dgw::bmi::Bmi* model) {
    delete model;
}

} // extern "C"
