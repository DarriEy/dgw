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
#include "dgw/bmi/dgw_bmi.hpp"
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

    state_checkpoints_.clear();
    time_checkpoints_.clear();
    step_dts_.clear();
    last_dt_ = 0;
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

    // Set head_old = current head before solving (pre-step state)
    switch (config_.physics.governing_equation) {
        case GoverningEquation::Boussinesq:
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Confined:
            state_.as_2d().head_old = state_.as_2d().head;
            break;
        case GoverningEquation::TwoLayer:
            state_.as_two_layer().h1_old = state_.as_two_layer().h1;
            state_.as_two_layer().h2_old = state_.as_two_layer().h2;
            break;
        default:
            break;
    }

    // Store checkpoint before solving
    state_checkpoints_.push_back(state_.primary_state());
    time_checkpoints_.push_back(state_.time());
    step_dts_.push_back(dt);
    last_dt_ = dt;

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
            case GoverningEquation::MultiLayer:
                state_.as_multi_layer().advance_time(dt);
                break;
            case GoverningEquation::Richards3D:
                state_.as_richards().advance_time(dt);
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

namespace {

// Helper: compute parameter gradients at one converged step via FD on the residual
// Given adjoint lambda, accumulates -lambda^T * (dF/dtheta_i) for each param
void compute_step_param_grads_2d(
    PhysicsBase& physics,
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    Real dt,
    const Vector& lambda,
    Parameters& grad_accum,
    Real fd_eps = 1e-8
) {
    const Index n = mesh.n_cells();

    // Helper for central FD on a parameter vector
    auto perturb_2d = [&](
        const Vector& param_vec,
        auto make_perturbed,
        Vector& grad_out
    ) {
        if (grad_out.size() != n) grad_out = Vector::Zero(n);
        for (Index i = 0; i < n; ++i) {
            Real orig = param_vec(i);
            // Scale perturbation by parameter magnitude for good conditioning
            Real eps_i = fd_eps * (1.0 + std::abs(orig));
            Parameters p_plus = make_perturbed(params, i, orig + eps_i);
            Parameters p_minus = make_perturbed(params, i, orig - eps_i);
            Vector Fp(n), Fm(n);
            physics.compute_residual(state, p_plus, mesh, dt, Fp);
            physics.compute_residual(state, p_minus, mesh, dt, Fm);
            Vector dF = (Fp - Fm) / (2.0 * eps_i);
            grad_out(i) += -lambda.dot(dF);
        }
    };

    perturb_2d(params.as_2d().K,
        [](const Parameters& p, Index i, Real v) {
            Parameters pp = p; pp.as_2d().K(i) = v; return pp;
        }, grad_accum.as_2d().K);

    perturb_2d(params.as_2d().Sy,
        [](const Parameters& p, Index i, Real v) {
            Parameters pp = p; pp.as_2d().Sy(i) = v; return pp;
        }, grad_accum.as_2d().Sy);
}

void compute_step_param_grads_two_layer(
    PhysicsBase& physics,
    const State& state,
    const Parameters& params,
    const Mesh& mesh,
    Real dt,
    const Vector& lambda,
    Parameters& grad_accum,
    Real fd_eps = 1e-8
) {
    const Index n = state.primary_state().size();

    // Helper: central FD for a two-layer parameter vector
    auto perturb_param = [&](auto get_vec, auto set_vec, Vector& grad_out) {
        const Index nk = get_vec(params).size();
        if (grad_out.size() != nk) grad_out = Vector::Zero(nk);
        Parameters p_plus = params, p_minus = params;
        for (Index i = 0; i < nk; ++i) {
            Real orig = get_vec(params)(i);
            Real eps_i = fd_eps * std::max(1.0, std::abs(orig));
            set_vec(p_plus, i, orig + eps_i);
            set_vec(p_minus, i, orig - eps_i);
            Vector Fp(n), Fm(n);
            physics.compute_residual(state, p_plus, mesh, dt, Fp);
            physics.compute_residual(state, p_minus, mesh, dt, Fm);
            Vector dF = (Fp - Fm) / (2.0 * eps_i);
            grad_out(i) += -lambda.dot(dF);
            set_vec(p_plus, i, orig);
            set_vec(p_minus, i, orig);
        }
    };

    perturb_param(
        [](const Parameters& p) -> const Vector& { return p.as_two_layer().K1; },
        [](Parameters& p, Index i, Real v) { p.as_two_layer().K1(i) = v; },
        grad_accum.as_two_layer().K1);
    perturb_param(
        [](const Parameters& p) -> const Vector& { return p.as_two_layer().K2; },
        [](Parameters& p, Index i, Real v) { p.as_two_layer().K2(i) = v; },
        grad_accum.as_two_layer().K2);
    perturb_param(
        [](const Parameters& p) -> const Vector& { return p.as_two_layer().Sy; },
        [](Parameters& p, Index i, Real v) { p.as_two_layer().Sy(i) = v; },
        grad_accum.as_two_layer().Sy);
    perturb_param(
        [](const Parameters& p) -> const Vector& { return p.as_two_layer().Ss2; },
        [](Parameters& p, Index i, Real v) { p.as_two_layer().Ss2(i) = v; },
        grad_accum.as_two_layer().Ss2);
}

} // anonymous namespace

Parameters DGW::compute_gradients(const Vector& loss_gradient) const {
    Parameters grads(config_.physics.governing_equation);
    Real dt = last_dt_ > 0 ? last_dt_ : config_.solver.dt_initial;

    // Reconstruct state with correct h_old from the last checkpoint
    State eval_state = state_;
    if (!state_checkpoints_.empty()) {
        const Vector& h_old = state_checkpoints_.back();
        switch (config_.physics.governing_equation) {
            case GoverningEquation::Boussinesq:
            case GoverningEquation::LinearDiffusion:
            case GoverningEquation::Confined:
                eval_state.as_2d().head_old = h_old;
                break;
            case GoverningEquation::TwoLayer:
                eval_state.as_two_layer().h1_old = h_old.head(h_old.size() / 2);
                eval_state.as_two_layer().h2_old = h_old.tail(h_old.size() / 2);
                break;
            default:
                break;
        }
    }

    // 1. Compute Jacobian J = dF/dh at the converged state
    SparseMatrix jac = physics_->allocate_jacobian(*mesh_);
    physics_->compute_jacobian(eval_state, params_, *mesh_, dt, jac);

    // 2. Solve J^T * lambda = loss_gradient
    SparseMatrix jacT = jac.transpose();
    EigenLUSolver lu;
    lu.analyze_pattern(jacT);
    lu.factorize(jacT);
    Vector lambda(loss_gradient.size());
    lu.solve(loss_gradient, lambda);

    // 3. Compute parameter gradients via FD: grad_i = -lambda^T * (dF/dtheta_i)
    switch (config_.physics.governing_equation) {
        case GoverningEquation::Boussinesq:
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Confined:
            compute_step_param_grads_2d(
                *physics_, eval_state, params_, *mesh_, dt, lambda, grads);
            break;
        case GoverningEquation::TwoLayer:
            compute_step_param_grads_two_layer(
                *physics_, eval_state, params_, *mesh_, dt, lambda, grads);
            break;
        default:
            break;
    }

    return grads;
}

std::unordered_map<std::string, Vector> DGW::compute_gradients(
    const Vector& loss_gradient,
    const std::vector<std::string>& /*param_names*/
) const {
    std::unordered_map<std::string, Vector> result;
    Parameters grads = compute_gradients(loss_gradient);

    switch (config_.physics.governing_equation) {
        case GoverningEquation::TwoLayer:
            result["K1"] = grads.as_two_layer().K1;
            result["K2"] = grads.as_two_layer().K2;
            result["Sy"] = grads.as_two_layer().Sy;
            result["Ss2"] = grads.as_two_layer().Ss2;
            break;
        case GoverningEquation::LinearDiffusion:
        case GoverningEquation::Boussinesq:
        case GoverningEquation::Confined:
        default:
            result["K"] = grads.as_2d().K;
            result["Sy"] = grads.as_2d().Sy;
            break;
    }
    return result;
}

void DGW::forward_with_checkpoints() {
    // Checkpoints are already stored by step().
    // This method signals that checkpoints are ready for adjoint_pass.
}

Parameters DGW::adjoint_pass(const Vector& final_loss_gradient) const {
    Parameters grads(config_.physics.governing_equation);
    const Index n_steps = static_cast<Index>(step_dts_.size());

    if (n_steps == 0) {
        return compute_gradients(final_loss_gradient);
    }

    // state_checkpoints_[t] = pre-step state (h_old for step t)
    // After all steps, current state = final h_new
    // For step t: h_old = checkpoint[t], h_new = checkpoint[t+1] (or current if last)

    Vector lambda = final_loss_gradient;
    const Real fd_eps = 1e-7;

    for (Index t = n_steps - 1; t >= 0; --t) {
        Real dt = step_dts_[static_cast<size_t>(t)];
        Vector h_old = state_checkpoints_[static_cast<size_t>(t)];
        Vector h_new = (t == n_steps - 1)
            ? state_.primary_state()
            : state_checkpoints_[static_cast<size_t>(t + 1)];

        // Reconstruct the state at this step (converged state with h_new and h_old)
        State step_state = state_;
        step_state.primary_state() = h_new;
        switch (config_.physics.governing_equation) {
            case GoverningEquation::Boussinesq:
            case GoverningEquation::LinearDiffusion:
            case GoverningEquation::Confined:
                step_state.as_2d().head_old = h_old;
                break;
            case GoverningEquation::TwoLayer:
                step_state.as_two_layer().h1_old = h_old.head(h_old.size() / 2);
                step_state.as_two_layer().h2_old = h_old.tail(h_old.size() / 2);
                break;
            default:
                break;
        }

        // Compute Jacobian J = dF/dh at this step
        SparseMatrix jac = physics_->allocate_jacobian(*mesh_);
        physics_->compute_jacobian(step_state, params_, *mesh_, dt, jac);

        // Solve J^T * mu = lambda (adjoint system)
        SparseMatrix jacT = jac.transpose();
        EigenLUSolver lu;
        lu.analyze_pattern(jacT);
        lu.factorize(jacT);
        Vector mu(lambda.size());
        lu.solve(lambda, mu);

        // Accumulate parameter gradients at this step
        switch (config_.physics.governing_equation) {
            case GoverningEquation::Boussinesq:
            case GoverningEquation::LinearDiffusion:
            case GoverningEquation::Confined:
                compute_step_param_grads_2d(
                    *physics_, step_state, params_, *mesh_, dt, mu, grads, fd_eps);
                break;
            case GoverningEquation::TwoLayer:
                compute_step_param_grads_two_layer(
                    *physics_, step_state, params_, *mesh_, dt, mu, grads, fd_eps);
                break;
            default:
                break;
        }

        // Propagate adjoint backward: lambda_prev = -(dF/dh_old)^T * mu
        if (t > 0) {
            const Index n = h_old.size();
            Vector F0(n);
            physics_->compute_residual(step_state, params_, *mesh_, dt, F0);

            Vector lambda_new = Vector::Zero(n);
            for (Index i = 0; i < n; ++i) {
                State perturbed_state = step_state;
                switch (config_.physics.governing_equation) {
                    case GoverningEquation::Boussinesq:
                    case GoverningEquation::LinearDiffusion:
                    case GoverningEquation::Confined:
                        perturbed_state.as_2d().head_old(i) += fd_eps;
                        break;
                    case GoverningEquation::TwoLayer:
                        if (i < n / 2)
                            perturbed_state.as_two_layer().h1_old(i) += fd_eps;
                        else
                            perturbed_state.as_two_layer().h2_old(i - n / 2) += fd_eps;
                        break;
                    default:
                        break;
                }
                Vector F_pert(n);
                physics_->compute_residual(perturbed_state, params_, *mesh_, dt, F_pert);
                Vector dF_dhold_i = (F_pert - F0) / fd_eps;
                lambda_new(i) = -mu.dot(dF_dhold_i);
            }
            lambda = lambda_new;
        }
    }

    return grads;
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
        cached_recharge_ = Eigen::Map<Vector>(s, n);
        model_->set_recharge(cached_recharge_);
    } else if (name == "stream_stage") {
        cached_stream_stage_ = Eigen::Map<Vector>(s, n);
        model_->set_stream_stage(cached_stream_stage_);
    } else if (name == "pumping") {
        cached_pumping_ = Eigen::Map<Vector>(s, n);
        model_->set_pumping(cached_pumping_);
    } else {
        throw std::runtime_error("Cannot set variable: " + name);
    }
}

void DGW_BMI::SetValueAtIndices(
    std::string name, int* inds, int count, void* src
) {
    auto* s = static_cast<double*>(src);
    Index n = model_->mesh().n_cells();

    if (name == "recharge") {
        if (cached_recharge_.size() != n) {
            cached_recharge_ = Vector::Zero(n);
        }
        for (int i = 0; i < count; ++i) {
            if (inds[i] >= 0 && inds[i] < n) {
                cached_recharge_(inds[i]) = s[i];
            }
        }
        model_->set_recharge(cached_recharge_);
    } else if (name == "stream_stage") {
        if (cached_stream_stage_.size() != n) {
            cached_stream_stage_ = Vector::Zero(n);
        }
        for (int i = 0; i < count; ++i) {
            if (inds[i] >= 0 && inds[i] < n) {
                cached_stream_stage_(inds[i]) = s[i];
            }
        }
        model_->set_stream_stage(cached_stream_stage_);
    } else if (name == "pumping") {
        if (cached_pumping_.size() != n) {
            cached_pumping_ = Vector::Zero(n);
        }
        for (int i = 0; i < count; ++i) {
            if (inds[i] >= 0 && inds[i] < n) {
                cached_pumping_(inds[i]) = s[i];
            }
        }
        model_->set_pumping(cached_pumping_);
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

// DGW_NextGen factory functions are not yet implemented.
// DGW_NextGen is declared in dgw_bmi.hpp but requires a full implementation.
// Use DGW_BMI (via create_bmi/destroy_bmi) for NextGen integration.

} // extern "C"
