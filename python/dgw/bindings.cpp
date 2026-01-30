/**
 * @file bindings.cpp
 * @brief Python bindings for dGW using pybind11
 * 
 * Provides Python interface for:
 * - Model creation and configuration
 * - Time stepping and simulation
 * - State access and modification
 * - Gradient computation via Enzyme
 * - JAX integration through custom_vjp
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "dgw/dgw.hpp"
#include <cstring>

namespace py = pybind11;
using namespace dgw;

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert Eigen Vector to NumPy array
py::array_t<double> vector_to_numpy(const Vector& vec) {
    return py::array_t<double>(vec.size(), vec.data());
}

/// Convert NumPy array to Eigen Vector  
Vector numpy_to_vector(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    Vector vec(buf.size);
    std::memcpy(vec.data(), buf.ptr, buf.size * sizeof(double));
    return vec;
}

// ============================================================================
// Python Module Definition
// ============================================================================

PYBIND11_MODULE(dgw_py, m) {
    m.doc() = R"pbdoc(
        dGW: Differentiable Groundwater Model
        =====================================
        
        A C++ groundwater model with automatic differentiation via Enzyme.
        
        Features:
        - Multiple physics options (Boussinesq, Richards, Two-Layer)
        - Unstructured mesh support
        - BMI compliance for NextGen
        - Gradient computation for optimization/ML
        
        Example:
            >>> import dgw_py as dgw
            >>> model = dgw.DGW.from_config("config.yaml")
            >>> model.run()
            >>> grads = model.compute_gradients(loss_grad)
    )pbdoc";
    
    // ========================================================================
    // Enums
    // ========================================================================
    
    py::enum_<GoverningEquation>(m, "GoverningEquation")
        .value("LinearDiffusion", GoverningEquation::LinearDiffusion)
        .value("Boussinesq", GoverningEquation::Boussinesq)
        .value("Confined", GoverningEquation::Confined)
        .value("TwoLayer", GoverningEquation::TwoLayer)
        .value("MultiLayer", GoverningEquation::MultiLayer)
        .value("Richards3D", GoverningEquation::Richards3D)
        .export_values();
    
    py::enum_<TransmissivityMethod>(m, "TransmissivityMethod")
        .value("Standard", TransmissivityMethod::Standard)
        .value("Smoothed", TransmissivityMethod::Smoothed)
        .value("Harmonic", TransmissivityMethod::Harmonic)
        .value("Upstream", TransmissivityMethod::Upstream)
        .export_values();
    
    py::enum_<StreamExchangeMethod>(m, "StreamExchangeMethod")
        .value("Conductance", StreamExchangeMethod::Conductance)
        .value("ConductanceClogging", StreamExchangeMethod::ConductanceClogging)
        .value("KinematicLosing", StreamExchangeMethod::KinematicLosing)
        .value("SaturatedUnsaturated", StreamExchangeMethod::SaturatedUnsaturated)
        .export_values();
    
    py::enum_<VadoseMethod>(m, "VadoseMethod")
        .value("Direct", VadoseMethod::Direct)
        .value("ExponentialLag", VadoseMethod::ExponentialLag)
        .value("KinematicWave", VadoseMethod::KinematicWave)
        .value("FullRichards", VadoseMethod::FullRichards)
        .export_values();
    
    py::enum_<StorageMethod>(m, "StorageMethod")
        .value("Constant", StorageMethod::Constant)
        .value("DepthDependent", StorageMethod::DepthDependent)
        .value("Hysteretic", StorageMethod::Hysteretic)
        .export_values();

    py::enum_<RetentionModel>(m, "RetentionModel")
        .value("VanGenuchten", RetentionModel::VanGenuchten)
        .value("BrooksCorey", RetentionModel::BrooksCorey)
        .value("ClappHornberger", RetentionModel::ClappHornberger)
        .value("Tabulated", RetentionModel::Tabulated)
        .export_values();
    
    py::enum_<BoundaryType>(m, "BoundaryType")
        .value("NoFlow", BoundaryType::NoFlow)
        .value("FixedHead", BoundaryType::FixedHead)
        .value("FixedFlux", BoundaryType::FixedFlux)
        .value("GeneralHead", BoundaryType::GeneralHead)
        .value("SeepageFace", BoundaryType::SeepageFace)
        .value("Drain", BoundaryType::Drain)
        .value("River", BoundaryType::River)
        .value("Recharge", BoundaryType::Recharge)
        .export_values();
    
    // ========================================================================
    // Result Types
    // ========================================================================
    
    py::class_<SolveResult>(m, "SolveResult")
        .def_readonly("converged", &SolveResult::converged)
        .def_readonly("iterations", &SolveResult::iterations)
        .def_readonly("final_residual", &SolveResult::final_residual)
        .def_readonly("solve_time_ms", &SolveResult::solve_time_ms)
        .def_readonly("message", &SolveResult::message)
        .def("__repr__", [](const SolveResult& r) {
            return "<SolveResult converged=" + std::to_string(r.converged) +
                   " iterations=" + std::to_string(r.iterations) + ">";
        });
    
    py::class_<StepResult>(m, "StepResult")
        .def_readonly("success", &StepResult::success)
        .def_readonly("dt_actual", &StepResult::dt_actual)
        .def_readonly("dt_next", &StepResult::dt_next)
        .def_readonly("newton_iters", &StepResult::newton_iters)
        .def_readonly("mass_balance_error", &StepResult::mass_balance_error);
    
    // ========================================================================
    // Configuration
    // ========================================================================
    
    py::class_<PhysicsDecisions>(m, "PhysicsDecisions")
        .def(py::init<>())
        .def_readwrite("governing_equation", &PhysicsDecisions::governing_equation)
        .def_readwrite("transmissivity", &PhysicsDecisions::transmissivity)
        .def_readwrite("storage", &PhysicsDecisions::storage)
        .def_readwrite("stream_exchange", &PhysicsDecisions::stream_exchange)
        .def_readwrite("vadose", &PhysicsDecisions::vadose)
        .def_readwrite("retention", &PhysicsDecisions::retention)
        .def_readwrite("n_layers", &PhysicsDecisions::n_layers);
    
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("max_newton_iterations", &SolverConfig::max_newton_iterations)
        .def_readwrite("newton_tolerance", &SolverConfig::newton_tolerance)
        .def_readwrite("dt_initial", &SolverConfig::dt_initial)
        .def_readwrite("dt_min", &SolverConfig::dt_min)
        .def_readwrite("dt_max", &SolverConfig::dt_max);
    
    py::class_<TimeConfig>(m, "TimeConfig")
        .def(py::init<>())
        .def_readwrite("start_time", &TimeConfig::start_time)
        .def_readwrite("end_time", &TimeConfig::end_time)
        .def_readwrite("output_interval", &TimeConfig::output_interval);
    
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_static("from_file", &Config::from_file)
        .def("to_file", &Config::to_file)
        .def("validate", &Config::validate)
        .def_readwrite("physics", &Config::physics)
        .def_readwrite("solver", &Config::solver)
        .def_readwrite("time", &Config::time);
    
    // ========================================================================
    // Mesh Classes
    // ========================================================================
    
    py::class_<Mesh, Ptr<Mesh>>(m, "Mesh")
        .def_static("from_file", &Mesh::from_file)
        .def("n_cells", &Mesh::n_cells)
        .def("n_faces", &Mesh::n_faces)
        .def("n_nodes", &Mesh::n_nodes)
        .def("n_layers", &Mesh::n_layers)
        .def("dimension", &Mesh::dimension)
        .def("cell_volume", &Mesh::cell_volume)
        .def("river_cells", &Mesh::river_cells)
        .def("boundary_faces", &Mesh::boundary_faces);
    
    py::class_<Mesh2D, Mesh, Ptr<Mesh2D>>(m, "Mesh2D")
        .def(py::init<>())
        .def("finalize", &Mesh2D::finalize)
        .def("set_hru_mapping", [](Mesh2D& self, py::array_t<double> data,
                                    py::array_t<int> row_ptr,
                                    py::array_t<int> col_idx,
                                    int n_rows, int n_cols) {
            // Build sparse matrix from CSR components
            py::buffer_info data_buf = data.request();
            py::buffer_info row_buf = row_ptr.request();
            py::buffer_info col_buf = col_idx.request();
            
            std::vector<SparseTriplet> triplets;
            double* data_ptr = static_cast<double*>(data_buf.ptr);
            int* row_ptr_ptr = static_cast<int*>(row_buf.ptr);
            int* col_ptr = static_cast<int*>(col_buf.ptr);
            
            for (int i = 0; i < n_rows; ++i) {
                for (int k = row_ptr_ptr[i]; k < row_ptr_ptr[i+1]; ++k) {
                    triplets.emplace_back(i, col_ptr[k], data_ptr[k]);
                }
            }
            
            SparseMatrix mat(n_rows, n_cols);
            mat.setFromTriplets(triplets.begin(), triplets.end());
            self.set_hru_mapping(mat);
        });
    
    // Mesh creation helpers
    m.def("create_structured_mesh_2d", &Mesh::create_structured_2d,
          py::arg("nx"), py::arg("ny"), py::arg("dx"), py::arg("dy"),
          py::arg("z_surface"), py::arg("z_bottom"),
          "Create a structured 2D mesh");
    
    // ========================================================================
    // State and Parameters
    // ========================================================================
    
    py::class_<State>(m, "State")
        .def(py::init<>())
        .def(py::init<GoverningEquation>())
        .def("physics", &State::physics)
        .def("time", &State::time)
        .def("pack", &State::pack)
        .def("unpack", &State::unpack)
        .def("get_head", [](const State& s) {
            return vector_to_numpy(s.primary_state());
        })
        .def("set_head", [](State& s, py::array_t<double> arr) {
            s.primary_state() = numpy_to_vector(arr);
        });
    
    py::class_<Parameters>(m, "Parameters")
        .def(py::init<>())
        .def(py::init<GoverningEquation>())
        .def("physics", &Parameters::physics)
        .def("pack_trainable", &Parameters::pack_trainable)
        .def("unpack_trainable", &Parameters::unpack_trainable)
        .def("n_trainable", &Parameters::n_trainable)
        .def("trainable_names", &Parameters::trainable_names)
        .def("load", &Parameters::load)
        .def("save", &Parameters::save);
    
    // ========================================================================
    // Main DGW Model
    // ========================================================================
    
    py::class_<DGW>(m, "DGW")
        .def(py::init<>())
        .def_static("from_config", py::overload_cast<const std::string&>(&DGW::from_config),
                    "Create model from configuration file")
        .def_static("from_config", py::overload_cast<const Config&>(&DGW::from_config),
                    "Create model from Config object")
        
        // Setup
        .def("set_mesh", &DGW::set_mesh)
        .def("set_physics", &DGW::set_physics)
        .def("set_parameters", &DGW::set_parameters)
        .def("set_config", &DGW::set_config)
        .def("initialize", py::overload_cast<>(&DGW::initialize))
        
        // Time stepping
        .def("step", py::overload_cast<>(&DGW::step))
        .def("step", py::overload_cast<Real>(&DGW::step))
        .def("step_until", &DGW::step_until)
        .def("run", &DGW::run)
        
        // Coupling
        .def("set_recharge", [](DGW& self, py::array_t<double> arr) {
            self.set_recharge(numpy_to_vector(arr));
        })
        .def("set_recharge_from_hrus", [](DGW& self, py::array_t<double> arr) {
            self.set_recharge_from_hrus(numpy_to_vector(arr));
        })
        .def("set_stream_stage", [](DGW& self, py::array_t<double> arr) {
            self.set_stream_stage(numpy_to_vector(arr));
        })
        .def("set_pumping", [](DGW& self, py::array_t<double> arr) {
            self.set_pumping(numpy_to_vector(arr));
        })
        .def("get_water_table_depth", [](const DGW& self) {
            return vector_to_numpy(self.get_water_table_depth());
        })
        .def("get_water_table_depth_hrus", [](const DGW& self) {
            return vector_to_numpy(self.get_water_table_depth_hrus());
        })
        .def("get_stream_exchange", [](const DGW& self) {
            return vector_to_numpy(self.get_stream_exchange());
        })
        
        // State access
        .def("state", py::overload_cast<>(&DGW::state), py::return_value_policy::reference)
        .def("head", [](const DGW& self) {
            return vector_to_numpy(self.head());
        })
        .def("time", &DGW::time)
        .def("mesh", &DGW::mesh, py::return_value_policy::reference)
        .def("parameters", py::overload_cast<>(&DGW::parameters), 
             py::return_value_policy::reference)
        .def("config", &DGW::config, py::return_value_policy::reference)
        
        // Outputs
        .def("total_storage", &DGW::total_storage)
        .def("mass_balance_error", &DGW::mass_balance_error)
        .def("face_fluxes", [](const DGW& self) {
            return vector_to_numpy(self.face_fluxes());
        })
        .def("write_output", &DGW::write_output)
        .def("write_netcdf", &DGW::write_netcdf)
        
        // Automatic Differentiation
        .def("compute_gradients", [](const DGW& self, py::array_t<double> loss_grad) {
            return self.compute_gradients(numpy_to_vector(loss_grad));
        }, "Compute parameter gradients via adjoint method")
        .def("forward_with_checkpoints", &DGW::forward_with_checkpoints)
        .def("adjoint_pass", [](const DGW& self, py::array_t<double> loss_grad) {
            return self.adjoint_pass(numpy_to_vector(loss_grad));
        })
        .def("check_gradients_fd", [](const DGW& self, py::array_t<double> loss_grad, Real eps) {
            return self.check_gradients_fd(numpy_to_vector(loss_grad), eps);
        }, py::arg("loss_gradient"), py::arg("epsilon") = 1e-7);
    
    // ========================================================================
    // BMI Interface
    // ========================================================================
    
    py::class_<DGW_BMI>(m, "DGW_BMI")
        .def(py::init<>())
        .def("Initialize", &DGW_BMI::Initialize)
        .def("Update", &DGW_BMI::Update)
        .def("UpdateUntil", &DGW_BMI::UpdateUntil)
        .def("Finalize", &DGW_BMI::Finalize)
        .def("GetComponentName", &DGW_BMI::GetComponentName)
        .def("GetInputVarNames", &DGW_BMI::GetInputVarNames)
        .def("GetOutputVarNames", &DGW_BMI::GetOutputVarNames)
        .def("GetCurrentTime", &DGW_BMI::GetCurrentTime)
        .def("GetStartTime", &DGW_BMI::GetStartTime)
        .def("GetEndTime", &DGW_BMI::GetEndTime)
        .def("GetTimeStep", &DGW_BMI::GetTimeStep)
        .def("GetTimeUnits", &DGW_BMI::GetTimeUnits)
        .def("GetVarUnits", &DGW_BMI::GetVarUnits)
        .def("GetGridType", &DGW_BMI::GetGridType)
        .def("GetGridSize", &DGW_BMI::GetGridSize)
        .def("GetValue", [](DGW_BMI& self, std::string name) {
            int size = self.GetVarNbytes(name) / sizeof(double);
            py::array_t<double> result(size);
            self.GetValue(name, result.mutable_data());
            return result;
        })
        .def("SetValue", [](DGW_BMI& self, std::string name, py::array_t<double> arr) {
            self.SetValue(name, arr.mutable_data());
        })
        // AD extensions
        .def("GetGradient", [](DGW_BMI& self, std::string out, std::string in) {
            // Would need to know sizes
            int size = self.GetGridSize(0);
            py::array_t<double> result(size);
            self.GetGradient(out, in, result.mutable_data());
            return result;
        })
        .def("RunAdjoint", [](DGW_BMI& self, std::string out, py::array_t<double> seed) {
            self.RunAdjoint(out, seed.mutable_data());
        });
    
    // ========================================================================
    // JAX Integration Helpers
    // ========================================================================
    
    m.def("boussinesq_step_fwd",
        [](py::array_t<double> h_old,
           py::array_t<double> K,
           py::array_t<double> Sy,
           py::array_t<double> recharge,
           py::object mesh_capsule,
           double dt) -> std::tuple<py::array_t<double>, py::tuple> {
            // Forward pass: returns (h_new, residuals_for_bwd)
            // This would be called from JAX's custom_vjp
            
            // Placeholder - actual implementation would:
            // 1. Create DGW model from mesh
            // 2. Set parameters from inputs
            // 3. Run one step
            // 4. Return new state and info needed for backward
            
            py::array_t<double> h_new(h_old.size());
            // Zero-initialize until actual computation is implemented
            std::memset(h_new.mutable_data(), 0, h_new.size() * sizeof(double));

            return std::make_tuple(
                h_new,
                py::make_tuple(h_old, h_new, K, Sy, recharge, dt)
            );
        },
        "Forward pass for JAX custom_vjp");
    
    m.def("boussinesq_step_bwd",
        [](py::tuple res, py::array_t<double> g) 
           -> std::tuple<py::array_t<double>, py::array_t<double>,
                        py::array_t<double>, py::array_t<double>,
                        py::none, py::none> {
            // Backward pass: given ∂L/∂h_new, return gradients
            
            // Unpack residuals
            auto h_old = res[0].cast<py::array_t<double>>();
            auto h_new = res[1].cast<py::array_t<double>>();
            auto K = res[2].cast<py::array_t<double>>();
            auto Sy = res[3].cast<py::array_t<double>>();
            auto recharge = res[4].cast<py::array_t<double>>();
            auto dt = res[5].cast<double>();
            
            // Compute adjoints using Enzyme-generated code
            int n = h_old.size();
            py::array_t<double> d_h_old(n);
            py::array_t<double> d_K(n);
            py::array_t<double> d_Sy(n);
            py::array_t<double> d_recharge(n);

            // Zero-initialize until actual Enzyme adjoint is implemented
            std::memset(d_h_old.mutable_data(), 0, n * sizeof(double));
            std::memset(d_K.mutable_data(), 0, n * sizeof(double));
            std::memset(d_Sy.mutable_data(), 0, n * sizeof(double));
            std::memset(d_recharge.mutable_data(), 0, n * sizeof(double));
            
            return std::make_tuple(
                d_h_old, d_K, d_Sy, d_recharge,
                py::none(), py::none()  // No gradient for mesh, dt
            );
        },
        "Backward pass for JAX custom_vjp");
    
    // ========================================================================
    // Version Info
    // ========================================================================
    
    m.attr("__version__") = "0.1.0";
    m.attr("__enzyme_enabled__") = 
#ifdef DGW_HAS_ENZYME
        true;
#else
        false;
#endif
}
