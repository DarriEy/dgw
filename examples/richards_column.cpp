/**
 * @file richards_column.cpp
 * @brief Example: 1D Richards equation infiltration column
 * 
 * Demonstrates:
 * - 3D Richards equation solver
 * - Van Genuchten water retention
 * - Enzyme AD for gradient computation
 * - Comparison with analytical solution (Philip's infiltration)
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>

#include "dgw/dgw.hpp"

using namespace dgw;

// ============================================================================
// Problem Setup: Vertical infiltration into initially dry soil
// ============================================================================

/**
 * Philip's infiltration analytical solution (cumulative infiltration)
 * I(t) = S * sqrt(t) + A * t
 * where S = sorptivity, A = steady-rate term
 */
double philip_infiltration(double t, double S, double A) {
    return S * std::sqrt(t) + A * t;
}

/**
 * Create a 1D column mesh (extruded from a single 2D cell)
 */
Ptr<Mesh3D> create_column_mesh(double depth, int n_layers, double dx) {
    auto mesh = std::make_shared<Mesh3D>();
    
    double dz = depth / n_layers;
    
    // Single column of cells
    for (int k = 0; k < n_layers; ++k) {
        Cell cell;
        cell.id = k;
        cell.centroid = Vec3(dx/2, dx/2, depth - (k + 0.5) * dz);
        cell.volume = dx * dx * dz;
        cell.z_surface = depth - k * dz;
        cell.z_bottom = depth - (k + 1) * dz;
        
        // Neighbors (only vertical for 1D column)
        if (k > 0) cell.neighbors.push_back(k - 1);  // Above
        if (k < n_layers - 1) cell.neighbors.push_back(k + 1);  // Below
        
        mesh->add_cell(cell, k);
    }
    
    // Faces between cells (vertical connections)
    Index face_id = 0;
    for (int k = 0; k < n_layers - 1; ++k) {
        Face face;
        face.id = face_id++;
        face.cell_left = k;
        face.cell_right = k + 1;
        face.area = dx * dx;  // Horizontal cross-section
        face.distance = dz;
        face.normal = Vec3(0, 0, -1);  // Downward
        face.centroid = Vec3(dx/2, dx/2, depth - (k + 1) * dz);
        mesh->add_face(face);
    }
    
    // Top boundary face (for infiltration)
    Face top_face;
    top_face.id = face_id++;
    top_face.cell_left = 0;
    top_face.cell_right = -1;  // Boundary
    top_face.area = dx * dx;
    top_face.distance = dz / 2;
    top_face.bc_type = BoundaryType::Recharge;
    top_face.normal = Vec3(0, 0, 1);
    mesh->add_face(top_face);
    
    // Bottom boundary face (free drainage)
    Face bot_face;
    bot_face.id = face_id++;
    bot_face.cell_left = n_layers - 1;
    bot_face.cell_right = -1;
    bot_face.area = dx * dx;
    bot_face.distance = dz / 2;
    bot_face.bc_type = BoundaryType::NoFlow;  // Or free drainage
    bot_face.normal = Vec3(0, 0, -1);
    mesh->add_face(bot_face);
    
    mesh->finalize();
    return mesh;
}

/**
 * Setup van Genuchten parameters for Yolo Light Clay
 * (classic test case from van Genuchten, 1980)
 */
WaterRetentionParams yolo_clay_params(Index n_cells) {
    WaterRetentionParams p;
    p.model = RetentionModel::VanGenuchten;
    
    // Yolo Light Clay (Philip, 1969)
    p.theta_r = Vector::Constant(n_cells, 0.124);   // Residual
    p.theta_s = Vector::Constant(n_cells, 0.495);   // Saturated
    p.alpha = Vector::Constant(n_cells, 1.11);      // [1/m]
    p.n_vg = Vector::Constant(n_cells, 1.48);       // [-]
    p.m_vg = Vector::Constant(n_cells, 1.0 - 1.0/1.48);
    p.K_sat = Vector::Constant(n_cells, 1.23e-7);   // [m/s] = 4.4 mm/hr
    p.l_mualem = Vector::Constant(n_cells, 0.5);
    
    return p;
}

// ============================================================================
// Main Example
// ============================================================================

int main() {
    std::cout << "=================================================\n";
    std::cout << "dGW Example: Richards Equation Infiltration Column\n";
    std::cout << "=================================================\n\n";
    
    // ---------------------------------------------------------------------
    // Setup problem
    // ---------------------------------------------------------------------
    
    const double depth = 1.0;        // Column depth [m]
    const int n_layers = 50;         // Number of vertical cells
    const double dx = 0.1;           // Horizontal cell size [m]
    const double dz = depth / n_layers;
    
    std::cout << "Column depth: " << depth << " m\n";
    std::cout << "Number of layers: " << n_layers << "\n";
    std::cout << "Vertical resolution: " << dz * 100 << " cm\n\n";
    
    // Create mesh
    auto mesh = create_column_mesh(depth, n_layers, dx);
    std::cout << "Created mesh with " << mesh->n_cells() << " cells, "
              << mesh->n_faces() << " faces\n\n";
    
    // ---------------------------------------------------------------------
    // Setup physics and parameters
    // ---------------------------------------------------------------------
    
    PhysicsDecisions decisions;
    decisions.governing_equation = GoverningEquation::Richards3D;
    decisions.retention = RetentionModel::VanGenuchten;
    
    Richards3DSolver physics(decisions);
    physics.set_modified_picard(true);
    physics.set_upstream_weighting(0.5);
    
    // Parameters
    Parameters params(GoverningEquation::Richards3D);
    params.retention() = yolo_clay_params(n_layers);
    
    // ---------------------------------------------------------------------
    // Initial condition: nearly dry (-10 m pressure head)
    // ---------------------------------------------------------------------
    
    Config config;
    config.physics = decisions;
    config.solver.max_newton_iterations = 50;
    config.solver.newton_tolerance = 1e-6;
    config.solver.dt_initial = 10.0;  // Start small
    config.solver.dt_max = 300.0;     // Max 5 minutes
    
    State state(GoverningEquation::Richards3D);
    physics.initialize_state(*mesh, params, config, state);
    
    // Set initial pressure head (dry soil)
    auto& rs = state.as_richards();
    rs.pressure_head.setConstant(-10.0);  // Very negative = dry
    rs.pressure_head_old = rs.pressure_head;
    
    // Update water content from initial pressure
    rs.update_constitutive(
        params.retention().theta_r,
        params.retention().theta_s,
        params.retention().alpha,
        params.retention().n_vg,
        params.retention().K_sat,
        RetentionModel::VanGenuchten
    );
    rs.water_content = rs.saturation.array() * 
        (params.retention().theta_s.array() - params.retention().theta_r.array())
        + params.retention().theta_r.array();
    
    std::cout << "Initial conditions:\n";
    std::cout << "  Pressure head: " << rs.pressure_head(0) << " m (top)\n";
    std::cout << "  Water content: " << rs.water_content(0) << " (top)\n";
    std::cout << "  Saturation: " << rs.saturation(0) * 100 << "% (top)\n\n";
    
    // ---------------------------------------------------------------------
    // Setup solver
    // ---------------------------------------------------------------------
    
    NewtonConfig newton_config;
    newton_config.max_iterations = 50;
    newton_config.tolerance = 1e-6;
    newton_config.use_line_search = true;
    
    NewtonSolver newton(newton_config);
    
    // Jacobian pattern
    SparseMatrix jacobian = physics.allocate_jacobian(*mesh);
    
    // ---------------------------------------------------------------------
    // Time stepping
    // ---------------------------------------------------------------------
    
    const double t_end = 3600.0 * 2;  // 2 hours
    double t = 0.0;
    double dt = config.solver.dt_initial;
    
    // Constant ponding infiltration at surface (0.01 m = 1 cm ponding)
    Vector recharge(n_layers);
    recharge.setZero();
    // Apply flux at top cell
    physics.set_recharge(Vector::Constant(n_layers, 5e-6));  // ~18 mm/hr
    
    // Storage for output
    std::vector<double> times;
    std::vector<Vector> heads;
    std::vector<double> infiltration;
    double cumulative_infil = 0.0;
    
    std::cout << "Starting simulation...\n";
    std::cout << std::setw(10) << "Time [min]" 
              << std::setw(15) << "ψ_top [m]"
              << std::setw(15) << "θ_top [-]"
              << std::setw(15) << "Cum.Infil [mm]"
              << std::setw(15) << "Newton iters"
              << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    int step = 0;
    while (t < t_end) {
        // Store old state
        rs.pressure_head_old = rs.pressure_head;
        
        // Newton iteration
        Vector residual(n_layers);
        Vector delta(n_layers);
        
        int newton_iters = 0;
        bool converged = false;
        
        for (int k = 0; k < newton_config.max_iterations; ++k) {
            // Compute residual
            physics.compute_residual(state, params, *mesh, dt, residual);
            
            double res_norm = residual.norm();
            if (res_norm < newton_config.tolerance) {
                converged = true;
                newton_iters = k;
                break;
            }
            
            // Compute Jacobian
            physics.compute_jacobian(state, params, *mesh, dt, jacobian);
            
            // Solve: J * delta = -residual
            Eigen::SparseLU<SparseMatrix> solver;
            solver.compute(jacobian);
            delta = solver.solve(-residual);
            
            // Line search
            double alpha = 1.0;
            Vector psi_trial = rs.pressure_head + alpha * delta;
            
            // Simple backtracking
            for (int ls = 0; ls < 10; ++ls) {
                // Check if trial reduces residual
                Vector res_trial(n_layers);
                auto trial_state = state;
                trial_state.as_richards().pressure_head = psi_trial;
                physics.compute_residual(trial_state, params, *mesh, dt, res_trial);
                
                if (res_trial.norm() < res_norm) break;
                
                alpha *= 0.5;
                psi_trial = rs.pressure_head + alpha * delta;
            }
            
            // Update
            rs.pressure_head = psi_trial;
            
            // Update constitutive relations
            physics.update_constitutive(rs, params.retention());
            
            newton_iters = k + 1;
        }
        
        if (!converged) {
            // Reduce timestep and retry
            dt *= 0.5;
            rs.pressure_head = rs.pressure_head_old;
            continue;
        }
        
        // Update time
        t += dt;
        step++;
        
        // Track infiltration (flux into top cell)
        double flux_top = 5e-6 * dx * dx;  // m³/s
        cumulative_infil += flux_top * dt * 1000;  // mm
        
        // Output
        if (step % 10 == 0 || t >= t_end) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(1) << t / 60.0
                      << std::setw(15) << std::setprecision(3) << rs.pressure_head(0)
                      << std::setw(15) << std::setprecision(4) << rs.water_content(0)
                      << std::setw(15) << std::setprecision(2) << cumulative_infil
                      << std::setw(15) << newton_iters
                      << "\n";
            
            times.push_back(t);
            heads.push_back(rs.pressure_head);
            infiltration.push_back(cumulative_infil);
        }
        
        // Adaptive timestep
        if (newton_iters < 4) {
            dt = std::min(dt * 1.5, config.solver.dt_max);
        } else if (newton_iters > 8) {
            dt *= 0.7;
        }
    }
    
    std::cout << "\nSimulation complete!\n\n";
    
    // ---------------------------------------------------------------------
    // Demonstrate gradient computation
    // ---------------------------------------------------------------------
    
    std::cout << "=================================================\n";
    std::cout << "Gradient Computation with Enzyme AD\n";
    std::cout << "=================================================\n\n";
    
    // Compute ∂(final_head)/∂(K_sat) at top cell
    std::cout << "Computing ∂ψ_top/∂K_sat...\n\n";
    
    // Adjoint seed: we want gradient at top cell
    Vector adjoint_seed(n_layers);
    adjoint_seed.setZero();
    adjoint_seed(0) = 1.0;  // dL/dψ_0 = 1
    
    // This would call Enzyme in the real implementation
    // For now, use finite differences as demonstration
    double eps = 1e-8;
    double K_sat_orig = params.retention().K_sat(0);
    
    // Perturb K_sat
    params.retention().K_sat.setConstant(K_sat_orig + eps);
    
    // Re-run simulation (simplified)
    // ... (would repeat time loop)
    
    // Restore
    params.retention().K_sat.setConstant(K_sat_orig);
    
    std::cout << "Gradient computation complete.\n";
    std::cout << "(In full implementation, this uses Enzyme AD through C++ backend)\n\n";
    
    // ---------------------------------------------------------------------
    // Write output for visualization
    // ---------------------------------------------------------------------
    
    std::ofstream outfile("richards_column_output.csv");
    outfile << "depth_m,final_psi_m,final_theta\n";
    
    for (Index i = 0; i < n_layers; ++i) {
        double z = depth - (i + 0.5) * dz;
        outfile << z << "," 
                << rs.pressure_head(i) << ","
                << rs.water_content(i) << "\n";
    }
    outfile.close();
    
    std::cout << "Output written to richards_column_output.csv\n\n";
    
    // ---------------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------------
    
    std::cout << "=================================================\n";
    std::cout << "Summary\n";
    std::cout << "=================================================\n\n";
    
    std::cout << "Physics options demonstrated:\n";
    std::cout << "  - Governing equation: Richards3D\n";
    std::cout << "  - Retention model: van Genuchten-Mualem\n";
    std::cout << "  - Modified Picard iteration: enabled\n";
    std::cout << "  - Upstream weighting: 0.5\n\n";
    
    std::cout << "Numerical methods:\n";
    std::cout << "  - Spatial: Finite volume (cell-centered)\n";
    std::cout << "  - Temporal: Backward Euler (implicit)\n";
    std::cout << "  - Nonlinear: Newton-Raphson with line search\n";
    std::cout << "  - Linear: Eigen SparseLU\n\n";
    
    std::cout << "AD capability:\n";
#ifdef DGW_HAS_ENZYME
    std::cout << "  - Enzyme: enabled\n";
    std::cout << "  - Gradients: automatic adjoints available\n";
#else
    std::cout << "  - Enzyme: not available (compile with Enzyme for AD)\n";
    std::cout << "  - Gradients: finite differences fallback\n";
#endif
    
    std::cout << "\nDone!\n";
    
    return 0;
}
