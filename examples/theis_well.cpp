/**
 * @file theis_well.cpp
 * @brief Example: Theis well solution comparison
 * 
 * Validates dGW against the analytical Theis solution for
 * transient drawdown from a pumping well in a confined aquifer.
 * 
 * This demonstrates:
 * - Model setup from code (not config file)
 * - Structured mesh creation
 * - Parameter specification
 * - Comparison with analytical solution
 */

#include <dgw/dgw.hpp>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace dgw;

/**
 * @brief Theis well function W(u)
 * 
 * W(u) = ∫_u^∞ (e^(-t) / t) dt
 * 
 * Approximated using series expansion.
 */
double theis_W(double u) {
    if (u < 1.0) {
        // Series expansion for small u
        const double gamma = 0.5772156649;  // Euler's constant
        double sum = -gamma - std::log(u);
        double term = u;
        for (int n = 1; n <= 10; ++n) {
            sum += term / n;
            term *= -u / (n + 1);
        }
        return sum;
    } else {
        // Asymptotic expansion for large u
        double sum = 0.0;
        double term = std::exp(-u) / u;
        sum = term;
        for (int n = 1; n <= 10; ++n) {
            term *= -n / u;
            sum += term;
        }
        return sum;
    }
}

/**
 * @brief Analytical Theis drawdown
 * 
 * s(r,t) = Q / (4π T) * W(r² S / (4 T t))
 * 
 * @param r Distance from well [m]
 * @param t Time since pumping started [s]
 * @param Q Pumping rate [m³/s]
 * @param T Transmissivity [m²/s]
 * @param S Storativity [-]
 */
double theis_drawdown(double r, double t, double Q, double T, double S) {
    double u = r * r * S / (4.0 * T * t);
    double W = theis_W(u);
    return Q / (4.0 * M_PI * T) * W;
}

int main() {
    std::cout << "=== dGW Theis Well Test ===" << std::endl;
    
    // Problem parameters
    const double T = 1e-3;          // Transmissivity [m²/s]
    const double K = 1e-4;          // Hydraulic conductivity [m/s]
    const double b = T / K;         // Aquifer thickness = 10 m
    const double S = 1e-4;          // Storativity (confined)
    const double Q = 0.01;          // Pumping rate [m³/s]
    const double h0 = 100.0;        // Initial head [m]
    
    // Domain: 1000m x 1000m, well at center
    const Index nx = 101;
    const Index ny = 101;
    const double dx = 20.0;  // 20m cells
    const double dy = 20.0;
    const double L = nx * dx;  // 2020m domain
    
    // Time settings
    const double t_end = 86400.0;   // 1 day
    const double dt = 600.0;        // 10 minute steps
    
    std::cout << "Domain: " << L << " x " << L << " m" << std::endl;
    std::cout << "Cells: " << nx << " x " << ny << std::endl;
    std::cout << "T = " << T << " m²/s, S = " << S << std::endl;
    std::cout << "Q = " << Q << " m³/s" << std::endl;
    
    // Create mesh
    Vector z_surface(nx * ny);
    Vector z_bottom(nx * ny);
    for (Index i = 0; i < nx * ny; ++i) {
        z_surface(i) = h0 + b;  // Surface above initial head
        z_bottom(i) = h0 - b;   // Bottom below initial head
    }
    
    auto mesh = Mesh::create_structured_2d(nx, ny, dx, dy, z_surface, z_bottom);
    
    // Create model
    DGW model;
    model.set_mesh(mesh);
    model.set_physics(GoverningEquation::Confined);  // Confined aquifer
    
    // Set parameters
    Parameters params(GoverningEquation::Confined);
    auto& p = params.as_2d();
    p.K.resize(nx * ny);
    p.Sy.resize(nx * ny);  // This is Ss for confined
    p.z_surface = z_surface;
    p.z_bottom = z_bottom;
    
    for (Index i = 0; i < nx * ny; ++i) {
        p.K(i) = K;
        p.Sy(i) = S / b;  // Ss = S / b for confined
    }
    
    model.set_parameters(params);
    
    // Configure
    Config config;
    config.physics.governing_equation = GoverningEquation::Confined;
    config.solver.dt_initial = dt;
    config.solver.newton_tolerance = 1e-8;
    config.time.start_time = 0.0;
    config.time.end_time = t_end;
    
    model.set_config(config);
    
    // Initialize
    model.initialize();
    
    // Set pumping at center cell
    Index well_cell = (ny / 2) * nx + (nx / 2);
    double x_well = (nx / 2 + 0.5) * dx;
    double y_well = (ny / 2 + 0.5) * dy;
    
    Vector pumping(nx * ny);
    pumping.setZero();
    pumping(well_cell) = Q;
    model.set_pumping(pumping);
    
    std::cout << "Well at cell " << well_cell << " (" << x_well << ", " << y_well << ")" << std::endl;
    
    // Output file for comparison
    std::ofstream outfile("theis_comparison.csv");
    outfile << "time,r,numerical,analytical,error_pct\n";
    
    // Observation radii
    std::vector<double> r_obs = {50, 100, 200, 400, 800};
    
    // Time stepping
    double time = 0.0;
    int step = 0;
    
    while (time < t_end) {
        auto result = model.step(dt);
        time += dt;
        step++;
        
        if (!result.success) {
            std::cerr << "Step " << step << " failed!" << std::endl;
            break;
        }
        
        // Compare at observation points
        if (step % 10 == 0) {  // Every 10 steps
            const auto& head = model.head();
            
            for (double r : r_obs) {
                // Find cell at distance r from well (along x-axis)
                Index obs_i = static_cast<Index>((x_well + r) / dx);
                Index obs_j = ny / 2;
                
                if (obs_i >= nx) continue;
                
                Index obs_cell = obs_j * nx + obs_i;
                double h_numerical = head(obs_cell);
                double s_numerical = h0 - h_numerical;
                
                double s_analytical = theis_drawdown(r, time, Q, T, S);
                double error_pct = 100.0 * std::abs(s_numerical - s_analytical) / 
                                   (std::abs(s_analytical) + 1e-10);
                
                outfile << time << "," << r << "," 
                        << s_numerical << "," << s_analytical << ","
                        << error_pct << "\n";
            }
        }
        
        if (step % 100 == 0) {
            std::cout << "Step " << step << ", time = " << time/3600.0 << " hr" << std::endl;
        }
    }
    
    outfile.close();
    
    // Final comparison
    std::cout << "\n=== Final Drawdown Comparison (t = " << time/3600.0 << " hr) ===" << std::endl;
    std::cout << "r [m]\tNumerical [m]\tAnalytical [m]\tError [%]" << std::endl;
    
    const auto& head = model.head();
    for (double r : r_obs) {
        Index obs_i = static_cast<Index>((x_well + r) / dx);
        if (obs_i >= nx) continue;
        
        Index obs_cell = (ny / 2) * nx + obs_i;
        double s_numerical = h0 - head(obs_cell);
        double s_analytical = theis_drawdown(r, time, Q, T, S);
        double error_pct = 100.0 * std::abs(s_numerical - s_analytical) / 
                           (std::abs(s_analytical) + 1e-10);
        
        std::cout << r << "\t" << s_numerical << "\t" << s_analytical << "\t" << error_pct << std::endl;
    }
    
    // Check mass balance
    double storage = model.total_storage();
    double pumped_volume = Q * time;
    std::cout << "\nTotal pumped: " << pumped_volume << " m³" << std::endl;
    std::cout << "Storage change: " << storage << " m³" << std::endl;
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    
    return 0;
}
