/**
 * @file test_gradients.cpp
 * @brief Test automatic differentiation via Enzyme
 * 
 * Verifies that Enzyme-computed gradients match finite differences.
 * This is crucial for ensuring correctness of the adjoint method
 * used in parameter optimization.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <dgw/dgw.hpp>
#include <iostream>
#include <random>

using namespace dgw;
using Catch::Approx;

// Tolerance for gradient comparisons
constexpr double GRAD_TOL = 1e-4;
constexpr double FD_EPSILON = 1e-6;

/**
 * @brief Simple loss function: mean squared error from target head
 */
double compute_loss(const Vector& head, const Vector& target) {
    double loss = 0.0;
    for (Index i = 0; i < head.size(); ++i) {
        double diff = head(i) - target(i);
        loss += diff * diff;
    }
    return loss / head.size();
}

/**
 * @brief Gradient of MSE loss w.r.t. head
 */
Vector compute_loss_gradient(const Vector& head, const Vector& target) {
    Vector grad(head.size());
    for (Index i = 0; i < head.size(); ++i) {
        grad(i) = 2.0 * (head(i) - target(i)) / head.size();
    }
    return grad;
}

TEST_CASE("Boussinesq kernel gradients", "[gradients][boussinesq]") {
    // Small test mesh: 5x5 cells
    const Index nx = 5, ny = 5;
    const Index n = nx * ny;
    const double dx = 100.0, dy = 100.0;
    
    // Random initial conditions
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    Vector z_surface(n), z_bottom(n);
    for (Index i = 0; i < n; ++i) {
        z_surface(i) = 100.0;
        z_bottom(i) = 50.0;
    }
    
    auto mesh = Mesh::create_structured_2d(nx, ny, dx, dy, z_surface, z_bottom);
    
    // Parameters
    Vector K(n), Sy(n), recharge(n);
    Vector h_old(n), h_target(n);
    
    for (Index i = 0; i < n; ++i) {
        K(i) = 1e-4 * (0.5 + dist(rng));
        Sy(i) = 0.1 + 0.1 * dist(rng);
        recharge(i) = 1e-8 * dist(rng);
        h_old(i) = 70.0 + 10.0 * dist(rng);
        h_target(i) = h_old(i) + 5.0 * (dist(rng) - 0.5);
    }
    
    const double dt = 3600.0;  // 1 hour
    
    // Create model
    DGW model;
    model.set_mesh(mesh);
    model.set_physics(GoverningEquation::Boussinesq);
    
    Parameters params(GoverningEquation::Boussinesq);
    params.as_2d().K = K;
    params.as_2d().Sy = Sy;
    params.as_2d().z_surface = z_surface;
    params.as_2d().z_bottom = z_bottom;
    model.set_parameters(params);
    
    Config config;
    config.physics.governing_equation = GoverningEquation::Boussinesq;
    config.solver.dt_initial = dt;
    config.solver.newton_tolerance = 1e-10;
    model.set_config(config);
    
    // Initialize with h_old
    model.initialize();
    model.state().as_2d().head = h_old;
    model.state().as_2d().head_old = h_old;
    model.set_recharge(recharge);
    
    // Run forward
    model.step(dt);
    Vector h_new = model.head();
    
    // Compute loss and gradient
    double loss = compute_loss(h_new, h_target);
    Vector loss_grad = compute_loss_gradient(h_new, h_target);
    
    SECTION("Gradient w.r.t. K (Enzyme vs finite difference)") {
        // Get Enzyme gradient
        Parameters enzyme_grads = model.compute_gradients(loss_grad);
        Vector dL_dK_enzyme = enzyme_grads.as_2d().K;
        
        // Finite difference gradient
        Vector dL_dK_fd(n);
        
        for (Index i = 0; i < n; ++i) {
            // Perturb K[i]
            double K_orig = K(i);
            
            // Forward perturbation
            K(i) = K_orig + FD_EPSILON;
            params.as_2d().K = K;
            model.set_parameters(params);
            model.state().as_2d().head = h_old;
            model.step(dt);
            double loss_plus = compute_loss(model.head(), h_target);
            
            // Backward perturbation
            K(i) = K_orig - FD_EPSILON;
            params.as_2d().K = K;
            model.set_parameters(params);
            model.state().as_2d().head = h_old;
            model.step(dt);
            double loss_minus = compute_loss(model.head(), h_target);
            
            // Central difference
            dL_dK_fd(i) = (loss_plus - loss_minus) / (2.0 * FD_EPSILON);
            
            // Restore
            K(i) = K_orig;
        }
        
        // Compare
        for (Index i = 0; i < n; ++i) {
            if (std::abs(dL_dK_fd(i)) > 1e-10) {
                double rel_error = std::abs(dL_dK_enzyme(i) - dL_dK_fd(i)) / 
                                   std::abs(dL_dK_fd(i));
                REQUIRE(rel_error < GRAD_TOL);
            } else {
                REQUIRE(std::abs(dL_dK_enzyme(i)) < 1e-8);
            }
        }
    }
    
    SECTION("Gradient w.r.t. Sy") {
        Parameters enzyme_grads = model.compute_gradients(loss_grad);
        Vector dL_dSy_enzyme = enzyme_grads.as_2d().Sy;
        
        Vector dL_dSy_fd(n);
        
        for (Index i = 0; i < n; ++i) {
            double Sy_orig = Sy(i);
            
            Sy(i) = Sy_orig + FD_EPSILON;
            params.as_2d().Sy = Sy;
            model.set_parameters(params);
            model.state().as_2d().head = h_old;
            model.step(dt);
            double loss_plus = compute_loss(model.head(), h_target);
            
            Sy(i) = Sy_orig - FD_EPSILON;
            params.as_2d().Sy = Sy;
            model.set_parameters(params);
            model.state().as_2d().head = h_old;
            model.step(dt);
            double loss_minus = compute_loss(model.head(), h_target);
            
            dL_dSy_fd(i) = (loss_plus - loss_minus) / (2.0 * FD_EPSILON);
            Sy(i) = Sy_orig;
        }
        
        for (Index i = 0; i < n; ++i) {
            if (std::abs(dL_dSy_fd(i)) > 1e-10) {
                double rel_error = std::abs(dL_dSy_enzyme(i) - dL_dSy_fd(i)) / 
                                   std::abs(dL_dSy_fd(i));
                REQUIRE(rel_error < GRAD_TOL);
            }
        }
    }
}

TEST_CASE("Richards kernel gradients", "[gradients][richards]") {
    // 3x3x3 mesh for 3D Richards
    const Index nx = 3, ny = 3, nz = 3;
    const Index n = nx * ny * nz;
    
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // van Genuchten parameters
    Vector theta_r(n), theta_s(n), alpha(n), n_vg(n), K_sat(n);
    Vector psi_old(n), psi_target(n);
    
    for (Index i = 0; i < n; ++i) {
        theta_r(i) = 0.05 + 0.02 * dist(rng);
        theta_s(i) = 0.4 + 0.05 * dist(rng);
        alpha(i) = 3.0 + dist(rng);  // 1/m
        n_vg(i) = 1.5 + 0.5 * dist(rng);
        K_sat(i) = 1e-5 * (0.5 + dist(rng));
        psi_old(i) = -1.0 - dist(rng);  // Unsaturated
        psi_target(i) = psi_old(i) + 0.1 * (dist(rng) - 0.5);
    }
    
    // Test van Genuchten constitutive relations
    SECTION("van Genuchten theta(psi) derivative") {
        for (Index i = 0; i < n; ++i) {
            double psi = psi_old(i);
            
            // Analytical C = dtheta/dpsi
            double C_analytical = retention::vg_C(psi, theta_r(i), theta_s(i), 
                                                   alpha(i), n_vg(i));
            
            // Finite difference
            double theta_plus = retention::vg_theta(psi + FD_EPSILON, theta_r(i), 
                                                     theta_s(i), alpha(i), n_vg(i));
            double theta_minus = retention::vg_theta(psi - FD_EPSILON, theta_r(i),
                                                      theta_s(i), alpha(i), n_vg(i));
            double C_fd = (theta_plus - theta_minus) / (2.0 * FD_EPSILON);
            
            double rel_error = std::abs(C_analytical - C_fd) / (std::abs(C_fd) + 1e-10);
            REQUIRE(rel_error < 1e-4);
        }
    }
    
    SECTION("van Genuchten K(psi) derivative") {
        for (Index i = 0; i < n; ++i) {
            double psi = psi_old(i);
            
            // Analytical dK/dpsi
            double dK_analytical = retention::vg_dK_dpsi(psi, K_sat(i), alpha(i), n_vg(i));
            
            // Finite difference
            double K_plus = retention::vg_K(psi + FD_EPSILON, K_sat(i), alpha(i), n_vg(i));
            double K_minus = retention::vg_K(psi - FD_EPSILON, K_sat(i), alpha(i), n_vg(i));
            double dK_fd = (K_plus - K_minus) / (2.0 * FD_EPSILON);
            
            double rel_error = std::abs(dK_analytical - dK_fd) / (std::abs(dK_fd) + 1e-10);
            REQUIRE(rel_error < 1e-3);
        }
    }
}

TEST_CASE("Smooth transmissivity derivative", "[gradients][boussinesq]") {
    const double K = 1e-4;
    const double z_bot = 50.0;
    const double eps = 0.01;
    
    std::vector<double> h_values = {49.9, 50.0, 50.005, 50.01, 50.02, 50.1, 51.0};
    
    for (double h : h_values) {
        double dT_dh_analytical = boussinesq_kernels::smooth_transmissivity_dh(h, K, z_bot, eps);
        
        double T_plus = boussinesq_kernels::smooth_transmissivity(h + FD_EPSILON, K, z_bot, eps);
        double T_minus = boussinesq_kernels::smooth_transmissivity(h - FD_EPSILON, K, z_bot, eps);
        double dT_dh_fd = (T_plus - T_minus) / (2.0 * FD_EPSILON);
        
        double abs_error = std::abs(dT_dh_analytical - dT_dh_fd);
        REQUIRE(abs_error < 1e-6);
    }
}

TEST_CASE("Stream exchange derivative", "[gradients][coupling]") {
    const double conductance = 1e-3;
    const double streambed_elev = 80.0;
    
    SECTION("Connected stream (h_gw > streambed)") {
        double h_gw = 85.0;
        double h_stream = 82.0;
        
        double dQ_dhgw = stream_kernels::conductance_dQ_dhgw(conductance);
        REQUIRE(dQ_dhgw == Approx(-conductance));
        
        // Finite difference check
        double Q_plus = stream_kernels::conductance_exchange(h_gw + FD_EPSILON, h_stream, conductance);
        double Q_minus = stream_kernels::conductance_exchange(h_gw - FD_EPSILON, h_stream, conductance);
        double dQ_fd = (Q_plus - Q_minus) / (2.0 * FD_EPSILON);
        
        REQUIRE(dQ_dhgw == Approx(dQ_fd));
    }
    
    SECTION("Disconnected stream (h_gw < streambed)") {
        double h_gw = 75.0;  // Below streambed
        double h_stream = 82.0;
        
        double dQ_dhgw = stream_kernels::kinematic_losing_dQ_dhgw(h_gw, streambed_elev, conductance);
        REQUIRE(dQ_dhgw == Approx(0.0));  // No dependence when disconnected
        
        // Verify exchange is constant w.r.t. h_gw when disconnected
        double Q_plus = stream_kernels::kinematic_losing_exchange(h_gw + 1.0, h_stream, 
                                                                   streambed_elev, conductance);
        double Q_base = stream_kernels::kinematic_losing_exchange(h_gw, h_stream,
                                                                   streambed_elev, conductance);
        REQUIRE(Q_plus == Approx(Q_base));
    }
}

TEST_CASE("Multi-step adjoint accumulation", "[gradients][adjoint]") {
    // Test that gradients accumulate correctly over multiple time steps
    const Index nx = 3, ny = 3;
    const Index n = nx * ny;
    const double dx = 100.0, dy = 100.0;
    const double dt = 3600.0;
    const int n_steps = 5;
    
    Vector z_surface(n), z_bottom(n);
    for (Index i = 0; i < n; ++i) {
        z_surface(i) = 100.0;
        z_bottom(i) = 50.0;
    }
    
    auto mesh = Mesh::create_structured_2d(nx, ny, dx, dy, z_surface, z_bottom);
    
    Vector K(n), Sy(n);
    K.setConstant(1e-4);
    Sy.setConstant(0.2);
    
    // Create model
    DGW model;
    model.set_mesh(mesh);
    model.set_physics(GoverningEquation::Boussinesq);
    
    Parameters params(GoverningEquation::Boussinesq);
    params.as_2d().K = K;
    params.as_2d().Sy = Sy;
    params.as_2d().z_surface = z_surface;
    params.as_2d().z_bottom = z_bottom;
    model.set_parameters(params);
    
    Config config;
    config.solver.dt_initial = dt;
    model.set_config(config);
    
    // Initial condition
    Vector h_init(n);
    h_init.setConstant(75.0);
    
    // Target at final time
    Vector h_target(n);
    h_target.setConstant(74.0);  // Slight drawdown
    
    // Forward with checkpointing
    model.initialize();
    model.state().as_2d().head = h_init;
    
    for (int step = 0; step < n_steps; ++step) {
        model.step(dt);
    }
    
    Vector h_final = model.head();
    double loss = compute_loss(h_final, h_target);
    Vector loss_grad = compute_loss_gradient(h_final, h_target);
    
    // Get gradients via adjoint
    model.forward_with_checkpoints();  // Re-run with checkpointing
    Parameters adjoint_grads = model.adjoint_pass(loss_grad);
    
    // Verify with finite differences (just check K[0])
    Index test_idx = 0;
    double K_orig = K(test_idx);
    
    K(test_idx) = K_orig + FD_EPSILON;
    params.as_2d().K = K;
    model.set_parameters(params);
    model.initialize();
    model.state().as_2d().head = h_init;
    for (int step = 0; step < n_steps; ++step) model.step(dt);
    double loss_plus = compute_loss(model.head(), h_target);
    
    K(test_idx) = K_orig - FD_EPSILON;
    params.as_2d().K = K;
    model.set_parameters(params);
    model.initialize();
    model.state().as_2d().head = h_init;
    for (int step = 0; step < n_steps; ++step) model.step(dt);
    double loss_minus = compute_loss(model.head(), h_target);
    
    double dL_dK_fd = (loss_plus - loss_minus) / (2.0 * FD_EPSILON);
    double dL_dK_adjoint = adjoint_grads.as_2d().K(test_idx);
    
    double rel_error = std::abs(dL_dK_adjoint - dL_dK_fd) / (std::abs(dL_dK_fd) + 1e-10);
    REQUIRE(rel_error < 0.01);  // 1% tolerance for multi-step
}
