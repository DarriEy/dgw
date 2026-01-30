/**
 * @file types.hpp
 * @brief Core type definitions for dGW
 * 
 * This file defines the fundamental types used throughout dGW,
 * including scalar types, array types, and configuration enums.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstdint>
#include <memory>
#include <vector>
#include <array>
#include <variant>
#include <optional>
#include <functional>
#include <string>
#include <string_view>

namespace dgw {

// ============================================================================
// Scalar Types
// ============================================================================

using Real = double;
using Index = int64_t;
using Size = size_t;

// ============================================================================
// Array Types (Eigen-based)
// ============================================================================

// Dense vectors
using Vector = Eigen::VectorXd;
using VectorI = Eigen::VectorXi;
using VectorRef = Eigen::Ref<Vector>;
using VectorConstRef = Eigen::Ref<const Vector>;

// Dense matrices
using Matrix = Eigen::MatrixXd;
using MatrixRef = Eigen::Ref<Matrix>;
using MatrixConstRef = Eigen::Ref<const Matrix>;

// Sparse matrices (CSR format for solver efficiency)
using SparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using SparseTriplet = Eigen::Triplet<Real>;

// Fixed-size vectors for coordinates
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;

// ============================================================================
// Physics Decision Enums (SUMMA-style)
// ============================================================================

/**
 * @brief Governing equation for groundwater flow
 */
enum class GoverningEquation {
    LinearDiffusion,    ///< Sy ∂h/∂t = T ∇²h + R (constant T)
    Boussinesq,         ///< Sy ∂h/∂t = ∇·(K(h-z_bot)∇h) + R (unconfined, nonlinear)
    Confined,           ///< Ss·b ∂h/∂t = ∇·(T∇h) + R (confined, linear)
    TwoLayer,           ///< Coupled unconfined + confined with leakage
    MultiLayer,         ///< N-layer system with vertical exchange
    Richards3D,         ///< Full 3D variably-saturated flow
};

/**
 * @brief Transmissivity computation method
 */
enum class TransmissivityMethod {
    Standard,           ///< T = K * (h - z_bot), clipped at zero
    Smoothed,           ///< Smooth transition near zero thickness
    Harmonic,           ///< Harmonic mean for layered K
    Upstream,           ///< Upstream weighting for stability
};

/**
 * @brief Specific yield / storage behavior
 */
enum class StorageMethod {
    Constant,           ///< Sy = Sy(x,y) spatially variable but constant in time
    DepthDependent,     ///< Sy = f(depth to water table)
    Hysteretic,         ///< Different for wetting vs draining
};

/**
 * @brief Stream-aquifer exchange formulation
 */
enum class StreamExchangeMethod {
    Conductance,        ///< Q = C*(h_stream - h_gw)
    ConductanceClogging,///< Adds clogging layer resistance
    KinematicLosing,    ///< Always losing (arid disconnected streams)
    SaturatedUnsaturated, ///< Smooth transition between connected/disconnected
};

/**
 * @brief Vadose zone delay for recharge
 */
enum class VadoseMethod {
    Direct,             ///< Instant transmission to water table
    ExponentialLag,     ///< τ = f(depth), delayed response
    KinematicWave,      ///< 1D unsaturated flow approximation
    FullRichards,       ///< Coupled with Richards3D
};

/**
 * @brief Soil water retention curve model (for Richards)
 */
enum class RetentionModel {
    VanGenuchten,       ///< θ(ψ) and K(ψ) from van Genuchten-Mualem
    BrooksCorey,        ///< θ(ψ) and K(ψ) from Brooks-Corey
    ClappHornberger,    ///< Simplified for land surface models
    Tabulated,          ///< User-supplied tables
};

/**
 * @brief Boundary condition types
 */
enum class BoundaryType {
    NoFlow,             ///< ∂h/∂n = 0 (Neumann zero flux)
    FixedHead,          ///< h = h_specified (Dirichlet)
    FixedFlux,          ///< q·n = q_specified (Neumann)
    GeneralHead,        ///< Q = C*(h_external - h) (Robin)
    SeepageFace,        ///< h = z if saturated, else no flow
    Drain,              ///< One-way flow out when h > z_drain
    River,              ///< Stream-aquifer exchange
    Recharge,           ///< Top boundary with infiltration
};

/**
 * @brief Mesh topology type
 */
enum class MeshType {
    Structured2D,       ///< Regular grid (i,j indexing)
    Unstructured2D,     ///< Voronoi/triangular (general connectivity)
    Structured3D,       ///< Regular grid (i,j,k)
    Unstructured3D,     ///< Tetrahedral/hexahedral
    Layered,            ///< 2D mesh extruded into layers
};

/**
 * @brief Linear solver backend
 */
enum class LinearSolver {
    EigenLU,            ///< Eigen's SparseLU (direct)
    EigenCholesky,      ///< Eigen's SimplicialLDLT (symmetric)
    EigenCG,            ///< Eigen's ConjugateGradient (iterative)
    EigenBiCGSTAB,      ///< Eigen's BiCGSTAB (non-symmetric)
#ifdef DGW_HAS_PETSC
    PETScKSP,           ///< PETSc Krylov solver
    PETScDirect,        ///< PETSc direct solver (MUMPS/SuperLU)
#endif
};

/**
 * @brief Nonlinear solver method
 */
enum class NonlinearSolver {
    Newton,             ///< Full Newton-Raphson
    Picard,             ///< Simple iteration (linearized)
    NewtonLineSearch,   ///< Newton with backtracking line search
    TrustRegion,        ///< Trust region Newton
};

/**
 * @brief Time stepping method
 */
enum class TimeSteppingMethod {
    BackwardEuler,      ///< First-order implicit (robust)
    CrankNicolson,      ///< Second-order (may oscillate)
    BDF2,               ///< Second-order backward differentiation
    Adaptive,           ///< Error-controlled adaptive stepping
};

// ============================================================================
// Forward Declarations
// ============================================================================

class Mesh;
class Mesh2D;
class Mesh3D;
class MeshLayered;

class State;
class Parameters;
class Config;

class PhysicsBase;
class LinearDiffusion;
class BoussinesqSolver;
class ConfinedSolver;
class TwoLayerSolver;
class Richards3DSolver;

class NewtonSolver;
class LinearSolverBase;
class TimeStepper;

class StreamAquiferExchange;
class RechargeHandler;
class Remapper;
class BoundaryConditions;

// ============================================================================
// Smart Pointer Aliases
// ============================================================================

template<typename T>
using Ptr = std::shared_ptr<T>;

template<typename T>
using UniquePtr = std::unique_ptr<T>;

template<typename T>
using WeakPtr = std::weak_ptr<T>;

// ============================================================================
// Function Types for Callbacks
// ============================================================================

/// Residual function: F(h) = 0
using ResidualFunc = std::function<void(const Vector& h, Vector& residual)>;

/// Jacobian function: J = ∂F/∂h
using JacobianFunc = std::function<void(const Vector& h, SparseMatrix& jacobian)>;

/// Convergence callback: called each iteration with (iter, residual_norm)
using ConvergenceCallback = std::function<bool(Index iter, Real norm)>;

/// Output callback: called each time step
using OutputCallback = std::function<void(Real time, const State& state)>;

// ============================================================================
// Result Types
// ============================================================================

/**
 * @brief Result of a solve operation
 */
struct SolveResult {
    bool converged = false;
    Index iterations = 0;
    Real final_residual = 0.0;
    Real solve_time_ms = 0.0;
    std::string message;
};

/**
 * @brief Time step result
 */
struct StepResult {
    bool success = false;
    Real dt_actual = 0.0;
    Real dt_next = 0.0;
    Index newton_iters = 0;
    Real mass_balance_error = 0.0;
};

// ============================================================================
// Constants
// ============================================================================

namespace constants {
    constexpr Real GRAVITY = 9.80665;           ///< m/s²
    constexpr Real WATER_DENSITY = 1000.0;      ///< kg/m³
    constexpr Real WATER_COMPRESSIBILITY = 4.4e-10; ///< 1/Pa
    constexpr Real EPSILON = 1e-15;             ///< Numerical zero
    constexpr Real PI = 3.14159265358979323846;
}

// ============================================================================
// Enzyme AD Helpers
// ============================================================================

#ifdef DGW_HAS_ENZYME

// Enzyme external function declarations
extern "C" {
    int enzyme_dup;
    int enzyme_dupnoneed;
    int enzyme_out;
    int enzyme_const;
}

/// Mark a value as active (needs gradient)
template<typename T>
T __enzyme_autodiff(void*, ...);

/// Forward mode AD
template<typename T>
T __enzyme_fwddiff(void*, ...);

#endif // DGW_HAS_ENZYME

} // namespace dgw
