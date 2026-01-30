"""
dgw/jax_wrapper.py - JAX integration for dGW

Provides differentiable groundwater simulation through JAX custom_vjp,
enabling gradient-based parameter calibration and neural network coupling.

Usage:
    import jax
    import jax.numpy as jnp
    from dgw import jax_wrapper as dgw_jax
    
    # Create differentiable forward model
    @jax.custom_vjp
    def simulate(params, forcings, initial_state):
        return dgw_jax.forward(params, forcings, initial_state)
    
    # Gradients flow through
    grad_fn = jax.grad(lambda p: loss(simulate(p, forcings, init), obs))
"""

import numpy as np
from typing import Dict, Tuple, Optional, NamedTuple
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_vjp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import dgw_core
    HAS_DGW = True
except ImportError:
    HAS_DGW = False
    print("Warning: dgw_core not found. Build with: cmake .. && make")


class DGWState(NamedTuple):
    """Differentiable groundwater state."""
    head: np.ndarray          # Hydraulic head [m]
    time: float               # Current time [s]
    storage: float            # Total storage [m³]


class DGWParams(NamedTuple):
    """Trainable groundwater parameters."""
    K: np.ndarray             # Hydraulic conductivity [m/s]
    Sy: np.ndarray            # Specific yield [-]
    streambed_K: Optional[np.ndarray] = None  # Streambed K [m/s]


class DGWForcings(NamedTuple):
    """Time-varying forcings."""
    recharge: np.ndarray      # Recharge rate [m/s], shape (n_times, n_cells)
    river_stage: Optional[np.ndarray] = None  # River stage [m]
    pumping: Optional[np.ndarray] = None      # Pumping [m³/s]


class DGWModel:
    """
    JAX-compatible wrapper for dGW groundwater model.
    
    Provides forward simulation with automatic differentiation through
    Enzyme-generated adjoints exposed via JAX custom_vjp.
    
    Example:
        model = DGWModel.from_config("config.yaml")
        
        # Forward simulation
        final_state = model.simulate(params, forcings, dt=3600.0, n_steps=24)
        
        # With gradients
        def loss_fn(params):
            pred = model.simulate(params, forcings)
            return jnp.mean((pred.head - observed)**2)
        
        grads = jax.grad(loss_fn)(params)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize model from config file."""
        if not HAS_DGW:
            raise RuntimeError("dgw_core not available")
        
        self._model = dgw_core.DGW()
        self._initialized = False
        
        if config_file:
            self.initialize(config_file)
    
    def initialize(self, config_file: str):
        """Initialize from configuration file."""
        self._model.initialize(config_file)
        self._initialized = True
        self._n_cells = self._model.mesh.n_cells()
    
    @classmethod
    def from_config(cls, config_file: str) -> 'DGWModel':
        """Create model from config file."""
        model = cls()
        model.initialize(config_file)
        return model
    
    @property
    def n_cells(self) -> int:
        """Number of mesh cells."""
        return self._n_cells
    
    @property
    def mesh(self):
        """Access mesh object."""
        return self._model.mesh
    
    def get_state(self) -> DGWState:
        """Get current model state."""
        head = np.array(self._model.get_value("head"))
        time = self._model.get_current_time()
        storage = self._model.get_total_storage()
        return DGWState(head=head, time=time, storage=storage)
    
    def set_state(self, state: DGWState):
        """Set model state."""
        self._model.set_value("head", state.head)
    
    def step(self, dt: float, 
             recharge: Optional[np.ndarray] = None,
             river_stage: Optional[np.ndarray] = None,
             pumping: Optional[np.ndarray] = None) -> DGWState:
        """
        Advance model by one timestep.
        
        Args:
            dt: Timestep [s]
            recharge: Recharge rate [m/s]
            river_stage: River stage [m]
            pumping: Pumping rate [m³/s]
        
        Returns:
            New model state
        """
        # Set forcings
        if recharge is not None:
            self._model.set_recharge(recharge)
        if river_stage is not None:
            self._model.set_river_stage(river_stage)
        if pumping is not None:
            self._model.set_value("pumping", pumping)
        
        # Step forward
        self._model.step(dt)
        
        return self.get_state()
    
    def simulate(self, 
                 params: DGWParams,
                 forcings: DGWForcings,
                 dt: float = 3600.0,
                 n_steps: int = 24,
                 initial_state: Optional[DGWState] = None) -> DGWState:
        """
        Run forward simulation.
        
        Args:
            params: Model parameters
            forcings: Time-varying forcings
            dt: Timestep [s]
            n_steps: Number of steps
            initial_state: Initial state (uses current if None)
        
        Returns:
            Final state after simulation
        """
        # Set parameters
        self._model.set_value("hydraulic_conductivity", params.K)
        self._model.set_value("specific_yield", params.Sy)
        if params.streambed_K is not None:
            self._model.set_value("streambed_conductivity", params.streambed_K)
        
        # Set initial state
        if initial_state is not None:
            self.set_state(initial_state)
        
        # Time loop
        for i in range(n_steps):
            # Get forcings for this step
            recharge = forcings.recharge[i] if forcings.recharge.ndim > 1 else forcings.recharge
            river_stage = None
            if forcings.river_stage is not None:
                river_stage = forcings.river_stage[i] if forcings.river_stage.ndim > 1 else forcings.river_stage
            
            self.step(dt, recharge=recharge, river_stage=river_stage)
        
        return self.get_state()
    
    def compute_gradients(self,
                          params: DGWParams,
                          forcings: DGWForcings,
                          observed: np.ndarray,
                          dt: float = 3600.0,
                          n_steps: int = 24) -> DGWParams:
        """
        Compute parameter gradients using adjoint method.
        
        This uses Enzyme AD through the C++ backend.
        
        Args:
            params: Current parameters
            forcings: Forcings used in simulation
            observed: Observed head values
            dt, n_steps: Simulation settings
        
        Returns:
            Parameter gradients (same structure as params)
        """
        # Forward simulation
        final_state = self.simulate(params, forcings, dt, n_steps)
        
        # Compute loss gradient: ∂L/∂h = 2*(h - obs)
        loss_grad = 2.0 * (final_state.head - observed)
        
        # Run adjoint
        self._model.run_adjoint("head", loss_grad)
        
        # Extract parameter gradients
        dK = np.array(self._model.get_parameter_gradient("hydraulic_conductivity"))
        dSy = np.array(self._model.get_parameter_gradient("specific_yield"))
        
        dStreamK = None
        if params.streambed_K is not None:
            dStreamK = np.array(self._model.get_parameter_gradient("streambed_conductivity"))
        
        return DGWParams(K=dK, Sy=dSy, streambed_K=dStreamK)


# =============================================================================
# JAX Integration
# =============================================================================

if HAS_JAX and HAS_DGW:
    
    # Global model instance for JAX functions
    _global_model: Optional[DGWModel] = None
    
    def set_global_model(model: DGWModel):
        """Set the global model for JAX functions."""
        global _global_model
        _global_model = model
    
    @partial(custom_vjp, nondiff_argnums=(1, 2, 3))
    def dgw_simulate(params_flat: jnp.ndarray,
                     forcings: DGWForcings,
                     dt: float,
                     n_steps: int) -> jnp.ndarray:
        """
        JAX-differentiable groundwater simulation.
        
        Args:
            params_flat: Flattened parameter array [K, Sy]
            forcings: Forcing data
            dt: Timestep [s]
            n_steps: Number of steps
        
        Returns:
            Final head as JAX array
        """
        if _global_model is None:
            raise RuntimeError("Call set_global_model() first")
        
        # Unpack parameters
        n_cells = _global_model.n_cells
        K = np.array(params_flat[:n_cells])
        Sy = np.array(params_flat[n_cells:2*n_cells])
        params = DGWParams(K=K, Sy=Sy)
        
        # Convert forcings to numpy
        forcings_np = DGWForcings(
            recharge=np.array(forcings.recharge),
            river_stage=np.array(forcings.river_stage) if forcings.river_stage is not None else None
        )
        
        # Run simulation
        final_state = _global_model.simulate(params, forcings_np, dt, n_steps)
        
        return jnp.array(final_state.head)
    
    def dgw_simulate_fwd(params_flat, forcings, dt, n_steps):
        """Forward pass, saving residuals for backward."""
        head = dgw_simulate(params_flat, forcings, dt, n_steps)
        # Save what we need for backward pass
        return head, (params_flat, forcings, dt, n_steps, head)
    
    def dgw_simulate_bwd(forcings, dt, n_steps, res, g):
        """Backward pass using Enzyme adjoints."""
        params_flat, _, _, _, head = res
        
        if _global_model is None:
            raise RuntimeError("Call set_global_model() first")
        
        # Unpack parameters
        n_cells = _global_model.n_cells
        K = np.array(params_flat[:n_cells])
        Sy = np.array(params_flat[n_cells:2*n_cells])
        params = DGWParams(K=K, Sy=Sy)
        
        # Convert adjoint seed (g = ∂L/∂head)
        adjoint_seed = np.array(g)
        
        # Re-run forward (needed for adjoint computation)
        forcings_np = DGWForcings(
            recharge=np.array(forcings.recharge),
            river_stage=np.array(forcings.river_stage) if forcings.river_stage is not None else None
        )
        _global_model.simulate(params, forcings_np, dt, n_steps)
        
        # Run adjoint through C++ Enzyme
        _global_model._model.run_adjoint("head", adjoint_seed)
        
        # Get parameter gradients
        dK = np.array(_global_model._model.get_parameter_gradient("hydraulic_conductivity"))
        dSy = np.array(_global_model._model.get_parameter_gradient("specific_yield"))
        
        # Pack gradients
        grad_params = jnp.concatenate([jnp.array(dK), jnp.array(dSy)])
        
        return (grad_params,)
    
    dgw_simulate.defvjp(dgw_simulate_fwd, dgw_simulate_bwd)
    
    def create_loss_function(observed: np.ndarray,
                             forcings: DGWForcings,
                             dt: float = 3600.0,
                             n_steps: int = 24):
        """
        Create a JAX-differentiable loss function.
        
        Args:
            observed: Observed head values [n_cells]
            forcings: Forcing data
            dt, n_steps: Simulation settings
        
        Returns:
            Loss function: params_flat -> scalar loss
        """
        obs = jnp.array(observed)
        
        def loss_fn(params_flat):
            pred = dgw_simulate(params_flat, forcings, dt, n_steps)
            return jnp.mean((pred - obs)**2)
        
        return loss_fn


# =============================================================================
# Optimization Utilities
# =============================================================================

def calibrate_model(model: DGWModel,
                    observed: np.ndarray,
                    forcings: DGWForcings,
                    initial_params: DGWParams,
                    dt: float = 3600.0,
                    n_steps: int = 24,
                    max_iterations: int = 100,
                    learning_rate: float = 1e-3,
                    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                    verbose: bool = True) -> Tuple[DGWParams, list]:
    """
    Calibrate model parameters using gradient descent.
    
    Args:
        model: Initialized DGW model
        observed: Observed head values
        forcings: Forcing data
        initial_params: Starting parameters
        dt, n_steps: Simulation settings
        max_iterations: Max optimization iterations
        learning_rate: Adam learning rate
        bounds: Parameter bounds {'K': (1e-8, 1e-2), 'Sy': (0.01, 0.4)}
        verbose: Print progress
    
    Returns:
        Tuple of (optimized_params, loss_history)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required for calibration")
    
    import jax.numpy as jnp
    from jax.example_libraries import optimizers
    
    # Set global model
    set_global_model(model)
    
    # Default bounds
    if bounds is None:
        bounds = {
            'K': (1e-10, 1e-1),
            'Sy': (0.01, 0.5)
        }
    
    # Create loss function
    loss_fn = create_loss_function(observed, forcings, dt, n_steps)
    grad_fn = jax.grad(loss_fn)
    
    # Initialize optimizer
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    
    # Pack initial parameters
    n_cells = model.n_cells
    params_flat = jnp.concatenate([
        jnp.array(initial_params.K),
        jnp.array(initial_params.Sy)
    ])
    
    opt_state = opt_init(params_flat)
    loss_history = []
    
    # Optimization loop
    for i in range(max_iterations):
        params = get_params(opt_state)
        
        # Compute loss and gradients
        loss = loss_fn(params)
        grads = grad_fn(params)
        
        loss_history.append(float(loss))
        
        if verbose and i % 10 == 0:
            print(f"Iteration {i}: loss = {loss:.6f}")
        
        # Update
        opt_state = opt_update(i, grads, opt_state)
        
        # Apply bounds
        params = get_params(opt_state)
        K = jnp.clip(params[:n_cells], bounds['K'][0], bounds['K'][1])
        Sy = jnp.clip(params[n_cells:], bounds['Sy'][0], bounds['Sy'][1])
        params = jnp.concatenate([K, Sy])
        opt_state = opt_init(params)  # Reset with clipped params
        
        # Convergence check
        if len(loss_history) > 10:
            recent = loss_history[-10:]
            if abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-10) < 1e-6:
                if verbose:
                    print(f"Converged at iteration {i}")
                break
    
    # Unpack final parameters
    final_params = get_params(opt_state)
    final_K = np.array(final_params[:n_cells])
    final_Sy = np.array(final_params[n_cells:])
    
    return DGWParams(K=final_K, Sy=final_Sy), loss_history


# =============================================================================
# Utility Functions
# =============================================================================

def nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    return 1 - np.sum((observed - simulated)**2) / np.sum((observed - np.mean(observed))**2)


def rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Root Mean Square Error."""
    return np.sqrt(np.mean((observed - simulated)**2))


def kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Kling-Gupta Efficiency."""
    r = np.corrcoef(observed, simulated)[0, 1]
    alpha = np.std(simulated) / np.std(observed)
    beta = np.mean(simulated) / np.mean(observed)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
