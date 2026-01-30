"""
dGW Python Interface with JAX Integration
=========================================

This module provides a Pythonic interface to dGW with:
- JAX-compatible differentiable operations via custom_vjp
- NumPy interface for non-JAX workflows  
- Coupling helpers for SUMMA and dRoute

Example:
    >>> import dgw
    >>> import jax.numpy as jnp
    >>> 
    >>> # Create model
    >>> model = dgw.DGW.from_config("config.yaml")
    >>> 
    >>> # JAX-differentiable step
    >>> h_new = dgw.jax.boussinesq_step(h_old, K, Sy, recharge, model.mesh, dt)
    >>> 
    >>> # Compute gradients
    >>> grad_fn = jax.grad(loss_fn)
    >>> param_grads = grad_fn(params)
"""

from __future__ import annotations
from typing import Tuple, Optional, NamedTuple, Dict, Any
import numpy as np

# Import C++ bindings
try:
    from . import dgw_py as _dgw
except ImportError:
    import dgw_py as _dgw

# Re-export core classes
DGW = _dgw.DGW
DGW_BMI = _dgw.DGW_BMI
Config = _dgw.Config
State = _dgw.State
Parameters = _dgw.Parameters
Mesh = _dgw.Mesh
Mesh2D = _dgw.Mesh2D

# Re-export enums
GoverningEquation = _dgw.GoverningEquation
TransmissivityMethod = _dgw.TransmissivityMethod
StreamExchangeMethod = _dgw.StreamExchangeMethod
VadoseMethod = _dgw.VadoseMethod
StorageMethod = _dgw.StorageMethod
RetentionModel = _dgw.RetentionModel
BoundaryType = _dgw.BoundaryType

# Version
__version__ = _dgw.__version__
__enzyme_enabled__ = _dgw.__enzyme_enabled__


class StepResult(NamedTuple):
    """Result from a time step."""
    success: bool
    dt_actual: float
    dt_next: float
    newton_iters: int
    mass_balance_error: float


def create_structured_mesh(
    nx: int, ny: int, 
    dx: float, dy: float,
    z_surface: np.ndarray,
    z_bottom: np.ndarray
) -> Mesh:
    """
    Create a structured 2D mesh.
    
    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y directions
    dx, dy : float
        Cell spacing in x and y directions [m]
    z_surface : ndarray
        Surface elevation at each cell [m]
    z_bottom : ndarray
        Aquifer bottom elevation at each cell [m]
    
    Returns
    -------
    Mesh
        Structured 2D mesh
    """
    return _dgw.create_structured_mesh_2d(nx, ny, dx, dy, z_surface, z_bottom)


# =============================================================================
# JAX Integration
# =============================================================================

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_vjp
    
    _HAS_JAX = True
    
    @custom_vjp
    def boussinesq_step(
        h_old: jnp.ndarray,
        K: jnp.ndarray, 
        Sy: jnp.ndarray,
        recharge: jnp.ndarray,
        mesh_data: Dict[str, Any],
        dt: float
    ) -> jnp.ndarray:
        """
        JAX-differentiable Boussinesq time step.
        
        Computes one implicit time step of the Boussinesq equation:
            Sy * (h_new - h_old) / dt = div(K * (h - z_bot) * grad(h)) + R
        
        Parameters
        ----------
        h_old : jnp.ndarray
            Head at previous timestep [m]
        K : jnp.ndarray
            Hydraulic conductivity at each cell [m/s]
        Sy : jnp.ndarray
            Specific yield at each cell [-]
        recharge : jnp.ndarray
            Recharge rate at each cell [m/s]
        mesh_data : dict
            Mesh connectivity data (must be static)
        dt : float
            Time step [s]
        
        Returns
        -------
        jnp.ndarray
            Head at new timestep [m]
        
        Notes
        -----
        This function calls the C++ solver internally and uses Enzyme
        for automatic differentiation in the backward pass.
        """
        # Call C++ forward pass
        h_new_np, _ = _dgw.boussinesq_step_fwd(
            np.asarray(h_old),
            np.asarray(K),
            np.asarray(Sy),
            np.asarray(recharge),
            mesh_data,
            float(dt)
        )
        return jnp.array(h_new_np)
    
    def _boussinesq_step_fwd(h_old, K, Sy, recharge, mesh_data, dt):
        """Forward pass: compute h_new and save residuals for backward."""
        h_new_np, residuals = _dgw.boussinesq_step_fwd(
            np.asarray(h_old),
            np.asarray(K),
            np.asarray(Sy),
            np.asarray(recharge),
            mesh_data,
            float(dt)
        )
        h_new = jnp.array(h_new_np)
        # Package residuals for backward pass
        res = (h_old, h_new, K, Sy, recharge, mesh_data, dt)
        return h_new, res
    
    def _boussinesq_step_bwd(res, g):
        """Backward pass: compute gradients using Enzyme adjoint."""
        h_old, h_new, K, Sy, recharge, mesh_data, dt = res
        
        # Call C++ backward pass (Enzyme-generated)
        d_h_old, d_K, d_Sy, d_recharge, _, _ = _dgw.boussinesq_step_bwd(
            (np.asarray(h_old), np.asarray(h_new), 
             np.asarray(K), np.asarray(Sy), np.asarray(recharge), dt),
            np.asarray(g)
        )
        
        return (
            jnp.array(d_h_old),
            jnp.array(d_K),
            jnp.array(d_Sy),
            jnp.array(d_recharge),
            None,  # No gradient for mesh_data
            None   # No gradient for dt
        )
    
    boussinesq_step.defvjp(_boussinesq_step_fwd, _boussinesq_step_bwd)
    
    
    @custom_vjp
    def richards_step(
        psi_old: jnp.ndarray,
        K_sat: jnp.ndarray,
        theta_r: jnp.ndarray,
        theta_s: jnp.ndarray,
        alpha: jnp.ndarray,
        n_vg: jnp.ndarray,
        recharge: jnp.ndarray,
        mesh_data: Dict[str, Any],
        dt: float
    ) -> jnp.ndarray:
        """
        JAX-differentiable Richards equation time step.
        
        Computes one implicit time step of the 3D Richards equation:
            C(psi) * dpsi/dt = div(K(psi) * grad(psi + z)) + S
        
        Parameters
        ----------
        psi_old : jnp.ndarray
            Pressure head at previous timestep [m]
        K_sat : jnp.ndarray
            Saturated hydraulic conductivity [m/s]
        theta_r, theta_s : jnp.ndarray
            Residual and saturated water content [-]
        alpha, n_vg : jnp.ndarray
            van Genuchten parameters [1/m] and [-]
        recharge : jnp.ndarray
            Surface recharge [m/s]
        mesh_data : dict
            3D mesh connectivity
        dt : float
            Time step [s]
        
        Returns
        -------
        jnp.ndarray
            Pressure head at new timestep [m]
        """
        # Placeholder - actual implementation calls C++
        return psi_old  # Would call _dgw.richards_step_fwd
    
    def _richards_step_fwd(psi_old, K_sat, theta_r, theta_s, alpha, n_vg, 
                           recharge, mesh_data, dt):
        """Forward pass for Richards equation."""
        # Would call C++ and return (psi_new, residuals)
        psi_new = psi_old  # Placeholder
        res = (psi_old, psi_new, K_sat, theta_r, theta_s, alpha, n_vg, 
               recharge, mesh_data, dt)
        return psi_new, res
    
    def _richards_step_bwd(res, g):
        """Backward pass for Richards equation."""
        # Would call Enzyme-generated adjoint
        psi_old, psi_new, K_sat, theta_r, theta_s, alpha, n_vg, \
            recharge, mesh_data, dt = res
        n = len(psi_old)
        return (
            jnp.zeros(n), jnp.zeros(n), jnp.zeros(n), jnp.zeros(n),
            jnp.zeros(n), jnp.zeros(n), jnp.zeros(n), None, None
        )
    
    richards_step.defvjp(_richards_step_fwd, _richards_step_bwd)
    
    
    def run_simulation_jax(
        model: DGW,
        n_steps: int,
        callback: Optional[callable] = None
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Run differentiable simulation using JAX.
        
        Parameters
        ----------
        model : DGW
            Initialized model
        n_steps : int
            Number of time steps
        callback : callable, optional
            Function called each step: callback(step, state)
        
        Returns
        -------
        final_state : jnp.ndarray
            Final head/pressure field
        diagnostics : dict
            Time series of storage, mass balance, etc.
        """
        # Extract mesh data for JAX (must be static)
        mesh = model.mesh()
        mesh_data = {
            'n_cells': mesh.n_cells(),
            'neighbors': ...,  # Would extract connectivity
            'face_areas': ...,
            'cell_volumes': ...
        }
        
        # Get initial state and parameters
        h = jnp.array(model.head())
        params = model.parameters()
        K = jnp.array(params.as_2d().K)
        Sy = jnp.array(params.as_2d().Sy)
        
        dt = model.config().solver.dt_initial
        storage_history = []
        
        for step in range(n_steps):
            recharge = jnp.array(model.state().as_2d().recharge_flux)
            
            # Differentiable step
            h = boussinesq_step(h, K, Sy, recharge, mesh_data, dt)
            
            # Track storage
            storage = jnp.sum(Sy * h * jnp.array([mesh.cell_volume(i) 
                                                   for i in range(mesh.n_cells())]))
            storage_history.append(storage)
            
            if callback:
                callback(step, h)
        
        return h, {'storage': jnp.stack(storage_history)}

except ImportError:
    _HAS_JAX = False
    
    def boussinesq_step(*args, **kwargs):
        raise ImportError("JAX not available. Install with: pip install jax jaxlib")
    
    def richards_step(*args, **kwargs):
        raise ImportError("JAX not available. Install with: pip install jax jaxlib")


# =============================================================================
# NumPy Interface
# =============================================================================

class NumPyModel:
    """
    NumPy-only interface to dGW (no JAX required).
    
    Example:
        >>> model = dgw.NumPyModel.from_config("config.yaml")
        >>> model.step()
        >>> h = model.get_head()
    """
    
    def __init__(self, dgw_model: Optional[DGW] = None):
        self._model = dgw_model or DGW()
    
    @classmethod
    def from_config(cls, config_file: str) -> 'NumPyModel':
        """Create model from config file."""
        return cls(DGW.from_config(config_file))
    
    def initialize(self):
        """Initialize model."""
        self._model.initialize()
    
    def step(self, dt: Optional[float] = None) -> StepResult:
        """Advance one time step."""
        if dt is not None:
            result = self._model.step(dt)
        else:
            result = self._model.step()
        return StepResult(
            success=result.success,
            dt_actual=result.dt_actual,
            dt_next=result.dt_next,
            newton_iters=result.newton_iters,
            mass_balance_error=result.mass_balance_error
        )
    
    def run(self):
        """Run to end time."""
        self._model.run()
    
    def get_head(self) -> np.ndarray:
        """Get current head field."""
        return np.array(self._model.head())
    
    def set_recharge(self, recharge: np.ndarray):
        """Set recharge field."""
        self._model.set_recharge(recharge)
    
    def set_stream_stage(self, stage: np.ndarray):
        """Set stream stage."""
        self._model.set_stream_stage(stage)
    
    def set_pumping(self, pumping: np.ndarray):
        """Set pumping rates."""
        self._model.set_pumping(pumping)
    
    def get_water_table_depth(self) -> np.ndarray:
        """Get water table depth."""
        return np.array(self._model.get_water_table_depth())
    
    def get_stream_exchange(self) -> np.ndarray:
        """Get stream-aquifer exchange."""
        return np.array(self._model.get_stream_exchange())
    
    def total_storage(self) -> float:
        """Get total storage."""
        return self._model.total_storage()
    
    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._model.time()
    
    @property
    def mesh(self) -> Mesh:
        """Model mesh."""
        return self._model.mesh()
    
    @property
    def config(self) -> Config:
        """Model configuration."""
        return self._model.config()


# =============================================================================
# Coupling Helpers
# =============================================================================

class SUMMACoupler:
    """
    Helper for coupling dGW with SUMMA land surface model.
    
    Handles:
    - Remapping recharge from SUMMA HRUs to GW cells
    - Remapping water table depth from GW cells to HRUs
    - Time synchronization
    """
    
    def __init__(self, dgw_model: DGW, hru_to_cell_map: np.ndarray):
        self.dgw = dgw_model
        self._hru_to_cell = hru_to_cell_map
        
        # Build cell to HRU map (transpose, area-weighted)
        self._cell_to_hru = self._build_cell_to_hru_map()
    
    def _build_cell_to_hru_map(self) -> np.ndarray:
        """Build reverse mapping matrix."""
        # Area-weighted average
        from scipy import sparse
        if sparse.issparse(self._hru_to_cell):
            return self._hru_to_cell.T.tocsr()
        else:
            return self._hru_to_cell.T
    
    def send_recharge(self, hru_drainage: np.ndarray):
        """
        Send drainage from SUMMA HRUs to GW cells.
        
        Parameters
        ----------
        hru_drainage : ndarray
            Drainage at soil bottom from each HRU [m/s]
        """
        # Remap to cells
        cell_recharge = self._hru_to_cell @ hru_drainage
        self.dgw.set_recharge(cell_recharge)
    
    def receive_water_table(self) -> np.ndarray:
        """
        Receive water table depth for SUMMA HRUs.
        
        Returns
        -------
        ndarray
            Water table depth at each HRU [m]
        """
        cell_wtd = self.dgw.get_water_table_depth()
        hru_wtd = self._cell_to_hru @ cell_wtd
        return hru_wtd


class RoutingCoupler:
    """
    Helper for coupling dGW with routing model (dRoute).
    
    Handles:
    - Setting stream stage from routing
    - Getting stream-aquifer exchange for routing
    """
    
    def __init__(self, dgw_model: DGW, reach_to_cell_map: np.ndarray):
        self.dgw = dgw_model
        self._reach_to_cell = reach_to_cell_map
    
    def send_stage(self, reach_stage: np.ndarray):
        """
        Send stream stage from routing to GW.
        
        Parameters
        ----------
        reach_stage : ndarray
            Stage at each routing reach [m]
        """
        # Map reaches to cells
        cell_stage = np.zeros(self.dgw.mesh().n_cells())
        for reach, cell in enumerate(self._reach_to_cell):
            if cell >= 0:
                cell_stage[cell] = reach_stage[reach]
        self.dgw.set_stream_stage(cell_stage)
    
    def receive_exchange(self) -> np.ndarray:
        """
        Receive stream-aquifer exchange for routing.

        Returns
        -------
        ndarray
            Exchange at each reach [m³/s], positive = gaining
        """
        cell_exchange = np.array(self.dgw.get_stream_exchange())
        # Remap from cell exchange to reach exchange
        n_reaches = len(self._reach_to_cell)
        reach_exchange = np.zeros(n_reaches)
        for reach, cell in enumerate(self._reach_to_cell):
            if cell >= 0:
                reach_exchange[reach] = cell_exchange[cell]
        return reach_exchange
