# dGW - Differentiable Groundwater Model

**State-of-the-art groundwater modeling with automatic differentiation**

dGW is a modern C++ groundwater model featuring:
- 🧮 **SUMMA-style physics optionality** - Choose governing equations at runtime
- 🔬 **Enzyme automatic differentiation** - Exact gradients for optimization
- 🔗 **BMI compliance** - Direct integration with NextGen framework
- 🌐 **Unstructured mesh support** - River-conforming Voronoi cells
- 🐍 **Python/JAX bindings** - Seamless ML workflow integration

## Physics Options

| Governing Equation | Description | Use Case |
|-------------------|-------------|----------|
| `LinearDiffusion` | Constant transmissivity | Quick regional estimates |
| `Boussinesq` | Nonlinear unconfined | Standard water table problems |
| `Confined` | Linear confined flow | Deep aquifer systems |
| `TwoLayer` | Coupled unconfined + confined | Global-scale with GRACE validation |
| `MultiLayer` | N-layer with leakage | Complex stratigraphy |
| `Richards3D` | Full 3D variably-saturated | Vadose zone, infiltration, capillary rise |

### Supporting Options

**Transmissivity Methods:**
- `Standard` - Direct T = K × saturated_thickness
- `Smoothed` - Stable near dry cells
- `Upstream` - Numerical stability for steep gradients

**Water Retention (Richards):**
- `VanGenuchten` - Most common, well-tested
- `BrooksCorey` - Alternative parameterization
- `ClappHornberger` - Simplified for land surface coupling

**Stream-Aquifer Exchange:**
- `Conductance` - Simple Q = C(h_stream - h_gw)
- `ConductanceClogging` - With clogging layer
- `KinematicLosing` - Disconnected streams (arid regions)

## Quick Start

### C++ Usage

```cpp
#include <dgw/dgw.hpp>

int main() {
    // Create model via BMI interface
    dgw::DGW model;
    model.Initialize("config.yaml");
    
    // Coupling with external models
    model.set_recharge(recharge_from_summa);
    model.set_river_stage(stage_from_droute);
    
    // Time stepping
    while (model.GetCurrentTime() < model.GetEndTime()) {
        model.Update();
        
        // Get outputs for coupling
        auto wt_depth = model.get_water_table_depth();
        auto river_flux = model.get_stream_exchange();
    }
    
    model.Finalize();
    return 0;
}
```

### Python Usage

```python
import dgw_core as dgw
import numpy as np

# Initialize
model = dgw.DGW()
model.initialize("config.yaml")

# Run simulation
while model.get_current_time() < model.get_end_time():
    model.set_recharge(recharge_array)
    model.update()

# Get results
head = model.get_value("head")
wt_depth = model.get_water_table_depth()
```

### JAX Integration for Calibration

```python
from dgw.jax_wrapper import DGWModel, calibrate_model, DGWParams, DGWForcings

# Setup
model = DGWModel.from_config("config.yaml")

# Initial parameters
params = DGWParams(
    K=np.full(model.n_cells, 1e-4),
    Sy=np.full(model.n_cells, 0.2)
)

forcings = DGWForcings(
    recharge=recharge_data,
    river_stage=stage_data
)

# Calibrate with gradient descent
optimal_params, loss_history = calibrate_model(
    model, observed_heads, forcings, params,
    max_iterations=100,
    learning_rate=1e-3
)
```

## Configuration (YAML)

```yaml
physics:
  governing_equation: Boussinesq
  transmissivity: Smoothed
  stream_exchange: ConductanceClogging
  vadose: ExponentialLag
  
mesh:
  file: domain_mesh.nc
  type: VORONOI

solver:
  method: Newton
  linear_solver: EigenCholesky
  max_iterations: 50
  tolerance: 1e-6
  
  time_stepping:
    method: Adaptive
    dt_min: 60.0
    dt_max: 86400.0

output:
  variables: [head, water_table_depth, stream_exchange]
  frequency: daily
  format: netcdf
```

## Building

### Requirements

- C++20 compiler (GCC 11+, Clang 14+)
- CMake 3.20+
- Eigen3
- Optional: Enzyme (for AD), PETSc (for large-scale), pybind11 (for Python)

### Build Steps

```bash
mkdir build && cd build
cmake .. -DDGW_ENABLE_ENZYME=ON -DDGW_ENABLE_PYTHON=ON
make -j8
make install
```

### With Enzyme AD

```bash
# Install Enzyme first
git clone https://github.com/EnzymeAD/Enzyme
cd Enzyme/enzyme && mkdir build && cd build
cmake .. -DLLVM_DIR=/path/to/llvm
make && make install

# Then build dGW with Enzyme
cmake .. -DDGW_ENABLE_ENZYME=ON -DEnzyme_DIR=/path/to/enzyme
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ Python   │  │  JAX     │  │  C++     │  │   BMI/       │ │
│  │ Bindings │  │ Wrapper  │  │  Direct  │  │   NextGen    │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘ │
└───────┼─────────────┼─────────────┼───────────────┼─────────┘
        │             │             │               │
        └─────────────┴──────┬──────┴───────────────┘
                             │
┌────────────────────────────┼────────────────────────────────┐
│                    C++ Core (libdgw)                         │
│  ┌─────────────────────────┴───────────────────────────┐    │
│  │                 Physics Modules                       │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐  │    │
│  │  │ Linear  │ │Boussinesq│ │Two-Layer│ │ Richards  │  │    │
│  │  │Diffusion│ │         │ │         │ │    3D     │  │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └───────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                             │                                │
│  ┌─────────────────────────┴───────────────────────────┐    │
│  │              Solver Infrastructure                    │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐  │    │
│  │  │ Newton  │ │ Linear  │ │  Time   │ │  Mesh     │  │    │
│  │  │ Solver  │ │ Solvers │ │ Stepper │ │ Handling  │  │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └───────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                             │                                │
│  ┌─────────────────────────┴───────────────────────────┐    │
│  │              Enzyme AD Layer                          │    │
│  │      (Compile-time automatic adjoint generation)      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Coupling with SYMFLUENCE

dGW integrates with cSUMMA (land surface) and dRoute (routing):

```cpp
void coupled_step(SUMMA& summa, DGW& gw, dRoute& routing, double dt) {
    // 1. Land surface provides recharge
    summa.step(dt);
    gw.set_recharge(summa.get_drainage());
    
    // 2. Groundwater with river coupling
    gw.set_river_stage(routing.get_stage());
    gw.step(dt);
    
    // 3. Route surface + groundwater fluxes
    routing.step(summa.get_runoff() + gw.get_stream_exchange(), dt);
    
    // 4. Feedback: water table affects land surface
    summa.set_water_table_depth(gw.get_water_table_depth());
}
```

## Gradient-Based Calibration

dGW computes exact gradients via Enzyme AD:

```
Forward:  F(θ, h_old) → h_new    # Physics simulation
Adjoint:  λ = (∂L/∂h_new)        # From loss function
          ∂L/∂θ = λᵀ (∂F/∂θ)    # Parameter sensitivity
```

This enables:
- Efficient calibration against GRACE gravity observations
- Neural network hybrid models
- Uncertainty quantification via Hamiltonian Monte Carlo

## Examples

See `examples/` for:
- `theis_well.cpp` - Pumping test with analytical comparison
- `richards_column.cpp` - Infiltration with van Genuchten
- `coupled_summa.cpp` - Land surface coupling demo
- `jax_calibration.py` - Gradient-based parameter optimization

## Testing

```bash
cd build
ctest --output-on-failure
```

Tests include:
- Analytical solution comparisons (Theis, Dupuit, Philip)
- Gradient verification (Enzyme vs finite differences)
- Mass balance checks
- BMI compliance tests

## References

- Clark et al. (2015) - SUMMA approach to physics optionality
- de Graaf et al. (2015) - Two-layer global groundwater
- Moses & Churavy (2020) - Enzyme AD
- Hutton et al. (2020) - BMI 2.0 specification

## License

BSD 3-Clause

## Contributing

See CONTRIBUTING.md for guidelines.

## Citation

```bibtex
@software{dgw2025,
  title = {dGW: Differentiable Groundwater Model},
  author = {SYMFLUENCE Team},
  year = {2025},
  url = {https://github.com/SYMFLUENCE/dgw}
}
```
