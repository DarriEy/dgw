# dGW

Differentiable groundwater model. C++20. Unstructured meshes. Enzyme AD for gradients. BMI-compliant.

## Physics

Pick at runtime:

| Equation | What it does |
|---|---|
| `LinearDiffusion` | Constant transmissivity |
| `Boussinesq` | Nonlinear unconfined flow |
| `Confined` | Linear confined flow |
| `TwoLayer` | Coupled unconfined + confined |
| `Richards3D` | Full 3D variably-saturated |

Water retention: Van Genuchten, Brooks-Corey, Clapp-Hornberger.
Stream-aquifer exchange: conductance, clogging layer, kinematic losing.

## Build

Needs C++20, CMake 3.20+, Eigen3. Optional: Enzyme, PETSc, pybind11.

```bash
mkdir build && cd build
cmake ..
make -j8
```

For AD gradients, add `-DDGW_ENABLE_ENZYME=ON -DEnzyme_DIR=/path/to/enzyme`.

## Usage

```cpp
#include <dgw/dgw.hpp>

dgw::DGW model;
model.set_mesh(mesh);
model.set_physics(dgw::GoverningEquation::Boussinesq);
model.set_parameters(params);
model.initialize();

while (model.time() < end_time) {
    model.set_recharge(recharge);
    model.step();
}

auto gradients = model.compute_gradients(loss_gradient);
```

BMI works too:

```cpp
dgw::DGW_BMI bmi;
bmi.Initialize("config.yaml");
bmi.Update();
bmi.Finalize();
```

## Config

```yaml
physics:
  governing_equation: Boussinesq
  transmissivity: Smoothed
  stream_exchange: ConductanceClogging

mesh:
  file: domain.nc
  type: VORONOI

solver:
  method: Newton
  tolerance: 1e-6
  time_stepping:
    method: Adaptive
    dt_min: 60.0
    dt_max: 86400.0
```

## Tests

```bash
cd build && ctest --output-on-failure
```

## References

- Clark et al. (2015) -- SUMMA physics optionality
- de Graaf et al. (2015) -- Two-layer global groundwater
- Moses & Churavy (2020) -- Enzyme AD
- Hutton et al. (2020) -- BMI 2.0

## License

BSD 3-Clause
