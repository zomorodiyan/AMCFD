# Step 1: Types & Config (Taichi Implementation)

## Overview

This step implements the foundational data structures and input/output handling for the AM-CFD Taichi implementation. The design leverages Taichi's `ti.field()` for GPU-compatible arrays while using Python dataclasses for parameter containers.

## Key Differences from JAX Version

### Data Structures

| JAX | Taichi | Reason |
|-----|--------|--------|
| `NamedTuple` | `@dataclass` | Mutable fields for Taichi compatibility |
| `jnp.Array` | `ti.field()` | GPU memory management |
| Immutable state | `@ti.data_oriented` classes | Taichi's kernel access pattern |

### Field Types

```python
# JAX uses JAX numpy arrays
uVel: Array  # shape (ni, nj, nk)

# Taichi uses ti.field()
self.uVel = ti.field(dtype=ti.f64, shape=(ni, nj, nk))
```

## Files

### `data_structures.py`

Contains all type definitions:

- **Parameter containers** (dataclasses):
  - `PhysicsParams`: Material properties
  - `SimulationParams`: Numerical parameters  
  - `LaserParams`: Laser configuration
  - `OutputConfig`: Output settings
  - `ToolPath`: Toolpath data

- **Taichi field containers** (`@ti.data_oriented` classes):
  - `GridParams`: Computational grid geometry
  - `State`: Primary flow field variables
  - `StatePrev`: Previous timestep values
  - `MaterialProps`: Spatially-varying properties
  - `DiscretCoeffs`: FVM discretization coefficients
  - `LaserState`: Laser position and heat flux

### `param.py`

Input/output handling:

- `parse_input()`: Parse Fortran namelist-style input files
- `load_toolpath()`: Load .crs toolpath files
- `write_output_header()`: Initialize output files
- `write_output_line()`: Append timestep data
- `write_tecplot()`: Write Tecplot visualization files

## Usage

```python
import taichi as ti
ti.init(arch=ti.gpu)  # or ti.cpu

from data_structures import PhysicsParams, SimulationParams, GridParams, State
from param import parse_input

# Parse input file
physics, simulation, laser, output = parse_input("inputfile/input_param.txt")

# Create Taichi fields
grid = GridParams(simulation.ni, simulation.nj, simulation.nk)
state = State(simulation.ni, simulation.nj, simulation.nk)

# Access in kernels
@ti.kernel
def initialize_temp(state: ti.template(), physics: ti.template()):
    for i, j, k in state.temp:
        state.temp[i, j, k] = physics.tpreheat
```

## Testing

Run tests with:

```bash
cd taichi
python test_step1.py
```

## Taichi-Specific Considerations

### Initialization

Taichi must be initialized before creating any fields:

```python
import taichi as ti
ti.init(arch=ti.gpu)  # CUDA GPU
# ti.init(arch=ti.cpu)  # CPU fallback
# ti.init(arch=ti.vulkan)  # Vulkan GPU
```

### Field Access in Kernels

Taichi fields must be accessed inside `@ti.kernel` or `@ti.func`:

```python
@ti.kernel
def compute_something(field: ti.template()):
    for i, j, k in field:
        # Loop automatically parallelized on GPU
        field[i, j, k] = ...
```

### Data Transfer

To move data between CPU and GPU:

```python
# GPU -> CPU
numpy_array = field.to_numpy()

# CPU -> GPU
field.from_numpy(numpy_array)
```

## Performance Notes

1. **Double precision**: Using `ti.f64` for accuracy in CFD
2. **Field layout**: Default struct-of-arrays (SoA) for better memory coalescing
3. **Kernel fusion**: Combine operations in single kernels where possible
