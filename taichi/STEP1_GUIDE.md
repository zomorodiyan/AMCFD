# Step 1: Types & Config (Taichi Implementation)

## Overview

This step implements the foundational data structures and input/output handling for the AM-CFD Taichi implementation. The design leverages Taichi's `ti.field()` for GPU-compatible arrays while using Python dataclasses for parameter containers.

## Input File Format

The Taichi implementation uses **YAML format** for input files, providing a clean and readable configuration:

```yaml
# Geometry
geometry:
  x:
    zones: 1
    zone_length_m: 4.0e-3
    cv_per_zone: 200
    cv_boundary_exponent: 1
  y:
    zones: 1
    zone_length_m: 4.0e-3
    cv_per_zone: 200
    cv_boundary_exponent: 1
  z:
    zones: 2
    zone_length_m: [0.5e-3, 0.2e-3]
    cv_per_zone: [10, 20]
    cv_boundary_exponent: [-1.5, 1]

# Material properties
material_properties:
  dens: 8440
  tsolid: 1563
  tliquid: 1623
  # ... more properties

# Numerical relaxation
numerical_relax:
  maxit: 50
  delt: 2e-5
  urfu: 0.7
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

- `parse_input()`: Parse YAML input files
- `_read_yaml()`: Read and flatten YAML structure
- `_parse_geometry_yaml()`: Parse geometry section
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

# Parse YAML input file
physics, simulation, laser, output = parse_input("inputfile/input_param.yaml")

# Create Taichi fields
grid = GridParams(simulation.ni, simulation.nj, simulation.nk)
state = State(simulation.ni, simulation.nj, simulation.nk)

# Access in kernels
@ti.kernel
def initialize_temp(state: ti.template(), tpreheat: ti.f64):
    for i, j, k in state.temp:
        state.temp[i, j, k] = tpreheat
```

## Testing

Run tests with:

```bash
cd taichi
source ../taichi_env/bin/activate
python test_step1.py
```

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Required packages:
- `taichi>=1.7.0` - GPU-accelerated computing
- `numpy>=1.24.0` - Numerical arrays
- `pyyaml>=6.0` - YAML parsing

## YAML vs Fortran Namelist

| Feature | YAML | Fortran Namelist |
|---------|------|------------------|
| Format | Hierarchical, indented | Flat with &sections |
| Arrays | Native `[a, b, c]` | Space-separated |
| Comments | `#` | `!` |
| Readability | High | Medium |
| Python parsing | `yaml.safe_load()` | Custom parser |

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
