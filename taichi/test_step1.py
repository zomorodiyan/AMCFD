"""
Test Step 1: Types & Config (Taichi Implementation)

Verifies that YAML input parsing works correctly and all parameters are extracted.
Uses the input_param.yaml file.
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

# Initialize Taichi before importing data_structures
import taichi as ti
ti.init(arch=ti.cpu)  # Use CPU for testing, can switch to ti.gpu for GPU

from data_structures import PhysicsParams, SimulationParams, LaserParams, GridParams, State
from param import parse_input, load_toolpath, _read_yaml, _temp_to_enthalpy_solid

# Path to YAML input file
INPUT_FILE = Path(__file__).parent / "inputfile" / "input_param.yaml"


def test_enthalpy_conversion():
    """Test enthalpy calculation."""
    # Test with typical steel values
    acpa = 0.0
    acpb = 500.0
    T = 1000.0
    
    H = _temp_to_enthalpy_solid(T, acpa, acpb)
    expected = 0.5 * acpa * T * T + acpb * T
    
    assert abs(H - expected) < 1e-10
    assert abs(H - 500000.0) < 1e-10  # 500 * 1000
    
    # Test with quadratic term
    acpa = 0.5
    H = _temp_to_enthalpy_solid(T, acpa, acpb)
    expected = 0.5 * 0.5 * 1000.0 * 1000.0 + 500.0 * 1000.0
    assert abs(H - expected) < 1e-10
    
    print("✓ Enthalpy conversion tests passed")


def test_physics_params():
    """Test PhysicsParams creation with defaults."""
    # Create with minimal parameters
    params = {
        'tsolid': 1563.0,
        'tliquid': 1623.0,
        'temppreheat': 300.0,
        'dens': 7800.0,
        'hlatent': 2.7e5,
    }
    
    from param import _create_physics_params
    physics = _create_physics_params(params)
    
    assert physics.tsolid == 1563.0
    assert physics.tliquid == 1623.0
    assert physics.rho == 7800.0
    assert physics.hlatent == 2.7e5
    assert physics.sigma == 5.67e-8  # Stefan-Boltzmann constant
    
    # Check computed enthalpy values
    assert physics.hsmelt > 0
    assert physics.hlcal > physics.hsmelt
    assert physics.hlcal == physics.hsmelt + physics.hlatent
    
    print("✓ PhysicsParams tests passed")


def test_simulation_params():
    """Test SimulationParams creation."""
    params = {
        'delt': 1.0e-6,
        'timax': 1.0e-3,
        'nx': 100,
        'ny': 100,
        'nz': 50,
    }
    
    from param import _create_simulation_params
    sim = _create_simulation_params(params)
    
    assert sim.delt == 1.0e-6
    assert sim.timax == 1.0e-3
    assert sim.ni == 100
    assert sim.nj == 100
    assert sim.nk == 50
    
    print("✓ SimulationParams tests passed")


def test_laser_params():
    """Test LaserParams creation and peak flux calculation."""
    import math
    
    params = {
        'alaspowvol': 200.0,
        'sourcerad': 50.0e-6,
        'alasetavol': 0.35,
    }
    
    from param import _create_laser_params
    laser = _create_laser_params(params)
    
    assert laser.power == 200.0
    assert laser.radius == 50.0e-6
    assert laser.absorptivity == 0.35
    
    # Verify peak flux calculation
    expected_peak = 2.0 * 200.0 * 0.35 / (math.pi * (50.0e-6)**2)
    assert abs(laser.peak_flux - expected_peak) / expected_peak < 1e-10
    
    print("✓ LaserParams tests passed")


def test_yaml_input_file():
    """Test parsing the YAML input file."""
    
    # Verify input file exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    # Parse the actual file
    physics, simulation, laser, output = parse_input(str(INPUT_FILE))
    
    # Verify grid dimensions from geometry section
    # From YAML: cv_per_zone: x=200, y=200, z=[10,20] -> total 200, 200, 30
    assert simulation.ni == 200, f"Expected ni=200, got {simulation.ni}"
    assert simulation.nj == 200, f"Expected nj=200, got {simulation.nj}"
    assert simulation.nk == 30, f"Expected nk=30, got {simulation.nk}"
    
    # Verify domain lengths
    # zone_length_m: x=4e-3, y=4e-3, z=[0.5e-3, 0.2e-3] -> 4mm, 4mm, 0.7mm
    assert abs(simulation.xlen - 4.0e-3) < 1e-10, f"Expected xlen=4mm, got {simulation.xlen}"
    assert abs(simulation.ylen - 4.0e-3) < 1e-10, f"Expected ylen=4mm, got {simulation.ylen}"
    assert abs(simulation.zlen - 0.7e-3) < 1e-10, f"Expected zlen=0.7mm, got {simulation.zlen}"
    
    # Verify material properties
    assert physics.rho == 8440.0, f"Expected rho=8440, got {physics.rho}"
    assert physics.tsolid == 1563.0, f"Expected tsolid=1563, got {physics.tsolid}"
    assert physics.tliquid == 1623.0, f"Expected tliquid=1623, got {physics.tliquid}"
    assert physics.hsmelt == 861e3, f"Expected hsmelt=861e3, got {physics.hsmelt}"
    
    # Verify laser parameters
    assert laser.power == 300.0, f"Expected power=300, got {laser.power}"
    assert abs(laser.absorptivity - 0.3512) < 1e-10, f"Expected absorptivity=0.3512, got {laser.absorptivity}"
    
    # Verify numerical parameters
    assert abs(simulation.delt - 2e-5) < 1e-10, f"Expected delt=2e-5, got {simulation.delt}"
    assert simulation.urf_vel == 0.7, f"Expected urfu=0.7, got {simulation.urf_vel}"
    assert simulation.max_iter == 50, f"Expected maxit=50, got {simulation.max_iter}"
    
    print("✓ YAML input file parsing tests passed")


def test_geometry_parsing():
    """Test parsing of geometry section from YAML."""
    from param import _read_yaml
    
    if not INPUT_FILE.exists():
        print("⚠ Skipping geometry test - input file not found")
        return
    
    params = _read_yaml(str(INPUT_FILE))
    
    # Check zone counts
    assert params['nzx'] == 1, f"Expected nzx=1, got {params['nzx']}"
    assert params['nzy'] == 1, f"Expected nzy=1, got {params['nzy']}"
    assert params['nzz'] == 2, f"Expected nzz=2, got {params['nzz']}"
    
    # Check zone lengths
    assert params['xzone'] == [4.0e-3], f"Expected xzone=[4e-3], got {params['xzone']}"
    assert params['yzone'] == [4.0e-3], f"Expected yzone=[4e-3], got {params['yzone']}"
    assert params['zzone'] == [0.5e-3, 0.2e-3], f"Expected zzone=[0.5e-3, 0.2e-3], got {params['zzone']}"
    
    # Check control volumes per zone
    assert params['ncvx'] == [200], f"Expected ncvx=[200], got {params['ncvx']}"
    assert params['ncvy'] == [200], f"Expected ncvy=[200], got {params['ncvy']}"
    assert params['ncvz'] == [10, 20], f"Expected ncvz=[10,20], got {params['ncvz']}"
    
    # Check computed totals
    assert params['nx'] == 200, f"Expected nx=200, got {params['nx']}"
    assert params['ny'] == 200, f"Expected ny=200, got {params['ny']}"
    assert params['nz'] == 30, f"Expected nz=30, got {params['nz']}"
    
    print("✓ Geometry section parsing tests passed")


def test_toolpath_loading():
    """Test loading toolpath from .crs file."""
    toolpath_file = Path(__file__).parent / "inputfile" / "B26-1.crs"
    
    if not toolpath_file.exists():
        print("⚠ Skipping toolpath test - B26-1.crs not found")
        return
    
    toolpath = load_toolpath(str(toolpath_file))
    
    # Check that data was loaded
    assert toolpath.n_segments == 6, f"Expected 6 segments, got {toolpath.n_segments}"
    
    # Check first segment (all zeros)
    assert float(toolpath.time[0]) == 0.0, f"Expected time[0]=0, got {toolpath.time[0]}"
    assert float(toolpath.x[0]) == 0.0, f"Expected x[0]=0, got {toolpath.x[0]}"
    assert int(toolpath.laser_on[0]) == 0, f"Expected laser_on[0]=0, got {toolpath.laser_on[0]}"
    
    # Check segment with laser on (index 2)
    assert int(toolpath.laser_on[2]) == 1, f"Expected laser_on[2]=1, got {toolpath.laser_on[2]}"
    
    # Check that x, y, z coordinates are reasonable (in mm range)
    assert float(toolpath.x[2]) > 0, f"Expected positive x[2], got {toolpath.x[2]}"
    
    print("✓ Toolpath loading tests passed")


def test_taichi_field_creation():
    """Test Taichi-specific field creation."""
    # Test GridParams creation
    ni, nj, nk = 10, 10, 5
    grid = GridParams(ni, nj, nk)
    
    assert grid.ni == ni
    assert grid.nj == nj
    assert grid.nk == nk
    
    # Verify field shapes
    assert grid.x.shape == (ni,)
    assert grid.y.shape == (nj,)
    assert grid.z.shape == (nk,)
    assert grid.vol.shape == (ni, nj, nk)
    
    # Test State creation
    state = State(ni, nj, nk)
    assert state.uVel.shape == (ni, nj, nk)
    assert state.temp.shape == (ni, nj, nk)
    assert state.enthalpy.shape == (ni, nj, nk)
    
    print("✓ Taichi field creation tests passed")


def test_taichi_field_operations():
    """Test basic Taichi field operations."""
    ni, nj, nk = 5, 5, 3
    
    # Create a simple field and fill it
    state = State(ni, nj, nk)
    
    # Fill with preheat temperature
    @ti.kernel
    def fill_temp(temp: ti.template(), value: ti.f64):
        for i, j, k in temp:
            temp[i, j, k] = value
    
    fill_temp(state.temp, 300.0)
    
    # Verify values
    temp_np = state.temp.to_numpy()
    assert temp_np.shape == (ni, nj, nk)
    assert abs(temp_np[0, 0, 0] - 300.0) < 1e-10
    assert abs(temp_np[ni-1, nj-1, nk-1] - 300.0) < 1e-10
    
    print("✓ Taichi field operations tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Step 1: Types & Config (Taichi Implementation)")
    print("=" * 60 + "\n")
    
    test_enthalpy_conversion()
    test_physics_params()
    test_simulation_params()
    test_laser_params()
    test_geometry_parsing()
    test_yaml_input_file()
    test_toolpath_loading()
    test_taichi_field_creation()
    test_taichi_field_operations()
    
    print("\n" + "=" * 60)
    print("All Step 1 tests passed! ✓")
    print(f"Input file used: {INPUT_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
