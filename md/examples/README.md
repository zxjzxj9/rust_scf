# Molecular Dynamics Examples

This directory contains examples demonstrating molecular dynamics simulations using the Lennard-Jones potential.

## Examples

### 1. Argon Melting Simulation (`argon_melting.rs`)

A comprehensive molecular dynamics simulation showing argon atoms undergoing a solid-to-liquid phase transition.

**Features:**
- Realistic physical parameters for argon (ε = 120 K, σ = 3.4 Å, mass = 39.948 u)
- FCC crystal lattice initialization
- Temperature ramping from 60K to 180K
- Nosé-Hoover thermostat for temperature control
- Melting detection via diffusion coefficient analysis
- Physical unit conversions and detailed output

**Run:**
```bash
cargo run --example argon_melting
```

**Note:** This is a long simulation (25,000 steps) - expect ~2-5 minutes runtime.

### 2. Argon Melting Demo (`argon_melting_demo.rs`)

A shortened version of the argon melting simulation for quick demonstration.

**Features:**
- Same physics as the full simulation
- Reduced system size (108 atoms vs 256)
- Shorter runtime (2,500 steps vs 25,000)
- More frequent output for visualization

**Run:**
```bash
cargo run --example argon_melting_demo
```

**Runtime:** ~10-30 seconds

## Understanding the Output

The simulations produce tabulated output showing:

- **Step**: Simulation step number
- **T (K)**: Temperature in Kelvin
- **T_red**: Temperature in reduced units (T* = kT/ε)
- **KE_red**: Kinetic energy in reduced units
- **PE_red**: Potential energy in reduced units  
- **Total_E**: Total energy (should be approximately conserved)
- **Diff(σ²/τ)**: Diffusion coefficient in reduced units

### Melting Detection

The simulations use the diffusion coefficient to detect phase transitions:
- **D < 0.001**: Solid phase (low mobility)
- **0.001 < D < 0.01**: Intermediate/transition phase
- **D > 0.01**: Liquid phase (high mobility)

### Physical Context

For argon:
- Melting point: ~84 K (experimental)
- In reduced units: T* ≈ 0.8-1.0
- The simulations heat from 60K → 180K to observe the transition

## Customization

You can modify the examples to:
- Change the system size (`n_cells` parameter)
- Adjust temperature range and ramping speed
- Use different LJ parameters for other noble gases
- Add analysis functions (radial distribution, structure factor, etc.)

### 3. NPT Ensemble Simulations (Barostat + Thermostat)

A collection of examples demonstrating constant pressure and temperature (NPT) molecular dynamics:

**Available NPT Examples:**
- `quick_lj_npt.rs` - Quick testing (8 atoms, 30s runtime)
- `lj_cluster_npt.rs` - Full cluster simulation with analysis (32 atoms, 3min)
- `single_atom_npt.rs` - Educational single-atom ideal gas
- `multi_atom_npt.rs` - Multi-atom LJ system with structure analysis

**Key Features:**
- Nosé-Hoover thermostat (temperature control)
- Parrinello-Rahman barostat (pressure control)
- Proper virial-based pressure calculation
- Volume fluctuations at constant pressure
- Phase transition studies (melting, evaporation)
- Coordination number and structural analysis

**Quick Start:**
```bash
# Quick test (30 seconds)
cargo run --example quick_lj_npt --release

# Full simulation with analysis (3 minutes)
cargo run --example lj_cluster_npt --release

# Educational single-atom
cargo run --example single_atom_npt --release
```

**Documentation:** See `NPT_EXAMPLES_README.md` and `LJ_CLUSTER_NPT_GUIDE.md` for detailed guides.

**NPT Ensemble Physics:**

In NPT simulations:
- **N** (particles): Fixed
- **P** (pressure): Controlled by barostat
- **T** (temperature): Controlled by thermostat
- **V** (volume): Fluctuates to maintain pressure

Pressure calculation includes virial:
```
P = (2K + W) / (3V)
```
where K = kinetic energy, W = virial from forces

---

## Physics Background

The simulations use:
- **Lennard-Jones potential**: V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
- **Nosé-Hoover thermostat**: Maintains target temperature
- **Parrinello-Rahman barostat**: Maintains target pressure (NPT only)
- **Verlet integration**: Time evolution of positions/velocities
- **Periodic boundary conditions**: Simulate bulk behavior
- **Reduced units**: Natural units where ε = σ = m = 1










