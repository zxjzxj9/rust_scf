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

## Physics Background

The simulations use:
- **Lennard-Jones potential**: V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
- **Nosé-Hoover thermostat**: Maintains target temperature
- **Verlet integration**: Time evolution of positions/velocities
- **Periodic boundary conditions**: Simulate bulk behavior
- **Reduced units**: Natural units where ε = σ = m = 1










