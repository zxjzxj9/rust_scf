# Molecular Dynamics (MD) Examples

This directory contains examples demonstrating classical molecular dynamics simulations.

## Examples

### 1. Argon Melting Simulation (`argon_melting.rs`)

A comprehensive molecular dynamics simulation showing argon atoms undergoing a solid-to-liquid phase transition.

**Features:**
- Realistic physical parameters for argon (ε = 120 K, σ = 3.4 Å, mass = 39.948 u)
- FCC crystal lattice initialization (256 atoms default, configurable)
- Temperature ramping from 60K to 180K
- Nosé-Hoover thermostat for temperature control
- Melting detection via diffusion coefficient analysis
- Physical unit conversions and detailed output

**Run:**
```bash
cd md
cargo run --example argon_melting --release
```

**Runtime:** ~2-5 minutes (25,000 steps)

**Output:**
- Temperature (K and reduced units)
- Kinetic and potential energies
- Diffusion coefficient (phase indicator)
- Total energy conservation check

**Melting Detection:**
- D < 0.001: Solid phase
- 0.001 < D < 0.01: Transition phase
- D > 0.01: Liquid phase

**Customization:**
```rust
// Change system size
let n_cells = 3;  // 3×3×3 FCC = 108 atoms

// Adjust temperature range
let initial_temp = 60.0;  // K
let final_temp = 200.0;   // K

// Modify heating schedule
let heating_start_step = 5000;
let heating_end_step = 20000;
```

---

### 2. Pressure Calculation Demo (`pressure_calculation_demo.rs`)

Demonstrates the calculation of pressure in molecular dynamics simulations, including both kinetic and virial contributions.

**Features:**
- Clear separation of kinetic and virial pressure terms
- Comparison with ideal gas law
- Educational output showing pressure components
- Useful for understanding pressure calculation in MD

**Run:**
```bash
cd md
cargo run --example pressure_calculation_demo --release
```

**Key Concepts:**

For **non-interacting** systems (ideal gas):
```
P = NkT/V
```

For **Lennard-Jones** systems:
```
P = (2K + W) / (3V)
```

where:
- K = kinetic energy
- W = virial = Σ_ij r_ij · F_ij
- V = volume

---

### 3. Langevin LJ Cluster (`langevin_lj_cluster.rs`)

Simulates a 38-atom Lennard-Jones cluster in reduced units while coupled to a Langevin heat bath. Useful for studying finite-size energy landscapes, exploring structural transitions, or demonstrating stochastic thermostats without periodic bulk effects.

**Features:**
- Custom Langevin integrator with friction + stochastic kicks
- Compact cluster builder (radius-sorted cubic lattice)
- Energy and radius monitoring every 250 steps
- Optional on-the-fly thermostat retargeting
- Re-centering to keep the aggregate away from box edges

**Run:**
```bash
cd md
cargo run --example langevin_lj_cluster --release
```

**Output Highlights:**
- Instantaneous reduced temperature `T*`
- Kinetic and potential energy per particle
- Average and maximum radial extent of the cluster
- Running averages over the reported frames

**Customization Hooks:**
```rust
let n_atoms = 38;          // change cluster size
let target_temp = 0.6;     // Langevin thermostat set point
let gamma = 1.5;           // friction (higher => stronger coupling)
let total_steps = 25_000;  // simulation length
let sample_interval = 250; // reporting cadence
```

---

## Physics Background

### Lennard-Jones Potential

The simulations use the Lennard-Jones potential to model van der Waals interactions:

```
V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
```

where:
- ε = well depth (energy scale)
- σ = collision diameter (length scale)

### Nosé-Hoover Thermostat

Controls temperature by coupling the system to a heat bath with dynamics:

```
dξ/dt = (T - T_target) / Q
```

where Q is the thermostat coupling parameter.

### Reduced Units

For computational efficiency, simulations often use reduced units where:
- ε = 1 (energy)
- σ = 1 (length)
- m = 1 (mass)
- k_B = 1 (Boltzmann constant)

---

## Performance Tips

1. **Always use release mode:**
   ```bash
   cargo run --example <name> --release
   ```
   ~10× faster than debug mode

2. **For large systems:** Consider reducing output frequency

3. **Parallelization:** Force calculations are parallelized using Rayon

---

## References

1. Allen, M. P. & Tildesley, D. J. (2017). *Computer Simulation of Liquids* (2nd ed.). Oxford University Press.
2. Nosé, S. (1984). "A unified formulation of the constant temperature molecular dynamics methods." *J. Chem. Phys.* **81**, 511.
3. Hoover, W. G. (1985). "Canonical dynamics: Equilibrium phase-space distributions." *Phys. Rev. A* **31**, 1695.

