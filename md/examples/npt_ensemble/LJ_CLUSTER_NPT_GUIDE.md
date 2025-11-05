# LJ Cluster NPT Simulation Guide

## Overview

This guide explains how to use the Lennard-Jones cluster NPT (constant Number, Pressure, Temperature) simulation code. The implementation uses:

- **Nosé-Hoover thermostat** for temperature control
- **Parrinello-Rahman barostat** for pressure control  
- **Proper virial-based pressure calculation** for accurate LJ interactions

## Physical Background

### NPT Ensemble

In the NPT ensemble, the system maintains:
- **N**: Number of particles (constant)
- **P**: Pressure (constant, controlled by barostat)
- **T**: Temperature (constant, controlled by thermostat)

The **volume** is allowed to fluctuate to maintain constant pressure.

### Pressure Calculation

For Lennard-Jones systems, pressure has two contributions:

```
P = (2K + W) / (3V)
```

where:
- `K` = kinetic energy (from thermal motion)
- `W` = virial (from inter-atomic forces)
- `V` = volume

The virial is computed as:
```
W = -1/3 * Σ_ij r_ij · F_ij
```

This is **crucial** for accurate NPT simulations of interacting systems!

### Lennard-Jones Potential

The LJ potential describes van der Waals interactions:

```
V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
```

where:
- `ε` = energy scale (well depth)
- `σ` = length scale (collision diameter)

In reduced units: `ε = 1`, `σ = 1`, `m = 1`, `k_B = 1`

## Running the Simulation

### Basic Usage

```bash
cd md
cargo run --example lj_cluster_npt --release
```

### Expected Output

The simulation will:
1. Create a face-centered cubic (FCC) cluster
2. Equilibrate at low temperature (solid-like)
3. Gradually heat the cluster
4. Monitor structural changes (melting, etc.)
5. Report final statistics and phase identification

### Output Table

```
┌────────┬─────────┬─────────┬─────────┬──────────┬──────────┬─────────┬─────────┐
│   Step │  T_inst │  P_inst │   P_tgt │   Volume │       PE │  Box_L  │  Coord# │
├────────┼─────────┼─────────┼─────────┼──────────┼──────────┼─────────┼─────────┤
```

Where:
- **Step**: Current timestep
- **T_inst**: Instantaneous temperature
- **P_inst**: Instantaneous pressure (with virial)
- **P_tgt**: Target pressure
- **Volume**: Current box volume
- **PE**: Potential energy
- **Box_L**: Box length (cubic box)
- **Coord#**: Average coordination number (neighbors within 1.5σ)

## Key Parameters

### System Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_cells` | 2 | FCC unit cells (2×2×2 = 32 atoms) |
| `lattice_constant` | 1.55σ | FCC lattice spacing |
| `initial_temp` | 0.5 ε/k_B | Starting temperature |
| `target_pressure` | 0.1 ε/σ³ | Target pressure |

### Integrator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `q_t` | 100.0 | Thermostat coupling (larger = gentler) |
| `q_p` | 2000.0 | Barostat coupling (larger = gentler) |
| `dt` | 0.002 | Time step in reduced units |

### Tuning Guidelines

**Thermostat coupling (q_t):**
- Smaller values (10-50): Fast temperature control, may cause oscillations
- Medium values (100-500): Balanced, good for most cases
- Larger values (>1000): Slow, gentle temperature control

**Barostat coupling (q_p):**
- Should be ~10-100× larger than q_t for stability
- Too small: Box size oscillations
- Too large: Slow pressure equilibration

**Time step (dt):**
- Typical: 0.001-0.005 τ (reduced time units)
- Smaller for high temperatures or strong forces
- Larger for well-equilibrated systems

## Understanding Results

### Phase Identification

The code analyzes three key indicators:

1. **Coordination Number**
   - Solid: > 10 neighbors
   - Liquid: 6-10 neighbors  
   - Gas: < 6 neighbors

2. **Potential Energy per Atom**
   - Very stable (solid): PE/atom < -4 ε
   - Moderately stable (liquid): -4 < PE/atom < -2 ε
   - High energy (gas): PE/atom > -2 ε

3. **Density**
   - High (condensed): ρ > 0.8 atoms/σ³
   - Medium: 0.3 < ρ < 0.8 atoms/σ³
   - Low (gas): ρ < 0.3 atoms/σ³

### Typical Behaviors

**Melting Transition:**
- Coordination number drops from ~12 to 6-8
- Potential energy increases
- Energy fluctuations increase
- Volume may expand

**Evaporation (low pressure):**
- Volume increases dramatically
- Coordination number drops to ~0-3
- Density decreases significantly

**Condensation (high pressure):**
- Volume decreases
- Coordination number increases
- Density increases

## Physical Insights

### Why Virial Matters

For a non-interacting ideal gas:
```
P = NkT/V  (no virial term)
```

For Lennard-Jones clusters:
```
P = (NkT + W)/V  (virial is significant!)
```

At high densities or low temperatures, the virial contribution can **dominate** the pressure. Without proper virial calculation, the NPT simulation will give incorrect volumes and densities.

### Temperature vs. Pressure Effects

- **High T, Low P**: Expanded gas-like cluster
- **High T, High P**: Dense liquid
- **Low T, Low P**: Weakly bound cluster (may evaporate)
- **Low T, High P**: Compressed solid

### Cluster Size Effects

Small clusters (< 50 atoms):
- Surface atoms dominate
- Lower melting points
- Larger fluctuations

Large clusters (> 100 atoms):
- Bulk-like behavior
- Sharper phase transitions
- More stable structures

## Modifications

### Change Cluster Size

```rust
let n_cells = 3;  // 3×3×3 = 108 atoms
```

### Adjust Pressure Range

```rust
// Low pressure (gas-like)
let target_pressure = 0.01;

// Medium pressure (liquid)
let target_pressure = 0.5;

// High pressure (compressed solid)
let target_pressure = 2.0;
```

### Temperature Ramping

```rust
// Heat slowly
let heating_start = 5000;
let heating_end = 25000;
let final_temp = 2.0;  // Well above melting

// Cool down instead
let final_temp = 0.3;  // Below melting
```

### Analysis Frequency

```rust
let output_interval = 200;    // More frequent output
let analysis_interval = 10;   // More statistics samples
```

## Troubleshooting

### Problem: Box size explodes or collapses

**Solution**: Adjust barostat coupling or clamp limits
```rust
let q_p = 5000.0;  // Gentler coupling
```

### Problem: Temperature oscillates wildly

**Solution**: Increase thermostat coupling
```rust
let q_t = 500.0;  // Smoother temperature control
```

### Problem: Pressure never reaches target

**Possible causes:**
1. System needs more equilibration time
2. Target pressure incompatible with temperature (e.g., high T + very high P)
3. Barostat coupling too weak

**Solution**: 
- Increase total simulation time
- Adjust temperature or pressure targets
- Reduce q_p for faster response

### Problem: Cluster evaporates

**Cause**: Pressure too low or temperature too high

**Solution**: Increase target pressure or decrease temperature

## Advanced Features

### Custom Initial Structures

You can replace the FCC cluster with:
- Simple cubic lattice
- Random positions
- Pre-melted configurations

### Anisotropic Pressure

Modify the barostat to apply different pressures in each direction:
```rust
let pressure_x = 0.1;
let pressure_y = 0.1;
let pressure_z = 0.5;  // Compression along z-axis
```

### Stress-Strain Analysis

Apply time-varying pressure to study mechanical response.

## References

1. **Nosé-Hoover Thermostat**: 
   - Nosé, S. (1984). J. Chem. Phys. 81, 511.
   - Hoover, W. G. (1985). Phys. Rev. A 31, 1695.

2. **Parrinello-Rahman Barostat**:
   - Parrinello, M. & Rahman, A. (1981). J. Appl. Phys. 52, 7182.

3. **Virial Pressure**:
   - Allen, M. P. & Tildesley, D. J. "Computer Simulation of Liquids" (2017).

4. **LJ Clusters**:
   - Wales, D. J. "Energy Landscapes" (2003).

## Example Output Interpretation

```
Final Analysis:
  Temperature: 1.502 ε/k_B (target: 1.500)     ✅ Thermostat working
  Pressure: 0.098 ε/σ³ (target: 0.100)         ✅ Barostat working
  Coordination number: 7.2                      ➡️ Liquid-like
  PE per atom: -3.2 ε                          ➡️ Moderately stable
  Density: 0.52 atoms/σ³                       ➡️ Medium density
```

This indicates a **liquid-like cluster** at the target (P,T) conditions!

---

**Happy simulating! 🎯**

