# NPT Ensemble Simulation Examples

This directory contains several examples demonstrating NPT (constant Number, Pressure, Temperature) molecular dynamics simulations for Lennard-Jones systems.

## Overview of NPT Ensemble

In the NPT ensemble:
- **N** (Number of particles): Fixed
- **P** (Pressure): Controlled by **barostat**
- **T** (Temperature): Controlled by **thermostat**
- **V** (Volume): **Fluctuates** to maintain constant pressure

### Thermostats and Barostats

Our implementation uses:
- **Nosé-Hoover thermostat**: Controls temperature by coupling to a heat bath
- **Parrinello-Rahman barostat**: Controls pressure by allowing box size changes

## Available Examples

### 1. `quick_lj_npt.rs` - Quick Testing ⚡

**Purpose**: Rapid prototyping and testing

**System**: 8 atoms (2×2×2 simple cubic)

**Runtime**: ~30 seconds

**Use when**:
- Testing parameter changes quickly
- Learning the NPT basics
- Debugging new features

**Run**:
```bash
cargo run --example quick_lj_npt --release
```

**Output**: Simple table showing T, P, box size evolution

---

### 2. `lj_cluster_npt.rs` - Full Cluster Simulation 🔬

**Purpose**: Comprehensive LJ cluster NPT simulation with detailed analysis

**System**: 32 atoms (2×2×2 FCC cluster, configurable)

**Runtime**: ~2-3 minutes

**Features**:
- Proper virial-based pressure calculation
- Temperature ramping (heating cycle)
- Structural analysis (coordination number)
- Phase identification (solid/liquid/gas)
- Energy fluctuation statistics

**Use when**:
- Studying phase transitions (melting, evaporation)
- Analyzing cluster thermodynamics
- Publishing-quality simulations
- Learning advanced NPT techniques

**Run**:
```bash
cargo run --example lj_cluster_npt --release
```

**Documentation**: See `LJ_CLUSTER_NPT_GUIDE.md` for detailed guide

---

### 3. `single_atom_npt.rs` - Single Atom Ideal Gas 🎯

**Purpose**: Educational - demonstrates NPT mechanics without interactions

**System**: 1 atom (ideal gas)

**Runtime**: ~1 minute

**Features**:
- Pure NPT dynamics without LJ interactions
- Validates ideal gas law: PV = nkT
- Temperature ramping
- Educational output and explanations

**Use when**:
- Understanding barostat/thermostat mechanics
- Validating NPT implementation
- Teaching statistical mechanics
- Debugging NPT algorithms

**Run**:
```bash
cargo run --example single_atom_npt --release
```

**Physical insight**: With one atom, pressure is purely kinetic. The barostat adjusts volume to maintain P = kT/V.

---

### 4. `multi_atom_npt.rs` - Multi-Atom LJ System 🌊

**Purpose**: Realistic multi-atom NPT with structure analysis

**System**: 27 atoms (3×3×3 simple cubic)

**Runtime**: ~2 minutes

**Features**:
- Lennard-Jones interactions
- Temperature ramping
- Radial distribution function
- Diffusion coefficient calculation
- Phase behavior analysis

**Use when**:
- Studying liquid/gas transitions
- Analyzing structural properties
- Comparing with single-atom case
- Production simulations

**Run**:
```bash
cargo run --example multi_atom_npt --release
```

---

## Comparison Table

| Example | Atoms | Interactions | Runtime | Complexity | Best For |
|---------|-------|--------------|---------|------------|----------|
| `quick_lj_npt` | 8 | LJ | 30s | Simple | Quick tests |
| `lj_cluster_npt` | 32 | LJ | 3min | Advanced | Research |
| `single_atom_npt` | 1 | None | 1min | Educational | Learning |
| `multi_atom_npt` | 27 | LJ | 2min | Moderate | Production |

## Key Concepts

### Virial Pressure

For **non-interacting** particles (ideal gas):
```
P = NkT/V
```

For **Lennard-Jones** systems:
```
P = (2K + W) / (3V)
```

where:
- `K` = kinetic energy
- `W` = virial = Σ_ij r_ij · F_ij
- `V` = volume

**Critical**: The `lj_cluster_npt` example uses **proper virial calculation** via `LennardJones::compute_pressure()`. This is essential for accurate NPT simulations!

### Coupling Parameters

**Thermostat coupling (Q_T)**:
- Controls how quickly temperature equilibrates
- Typical range: 50-500
- Larger = gentler, slower response
- Smaller = aggressive, may oscillate

**Barostat coupling (Q_P)**:
- Controls how quickly pressure equilibrates
- Typical range: 1000-5000
- Should be 10-100× larger than Q_T
- Too small = box oscillations
- Too large = slow equilibration

### Time Step

Typical values: `dt = 0.001 - 0.005` (reduced units)

Rule of thumb:
- High temperature → smaller dt
- Strong forces → smaller dt  
- Well-equilibrated → larger dt possible

## Common Use Cases

### 1. Study Melting Transition

Use: `lj_cluster_npt.rs`

Modify temperature schedule:
```rust
let initial_temp = 0.3;  // Solid
let final_temp = 1.5;    // Above melting
let heating_start = 5000;
let heating_end = 20000;
```

Monitor: Coordination number drops from ~12 to ~6-8 at melting

---

### 2. Compress/Expand System

Use: `lj_cluster_npt.rs` or `multi_atom_npt.rs`

Modify target pressure:
```rust
// Low pressure (expansion/evaporation)
let target_pressure = 0.01;

// High pressure (compression)
let target_pressure = 2.0;
```

Monitor: Volume and density changes

---

### 3. Validate Thermodynamics

Use: `single_atom_npt.rs`

Check ideal gas law:
```
PV/NkT ≈ 1
```

This validates that thermostat + barostat work correctly.

---

### 4. Explore Phase Diagram

Use: `lj_cluster_npt.rs`

Scan (P,T) space:
```rust
// Vary both pressure and temperature
for pressure in [0.01, 0.1, 0.5, 1.0, 2.0] {
    for temp in [0.5, 1.0, 1.5, 2.0] {
        // Run NPT simulation
        // Classify phase from coordination number
    }
}
```

---

## Troubleshooting

### Box Size Explodes

**Symptoms**: Box grows to 50+ σ

**Cause**: 
- Pressure target too low for given temperature
- System wants to evaporate

**Fix**:
- Increase target pressure
- Decrease temperature
- Increase barostat coupling (larger Q_P)

---

### Box Size Collapses

**Symptoms**: Box shrinks to minimum (2-3 σ)

**Cause**:
- Pressure target too high
- Strong compression

**Fix**:
- Decrease target pressure
- Check if this is physically reasonable
- May indicate phase transition to solid

---

### Temperature Oscillates

**Symptoms**: T fluctuates wildly around target

**Cause**: Thermostat coupling too weak

**Fix**:
```rust
let q_t = 500.0;  // Increase from default
```

---

### Pressure Never Stabilizes

**Symptoms**: P keeps drifting away from target

**Possible causes**:
1. Not enough equilibration time
2. Target (P,T) incompatible with system
3. Virial calculation incorrect

**Fixes**:
1. Increase `total_steps`
2. Adjust P or T to physically reasonable values
3. Ensure using `LennardJones::compute_pressure()` with virial

---

## Performance Tips

### For Large Systems (N > 100)

1. **Use release mode**:
```bash
cargo run --example lj_cluster_npt --release
```
~10× faster than debug mode

2. **Reduce output frequency**:
```rust
let output_interval = 1000;  // Instead of 500
```

3. **Parallelize** (already enabled via Rayon in `lj_pot.rs`)

### For Long Simulations

1. **Save checkpoints**:
```rust
if step % 10000 == 0 {
    // Save positions, velocities, box size
}
```

2. **Stream output to file**:
```bash
cargo run --example lj_cluster_npt --release > output.dat
```

---

## Extending the Examples

### Add Custom Analysis

In `lj_cluster_npt.rs`, add to the `ClusterAnalyzer`:
```rust
fn my_custom_metric(&self) -> f64 {
    // Your analysis here
}
```

### Change Initial Structure

Replace FCC cluster with:
```rust
// Random positions
let positions = (0..n_atoms).map(|_| {
    Vector3::new(
        rng.gen_range(0.0..box_size),
        rng.gen_range(0.0..box_size),
        rng.gen_range(0.0..box_size),
    )
}).collect();
```

### Anisotropic Pressure

Modify barostat to apply different pressures in each direction (requires code changes to allow independent box lengths).

---

## References

### Theory

1. **Nosé-Hoover Thermostat**:
   - Nosé, S. (1984). "A unified formulation of the constant temperature molecular dynamics methods." *J. Chem. Phys.* **81**, 511.
   
2. **Parrinello-Rahman Barostat**:
   - Parrinello, M. & Rahman, A. (1981). "Polymorphic transitions in single crystals: A new molecular dynamics method." *J. Appl. Phys.* **52**, 7182.

3. **Virial Theorem & Pressure**:
   - Allen, M. P. & Tildesley, D. J. (2017). *Computer Simulation of Liquids* (2nd ed.). Oxford University Press.

### Lennard-Jones Clusters

4. **Phase Behavior**:
   - Wales, D. J. (2003). *Energy Landscapes*. Cambridge University Press.
   
5. **Melting**:
   - Berry, R. S., et al. (1993). "From van der Waals to metallic bonding: Melting of small clusters." *Phys. Rev. A* **47**, 5120.

---

## Quick Start

**Complete beginner**: Start with `single_atom_npt.rs`

**Learning NPT**: Progress to `quick_lj_npt.rs`

**Research simulations**: Use `lj_cluster_npt.rs` with `LJ_CLUSTER_NPT_GUIDE.md`

**Production runs**: Use `multi_atom_npt.rs` or `lj_cluster_npt.rs` with custom parameters

---

## Summary

✅ **Four NPT examples** covering education → research

✅ **Proper virial pressure** in `lj_cluster_npt.rs`

✅ **Comprehensive analysis** tools

✅ **Well-documented** with guides and comments

✅ **Fast compilation** and execution

🎯 **Start simulating realistic cluster thermodynamics today!**




