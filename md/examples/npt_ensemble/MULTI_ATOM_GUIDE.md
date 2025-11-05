# Multi-Atom Molecular Dynamics Guide

This guide explains how to work with multiple atoms in molecular dynamics simulations, highlighting the key differences from single-atom systems.

## Key Concepts

### 1. **Inter-atomic Interactions**

**Single Atom:**
```rust
// No inter-atomic forces
struct SingleAtomForces;
impl ForceProvider for SingleAtomForces {
    fn compute_forces(&self, positions: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        vec![Vector3::zeros(); positions.len()] // Zero forces
    }
}
```

**Multiple Atoms:**
```rust
// Lennard-Jones interactions between all atom pairs
let lj = LennardJones::new(epsilon, sigma, box_lengths);
// Forces computed from: V(r) = 4ε[(σ/r)¹² - (σ/r)⁶]
```

### 2. **System Initialization**

**Positions:**
- **Single atom:** Place anywhere in the box
- **Multiple atoms:** Use structured arrangements (lattices) or random configurations

```rust
// Simple cubic lattice for multiple atoms
fn create_simple_cubic_lattice(n_per_side: usize, spacing: f64) -> Vec<Vector3<f64>> {
    let mut positions = Vec::new();
    for i in 0..n_per_side {
        for j in 0..n_per_side {
            for k in 0..n_per_side {
                positions.push(Vector3::new(
                    i as f64 * spacing,
                    j as f64 * spacing, 
                    k as f64 * spacing
                ));
            }
        }
    }
    positions
}
```

**Velocities:**
- **Single atom:** Sample from Maxwell-Boltzmann distribution
- **Multiple atoms:** Sample + remove center-of-mass motion

```rust
fn initialize_velocities(n_atoms: usize, temperature: f64) -> Vec<Vector3<f64>> {
    // 1. Sample individual velocities from Maxwell-Boltzmann
    // 2. Remove center-of-mass motion: Σᵢ mᵢvᵢ = 0  
    // 3. Scale to exact target temperature
}
```

### 3. **Pressure Calculation**

**Single Atom:**
```rust
// Pure ideal gas: P = nkT/V
let pressure = n_atoms * k_b * temperature / volume;
```

**Multiple Atoms:**
```rust  
// Ideal gas + virial contribution from interactions
let kinetic_pressure = n_atoms * k_b * temperature / volume;
let virial_pressure = virial_sum / (3.0 * volume); // From forces
let total_pressure = kinetic_pressure + virial_pressure;
```

### 4. **System Analysis**

Multi-atom systems enable rich structural analysis:

```rust
struct StructureAnalyzer {
    // Track positions over time
    position_history: VecDeque<Vec<Vector3<f64>>>,
}

impl StructureAnalyzer {
    // Radial distribution function g(r)
    fn radial_distribution(&self) -> Vec<f64>;
    
    // Diffusion coefficient from mean squared displacement  
    fn diffusion_coefficient(&self) -> f64;
    
    // Order parameters for phase identification
    fn bond_order_parameter(&self) -> f64;
}
```

## Examples Comparison

| Property | Single Atom | Multiple Atoms |
|----------|-------------|----------------|
| **Forces** | Zero or external only | Inter-atomic (LJ, Coulomb, etc.) |
| **Pressure** | P = nkT/V (ideal gas) | P = P_kinetic + P_virial |  
| **Phases** | No phase transitions | Solid ↔ Liquid ↔ Gas |
| **Structure** | No structure | Crystals, liquids, clusters |
| **Dynamics** | Simple ballistic | Collective motion, waves |

## Running the Examples

### Single Atom NPT:
```bash
cargo run --example single_atom_npt
```
- Demonstrates pure NPT mechanics
- Ideal gas law behavior (PV = nkT)
- Volume fluctuations under pressure control

### Multiple Atoms NPT:
```bash  
cargo run --example multi_atom_npt
```
- Realistic inter-atomic interactions
- Structure formation and analysis
- Phase-like behavior
- Collective dynamics

### Existing Melting Simulation:
```bash
cargo run --example argon_melting
```
- 256 atoms, NVT ensemble
- Solid → liquid phase transition
- Temperature ramping from 60K → 180K

## Key Physics Insights

### 1. **Emergent Behavior**
- Single atoms show only thermal motion
- Multiple atoms show collective phenomena:
  - Sound waves, diffusion, phase transitions

### 2. **Length Scales**
- **Single atom:** Only box size matters
- **Multiple atoms:** Inter-atomic spacing, correlation lengths, defects

### 3. **Time Scales**  
- **Single atom:** Only collision time with walls
- **Multiple atoms:** Vibrational periods, diffusion times, relaxation

### 4. **Thermodynamics**
- **Single atom:** Trivial (ideal gas)
- **Multiple atoms:** Rich phase diagrams, equations of state

## Advanced Topics

### Custom Force Fields
```rust
struct CustomForceField {
    lj_params: LennardJonesParams,
    coulomb_params: CoulombParams,
    bond_params: BondParams,
}
```

### Analysis Tools
```rust
// Pair correlation function
fn g_of_r(&self, positions: &[Vector3<f64>]) -> Vec<f64>;

// Structure factor  
fn structure_factor(&self, q_vectors: &[Vector3<f64>]) -> Vec<f64>;

// Velocity autocorrelation
fn velocity_autocorr(&self) -> Vec<f64>;
```

### Optimization for Large Systems
- **Neighbor lists** for efficient force calculations  
- **Cell lists** for spatial partitioning
- **Parallel computation** across atoms
- **GPU acceleration** for massive systems

## Troubleshooting

### Common Issues:
1. **System instability:** Reduce time step, gentler coupling
2. **Unphysical behavior:** Check initialization, force field parameters  
3. **Poor equilibration:** Longer runs, better initial conditions
4. **Box collapse:** Proper pressure coupling parameters

### Performance:
- **N² scaling:** Force computation is expensive for large N
- **Memory:** Position/velocity histories for analysis
- **I/O:** Frequent output can slow simulation

Ready to explore multi-atom dynamics! 🚀
