# Grand Canonical Monte Carlo (GCMC) for Lennard-Jones Particles

## Overview

This module implements Grand Canonical Monte Carlo (GCMC) simulations for Lennard-Jones particles in periodic boundary conditions. GCMC samples the grand canonical ensemble (μVT), where the chemical potential (μ), volume (V), and temperature (T) are fixed, while the number of particles (N) fluctuates.

## Theory

### Grand Canonical Ensemble

In the grand canonical ensemble, the system can exchange particles with a reservoir at fixed chemical potential μ. The probability of observing a state with N particles and energy E is:

```
P(N, E) ∝ exp(β(μN - E))
```

where β = 1/(k_B T).

### Monte Carlo Moves

GCMC uses three types of moves:

1. **Displacement Move** (Metropolis):
   - Randomly select a particle
   - Displace it by a random amount
   - Accept with probability: min(1, exp(-β ΔU))

2. **Insertion Move**:
   - Insert a particle at random position
   - Accept with probability: min(1, (V/(N+1)) exp(-β ΔU + β μ))

3. **Deletion Move**:
   - Remove a randomly selected particle
   - Accept with probability: min(1, (N/V) exp(-β ΔU - β μ))

### Lennard-Jones Potential

The LJ potential between particles i and j is:

```
U(r_ij) = 4ε[(σ/r_ij)^12 - (σ/r_ij)^6]
```

With cutoff at r_c = 2.5σ for computational efficiency.

## Usage

### Basic Example

```rust
use md::GCMC;

// Create GCMC simulation
let mut gcmc = GCMC::new(
    1.0,    // epsilon (energy parameter)
    1.0,    // sigma (length parameter)
    10.0,   // box_length
    1.5,    // temperature
    -3.0,   // chemical_potential
);

// Initialize with some particles
gcmc.initialize_random(0.3);  // density = 0.3

// Equilibration
gcmc.run(10_000);

// Production with sampling
for _ in 0..50_000 {
    gcmc.monte_carlo_step();
    if step % 100 == 0 {
        gcmc.sample();
    }
}

// Get results
println!("Average N: {}", gcmc.stats.avg_n_particles);
println!("Average density: {}", gcmc.density());
```

### Chemical Potential Sweep

To compute isotherms (density vs. chemical potential):

```rust
use md::{GCMC, parallel_gcmc_sweep};

// Create simulations at different μ values
let mu_values = vec![-5.0, -4.0, -3.0, -2.0, -1.0];
let gcmc_configs: Vec<GCMC> = mu_values.iter()
    .map(|&mu| GCMC::new(1.0, 1.0, 10.0, 1.5, mu))
    .collect();

// Run in parallel
let results = parallel_gcmc_sweep(
    &gcmc_configs,
    10_000,  // equilibration steps
    50_000,  // production steps
    100,     // sample interval
);

// Process results
for result in results {
    println!("μ = {:.2}: ⟨ρ⟩ = {:.4}", 
             result.chemical_potential, 
             result.avg_density);
}
```

### Non-cubic Boxes

For triclinic or non-orthogonal boxes:

```rust
use nalgebra::{Vector3, Matrix3};

let a = Vector3::new(10.0, 0.0, 0.0);
let b = Vector3::new(5.0, 8.66, 0.0);  // 60° angle
let c = Vector3::new(0.0, 0.0, 10.0);
let lattice = Matrix3::from_columns(&[a, b, c]);

let gcmc = GCMC::from_lattice(1.0, 1.0, lattice, 1.5, -3.0);
```

## Configuration Options

### Move Probabilities

Adjust the relative frequency of different move types:

```rust
// Default: [0.5, 0.25, 0.25] (displacement, insertion, deletion)
gcmc.set_move_probabilities(0.7, 0.15, 0.15);
```

**Guidelines:**
- High density: increase displacement moves (0.7, 0.15, 0.15)
- Low density: increase insertion/deletion (0.4, 0.3, 0.3)
- Balanced: equal insertion/deletion, moderate displacement

### Maximum Displacement

Control the step size for displacement moves:

```rust
gcmc.set_max_displacement(0.5);  // in units of σ
```

**Guidelines:**
- Target acceptance rate: 30-50%
- Too large: low acceptance
- Too small: slow sampling
- Typical: 0.3-0.5 σ

### Temperature

```rust
gcmc.set_temperature(1.5);
```

**Reduced units:** T* = k_B T / ε

**Phase behavior:**
- T* < 1.3: Gas-liquid coexistence possible
- T* ≈ 1.3: Near critical temperature
- T* > 1.3: Supercritical fluid

### Chemical Potential

```rust
gcmc.set_chemical_potential(-3.0);
```

**Reduced units:** μ* = μ / ε

**Guidelines:**
- More negative: fewer particles (gas)
- Less negative: more particles (liquid)
- Typical range: -6 to 0

## Output and Analysis

### Statistics

```rust
// Acceptance rates
println!("Displacement: {:.1}%", 
         100.0 * gcmc.stats.displacement_acceptance_rate());
println!("Insertion: {:.1}%", 
         100.0 * gcmc.stats.insertion_acceptance_rate());
println!("Deletion: {:.1}%", 
         100.0 * gcmc.stats.deletion_acceptance_rate());

// Averages
println!("⟨N⟩ = {:.2}", gcmc.stats.avg_n_particles);
println!("⟨E⟩ = {:.2}", gcmc.stats.avg_energy);
```

### Observables

```rust
// Instantaneous values
let n = gcmc.n_particles();
let density = gcmc.density();
let energy = gcmc.potential_energy();
let energy_per_particle = gcmc.potential_energy_per_particle();

// Particle positions
let positions = gcmc.get_positions();
```

### Results Analysis

```rust
let result = results[0];  // From parallel_gcmc_sweep

// Averages
println!("⟨N⟩ = {:.2}", result.avg_n_particles);
println!("⟨ρ⟩ = {:.6}", result.avg_density);
println!("⟨E/N⟩ = {:.4}", result.avg_energy_per_particle);

// Fluctuations
println!("σ(N) = {:.2}", result.n_particles_std());
println!("σ(ρ) = {:.6}", result.density_std());

// Time series
let n_samples = &result.n_particles_samples;
let density_samples = &result.density_samples;
let energy_samples = &result.energy_samples;
```

## Examples

### Example 1: Basic GCMC Simulation

```bash
cargo run --example gcmc_lj_demo
```

Features:
- Single chemical potential simulation
- Chemical potential sweep
- High and low density cases
- Statistics and acceptance rates

### Example 2: Phase Diagram

```bash
cargo run --example gcmc_phase_diagram
```

Features:
- Multiple temperature isotherms
- Phase behavior analysis
- Critical region detection
- Data export for plotting

## Physical Interpretation

### Density vs. Chemical Potential

The relationship ρ(μ) at fixed T:

1. **Gas phase** (low ρ, low μ):
   - Few particles
   - Weak interactions
   - Nearly ideal gas behavior

2. **Liquid phase** (high ρ, high μ):
   - Many particles
   - Strong interactions
   - Negative energy per particle

3. **Phase transition** (T < T_c):
   - Density jump at coexistence μ
   - Both phases present
   - Maxwell construction

4. **Critical point** (T ≈ T_c, μ ≈ μ_c):
   - Large density fluctuations
   - High compressibility
   - Critical opalescence

### Reduced Units

All quantities are in reduced units:

| Quantity | Reduced Unit | Relation |
|----------|--------------|----------|
| Length | σ | r* = r/σ |
| Energy | ε | E* = E/ε |
| Temperature | ε/k_B | T* = k_B T/ε |
| Density | σ⁻³ | ρ* = ρσ³ |
| Chemical potential | ε | μ* = μ/ε |
| Pressure | ε/σ³ | P* = Pσ³/ε |

## Performance Considerations

### Computational Cost

- Displacement: O(N) per attempt
- Insertion: O(N) per attempt
- Deletion: O(N) per attempt

Where N is the number of particles.

### Parallelization

Use `parallel_gcmc_sweep` for multiple independent simulations:

```rust
let results = parallel_gcmc_sweep(&configs, 10_000, 50_000, 100);
```

This uses Rayon to parallelize across different μ or T values.

### Memory Usage

- Positions: ~24N bytes (3 × f64 per particle)
- Minimal overhead for GCMC state

### Optimization Tips

1. **Cutoff**: Default 2.5σ is standard, reducing increases speed but affects accuracy

2. **Equilibration**: Rule of thumb: 10,000-50,000 steps depending on system

3. **Production**: 50,000-100,000 steps for good statistics

4. **Sampling interval**: Every 50-100 steps to reduce correlation

5. **Box size**: Larger boxes give better statistics but slower computation

## Validation

### Ideal Gas Limit

At high T and low ρ:
```
⟨N⟩ ≈ V exp(β μ) / Λ³
```
where Λ is thermal wavelength.

### Energy Check

For pure LJ with N particles:
- Ground state (T→0): E/N ≈ -6ε (FCC lattice)
- Gas phase: E/N ≈ 0
- Liquid phase: E/N ≈ -4 to -5 ε

### Acceptance Rates

Healthy simulation:
- Displacement: 30-50%
- Insertion: 0.1-10% (lower at high ρ)
- Deletion: 0.1-10% (higher at low ρ)

## Troubleshooting

### Problem: Acceptance rates too low

**Solution:**
- Reduce max_displacement
- Adjust move probabilities
- Check if μ is in reasonable range

### Problem: N → 0 or N → very large

**Solution:**
- Adjust chemical potential
- Check temperature is positive
- Verify initialization

### Problem: No convergence

**Solution:**
- Increase equilibration steps
- Check for overlapping particles at initialization
- Verify system parameters are physical

### Problem: Large fluctuations in N

**Possible causes:**
- Near critical point (physical)
- System too small
- Insufficient equilibration

## References

1. Frenkel, D., & Smit, B. (2002). *Understanding Molecular Simulation*. Academic Press.

2. Allen, M. P., & Tildesley, D. J. (1987). *Computer Simulation of Liquids*. Oxford University Press.

3. Adams, D. J. (1975). Grand canonical ensemble Monte Carlo for a Lennard-Jones fluid. *Molecular Physics*, 29(1), 307-311.

4. Norman, G. E., & Filinov, V. S. (1969). Investigations of phase transitions by a Monte-Carlo method. *High Temperature*, 7, 216.

## License

Part of the rust_scf molecular dynamics package.

