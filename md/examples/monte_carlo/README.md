# Monte Carlo Simulation Examples

This directory contains Monte Carlo simulation examples for various statistical mechanics systems.

## Overview

Monte Carlo methods use random sampling to compute statistical properties of many-body systems. Unlike molecular dynamics, MC does not simulate time evolution but directly samples from the Boltzmann distribution.

**Key Advantages:**
- Can easily sample rare configurations
- Natural for discrete systems (spins, lattice gases)
- Efficient for equilibrium properties
- Enables variable-N simulations (Grand Canonical)

## Directory Structure

```
monte_carlo/
├── README.md (this file)
├── ising/                          # Ising model spin systems
│   ├── README.md
│   ├── ising_2d_mc.rs             # 2D Ising model
│   ├── ising_3d_mc.rs             # 3D Ising model
│   ├── ising_4d_mc.rs             # 4D Ising model
│   ├── simple_ising_3d.rs         # Simplified 3D version
│   ├── critical_temperature_analysis.rs
│   ├── simple_tc_calculation.rs
│   ├── cluster_vs_metropolis.rs
│   └── wolff_algorithm_guide.rs
├── gcmc/                           # Grand Canonical MC
│   ├── README.md
│   ├── gcmc_quickstart.rs         # Quick introduction
│   ├── gcmc_lj_demo.rs            # Full GCMC simulation
│   └── gcmc_phase_diagram.rs      # Phase behavior
└── parallel_mc_benchmark.rs       # Performance testing
```

## Quick Start

### Ising Model Simulations

Study phase transitions in magnetic systems:

```bash
# Simple 3D ferromagnet
cargo run --example simple_ising_3d --release

# Detailed 2D analysis
cargo run --example ising_2d_mc --release

# Find critical temperature
cargo run --example critical_temperature_analysis --release
```

**See:** [ising/README.md](ising/README.md) for detailed documentation

### Grand Canonical Monte Carlo

Simulate systems with variable particle number:

```bash
# Quick start
cargo run --example gcmc_quickstart --release

# Full simulation
cargo run --example gcmc_lj_demo --release

# Phase diagram
cargo run --example gcmc_phase_diagram --release
```

**See:** [gcmc/README.md](gcmc/README.md) for detailed documentation

---

## Monte Carlo Fundamentals

### Metropolis-Hastings Algorithm

1. Propose a move (e.g., spin flip, particle displacement)
2. Calculate energy change: ΔE = E_new - E_old
3. Accept with probability: P = min(1, exp(-βΔE))
4. Update configuration if accepted

This ensures detailed balance and converges to Boltzmann distribution.

### Importance Sampling

Instead of uniform sampling, MC preferentially samples low-energy configurations weighted by exp(-βE), making it highly efficient for thermodynamic averages.

### Observables

Calculate ensemble averages:
```
⟨O⟩ = (1/N) Σᵢ O(config_i)
```

where configurations are sampled from the Boltzmann distribution.

---

## Comparison with Molecular Dynamics

| Feature | Monte Carlo | Molecular Dynamics |
|---------|-------------|-------------------|
| **Time evolution** | No | Yes (trajectories) |
| **Dynamics** | Unphysical | Physical |
| **Equilibrium** | Direct sampling | Time averaging |
| **Phase space** | Any ensemble | Usually NVE/NVT |
| **Rare events** | Accessible | Difficult |
| **Discrete systems** | Natural | Requires tricks |
| **Variable N** | Easy (GCMC) | Complex |

**Use MC when:**
- Equilibrium properties are sufficient
- System is discrete or has constraints
- Need to sample rare configurations
- Working in Grand Canonical ensemble

**Use MD when:**
- Dynamical properties needed (diffusion, spectra)
- Time-dependent processes
- Continuous potentials
- Transport coefficients

---

## Ensemble Types

### Canonical (NVT)
- Fixed: N, V, T
- Examples: All Ising models, some GCMC moves

### Grand Canonical (μVT)
- Fixed: μ, V, T
- Variable: N
- Examples: All GCMC simulations

### Microcanonical (NVE)
- Fixed: N, V, E
- Used in some specialized algorithms

---

## Performance Benchmarking

Test MC performance and scaling:

```bash
cargo run --example parallel_mc_benchmark --release
```

Compares:
- Serial vs. parallel implementations
- Different update schemes
- System size scaling

---

## Algorithm Selection Guide

### For Phase Transitions

- **Near T_c:** Use cluster algorithms (Wolff, Swendsen-Wang)
- **Far from T_c:** Simple Metropolis is fine
- **Unknown T_c:** Run `critical_temperature_analysis` first

### For Large Systems

- **N > 10⁶:** Use parallel updates when possible
- **Careful:** Ensure detailed balance preserved
- **Check:** Autocorrelation times

### For Variable-N Systems

- **Always:** Use GCMC
- **Phase coexistence:** Increase sampling significantly
- **Adsorption:** Scan chemical potential

---

## Common Pitfalls

### 1. Insufficient Equilibration

**Problem:** System hasn't reached equilibrium

**Solution:**
- Monitor observables vs. MC steps
- Look for drift or trends
- Increase equilibration time near T_c

### 2. Correlated Samples

**Problem:** Adjacent configurations are too similar

**Solution:**
- Measure autocorrelation time
- Sample every τ_auto steps
- Use cluster algorithms

### 3. Finite-Size Effects

**Problem:** Small systems don't show bulk behavior

**Solution:**
- Test multiple system sizes
- Use finite-size scaling theory
- L > 20-30 for most properties

### 4. Poor Acceptance Rates

**Problem:** Most moves rejected (or all accepted)

**Solution:**
- Tune step sizes (target 30-50%)
- Adjust parameters (T, μ, etc.)
- Check energy calculations

---

## Best Practices

1. **Always use release mode:**
   ```bash
   cargo run --example <name> --release
   ```

2. **Check detailed balance:** Ensure P(A→B) exp(-βE_A) = P(B→A) exp(-βE_B)

3. **Monitor acceptance rates:** Should be 20-60% for most moves

4. **Measure autocorrelations:** Sample spacing > τ_auto

5. **Use multiple runs:** Average over independent simulations

6. **Save configurations:** For later analysis or restart

---

## References

### General Monte Carlo

1. Metropolis, N., et al. (1953). "Equation of State Calculations by Fast Computing Machines." *J. Chem. Phys.* **21**, 1087.
2. Hastings, W. K. (1970). "Monte Carlo Sampling Methods Using Markov Chains." *Biometrika* **57**, 97.
3. Newman, M. E. J. & Barkema, G. T. (1999). *Monte Carlo Methods in Statistical Physics*. Oxford University Press.

### Advanced Techniques

4. Swendsen, R. H. & Wang, J.-S. (1987). "Nonuniversal critical dynamics in Monte Carlo simulations." *Phys. Rev. Lett.* **58**, 86.
5. Wolff, U. (1989). "Collective Monte Carlo Updating for Spin Systems." *Phys. Rev. Lett.* **62**, 361.

### Applications

6. Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation* (2nd ed.). Academic Press.
7. Landau, D. P. & Binder, K. (2014). *A Guide to Monte Carlo Simulations in Statistical Physics* (4th ed.). Cambridge University Press.

---

**Happy Monte Carlo simulating! 🎲**

