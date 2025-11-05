# Ising Model Monte Carlo Simulations

This directory contains Monte Carlo simulations of the Ising model in various dimensions.

## Overview

The Ising model is a mathematical model of ferromagnetism in statistical mechanics. It consists of discrete spins that can be in one of two states (+1 or -1), with interactions between neighboring spins.

**Hamiltonian:**
```
H = -J Σ_<ij> sᵢsⱼ - h Σᵢ sᵢ
```

where:
- J = coupling constant (J > 0 for ferromagnetic)
- h = external magnetic field
- sᵢ = spin at site i (±1)
- <ij> = nearest neighbor pairs

## Examples

### 1. 2D Ising Model (`ising_2d_mc.rs`)

Classic 2D Ising model with exact critical temperature known: **T_c ≈ 2.269 J/k_B**

**Run:**
```bash
cd md
cargo run --example ising_2d_mc --release
```

**Features:**
- Square lattice topology
- Periodic boundary conditions
- Metropolis-Hastings algorithm
- Observable calculations: magnetization, energy, susceptibility
- Configurable system size

---

### 2. 3D Ising Model (`ising_3d_mc.rs`)

Full-featured 3D Ising model simulation with detailed analysis.

**Run:**
```bash
cd md
cargo run --example ising_3d_mc --release
```

**Features:**
- Cubic lattice
- Temperature sweeps
- Critical behavior analysis
- Correlation functions
- Multiple observables tracked

**Critical Temperature:** T_c ≈ 4.511 J/k_B (3D)

---

### 3. Simple 3D Ising (`simple_ising_3d.rs`)

Simplified version for quick testing and learning.

**Run:**
```bash
cd md
cargo run --example simple_ising_3d --release
```

**Use when:**
- Learning Ising model basics
- Quick parameter testing
- Educational demonstrations

---

### 4. 4D Ising Model (`ising_4d_mc.rs`)

Four-dimensional Ising model for exploring mean-field regime.

**Run:**
```bash
cd md
cargo run --example ising_4d_mc --release
```

**Features:**
- Hypercubic lattice (4D)
- Mean-field-like behavior (d > 4 upper critical dimension)
- Faster convergence than 2D/3D
- Educational for understanding dimensionality effects

---

### 5. Critical Temperature Analysis (`critical_temperature_analysis.rs`)

Systematic analysis to determine critical temperature via finite-size scaling.

**Run:**
```bash
cd md
cargo run --example critical_temperature_analysis --release
```

**Features:**
- Multiple system sizes
- Temperature scan near T_c
- Binder cumulant analysis
- Finite-size scaling extrapolation
- Publication-quality analysis

**Runtime:** Several minutes (scans many temperatures and sizes)

---

### 6. Simple T_c Calculation (`simple_tc_calculation.rs`)

Quick estimation of critical temperature using basic observables.

**Run:**
```bash
cd md
cargo run --example simple_tc_calculation --release
```

**Method:** Finds peak in susceptibility χ(T) or specific heat C(T)

---

### 7. Cluster Algorithm Comparison (`cluster_vs_metropolis.rs`)

Compares Metropolis and Wolff cluster algorithms for efficiency.

**Run:**
```bash
cd md
cargo run --example cluster_vs_metropolis --release
```

**Key Insight:** Cluster algorithms reduce critical slowing down near T_c

---

### 8. Wolff Algorithm Guide (`wolff_algorithm_guide.rs`)

Detailed implementation and explanation of the Wolff cluster algorithm.

**Run:**
```bash
cd md
cargo run --example wolff_algorithm_guide --release
```

**Features:**
- Step-by-step algorithm explanation
- Performance comparisons
- Visual output of cluster formation
- Educational comments throughout

**Advantage:** O(N) autocorrelation time vs O(N²) for single-spin flip

---

## Comparison Table

| Example | Dimension | Complexity | T_c | Best For |
|---------|-----------|------------|-----|----------|
| `ising_2d_mc` | 2D | Medium | 2.269 | Classic studies |
| `ising_3d_mc` | 3D | High | 4.511 | Realistic systems |
| `simple_ising_3d` | 3D | Low | 4.511 | Quick tests |
| `ising_4d_mc` | 4D | Very High | ~6.7 | Dimensionality |
| `critical_temperature_analysis` | 2D/3D | Advanced | Various | Research |
| `simple_tc_calculation` | Any | Low | Various | Quick T_c |
| `cluster_vs_metropolis` | 2D/3D | Medium | Various | Benchmarking |
| `wolff_algorithm_guide` | 2D/3D | Medium | Various | Learning clusters |

---

## Key Concepts

### Phase Transition

- **T < T_c:** Ordered (ferromagnetic) phase, spontaneous magnetization
- **T = T_c:** Critical point, power-law correlations, diverging susceptibility
- **T > T_c:** Disordered (paramagnetic) phase, no net magnetization

### Observables

1. **Magnetization:** M = (1/N) Σᵢ sᵢ
2. **Energy:** E = -J Σ_<ij> sᵢsⱼ
3. **Susceptibility:** χ = N(⟨M²⟩ - ⟨M⟩²)/kT
4. **Specific Heat:** C = (⟨E²⟩ - ⟨E⟩²)/(kT²)

### Algorithms

**Metropolis-Hastings:**
- Single spin flip per step
- Accept/reject with probability min(1, exp(-ΔE/kT))
- Simple but slow near T_c

**Wolff Cluster:**
- Flip entire clusters of aligned spins
- Greatly reduces autocorrelation time
- Essential for accurate T_c determination

---

## Customization

### Change System Size

```rust
let l = 32;  // 32×32 for 2D, 32×32×32 for 3D
```

Larger systems → better finite-size scaling, but slower

### Adjust Temperature Range

```rust
let t_min = 1.0;
let t_max = 5.0;
let n_temps = 50;
```

### Equilibration and Sampling

```rust
let n_equilibration = 10000;  // Thermalization
let n_measurements = 50000;   // Sampling
let sample_interval = 10;     // Decorrelation
```

---

## Performance Tips

1. **Use release mode:** Essential for MC simulations
   ```bash
   cargo run --example <name> --release
   ```

2. **System size:** Start small (L=16-32), then scale up

3. **Near T_c:** Increase sampling for accurate observables

4. **Cluster algorithms:** Use Wolff for T ≈ T_c

---

## References

1. Ising, E. (1925). "Beitrag zur Theorie des Ferromagnetismus." *Z. Phys.* **31**, 253.
2. Onsager, L. (1944). "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition." *Phys. Rev.* **65**, 117.
3. Wolff, U. (1989). "Collective Monte Carlo Updating for Spin Systems." *Phys. Rev. Lett.* **62**, 361.
4. Swendsen, R. H. & Wang, J.-S. (1987). "Nonuniversal critical dynamics in Monte Carlo simulations." *Phys. Rev. Lett.* **58**, 86.
5. Newman, M. E. J. & Barkema, G. T. (1999). *Monte Carlo Methods in Statistical Physics*. Oxford University Press.

