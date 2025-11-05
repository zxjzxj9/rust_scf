# Grand Canonical Monte Carlo (GCMC) Simulations

This directory contains Grand Canonical Monte Carlo simulations for studying systems with variable particle number.

## Overview

In the Grand Canonical ensemble:
- **μ** (Chemical potential): Fixed - controls particle exchange with reservoir
- **V** (Volume): Fixed
- **T** (Temperature): Fixed
- **N** (Number of particles): **Fluctuates**

GCMC is ideal for studying:
- Adsorption/desorption phenomena
- Phase coexistence
- Fluid-vapor equilibria
- Porous materials and interfaces

## Examples

### 1. GCMC Quick Start (`gcmc_quickstart.rs`)

Minimal example to get started with GCMC simulations.

**Run:**
```bash
cd md
cargo run --example gcmc_quickstart --release
```

**Features:**
- Basic GCMC move types (insertion, deletion, displacement)
- Simple Lennard-Jones system
- Chemical potential control
- Educational output

**Runtime:** ~10-30 seconds

**Use when:**
- First learning GCMC
- Quick parameter testing
- Understanding μ-N relationship

---

### 2. GCMC LJ Demo (`gcmc_lj_demo.rs`)

Comprehensive GCMC simulation with detailed analysis.

**Run:**
```bash
cd md
cargo run --example gcmc_lj_demo --release
```

**Features:**
- Full GCMC implementation
- Multiple observables: N, ρ, P, energy
- Histogram analysis
- Acceptance rate monitoring
- Time series output

**Runtime:** ~1-2 minutes

**Use when:**
- Production simulations
- Understanding phase behavior
- Analyzing density fluctuations
- Testing different (μ, V, T) conditions

---

### 3. GCMC Phase Diagram (`gcmc_phase_diagram.rs`)

Systematic exploration of phase space to construct phase diagrams.

**Run:**
```bash
cd md
cargo run --example gcmc_phase_diagram --release
```

**Features:**
- Scans chemical potential at fixed T
- Multiple temperature isotherms
- Identifies phase transitions
- Density vs. μ curves
- Coexistence region detection

**Output:**
- μ-ρ phase diagrams
- Critical point estimation
- Coexistence densities

**Runtime:** ~5-15 minutes (many state points)

**Use when:**
- Mapping out phase behavior
- Finding critical points
- Studying vapor-liquid equilibrium
- Publication-quality phase diagrams

---

## Comparison Table

| Example | Complexity | Features | Runtime | Best For |
|---------|------------|----------|---------|----------|
| `gcmc_quickstart` | Low | Basic GCMC moves | 30s | Learning |
| `gcmc_lj_demo` | Medium | Full analysis | 2min | Production |
| `gcmc_phase_diagram` | High | Phase space scan | 15min | Research |

---

## Key Concepts

### GCMC Move Types

1. **Particle Insertion:**
   - Random position in volume
   - Accept with P = min(1, (V/(N+1)Λ³) exp(-βΔU + βμ))

2. **Particle Deletion:**
   - Random particle removed
   - Accept with P = min(1, (NΛ³/V) exp(-βΔU - βμ))

3. **Particle Displacement:**
   - Standard Metropolis move
   - Accept with P = min(1, exp(-βΔU))

### Chemical Potential (μ)

Controls the "drive" for particles to enter the system:
- **High μ:** More particles, high density
- **Low μ:** Fewer particles, low density
- **At coexistence:** Two phases with different ρ at same μ

### Thermal de Broglie Wavelength

```
Λ = h / √(2πmkT)
```

Sets the quantum concentration scale (often Λ = 1 in reduced units)

### Phase Behavior

**Typical μ-ρ curve:**
- Low μ: Gas phase (low ρ, small fluctuations)
- Intermediate μ: Coexistence plateau (large N fluctuations!)
- High μ: Liquid phase (high ρ, small fluctuations)

**Critical Point:**
- Above T_c: Smooth transition from gas to liquid
- Below T_c: First-order phase transition with coexistence

---

## Physical Observables

### Average Particle Number

```
⟨N⟩ = Σ_N N exp(-β(E_N - μN)) / Z
```

### Density

```
ρ = ⟨N⟩ / V
```

### Particle Number Fluctuations

```
⟨(ΔN)²⟩ = ⟨N²⟩ - ⟨N⟩²
```

Large near phase transitions!

### Pressure (from density)

Can extract pressure using equation of state or thermodynamic integration.

---

## Customization

### Change Chemical Potential Range

```rust
let mu_min = -5.0;
let mu_max = 0.0;
let n_mu_points = 50;
```

### Adjust Move Probabilities

```rust
let p_insert = 0.25;
let p_delete = 0.25;
let p_displace = 0.50;
```

Balance for ~50% acceptance rates

### System Size

```rust
let box_size = 10.0;  // σ units for LJ
```

Larger boxes → better statistics, but slower

### Equilibration

```rust
let n_equilibration = 50000;
let n_production = 200000;
let sample_interval = 100;
```

Near phase transitions need longer equilibration

---

## Practical Tips

### 1. Choosing μ

Start with rough estimates:
- Gas phase: μ < -3 ε (for LJ)
- Liquid phase: μ > -1 ε
- Coexistence: μ ≈ -2 ε (T-dependent)

### 2. Acceptance Rates

**Insertion/Deletion:**
- Too high (>80%): Increase μ range resolution
- Too low (<5%): Adjust μ step size or equilibrate longer

**Displacement:**
- Target: 30-50%
- Adjust step size: `dr_max`

### 3. Finite-Size Effects

- Small boxes: Surface effects dominate
- Large boxes: Better bulk behavior, but slower
- Periodic boundaries: Minimize surface

### 4. Detecting Coexistence

Signs of phase transition:
- Large particle number fluctuations
- Bimodal histograms in N
- Plateau in ρ(μ) curve
- Multiple "jumps" in time series

---

## Troubleshooting

### Problem: No particles inserted

**Cause:** μ too low or strong repulsions

**Fix:**
- Increase μ
- Check LJ parameters (ε, σ)
- Verify energy calculations

### Problem: System fills completely

**Cause:** μ too high

**Fix:**
- Decrease μ
- Check for energy calculation bugs
- Increase box size

### Problem: Large fluctuations, no convergence

**Cause:** Near critical point or coexistence

**Fix:**
- Increase simulation length significantly
- Use histogram reweighting
- Try finite-size scaling analysis

### Problem: Poor acceptance rates

**Cause:** Improper move parameters or μ choice

**Fix:**
- Tune displacement step size
- Adjust μ to avoid very dense/empty regions
- Balance move probabilities

---

## Advanced Topics

### Histogram Reweighting

Reuse simulation data to estimate properties at nearby μ:

```
⟨O⟩_μ' = Σ_i O_i exp(β(μ' - μ)N_i) / Σ_i exp(β(μ' - μ)N_i)
```

### Finite-Size Scaling

Critical exponents from system size dependence:
```
L → ∞: χ ∝ L^(γ/ν)
```

### Parallel Tempering

Run multiple replicas at different μ and exchange configurations.

---

## References

1. Adams, D. J. (1975). "Grand canonical ensemble Monte Carlo for a Lennard-Jones fluid." *Mol. Phys.* **29**, 307.
2. Frenkel, D. & Smit, B. (2001). *Understanding Molecular Simulation* (2nd ed.). Academic Press.
3. Allen, M. P. & Tildesley, D. J. (2017). *Computer Simulation of Liquids* (2nd ed.). Oxford University Press.
4. Panagiotopoulos, A. Z. (1987). "Direct determination of phase coexistence properties of fluids by Monte Carlo simulation in a new ensemble." *Mol. Phys.* **61**, 813.

---

**Ready to explore grand canonical simulations! 🎲**

