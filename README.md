# HowTos: Perform a Hartree-Fock SCF calculation

---

![](./imgs/scf_001.svg)

---
## 1. Input molecular geometry

- Specify atomic coordinates (usually in Cartesian coordinates)

- Define nuclear charges for each atom

- Set up molecular charge and spin multiplicity

![](./imgs/scf_002.svg)

## 2. Choose a basis set

- Select appropriate atomic orbital basis functions (e.g., STO-3G, 3-21G, 6-31G*)
 "
- Construct the overlap matrix S and kinetic energy matrix T

- Calculate nuclear-electron attraction integrals V

![](./imgs/scf_003.svg)

## 3. Form initial guess for density matrix P

- Often use extended Hückel theory or superposition of atomic densities

- Ensure the guess satisfies electron count constraints

- Convert density matrix to appropriate basis representation

![](./imgs/scf_004.svg)


## 4. Begin SCF iteration procedure:

![](./imgs/scf_005.svg)

### a. Construct Fock matrix F

- Calculate one-electron terms (T + V)

- Compute two-electron integrals

- Add Coulomb (J) and exchange (K) contributions



### b. Transform to orthogonal basis

- Solve $$\mathbf{S}\mathbf{X} = \mathbf{X}\mathbf{\lambda}$$ for transformation matrix X

- Form $$\mathbf{F}' = \mathbf{X}^{\dagger}\mathbf{F}\mathbf{X}$$



### c. Solve eigenvalue equation

- Diagonalize F' to get orbital coefficients C' and energies ε

- Transform back: $$\mathbf{C} = \mathbf{X}\mathbf{C}'$$



### d. Form new density matrix

- Populate orbitals according to aufbau principle

- Calculate $$\mathbf{P} = \mathbf{C}\mathbf{n}\mathbf{C}^{\dagger}$$ where n is occupation numbers


## 5. Check for convergence

- Compare energy change between iterations

- Check density matrix difference

- If not converged, return to step 4

- If converged, proceed to final analysis

![](./imgs/scf_006.svg)

## 6. Calculate final properties

- Total electronic energy

- Orbital energies and wavefunctions

- Population analysis

- Dipole moment and other molecular properties

![](./imgs/scf_007.svg)

---

## Molecular Dynamics Simulations

In addition to quantum chemistry calculations, this project includes molecular dynamics (MD) capabilities for classical simulations.

### Argon Melting Simulation

The `md/` package contains examples demonstrating argon phase transitions using the Lennard-Jones potential:

**Quick Demo:**
```bash
cd md
cargo run --example argon_melting_demo
```

**Full Simulation:**
```bash
cd md  
cargo run --example argon_melting
```

**Features:**
- Realistic argon parameters (ε = 120 K, σ = 3.4 Å)
- FCC crystal lattice initialization  
- Temperature ramping from 60K to 180K
- Nosé-Hoover thermostat
- Automatic melting detection via diffusion analysis
- Beautiful formatted output with physical units

**Example Output:**
```
┌────────┬─────────┬──────────┬──────────┬──────────┬──────────┬───────────┐
│   Step │   T (K) │    T_red │   KE_red │   PE_red │  Total_E │ Diff(σ²/τ) │
├────────┼─────────┼──────────┼──────────┼──────────┼──────────┼───────────┤
│      0 │    65.2 │    0.544 │  88.0790 │ -819.6056 │ -731.5266 │  0.000000 │
│   1300 │    96.2 │    0.801 │ 129.8143 │ -690.9616 │ -561.1473 │  0.117908 │
│   2000 │   340.1 │    2.834 │ 459.1283 │ -415.4981 │  43.6302 │  0.104667 │
└────────┴─────────┴──────────┴──────────┴──────────┴──────────┴───────────┘

🔥 System appears to be in LIQUID state (high diffusion)
```

The simulation shows argon melting around 100-120 K, matching experimental observations.

For more details, see [`md/examples/README.md`](md/examples/README.md).