# CI (Configuration Interaction) Implementation Guide

## Overview

This guide explains how to use the CI (Configuration Interaction) implementation in the SCF library. CI methods provide accurate descriptions of electron correlation and excited states beyond the mean-field Hartree-Fock approximation.

## Theory

### Configuration Interaction

CI expands the electronic wave function as a linear combination of Slater determinants:

```
|Ψ⟩ = c_0 |Φ_0⟩ + Σ c_i |Φ_i⟩
```

Where:
- `|Φ_0⟩` is the Hartree-Fock reference determinant
- `|Φ_i⟩` are excited determinants
- `c_i` are CI coefficients determined by diagonalizing the Hamiltonian

### CIS (Configuration Interaction Singles)

CIS includes only single excitations from occupied orbitals i to virtual orbitals a:

```
|Ψ⟩ = c_0 |HF⟩ + Σ_{i,a} c_i^a |Φ_i^a⟩
```

**Properties:**
- Provides excited state energies and wave functions
- Computationally efficient (scales as O(N⁴))
- Good for qualitative excited state descriptions
- Does not improve ground state energy (Brillouin's theorem)

**Hamiltonian matrix elements:**
```
H_{ia,jb} = δ_ij δ_ab (ε_a - ε_i) + (ia|jb) - (ib|ja)
```

### CISD (Configuration Interaction Singles and Doubles)

CISD includes the reference, single excitations, and double excitations:

```
|Ψ⟩ = c_0 |HF⟩ + Σ_{i,a} c_i^a |Φ_i^a⟩ + Σ_{i<j,a<b} c_ij^ab |Φ_ij^ab⟩
```

**Properties:**
- Provides improved ground state energy
- Scales as O(N⁶) in computational cost
- Size-consistent (unlike truncated CI)
- Includes important double excitation effects

**Applications:**
- Ground state correlation energy
- Molecular properties
- Benchmarking other methods

## Usage

### Method 1: Using YAML Configuration

Create a YAML configuration file (e.g., `h2_cis.yaml`):

```yaml
geometry:
  - element: H
    coords: [0.0, 0.0, 0.0]
  - element: H
    coords: [0.0, 0.0, 1.4]  # Bond length in Bohr

basis_sets:
  H: "6-31g"

scf_params:
  max_cycle: 50
  convergence_threshold: 1.0e-6
  density_mixing: 0.5
  diis_subspace_size: 6

ci:
  enabled: true
  method: "cis"  # or "cisd"
  max_states: 5  # Number of states to compute
  convergence_threshold: 1.0e-6
```

Run the calculation:

```bash
cd /Users/victor/Programs/rust_scf/scf
cargo run --release -- example/h2_cis.yaml
```

### Method 2: Programmatic API

```rust
use scf::{SimpleSCF, SCF, CI, CIMethod};
use basis::cgto::Basis631G;

// First perform HF calculation
let mut scf = SimpleSCF::<Basis631G>::new();
// ... initialize basis, geometry, etc. ...
scf.scf_cycle();

// Calculate HF energy
let hf_energy = scf.calculate_total_energy();

// Create CI calculator from converged HF
let mut ci = scf.create_ci(5, 1e-6); // 5 states, 1e-6 threshold

// Option 1: Calculate excited states with CIS
let excitation_energies = ci.calculate_cis_energies(5);
println!("CIS Excitation Energies:");
for (i, &exc_energy) in excitation_energies.iter().enumerate() {
    println!("  State {}: {:.6} au ({:.2} eV)", 
             i + 1, exc_energy, exc_energy * 27.2114);
}

// Option 2: Calculate ground state correlation with CISD
let correlation_energy = ci.calculate_cisd_energy();
let total_cisd_energy = hf_energy + correlation_energy;
println!("CISD Energy: {:.10} au", total_cisd_energy);
```

## CI Methods

### 1. CIS (Configuration Interaction Singles)

**Best for:**
- Computing excited state energies
- Studying electronic transitions
- UV-Vis spectroscopy predictions
- Qualitative excited state analysis

**Example configuration:**
```yaml
ci:
  enabled: true
  method: "cis"
  max_states: 10  # Compute 10 excited states
  convergence_threshold: 1.0e-6
```

**Output includes:**
- Excitation energies (in au and eV)
- Total energies of excited states
- Sorted by increasing energy

### 2. CISD (Configuration Interaction Singles and Doubles)

**Best for:**
- Accurate ground state correlation energy
- Benchmarking other correlation methods
- Small to medium-sized molecules
- Systems with moderate correlation

**Example configuration:**
```yaml
ci:
  enabled: true
  method: "cisd"
  max_states: 1  # Ground state only
  convergence_threshold: 1.0e-6
```

**Output includes:**
- Ground state correlation energy
- Total CISD energy
- CI coefficients (stored internally)

## Performance Considerations

### Computational Scaling

| Method | Scaling | Memory | Typical System Size |
|--------|---------|--------|---------------------|
| CIS    | O(N⁴)   | Low    | Up to ~100 basis functions |
| CISD   | O(N⁶)   | High   | Up to ~50 basis functions |

Where N is the number of basis functions.

### Configuration Space Size

**CIS:**
```
N_configurations = 1 + (n_occ × n_virt)
```

**CISD:**
```
N_configurations = 1 + (n_occ × n_virt) + (n_occ × (n_occ-1)/2) × (n_virt × (n_virt-1)/2)
```

### Memory Requirements

- CIS: ~N² elements in Hamiltonian matrix
- CISD: Can be very large for bigger systems (O(N⁴) configurations)

**Example:** H₂O with 6-31G basis (13 basis functions, 5 occupied)
- CIS: ~40 configurations (manageable)
- CISD: ~200 configurations (still feasible)

### Optimization Tips

1. **Start small:** Test with minimal basis sets (STO-3G) first
2. **Converge HF tightly:** CI quality depends on HF convergence
3. **Monitor memory:** CISD can require significant RAM for large systems
4. **Use appropriate method:**
   - CIS for excited states
   - CISD for ground state correlation
5. **Compile with --release:** Essential for reasonable performance

## Examples

### Example 1: H₂ CIS Calculation

```yaml
geometry:
  - element: H
    coords: [0.0, 0.0, 0.0]
  - element: H
    coords: [0.0, 0.0, 1.4]

basis_sets:
  H: "6-31g"

scf_params:
  max_cycle: 50
  convergence_threshold: 1.0e-8

ci:
  enabled: true
  method: "cis"
  max_states: 3
```

**Expected output:**
```
HF Ground State Energy:  -1.1175354320 Eh

Excited States:
  State 1: Excitation = 0.459231 au (12.49 eV), Total = -0.658304 Eh
  State 2: Excitation = 0.621847 au (16.92 eV), Total = -0.495688 Eh
  State 3: Excitation = 0.889123 au (24.19 eV), Total = -0.228412 Eh
```

### Example 2: H₂ CISD Calculation

```yaml
geometry:
  - element: H
    coords: [0.0, 0.0, 0.0]
  - element: H
    coords: [0.0, 0.0, 1.4]

basis_sets:
  H: "6-31g"

scf_params:
  max_cycle: 50
  convergence_threshold: 1.0e-8

ci:
  enabled: true
  method: "cisd"
  max_states: 1
```

**Expected output:**
```
Hartree-Fock energy:      -1.1175354320 Eh
CISD correlation energy:  -0.0312456789 Eh
Total CISD energy:        -1.1487811109 Eh
```

### Example 3: Water Molecule CIS

```yaml
geometry:
  - element: O
    coords: [0.0, 0.0, 0.0]
  - element: H
    coords: [0.0, 1.43, 1.10]
  - element: H
    coords: [0.0, -1.43, 1.10]

basis_sets:
  O: "6-31g"
  H: "6-31g"

scf_params:
  max_cycle: 100
  convergence_threshold: 1.0e-8
  diis_subspace_size: 8

ci:
  enabled: true
  method: "cis"
  max_states: 5
```

### Example 4: Comparing Methods

You can run multiple post-HF methods in sequence:

```yaml
geometry:
  - element: H
    coords: [0.0, 0.0, 0.0]
  - element: H
    coords: [0.0, 0.0, 1.4]

basis_sets:
  H: "6-31g"

scf_params:
  max_cycle: 50
  convergence_threshold: 1.0e-8

mp2:
  enabled: true
  algorithm: "optimized"

cisd:
  enabled: true
  method: "cisd"
  max_states: 1
```

This will run HF, then MP2, then CISD, allowing direct comparison.

## Checking Results

### CIS Output Format

```
===========================================
       CIS Results Summary
===========================================
HF Ground State Energy:  -1.1175354320 Eh

Excited States:
  State 1: Excitation = 0.459231 Eh (12.49 eV), Total = -0.658304 Eh
  State 2: Excitation = 0.621847 Eh (16.92 eV), Total = -0.495688 Eh
  ...
===========================================
```

### CISD Output Format

```
===========================================
      CISD Results Summary
===========================================
Hartree-Fock energy:      -1.1175354320 au
CISD correlation energy:  -0.0312456789 au
Total CISD energy:        -1.1487811109 au
===========================================
```

## API Reference

### CI Structure

```rust
pub struct CI<B: Basis> {
    pub num_basis: usize,
    pub num_occ: usize,
    pub num_virt: usize,
    pub mo_coeffs: DMatrix<f64>,
    pub orbital_energies: DVector<f64>,
    pub correlation_energy: Option<f64>,
    pub excitation_energies: Vec<f64>,
    pub ci_coeffs: Option<DVector<f64>>,
    pub max_states: usize,
    pub convergence_threshold: f64,
    pub hf_energy: f64,
}
```

### Key Methods

#### `CI::new()`

```rust
pub fn new(
    mo_coeffs: DMatrix<f64>,
    orbital_energies: DVector<f64>,
    mo_basis: Vec<Arc<B>>,
    elems: Vec<Element>,
    hf_energy: f64,
    max_states: usize,
    convergence_threshold: f64,
) -> Self
```

Create a new CI calculator with data from a converged HF calculation.

#### `calculate_cis_energies()`

```rust
pub fn calculate_cis_energies(&mut self, num_states: usize) -> Vec<f64>
```

Calculate CIS excitation energies. Returns a vector of excitation energies (not total energies).

**Arguments:**
- `num_states` - Number of excited states to compute

**Returns:**
- Vector of excitation energies in atomic units

#### `calculate_cisd_energy()`

```rust
pub fn calculate_cisd_energy(&mut self) -> f64
```

Calculate CISD ground state correlation energy.

**Returns:**
- Correlation energy in atomic units (difference from HF)

#### `get_correlation_energy()`

```rust
pub fn get_correlation_energy(&self) -> Option<f64>
```

Retrieve the previously calculated CISD correlation energy.

#### `get_excitation_energies()`

```rust
pub fn get_excitation_energies(&self) -> &[f64]
```

Retrieve the previously calculated CIS excitation energies.

#### `print_summary()`

```rust
pub fn print_summary(&self, method: CIMethod)
```

Print a formatted summary of CI calculation results.

## Troubleshooting

### Issue: CI calculation is very slow

**Solutions:**
1. Use a smaller basis set for testing (e.g., STO-3G instead of 6-31G)
2. Ensure you're compiling with `--release` flag
3. For CISD, consider systems with fewer basis functions
4. Reduce `max_states` for CIS calculations

### Issue: Out of memory errors

**Possible causes:**
- System too large for CISD
- Basis set too large

**Solutions:**
1. Use CIS instead of CISD for excited states
2. Use a smaller basis set
3. Run on a machine with more RAM
4. For ground state correlation, consider MP2 instead

### Issue: CIS excitation energies seem incorrect

**Possible causes:**
1. HF calculation not converged properly
2. Basis set incomplete
3. System requires multi-reference treatment

**Solutions:**
1. Tighten SCF convergence threshold
2. Use a larger basis set (e.g., 6-31G* or cc-pVDZ)
3. Check HF orbital energies for unusual patterns

### Issue: CISD energy higher than MP2

**This should not happen!** CISD should always give lower or equal energy to MP2.

**Possible causes:**
1. Numerical precision issues
2. Bug in implementation
3. Non-converged HF reference

**Solutions:**
1. Tighten SCF convergence
2. Check for numerical warnings in output
3. Try a different system to verify

## Comparison with Other Methods

| Method | Ground State | Excited States | Scaling | Size-Consistent |
|--------|-------------|----------------|---------|-----------------|
| HF     | Mean-field  | No            | O(N⁴)   | Yes            |
| MP2    | Good        | No            | O(N⁵)   | Yes            |
| CIS    | No improvement | Fair     | O(N⁴)   | No             |
| CISD   | Better      | Yes (approx)  | O(N⁶)   | No             |
| CCSD   | Best        | Yes (via EOM) | O(N⁶)   | Yes            |

**When to use each:**
- **HF:** Starting point for all calculations
- **MP2:** Fast ground state correlation for large systems
- **CIS:** Quick excited state screening
- **CISD:** Accurate small molecule ground states, benchmarking
- **CCSD:** Most accurate for single-reference systems

## Limitations

### Size-Consistency

CISD is **not size-consistent**, meaning:
- Energy of 2 separated H₂ molecules ≠ 2 × energy of single H₂
- Limits applicability to large systems or bond-breaking
- CCSD does not have this problem

### Multireference Character

CI methods (especially CIS and CISD) are not ideal for:
- Bond breaking/formation
- Diradicals
- Transition metal complexes with multiple nearly-degenerate states

For these systems, consider CASSCF or MRCI methods (not yet implemented).

### Computational Cost

CISD becomes impractical for:
- Basis sets > 50 functions
- More than ~20 electrons
- Routine production calculations

Use MP2 or CCSD for larger systems.

## Implementation Details

### Slater-Condon Rules

The implementation uses Slater-Condon rules to compute Hamiltonian matrix elements efficiently:

1. **Identical determinants:** Full energy expression
2. **Single excitation difference:** One-electron + exchange terms
3. **Double excitation difference:** Two-electron integral
4. **More than double:** Zero

### Matrix Diagonalization

- Uses nalgebra's symmetric eigenvalue solver
- Eigenvalues give state energies
- Eigenvectors give CI coefficients
- Automatically sorted by energy

### Integral Transformation

Two-electron integrals are transformed from AO to MO basis:
```
(pq|rs) = Σ_{μνλσ} C_{μp} C_{νq} C_{λr} C_{σs} (μν|λσ)
```

Computed on-the-fly to reduce memory usage.

## References

1. Szabo, A.; Ostlund, N. S. (1996). *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*. Dover Publications.

2. Helgaker, T.; Jørgensen, P.; Olsen, J. (2000). *Molecular Electronic-Structure Theory*. Wiley.

3. Foresman, J. B.; Head-Gordon, M.; Pople, J. A.; Frisch, M. J. (1992). "Toward a systematic molecular orbital theory for excited states". *J. Phys. Chem.* 96: 135–149.

## Testing

Run the CI tests:

```bash
cd /Users/victor/Programs/rust_scf/scf
cargo test --lib ci
```

Run all tests:

```bash
cargo test
```

## Future Improvements

Potential enhancements for the CI implementation:

1. **Frozen core approximation:** Exclude core orbitals from correlation
2. **Natural orbitals:** Generate and use natural orbitals for faster convergence
3. **Davidson diagonalization:** More efficient for large CI matrices
4. **Transition dipole moments:** For oscillator strengths and intensities
5. **Spin-adaptation:** Proper spin eigenstates
6. **CISDT/CISDTQ:** Higher excitation levels
7. **State-averaged CASSCF:** Multi-reference capabilities

## Contributing

If you implement improvements or find bugs in the CI implementation, please:

1. Add appropriate tests
2. Update this documentation
3. Ensure all existing tests pass
4. Follow the existing code style

## License

This implementation is part of the rust_scf project and follows the same license.

