# CCSD Implementation Guide

## Overview

This guide explains how to use the CCSD (Coupled Cluster Singles and Doubles) implementation in the SCF library. CCSD is a highly accurate post-Hartree-Fock method that systematically accounts for electron correlation effects through an exponential wavefunction ansatz.

## Theory

### Wavefunction Ansatz

CCSD uses an exponential ansatz for the wavefunction:

```
|Ψ⟩ = exp(T₁ + T₂) |Φ₀⟩
```

Where:
- `|Φ₀⟩` is the Hartree-Fock reference determinant
- `T₁` is the singles excitation operator: T₁ = Σᵢₐ tᵢᵃ aᵢᵃ†aᵢ
- `T₂` is the doubles excitation operator: T₂ = ¼ Σᵢⱼₐᵦ tᵢⱼᵃᵇ aₐ†aᵦ†aⱼaᵢ

### CCSD Equations

The T amplitudes are determined by solving the coupled equations:

**Singles equation:**
```
0 = ⟨Φᵢᵃ| H̄ |Φ₀⟩
```

**Doubles equation:**
```
0 = ⟨Φᵢⱼᵃᵇ| H̄ |Φ₀⟩
```

Where `H̄ = exp(-T₁ - T₂) H exp(T₁ + T₂)` is the similarity-transformed Hamiltonian.

### CCSD Energy

The correlation energy is calculated as:

```
E_CCSD = Σᵢₐ fᵢₐ tᵢᵃ + ¼ Σᵢⱼₐᵦ ⟨ij||ab⟩ tᵢⱼᵃᵇ + ½ Σᵢⱼₐᵦ ⟨ij||ab⟩ tᵢᵃ tⱼᵇ
```

Where:
- `i, j` are occupied molecular orbitals
- `a, b` are virtual (unoccupied) molecular orbitals
- `⟨ij||ab⟩ = (ia|jb) - (ib|ja)` are antisymmetrized two-electron integrals
- `fᵢₐ` are Fock matrix elements

The total CCSD energy is:
```
E_total = E_HF + E_CCSD_correlation
```

## Usage

### Method 1: Using YAML Configuration

Create a YAML configuration file (e.g., `h2_ccsd.yaml`):

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
  convergence_threshold: 1.0e-8
  density_mixing: 0.5
  diis_subspace_size: 6

ccsd:
  enabled: true
  max_iterations: 50
  convergence_threshold: 1.0e-7
```

Run the calculation:

```bash
cd /Users/victor/Programs/rust_scf/scf
cargo run --release -- example/h2_ccsd.yaml
```

### Method 2: Programmatic API

```rust
use scf::{SimpleSCF, SCF, CCSD};
use basis::cgto::Basis631G;

// First perform HF calculation
let mut scf = SimpleSCF::<Basis631G>::new();
// ... initialize basis, geometry, etc. ...
scf.scf_cycle();

// Calculate HF energy
let hf_energy = scf.calculate_total_energy();

// Create CCSD calculator from converged HF
let max_iterations = 50;
let convergence_threshold = 1e-7;
let mut ccsd = scf.create_ccsd(max_iterations, convergence_threshold);

// Solve CCSD equations
let correlation_energy = ccsd.solve();

// Total CCSD energy
let total_ccsd_energy = hf_energy + correlation_energy;

println!("HF Energy:             {:.10} Eh", hf_energy);
println!("CCSD Correlation:      {:.10} Eh", correlation_energy);
println!("Total CCSD Energy:     {:.10} Eh", total_ccsd_energy);

// Check T1 diagnostic for multireference character
let t1_diag = ccsd.t1_diagnostic();
println!("T1 diagnostic:         {:.6}", t1_diag);
if t1_diag > 0.02 {
    println!("Warning: T1 > 0.02 suggests significant multireference character");
}
```

## Configuration Parameters

### CCSD-Specific Parameters

```yaml
ccsd:
  enabled: true                    # Enable CCSD calculation
  max_iterations: 50               # Maximum number of CCSD iterations
  convergence_threshold: 1.0e-7    # Convergence threshold for amplitudes
```

- **enabled**: Set to `true` to perform CCSD calculation after HF converges
- **max_iterations**: Maximum number of iterations for CCSD amplitude updates (default: 50)
- **convergence_threshold**: RMS change in amplitudes below which calculation is considered converged (default: 1e-7)

### Recommended SCF Parameters for CCSD

For accurate CCSD calculations, use tight SCF convergence:

```yaml
scf_params:
  max_cycle: 100
  convergence_threshold: 1.0e-8    # Tight HF convergence recommended
  density_mixing: 0.5
  diis_subspace_size: 8
```

## Performance Considerations

### Computational Cost

1. **Scaling**: CCSD scales as O(N⁶) with system size, making it significantly more expensive than MP2 (O(N⁵))
2. **Memory**: Stores T1 (N_occ × N_virt) and T2 (N_occ² × N_virt²) amplitudes
3. **Iterations**: Typically requires 15-30 iterations for convergence

### Memory Requirements

Approximate memory usage for amplitudes:

```
Memory (GB) ≈ 8 × (N_occ × N_virt + N_occ² × N_virt²) / 1_073_741_824
```

For a system with 10 occupied and 40 virtual orbitals:
- T1: 10 × 40 = 400 elements
- T2: 10² × 40² = 160,000 elements
- Total: ~1.3 MB (very manageable)

For a system with 50 occupied and 200 virtual orbitals:
- T1: 50 × 200 = 10,000 elements
- T2: 50² × 200² = 100,000,000 elements
- Total: ~800 MB (substantial but feasible)

### Parallelization

- T2 amplitude updates are parallelized over occupied orbital pairs using Rayon
- Each worker thread processes different (i,j) pairs independently
- Efficient use of multi-core processors

### Basis Set Selection

For CCSD calculations, basis set choice is critical:

- **Minimal (STO-3G)**: Fast but qualitatively incorrect
- **Double-zeta (6-31G)**: Good for testing, moderate accuracy
- **Double-zeta + polarization (6-31G*)**: Better for quantitative results
- **Triple-zeta (cc-pVTZ)**: High accuracy but very expensive
- **Quadruple-zeta (cc-pVQZ)**: Benchmark quality, only for small systems

## Examples

### Water Molecule

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

ccsd:
  enabled: true
  max_iterations: 50
  convergence_threshold: 1.0e-7
```

### Expected Output

After running a CCSD calculation, you should see output like:

```
===========================================
     CCSD Initialization
===========================================
Number of basis functions: 13
Number of occupied orbitals: 5
Number of virtual orbitals: 8
Total electrons: 10
Max iterations: 50
Convergence threshold: 1.00e-7
T1 amplitude matrix size: 5 x 8
T2 amplitude tensor size: 5 x 5 x 8 x 8
Total memory for amplitudes: 0.02 MB
===========================================

Initializing T2 amplitudes with MP2 guess...
  Initialized T2 for 5/5 occupied orbitals
T2 initialization complete.

===========================================
     Starting CCSD Iterations
===========================================
Initial energy (MP2-like): -0.218456123456 Eh

 Iter            E_CCSD                ΔE           RMS(T)
------------------------------------------------------------
    1   -0.220145678901   -0.001689555445    0.0123456789
    2   -0.221234567890   -0.001088888989    0.0087654321
    3   -0.221567890123   -0.000333322233    0.0045678901
...
   18   -0.221987654321   -0.000000000012    0.0000000456

===========================================
       CCSD Converged!
===========================================
Final CCSD correlation energy: -0.221987654321 Eh
Number of iterations: 18
Final RMS change: 0.0000000456
Final energy change: -0.000000000012 Eh
===========================================

===========================================
        CCSD Results Summary
===========================================
Hartree-Fock energy:       -76.0267894567 Eh
CCSD correlation energy:    -0.2219876543 Eh
Total CCSD energy:         -76.2487771110 Eh
===========================================
```

## API Reference

### CCSD Structure

```rust
pub struct CCSD<B: Basis> {
    pub num_basis: usize,           // Number of basis functions
    pub num_occ: usize,             // Number of occupied orbitals
    pub num_virt: usize,            // Number of virtual orbitals
    pub mo_coeffs: DMatrix<f64>,    // MO coefficients
    pub orbital_energies: DVector<f64>, // Orbital energies
    pub t1: DMatrix<f64>,           // T1 amplitudes
    pub t2: Vec<f64>,               // T2 amplitudes (flattened)
    pub correlation_energy: Option<f64>, // Calculated correlation energy
    pub max_iterations: usize,      // Maximum iterations
    pub convergence_threshold: f64, // Convergence threshold
}
```

### Key Methods

#### `CCSD::new()`

```rust
pub fn new(
    mo_coeffs: DMatrix<f64>,
    orbital_energies: DVector<f64>,
    mo_basis: Vec<Arc<B>>,
    elems: Vec<Element>,
    max_iterations: usize,
    convergence_threshold: f64,
) -> Self
```

Create a new CCSD calculator with data from a converged HF calculation.

#### `solve()`

```rust
pub fn solve(&mut self) -> f64
```

Solve the CCSD equations iteratively until convergence. Returns the correlation energy in atomic units (Hartree).

**Algorithm:**
1. Initialize T2 amplitudes using MP2 guess
2. Iteratively update T1 and T2 amplitudes
3. Calculate energy at each iteration
4. Check for convergence based on RMS change in amplitudes
5. Return converged correlation energy

#### `get_correlation_energy()`

```rust
pub fn get_correlation_energy(&self) -> Option<f64>
```

Retrieve the previously calculated correlation energy.

#### `print_summary()`

```rust
pub fn print_summary(&self, hf_energy: f64)
```

Print a formatted summary of the CCSD calculation results.

#### `t1_diagnostic()`

```rust
pub fn t1_diagnostic(&self) -> f64
```

Calculate the T1 diagnostic, which measures the importance of single excitations and can indicate multireference character.

**Interpretation:**
- T1 < 0.02: Single-reference character, CCSD is appropriate
- 0.02 < T1 < 0.05: Borderline case, results may be less reliable
- T1 > 0.05: Significant multireference character, consider multireference methods (CASSCF, MRCI)

## Troubleshooting

### Issue: CCSD calculation is very slow

**Solutions:**
1. Use a smaller basis set for testing (e.g., STO-3G, 6-31G)
2. Ensure you're compiling with `--release` flag
3. Check system size - CCSD becomes prohibitive for > 30 basis functions
4. Verify parallelization is working (should use all CPU cores)

### Issue: CCSD does not converge

**Possible causes and solutions:**
1. **Poor HF reference**: Ensure SCF converges tightly (threshold < 1e-8)
2. **Difficult electronic structure**: Try tighter convergence threshold or more iterations
3. **Multireference character**: Check T1 diagnostic - if > 0.05, CCSD may not be appropriate
4. **Numerical instabilities**: Ensure geometry is reasonable (no very short bond distances)

### Issue: CCSD energy seems incorrect

**Diagnostic steps:**
1. Verify HF energy is correct first
2. Compare with MP2 - CCSD correlation should be more negative than MP2
3. Check T1 diagnostic - high values indicate problems
4. Ensure calculation converged (check final RMS change)
5. Try a smaller, well-characterized system (e.g., H₂, He₂) for validation

### Issue: Out of memory errors

**Solutions:**
1. Use a smaller basis set
2. Run on a machine with more RAM
3. For very large systems (> 100 basis functions), CCSD may not be feasible
4. Consider approximate methods like MP2 or DFT

## Comparison with Other Methods

| Method | Scaling | Accuracy | Use Case |
|--------|---------|----------|----------|
| HF | O(N⁴) | Reference | Starting point |
| MP2 | O(N⁵) | Good | Quick correlation |
| CCSD | O(N⁶) | Excellent | Accurate results |
| CCSD(T) | O(N⁷) | Gold standard | Benchmark (not yet implemented) |

### When to Use CCSD

**Use CCSD when:**
- High accuracy is required
- System size is manageable (< 20 atoms with double-zeta basis)
- Single-reference character (T1 diagnostic < 0.02)
- Comparing with experimental data
- Benchmarking other methods

**Use MP2 instead when:**
- Larger systems (> 30 atoms)
- Quick estimates needed
- Computational resources are limited

**Use multireference methods when:**
- T1 diagnostic > 0.05
- Near-degenerate states
- Bond breaking
- Transition states with multireference character

## Validation

### Reference Data for H₂ (STO-3G, R = 1.4 Bohr)

- HF Energy: ~-1.117 Eh
- MP2 Correlation: ~-0.026 Eh
- CCSD Correlation: ~-0.027 Eh (slightly more negative than MP2)

### Reference Data for H₂O (6-31G, experimental geometry)

- HF Energy: ~-76.027 Eh
- MP2 Correlation: ~-0.204 Eh
- CCSD Correlation: ~-0.222 Eh

## Implementation Details

### Amplitude Initialization

T2 amplitudes are initialized using MP2 guess:

```
tᵢⱼᵃᵇ(0) = (ia|jb) / (εᵢ + εⱼ - εₐ - εᵦ)
```

T1 amplitudes start at zero:

```
tᵢᵃ(0) = 0
```

### Amplitude Update Scheme

**Simplified equations (actual implementation includes more terms):**

**T1 update:**
```
tᵢᵃ(new) = [fᵢₐ + intermediate_terms] / Dᵢᵃ
```

**T2 update:**
```
tᵢⱼᵃᵇ(new) = [⟨ij||ab⟩ + intermediate_terms] / Dᵢⱼᵃᵇ
```

Where:
- Dᵢᵃ = εᵢ - εₐ (singles denominator)
- Dᵢⱼᵃᵇ = εᵢ + εⱼ - εₐ - εᵦ (doubles denominator)

### Convergence Criteria

The calculation is considered converged when:

1. RMS change in amplitudes < threshold
2. Energy change < threshold

```
RMS = √[Σ(tᵢᵃ(new) - tᵢᵃ(old))² + Σ(tᵢⱼᵃᵇ(new) - tᵢⱼᵃᵇ(old))²] / N_amplitudes
```

## Advanced Topics

### T1 and T2 Diagnostics

The T1 diagnostic is defined as:

```
T1 = ||T₁|| / √(2 × N_occ)
```

This provides a measure of the importance of single excitations and can indicate:
- Non-dynamical correlation effects
- Multireference character
- Suitability of single-reference methods

### DIIS Acceleration (Future Enhancement)

Currently, the CCSD implementation uses simple iterative updates. Future versions may implement DIIS (Direct Inversion in the Iterative Subspace) acceleration to improve convergence.

### Perturbative Triples: CCSD(T) (Future Enhancement)

The gold standard of quantum chemistry, CCSD(T), adds a perturbative correction for triple excitations. This is planned for future implementation.

## References

1. **Original CCSD papers:**
   - Purvis, G. D.; Bartlett, R. J. (1982). "A full coupled-cluster singles and doubles model: The inclusion of disconnected triples". J. Chem. Phys. 76: 1910.
   - Raghavachari, K.; Trucks, G. W.; Pople, J. A.; Head-Gordon, M. (1989). "A fifth-order perturbation comparison of electron correlation theories". Chem. Phys. Lett. 157: 479.

2. **Textbooks:**
   - Shavitt, I.; Bartlett, R. J. (2009). Many-Body Methods in Chemistry and Physics. Cambridge University Press.
   - Crawford, T. D.; Schaefer, H. F. (2000). "An Introduction to Coupled Cluster Theory for Computational Chemists". Reviews in Computational Chemistry, Vol. 14.

3. **Review articles:**
   - Bartlett, R. J.; Musiał, M. (2007). "Coupled-cluster theory in quantum chemistry". Rev. Mod. Phys. 79: 291.

## Testing

Run the CCSD tests:

```bash
cd /Users/victor/Programs/rust_scf/scf
cargo test --lib ccsd
```

Run all tests:

```bash
cargo test
```

## Future Improvements

Potential enhancements for the CCSD implementation:

1. **DIIS acceleration**: Implement DIIS for amplitude updates to improve convergence
2. **Frozen core approximation**: Option to freeze core orbitals
3. **CCSD(T)**: Add perturbative triples correction for "gold standard" accuracy
4. **Density-fitting**: Use density-fitting approximation to reduce O(N⁴) integral storage
5. **Local correlation**: Implement local CCSD for better scaling
6. **Analytic gradients**: CCSD gradients for geometry optimization
7. **EOM-CCSD**: Equation-of-motion CCSD for excited states
8. **Spin-unrestricted CCSD**: Extend to UHF references (UCCSD)
9. **Lambda equations**: Solve for Λ amplitudes for properties

## Contributing

If you implement improvements or find bugs in the CCSD implementation, please:

1. Add appropriate tests
2. Update this documentation
3. Ensure all existing tests pass
4. Follow the existing code style
5. Benchmark against reference implementations

## License

This implementation is part of the rust_scf project and follows the same license.

