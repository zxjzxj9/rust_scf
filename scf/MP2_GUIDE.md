# MP2 Implementation Guide

## Overview

This guide explains how to use the MP2 (Møller-Plesset perturbation theory, second order) implementation in the SCF library. MP2 is a post-Hartree-Fock method that adds electron correlation corrections to improve the accuracy of quantum chemistry calculations.

## Theory

MP2 calculates the correlation energy using second-order perturbation theory:

```
E_MP2 = Σ_{i<j,a<b} [(ia|jb) * (2*(ia|jb) - (ib|ja))] / (ε_i + ε_j - ε_a - ε_b)
```

Where:
- `i, j` are occupied molecular orbitals
- `a, b` are virtual (unoccupied) molecular orbitals
- `(ia|jb)` are two-electron repulsion integrals in MO basis
- `ε` are orbital energies from Hartree-Fock

The total MP2 energy is:
```
E_total = E_HF + E_MP2_correlation
```

## Usage

### Method 1: Using YAML Configuration

Create a YAML configuration file (e.g., `h2_mp2.yaml`):

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

mp2:
  enabled: true
  algorithm: "optimized"  # or "direct"
```

Run the calculation:

```bash
cd /Users/victor/Programs/rust_scf/scf
cargo run --release -- example/h2_mp2.yaml
```

### Method 2: Programmatic API

```rust
use scf::{SimpleSCF, SCF, MP2};
use basis::cgto::Basis631G;

// First perform HF calculation
let mut scf = SimpleSCF::<Basis631G>::new();
// ... initialize basis, geometry, etc. ...
scf.scf_cycle();

// Calculate HF energy
let hf_energy = scf.calculate_total_energy();

// Create MP2 calculator from converged HF
let mut mp2 = scf.create_mp2();

// Calculate MP2 correlation energy
let correlation_energy = mp2.calculate_mp2_energy();

// Total MP2 energy
let total_mp2_energy = hf_energy + correlation_energy;

println!("HF Energy:          {:.10} Eh", hf_energy);
println!("MP2 Correlation:    {:.10} Eh", correlation_energy);
println!("Total MP2 Energy:   {:.10} Eh", total_mp2_energy);
```

## MP2 Algorithms

The implementation provides two algorithms:

### 1. Optimized Algorithm (Recommended)

```yaml
mp2:
  enabled: true
  algorithm: "optimized"
```

- More efficient for production calculations
- Uses better parallelization
- Suitable for larger molecules

### 2. Direct Algorithm

```yaml
mp2:
  enabled: true
  algorithm: "direct"
```

- Straightforward O(N^8) implementation
- Useful for understanding the algorithm
- Better for very small systems or debugging

## Performance Considerations

1. **Computational Cost**: MP2 scales as O(N^5) with system size, making it significantly more expensive than Hartree-Fock

2. **Memory Requirements**: The implementation computes integrals on-the-fly to reduce memory usage, but large basis sets will still require substantial memory

3. **Parallelization**: Both algorithms use Rayon for parallel computation across occupied orbital pairs

4. **Basis Set Selection**: 
   - Small basis sets (STO-3G): Fast but less accurate
   - Medium basis sets (6-31G, 6-31G*): Good balance
   - Large basis sets (cc-pVDZ, cc-pVTZ): More accurate but much slower

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

mp2:
  enabled: true
  algorithm: "optimized"
```

### Checking Results

After running an MP2 calculation, you should see output like:

```
===========================================
       Starting MP2 Calculation
===========================================
MP2 Initialization:
  Number of basis functions: 5
  Number of occupied orbitals: 1
  Number of virtual orbitals: 4
  Total electrons: 2

Computing MP2 energy for 1 occupied and 4 virtual orbitals...
Processed 1/1 occupied orbital pairs, current E_MP2 = -0.026489431234 Eh

===========================================
        MP2 Results Summary
===========================================
Hartree-Fock energy:       -1.1175354320 au
MP2 correlation energy:    -0.0264894312 au
Total MP2 energy:          -1.1440248632 au
===========================================
```

## API Reference

### MP2 Structure

```rust
pub struct MP2<B: Basis> {
    pub num_basis: usize,      // Number of basis functions
    pub num_occ: usize,        // Number of occupied orbitals
    pub num_virt: usize,       // Number of virtual orbitals
    pub mo_coeffs: DMatrix<f64>,       // MO coefficients
    pub orbital_energies: DVector<f64>, // Orbital energies
    // ... other fields
}
```

### Key Methods

#### `MP2::new()`

```rust
pub fn new(
    mo_coeffs: DMatrix<f64>,
    orbital_energies: DVector<f64>,
    mo_basis: Vec<Arc<B>>,
    elems: Vec<Element>,
) -> Self
```

Create a new MP2 calculator with data from a converged HF calculation.

#### `calculate_mp2_energy()`

```rust
pub fn calculate_mp2_energy(&mut self) -> f64
```

Calculate MP2 correlation energy using the optimized algorithm. Returns the correlation energy in atomic units (Hartree).

#### `calculate_mp2_energy_direct()`

```rust
pub fn calculate_mp2_energy_direct(&mut self) -> f64
```

Calculate MP2 correlation energy using the direct O(N^8) algorithm. Useful for small systems and validation.

#### `get_correlation_energy()`

```rust
pub fn get_correlation_energy(&self) -> Option<f64>
```

Retrieve the previously calculated correlation energy.

#### `print_summary()`

```rust
pub fn print_summary(&self, hf_energy: f64)
```

Print a formatted summary of the MP2 calculation results.

## Troubleshooting

### Issue: MP2 calculation is very slow

**Solutions:**
1. Use a smaller basis set for testing
2. Ensure you're compiling with `--release` flag
3. Try the "optimized" algorithm instead of "direct"
4. Check that Rayon parallelization is working (should use all CPU cores)

### Issue: Correlation energy seems incorrect

**Possible causes:**
1. HF calculation not converged - check SCF convergence first
2. Numerical precision issues - try tighter SCF convergence threshold
3. Basis set incompatibility - ensure basis set is appropriate for the system

### Issue: Out of memory errors

**Solutions:**
1. Use a smaller basis set
2. Run on a machine with more RAM
3. For very large systems, consider implementing disk-based integral storage (not yet implemented)

## Implementation Details

### Integral Transformation

The implementation transforms atomic orbital (AO) integrals to molecular orbital (MO) basis using:

```
(pq|rs) = Σ_{μνλσ} C_{μp} C_{νq} C_{λr} C_{σs} (μν|λσ)
```

This is done on-the-fly to reduce memory requirements.

### Parallelization Strategy

- Outer loop over occupied orbital pairs (i,j) is parallelized
- Inner loops over virtual orbitals (a,b) are computed serially for each (i,j) pair
- Each worker thread processes different (i,j) pairs independently

### Numerical Considerations

1. Small denominators: The implementation checks for near-zero energy denominators and skips those terms
2. Coefficient screening: Very small MO coefficients (< 1e-10) are skipped to improve performance
3. Symmetry: The code properly handles i=j cases to avoid double-counting

## References

1. Møller, C.; Plesset, M. S. (1934). "Note on an Approximation Treatment for Many-Electron Systems". Physical Review. 46 (7): 618–622.

2. Szabo, A.; Ostlund, N. S. (1996). Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory. Dover Publications.

3. Helgaker, T.; Jørgensen, P.; Olsen, J. (2000). Molecular Electronic-Structure Theory. Wiley.

## Future Improvements

Potential enhancements for the MP2 implementation:

1. **Efficient integral transformation**: Implement quarter-transformation or density-fitting methods to reduce O(N^8) cost to O(N^5)

2. **Disk-based storage**: For very large systems, store transformed integrals on disk

3. **Frozen core approximation**: Option to freeze core orbitals in the correlation treatment

4. **Natural orbitals**: Generate and store MP2 natural orbitals

5. **Gradient implementation**: Analytic gradients for MP2 geometry optimization

6. **Spin-unrestricted MP2**: Extend to UMP2 for open-shell systems

7. **Local correlation methods**: Implement local MP2 variants for better scaling

## Testing

Run the MP2 tests:

```bash
cd /Users/victor/Programs/rust_scf/scf
cargo test --lib mp2
```

Run all tests:

```bash
cargo test
```

## Contributing

If you implement improvements or find bugs in the MP2 implementation, please:

1. Add appropriate tests
2. Update this documentation
3. Ensure all existing tests pass
4. Follow the existing code style

## License

This implementation is part of the rust_scf project and follows the same license.


