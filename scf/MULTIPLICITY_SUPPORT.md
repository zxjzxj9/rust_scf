# Spin Multiplicity Support in rust_scf

## Overview

The `simple_spin.rs` module provides full support for spin-polarized calculations with arbitrary multiplicities and molecular charges through an Unrestricted Hartree-Fock (UHF) implementation.

## Current Capabilities

### ✅ Fully Implemented Features

1. **Spin Multiplicity Support**
   - Handles any valid spin multiplicity (2S+1)
   - Singlet (S=0, multiplicity=1): All electrons paired
   - Doublet (S=1/2, multiplicity=2): One unpaired electron  
   - Triplet (S=1, multiplicity=3): Two unpaired electrons
   - Higher multiplicities as allowed by electron count

2. **Charge Support**
   - Handles cations (positive charge)
   - Handles anions (negative charge)
   - Properly accounts for charge in electron counting

3. **Validation**
   - Validates multiplicity is physically possible for given electron count
   - Ensures (N_electrons - unpaired_electrons) is even (pairing requirement)
   - Provides clear error messages for invalid configurations

4. **UHF Implementation**
   - Separate alpha and beta spin orbitals
   - Independent Fock matrices for each spin
   - Proper Coulomb (J) and Exchange (K) integrals with spin coupling
   - Converged SCF cycles with DIIS acceleration

5. **Force Calculations**
   - Nuclear-nuclear repulsion forces
   - Electron-nuclear attraction forces  
   - Two-electron integral derivatives
   - Pulay forces (basis function derivatives)
   - Energy-weighted density matrices for accurate forces

## Usage

### Command Line Interface

```bash
# Using configuration file with multiplicity
cargo run -- --config-file example/o2_triplet.yaml

# Override multiplicity via command line
cargo run -- --config-file example/h2o.yaml --multiplicity 3 --spin-polarized

# Specify charge and multiplicity
cargo run -- --config-file example/molecule.yaml --charge 1 --multiplicity 2 --spin-polarized

# Force spin-polarized calculation (UHF) even for singlets
cargo run -- --config-file example/h2o.yaml --spin-polarized
```

### Configuration File Format

```yaml
geometry:
  - element: O
    coords: [0.0, 0.0, 0.0]
  - element: O
    coords: [0.0, 0.0, 1.2]

basis_sets:
  O: "6-31g"

scf_params:
  density_mixing: 0.3
  max_cycle: 200
  convergence_threshold: 1e-6

# Spin and charge configuration
multiplicity: 3  # Triplet state (2 unpaired electrons)
charge: 0        # Neutral molecule
```

## Examples

### 1. Oxygen Molecule (O₂) - Triplet Ground State
```yaml
# O2 has 16 electrons, ground state is triplet
multiplicity: 3  # 2 unpaired electrons
charge: 0
```

### 2. Methyl Radical (CH₃•) - Doublet
```yaml
# CH3 radical has 7 electrons (one unpaired)
multiplicity: 2  # 1 unpaired electron  
charge: 0
```

### 3. Hydronium Cation (H₃O⁺) - Singlet
```yaml
# H3O+ has 10 electrons (11 protons - 1 charge)
multiplicity: 1  # All paired
charge: 1        # Cation
```

### 4. Hydroxide Anion (OH⁻) - Singlet
```yaml
# OH- has 10 electrons (9 protons + 1 electron)
multiplicity: 1  # All paired
charge: -1       # Anion
```

## Technical Details

### Electron Distribution

The number of alpha and beta electrons is calculated as:
```
total_electrons = nuclear_charge - molecular_charge
unpaired_electrons = multiplicity - 1
n_alpha = (total_electrons + unpaired_electrons) / 2
n_beta = (total_electrons - unpaired_electrons) / 2
```

### Spin Contamination

The current UHF implementation may exhibit spin contamination for non-ground state multiplicities. This is a known limitation of UHF methods where the wavefunction is not an eigenfunction of the S² operator.

### When to Use Spin-Polarized Calculations

The code automatically uses `SpinSCF` (UHF) when:
- Multiplicity > 1 (any unpaired electrons)
- Charge ≠ 0 (ions)
- `--spin-polarized` flag is explicitly set

For neutral singlets, it defaults to `SimpleSCF` (RHF) unless forced otherwise.

## Implementation Notes

### Key Components

1. **SpinSCF Structure** (`simple_spin.rs`)
   - Maintains separate alpha/beta density matrices
   - Independent molecular orbital coefficients
   - Separate energy levels for each spin

2. **Validation** 
   - `validate_multiplicity()` ensures physical validity
   - Checks electron count consistency
   - Validates pairing requirements

3. **SCF Convergence**
   - Uses DIIS acceleration for both spins
   - Density mixing to improve convergence
   - Independent convergence criteria for alpha/beta

### Force Calculation Considerations

Due to approximations in basis set derivative implementations (particularly `dTab_dR` and `dVab_dRbasis`), force calculations may have larger errors (~0.04-0.1 au) for complex basis functions. The overlap derivatives (`dSab_dR`) are exact.

## Testing

Run the included test script to verify multiplicity support:

```bash
cd scf
./test_multiplicity.sh
```

This will test:
1. O₂ triplet (ground state)
2. CH₃ radical (doublet)
3. H₃O⁺ cation (charged singlet)
4. Command-line multiplicity override

## Future Improvements

Potential enhancements could include:
- Restricted Open-shell HF (ROHF) for better spin purity
- Automatic multiplicity determination based on molecular structure
- Spin projection methods to reduce contamination
- More sophisticated initial guess strategies for open-shell systems



