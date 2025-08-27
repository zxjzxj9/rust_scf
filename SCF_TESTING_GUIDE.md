# SCF Testing Guide

## Overview

This guide provides comprehensive instructions for using the SCF (Self-Consistent Field) package to run various SCF calculations and tests. The SCF package supports both restricted (RHF) and unrestricted (UHF) Hartree-Fock calculations, geometry optimization, and various molecular systems.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Usage](#basic-usage)
3. [Configuration Format](#configuration-format)
4. [Command Line Options](#command-line-options)
5. [Test Examples](#test-examples)
6. [Geometry Optimization](#geometry-optimization)
7. [Spin-Polarized Calculations](#spin-polarized-calculations)
8. [Performance and Troubleshooting](#performance-and-troubleshooting)

## Installation and Setup

### Prerequisites
- Rust (latest stable version)
- Cargo package manager
- Internet connection (for automatic basis set fetching)

### Building the SCF Package

```bash
cd /path/to/rust_scf/scf
cargo build --release
```

For development and testing:
```bash
cargo build
```

## Basic Usage

### Running SCF Calculations

The basic command structure is:

```bash
cargo run -- --config-file <config.yaml> [OPTIONS]
```

For release builds:
```bash
cargo run --release -- --config-file <config.yaml> [OPTIONS]
```

### Quick Test Examples

```bash
# Simple H2O calculation
cargo run -- --config-file example/h2o.yaml

# H2O with geometry optimization
cargo run -- --config-file example/h2o_optim.yaml

# Triplet oxygen molecule
cargo run -- --config-file example/o2_triplet.yaml

# Methyl radical (doublet state)
cargo run -- --config-file example/ch3_radical.yaml
```

## Configuration Format

### YAML Configuration Structure

SCF calculations are configured using YAML files with the following structure:

```yaml
# Molecular geometry
geometry:
  - element: O
    coords: [0.0, 0.0, 0.0]
  - element: H
    coords: [0.757, 0.586, 0.0]
  - element: H
    coords: [-0.757, 0.586, 0.0]

# Basis set specifications
basis_sets:
  O: "6-31g"
  H: "6-31g"

# SCF calculation parameters
scf_params:
  density_mixing: 0.5
  max_cycle: 100
  diis_subspace_size: 8
  convergence_threshold: 1e-6

# Optional: Geometry optimization
optimization:
  enabled: true
  algorithm: "cg"  # or "sd"
  max_iterations: 50
  convergence_threshold: 1e-4
  step_size: 0.1

# Optional: System properties
charge: 0        # Total molecular charge
multiplicity: 1  # Spin multiplicity (2S+1)
```

### Configuration Examples

#### Basic Water Molecule (RHF)
```yaml
geometry:
  - element: O
    coords: [0.000000, 0.000000, 0.000000]
  - element: H
    coords: [0.757000, 0.586000, 0.000000]
  - element: H
    coords: [-0.757000, 0.586000, 0.000000]

basis_sets:
  O: "6-31g"
  H: "6-31g"

scf_params:
  density_mixing: 0.5
  max_cycle: 100
  convergence_threshold: 1e-6
```

#### Triplet Oxygen (UHF)
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

multiplicity: 3  # Triplet state
charge: 0
```

## Command Line Options

### Basic Options

- `--config-file, -c`: Path to YAML configuration file (default: `config.yaml`)
- `--output, -o`: Output file path (default: stdout)

### SCF Parameter Overrides

- `--density-mixing`: Override density mixing parameter
- `--max-cycle`: Override maximum SCF cycles
- `--diis-subspace-size`: Override DIIS subspace size
- `--convergence-threshold`: Override SCF convergence threshold

### System Properties

- `--charge`: Override molecular charge
- `--multiplicity`: Override spin multiplicity
- `--spin-polarized`: Force spin-polarized (UHF) calculation

### Geometry Optimization

- `--optimize`: Enable geometry optimization
- `--opt-algorithm`: Optimization algorithm ("cg" or "sd")
- `--opt-max-iterations`: Maximum optimization iterations
- `--opt-convergence`: Optimization convergence threshold
- `--opt-step-size`: Optimization step size

### Example Commands

```bash
# Override SCF parameters
cargo run -- --config-file example/h2o.yaml --density-mixing 0.3 --max-cycle 150

# Force UHF calculation
cargo run -- --config-file example/h2o.yaml --spin-polarized --multiplicity 3

# Enable optimization with custom parameters
cargo run -- --config-file example/h2o.yaml --optimize --opt-algorithm cg --opt-max-iterations 100

# Save output to file
cargo run -- --config-file example/h2o.yaml --output h2o_results.log
```

## Test Examples

### 1. Simple Molecules

#### Hydrogen Molecule
```yaml
geometry:
  - element: H
    coords: [0.0, 0.0, 0.0]
  - element: H
    coords: [0.0, 0.0, 1.4]

basis_sets:
  H: "sto-3g"

scf_params:
  density_mixing: 0.5
  max_cycle: 50
  convergence_threshold: 1e-6
```

#### Water Molecule
Use the provided `example/h2o.yaml` configuration.

### 2. Running Test Suites

#### Quick Test
```bash
# Simple H2 test
cargo run -- --config-file test_config.yaml
```

#### Comprehensive Multiplicity Tests
```bash
# Run all spin-polarized tests
./test_multiplicity.sh
```

#### Custom Test Configuration
Create a test configuration and run:
```bash
cat > my_test.yaml << EOF
geometry:
  - element: H
    coords: [0.0, 0.0, 0.0]
  - element: H
    coords: [0.0, 0.0, 1.4]
basis_sets:
  H: "sto-3g"
scf_params:
  max_cycle: 10
  convergence_threshold: 1e-4
EOF

cargo run -- --config-file my_test.yaml
```

### 3. Validation Tests

#### Energy Verification
```bash
# Compare with reference energies
cargo run -- --config-file example/h2o.yaml | grep "Total energy"
```

#### Force Validation
```bash
# Run force demonstration
cargo run --example force_demo
```

## Geometry Optimization

### Basic Optimization

Enable optimization in your YAML configuration:

```yaml
optimization:
  enabled: true
  algorithm: "cg"  # Conjugate Gradient
  max_iterations: 50
  convergence_threshold: 1e-4
  step_size: 0.1
```

Or use command line:
```bash
cargo run -- --config-file example/h2o.yaml --optimize
```

### Optimization Examples

#### Water Molecule Optimization
```bash
# Using provided optimization example
cargo run -- --config-file example/h2o_optim.yaml

# With custom parameters
cargo run -- --config-file example/h2o.yaml \
    --optimize \
    --opt-algorithm cg \
    --opt-max-iterations 100 \
    --opt-convergence 1e-5
```

#### Standalone Optimization Example
```bash
# Run the comprehensive H2O optimization example
cargo run --example h2o_geometry_optimization
```

### Optimization Algorithms

1. **Conjugate Gradient (cg)**: Recommended for most cases
2. **Steepest Descent (sd)**: More robust but slower convergence

### Expected Results

For H2O optimization:
- O-H bond length: ~1.8 bohr (~0.96 Å)
- H-O-H angle: ~104.5°
- Energy lowering: Several kcal/mol from distorted geometry

## Spin-Polarized Calculations

### When to Use UHF

Use unrestricted Hartree-Fock (UHF) for:
- Open-shell systems (radicals)
- Systems with multiplicity > 1
- Charged systems (ions)

### Configuration Examples

#### Methyl Radical (Doublet)
```yaml
geometry:
  - element: C
    coords: [0.0, 0.0, 0.0]
  - element: H
    coords: [1.079, 0.0, 0.0]
  - element: H
    coords: [-0.539, 0.934, 0.0]
  - element: H
    coords: [-0.539, -0.934, 0.0]

basis_sets:
  C: "6-31g"
  H: "6-31g"

scf_params:
  density_mixing: 0.4
  max_cycle: 150
  convergence_threshold: 1e-6

multiplicity: 2  # Doublet state
charge: 0
```

#### Oxygen Molecule (Triplet)
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

multiplicity: 3  # Triplet state
charge: 0
```

### Running Spin-Polarized Tests

```bash
# Automatic detection (based on multiplicity/charge)
cargo run -- --config-file example/o2_triplet.yaml

# Force UHF calculation
cargo run -- --config-file example/h2o.yaml --spin-polarized --multiplicity 3

# Test suite for various multiplicities
./test_multiplicity.sh
```

### Understanding UHF Output

UHF calculations provide separate energy levels for alpha and beta electrons:
```
Alpha electrons:
  Level 1: -20.5234 au
  Level 2: -1.3456 au
  ...
Beta electrons:
  Level 1: -20.5234 au
  Level 2: -1.2345 au
  ...
```

## Performance and Troubleshooting

### Performance Optimization

#### Compilation
```bash
# Use release mode for production runs
cargo build --release
cargo run --release -- --config-file example/h2o.yaml
```

#### Parallel Execution
The code uses Rayon for parallelization. Set thread count:
```bash
export RAYON_NUM_THREADS=4
cargo run -- --config-file example/h2o.yaml
```

#### Benchmark Tests
```bash
# Run parallel benchmarks
cargo run --example benchmark_parallel
```

### Common Issues and Solutions

#### 1. SCF Convergence Problems

**Symptoms**: SCF doesn't converge within max_cycle iterations

**Solutions**:
```bash
# Lower density mixing
cargo run -- --config-file example/problem.yaml --density-mixing 0.1

# Increase max cycles
cargo run -- --config-file example/problem.yaml --max-cycle 300

# Tighter convergence for difficult cases
cargo run -- --config-file example/problem.yaml --convergence-threshold 1e-8
```

#### 2. Basis Set Fetch Failures

**Symptoms**: Network errors when fetching basis sets

**Solutions**:
- Check internet connection
- Use local basis set files in `tests/basis_sets/`
- Verify basis set names (e.g., "6-31g", "sto-3g")

#### 3. Geometry Optimization Not Converging

**Symptoms**: Optimization exceeds max iterations

**Solutions**:
```bash
# Increase iterations
cargo run -- --config-file example/h2o.yaml --optimize --opt-max-iterations 200

# Relax convergence
cargo run -- --config-file example/h2o.yaml --optimize --opt-convergence 1e-3

# Try steepest descent
cargo run -- --config-file example/h2o.yaml --optimize --opt-algorithm sd
```

#### 4. Force Accuracy Issues

**Note**: Current implementation has limitations in force derivatives (see memory note about approximations).

**Solutions**:
- Use relaxed force convergence thresholds
- Expect larger errors (~0.04-0.1 au) for complex basis functions
- Tighter SCF convergence can help

### Memory and Disk Usage

- **Small molecules** (< 10 atoms): Minimal requirements
- **Larger systems**: Memory usage scales as O(N²-N⁴) depending on operation
- **Basis set caching**: Automatically caches downloaded basis sets
- **Output files**: Can be large for verbose calculations

### Performance Tips

1. **Use release builds** for production calculations
2. **Start with smaller basis sets** (STO-3G) for testing
3. **Monitor convergence** patterns to optimize parameters
4. **Use appropriate multiplicity** to avoid unnecessary UHF calculations
5. **Cache basis sets** locally to avoid repeated downloads

### Debugging Output

Enable debug output:
```bash
# Debug information is automatically included
cargo run -- --config-file example/h2o.yaml
```

The program provides detailed logging including:
- Basis set fetching status
- SCF cycle information
- Geometry optimization progress
- Final energies and molecular orbitals

## Advanced Usage

### Custom Basis Sets

Place custom basis set files in `tests/basis_sets/` directory:
```
tests/basis_sets/
├── 6-31g.h.nwchem
├── 6-31g.c.nwchem
├── sto-3g.h.nwchem
└── sto-3g.o.nwchem
```

### Integration with Other Tools

#### Energy Verification Scripts
```bash
# Verify H2 energy against reference
python verify_h2_energy.py
```

#### Symbolic Validation
```bash
cd symbol/
pip install -r requirements.txt
python kinetic_validate.py
```

### Example Workflows

#### 1. New Molecule Testing Workflow
```bash
# 1. Create configuration
cat > new_molecule.yaml << EOF
geometry:
  - element: ...
# ... configuration
EOF

# 2. Quick test with small basis
# Edit config to use "sto-3g"
cargo run -- --config-file new_molecule.yaml

# 3. Production run with larger basis
# Edit config to use "6-31g"
cargo run --release -- --config-file new_molecule.yaml

# 4. Optimization if needed
cargo run --release -- --config-file new_molecule.yaml --optimize
```

#### 2. Method Validation Workflow
```bash
# 1. Run with different parameters
for mixing in 0.1 0.3 0.5 0.7; do
    echo "Testing density_mixing = $mixing"
    cargo run -- --config-file test.yaml --density-mixing $mixing
done

# 2. Compare with reference calculations
# 3. Document results
```

## Conclusion

This guide covers the essential aspects of running SCF tests and calculations. For more specific examples, refer to:
- `/example/` directory for configuration files
- `/examples/` directory for Rust code examples
- `README_H2O_OPTIMIZATION.md` for detailed optimization examples
- Source code documentation for implementation details

Remember to use appropriate convergence thresholds and basis sets for your specific application, and always validate results against known benchmarks when possible.
