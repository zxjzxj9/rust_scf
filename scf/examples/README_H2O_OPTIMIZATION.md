# H2O Geometry Optimization Example

This directory contains a comprehensive example for performing geometry optimization on a water (H2O) molecule using the SCF package.

## Files

- `h2o_geometry_optimization.rs` - Standalone Rust example demonstrating H2O optimization
- `../example/h2o_optimization_example.yaml` - YAML configuration for H2O optimization
- This README file

## Running the Examples

### Method 1: Using the Standalone Rust Example

```bash
# From the scf directory
cargo run --example h2o_geometry_optimization
```

This will:
- Set up an H2O molecule with a distorted initial geometry
- Load 6-31G basis sets for O and H (from local files or Basis Set Exchange)
- Run SCF calculation to get initial energy
- Perform conjugate gradient geometry optimization
- Analyze the final geometry and compare with experimental values

### Method 2: Using the YAML Configuration

```bash
# From the scf directory
cargo run -- --config-file example/h2o_optimization_example.yaml --optimize
```

Or with command-line overrides:

```bash
cargo run -- --config-file example/h2o_optimization_example.yaml \
    --optimize \
    --opt-max-iterations 100 \
    --opt-convergence 1e-5
```

## What the Optimization Does

### Initial Setup
- **Molecule**: H2O (water)
- **Initial geometry**: Intentionally distorted from equilibrium
  - O-H distances: ~1.3 bohr (longer than equilibrium ~1.8 bohr)
  - H-O-H angle: ~126° (wider than equilibrium ~104.5°)
- **Basis set**: 6-31G for both O and H atoms
- **Method**: Restricted Hartree-Fock (RHF)

### Optimization Process
1. **SCF Convergence**: First converges the electronic structure for the initial geometry
2. **Force Calculation**: Computes analytical forces on each atom
3. **Geometry Update**: Uses conjugate gradient method to update atomic positions
4. **Iteration**: Repeats until forces are below convergence threshold

### Expected Results
- **Final O-H distance**: ~1.8 bohr (~0.96 Å)
- **Final H-O-H angle**: ~104.5°
- **Energy lowering**: Several kcal/mol from initial distorted geometry

## Understanding the Output

### Geometry Analysis
The example provides detailed analysis including:
- Bond distances in both bohr and Angstrom units
- Bond angles in degrees
- Comparison with experimental values

### Force Analysis
- Initial and final forces on each atom
- Maximum force magnitude
- Convergence status

### Energy Analysis
- Initial and final energies
- Energy lowering due to optimization
- Conversion to kcal/mol for chemical intuition

## Customization

### Modifying the Initial Geometry
Edit the `initial_coords` vector in the Rust example:
```rust
let initial_coords = vec![
    Vector3::new(0.0, 0.0, 0.0),      // O position
    Vector3::new(x1, y1, z1),         // H1 position
    Vector3::new(x2, y2, z2),         // H2 position
];
```

### Changing Optimization Parameters
- **Convergence threshold**: Lower values (e.g., 1e-5) for tighter optimization
- **Max iterations**: Increase for difficult cases
- **Density mixing**: Lower values (0.1-0.3) for better SCF stability
- **Algorithm**: Switch between "cg" (conjugate gradient) and "sd" (steepest descent)

### Using Different Basis Sets
Modify the basis set loading to use different basis sets (requires appropriate files):
```rust
let o_basis = load_basis_for_element("O")?;  // Will fetch 6-31G
```

## Performance Notes

### Computational Requirements
- **Time**: Typically 1-5 minutes on modern hardware
- **Memory**: Modest requirements due to small molecule size
- **Network**: May fetch basis sets from Basis Set Exchange if local files unavailable

### Known Limitations
- Force calculations use approximations (see implementation notes in code)
- Convergence may be slower for highly distorted initial geometries
- Higher-level methods (MP2, DFT) would give more accurate results

## Troubleshooting

### Common Issues
1. **Basis set fetch failures**: Check internet connection or use local basis set files
2. **SCF convergence problems**: Try lower density mixing or more cycles
3. **Optimization not converging**: Increase max iterations or relax convergence threshold

### Force Accuracy
The current implementation uses approximations in force calculations. For production work:
- Consider tighter SCF convergence
- Use more sophisticated basis sets
- Implement exact force derivatives

## Educational Value

This example demonstrates:
- Setting up molecular geometry optimization
- Understanding force-based optimization
- Analyzing molecular structure parameters
- Comparing computational results with experiment
- Working with basis sets and SCF methods

## Next Steps

After running this example, you might want to:
1. Try different initial geometries
2. Optimize larger molecules
3. Experiment with different basis sets
4. Implement analytical frequency calculations
5. Add solvation effects
