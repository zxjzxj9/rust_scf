# YAML Configuration System for MD Simulations

This document describes the YAML-based configuration system for molecular dynamics simulations in the `md` crate. The system provides a flexible, readable way to configure complex MD simulations without hardcoding parameters.

## Quick Start

1. **Run the demo example with a configuration file:**
   ```bash
   cargo run --example yaml_config_demo argon_nvt.yaml
   ```

2. **Available example configurations:**
   - `argon_nvt.yaml` - NVT simulation of argon atoms
   - `argon_npt.yaml` - NPT simulation with temperature ramping  
   - `water_cluster_nvt.yaml` - Small water cluster (explicit positions)
   - `high_pressure_npt.yaml` - High-pressure compression simulation
   - `random_gas_nvt.yaml` - Random gas with many atoms

## Configuration Structure

### Complete Example (argon_nvt.yaml)
```yaml
system:
  positions:
    type: cubic_lattice
    n_per_side: 3
    spacing: 1.2
  velocities:
    type: maxwell_boltzmann
    temperature: 0.8
    seed: 42
  masses: 1.0
  box_setup:
    lengths: [4.5, 4.5, 4.5]
    periodic: true

simulation:
  ensemble:
    type: nvt
    target_temperature: 0.8
    thermostat_coupling: 100.0
    k_boltzmann: 1.0
  integration:
    time_step: 0.002
    total_steps: 20000

potential:
  type: lennard_jones
  epsilon: 1.0
  sigma: 1.0
  cutoff: 2.5

output:
  output_interval: 400
  analysis_interval: 200
  save_trajectory: false
  trajectory_interval: 1000
```

## Configuration Sections

### System Configuration

#### Positions
Configure initial atomic positions using one of three methods:

**1. Cubic Lattice**
```yaml
positions:
  type: cubic_lattice
  n_per_side: 3          # Creates 3³ = 27 atoms
  spacing: 1.2           # Distance between atoms
  offset: [0.5, 0.5, 0.5]  # Optional offset from origin
```

**2. Explicit Coordinates**
```yaml
positions:
  type: explicit
  coords:
    - [0.0, 0.0, 0.0]    # Atom 1
    - [1.0, 0.0, 0.0]    # Atom 2
    - [0.0, 1.0, 0.0]    # Atom 3
```

**3. Random Positions**
```yaml
positions:
  type: random
  n_atoms: 50            # Number of atoms
  min_distance: 1.5      # Minimum separation
```

#### Velocities
Configure initial velocities:

**1. Maxwell-Boltzmann Distribution**
```yaml
velocities:
  type: maxwell_boltzmann
  temperature: 0.8       # Temperature for distribution
  seed: 42              # Optional random seed
```

**2. Explicit Velocities**
```yaml
velocities:
  type: explicit
  velocities:
    - [1.0, 0.0, 0.0]    # Atom 1 velocity
    - [-1.0, 0.0, 0.0]   # Atom 2 velocity
```

**3. Zero Velocities**
```yaml
velocities:
  type: zero
```

#### Masses and Box Setup
```yaml
masses: 1.0              # Uniform mass for all atoms
# OR
masses: [1.0, 16.0, 1.0] # Individual masses

box_setup:
  lengths: [10.0, 10.0, 10.0]  # Box dimensions
  periodic: true               # Periodic boundary conditions
```

### Simulation Configuration

#### NVT Ensemble (Canonical)
```yaml
simulation:
  ensemble:
    type: nvt
    target_temperature: 1.0      # Target temperature
    thermostat_coupling: 100.0   # Nosé-Hoover coupling (Q_t)
    k_boltzmann: 1.0            # Boltzmann constant
```

#### NPT Ensemble (Isothermal-Isobaric)
```yaml
simulation:
  ensemble:
    type: npt
    target_temperature: 1.0      # Target temperature
    target_pressure: 0.5         # Target pressure
    thermostat_coupling: 200.0   # Nosé-Hoover coupling (Q_t)  
    barostat_coupling: 1000.0    # Parrinello-Rahman coupling (Q_p)
    k_boltzmann: 1.0
```

#### Integration and Temperature Scheduling
```yaml
simulation:
  integration:
    time_step: 0.002      # Integration time step
    total_steps: 20000    # Total simulation steps
    
  # Optional temperature ramping
  temperature_schedule:
    initial_temperature: 0.8
    final_temperature: 1.2
    ramp_steps: 10000     # Steps for temperature change
```

### Potential Configuration

#### Lennard-Jones Potential
```yaml
potential:
  type: lennard_jones
  epsilon: 1.0          # Well depth (ε)
  sigma: 1.0            # Collision diameter (σ)  
  cutoff: 2.5           # Cutoff distance (in σ units)
```

### Output Configuration
```yaml
output:
  output_interval: 400      # Print results every N steps
  analysis_interval: 200    # Update analysis every N steps
  save_trajectory: true     # Save trajectory data
  trajectory_interval: 1000 # Save trajectory every N steps
```

## Using the Configuration System in Code

### Loading and Running
```rust
use md::{MdConfig, LennardJones, NoseHooverVerlet, EnsembleConfig};

// Load configuration
let config = MdConfig::from_file("my_simulation.yaml")?;

// Generate system components
let positions = config.generate_positions()?;
let velocities = config.generate_velocities(positions.len())?;
let masses = config.generate_masses(positions.len())?;

// Create potential
let box_lengths = Vector3::new(
    config.system.box_setup.lengths[0],
    config.system.box_setup.lengths[1], 
    config.system.box_setup.lengths[2],
);
let lj = LennardJones::new(epsilon, sigma, box_lengths);

// Create integrator based on ensemble
match &config.simulation.ensemble {
    EnsembleConfig::NVT { target_temperature, thermostat_coupling, k_boltzmann } => {
        let mut integrator = NoseHooverVerlet::new(
            positions, velocities, masses, lj, 
            *thermostat_coupling, *target_temperature, *k_boltzmann
        );
        // Run simulation...
    }
    EnsembleConfig::NPT { .. } => {
        // Create NPT integrator...
    }
}
```

### Temperature Scheduling
```rust
for step in 0..total_steps {
    // Get current target temperature from schedule
    let current_target_temp = config.get_target_temperature(step);
    integrator.set_target_temperature(current_target_temp);
    
    integrator.step(dt);
}
```

## Advanced Examples

### Multi-Component System
```yaml
system:
  positions:
    type: explicit
    coords:
      # Water molecule 1
      - [0.0, 0.0, 0.0]      # O
      - [0.757, 0.586, 0.0]  # H  
      - [-0.757, 0.586, 0.0] # H
      # Water molecule 2
      - [3.0, 0.0, 0.0]      # O
      - [3.757, 0.586, 0.0]  # H
      - [2.243, 0.586, 0.0]  # H
  masses: [16.0, 1.0, 1.0, 16.0, 1.0, 1.0]  # O, H, H, O, H, H
```

### High-Pressure Simulation
```yaml
simulation:
  ensemble:
    type: npt
    target_temperature: 1.0
    target_pressure: 5.0      # High pressure
    thermostat_coupling: 150.0
    barostat_coupling: 500.0  # Responsive barostat
```

### Gas-Phase Simulation  
```yaml
system:
  positions:
    type: random
    n_atoms: 100
    min_distance: 2.0
  box_setup:
    lengths: [20.0, 20.0, 20.0]  # Large box = low density
    
simulation:
  ensemble:
    type: nvt
    target_temperature: 3.0      # High temperature
```

## Validation and Error Handling

The configuration system includes comprehensive validation:
- **Parameter bounds**: All physical parameters must be positive
- **Consistency checks**: Number of atoms must match across positions/velocities/masses
- **Simulation stability**: Time steps and coupling constants are validated
- **File format**: YAML syntax and required fields are validated

Error messages provide clear guidance:
```
❌ Error loading configuration: Target temperature must be positive
❌ Error generating positions: Could not generate random positions with minimum distance constraint  
❌ Error generating velocities: Number of explicit velocities (5) doesn't match number of atoms (8)
```

## Best Practices

1. **Start with example configs**: Modify existing examples rather than writing from scratch
2. **Use appropriate time steps**: Smaller for light atoms (H) or high pressures
3. **Choose coupling constants wisely**: 
   - Thermostat: 50-500 (smaller = faster response)
   - Barostat: 500-2000 (NPT only)
4. **Set reasonable cutoffs**: 2.5σ for LJ is standard
5. **Use temperature scheduling**: For equilibration and phase transitions
6. **Save trajectories sparingly**: Only when needed for detailed analysis

## Running Simulations

```bash
# Run with specific config
cargo run --example yaml_config_demo examples/argon_nvt.yaml

# Run different examples
cargo run --example yaml_config_demo examples/argon_npt.yaml
cargo run --example yaml_config_demo examples/high_pressure_npt.yaml
cargo run --example yaml_config_demo examples/random_gas_nvt.yaml
```

This system makes it easy to run many different simulation types by just changing configuration files, without recompiling code!







