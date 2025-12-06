use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Configuration for molecular dynamics simulations
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct MdConfig {
    /// System setup parameters
    pub system: SystemConfig,
    /// Simulation ensemble and integration parameters
    pub simulation: SimulationConfig,
    /// Potential energy parameters
    pub potential: PotentialConfig,
    /// Output and analysis settings
    pub output: OutputConfig,
}

/// System setup configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SystemConfig {
    /// Initial atomic positions (can be explicit or generated)
    pub positions: PositionConfig,
    /// Initial velocity configuration
    pub velocities: VelocityConfig,
    /// Atomic masses (per atom or single value for all)
    pub masses: MassConfig,
    /// Simulation box configuration
    pub box_setup: BoxConfig,
}

/// Position configuration options
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum PositionConfig {
    /// Explicit list of positions
    #[serde(rename = "explicit")]
    Explicit { coords: Vec<[f64; 3]> },
    /// Generate simple cubic lattice
    #[serde(rename = "cubic_lattice")]
    CubicLattice {
        /// Number of atoms per side
        n_per_side: usize,
        /// Lattice spacing
        spacing: f64,
        /// Optional offset from origin
        offset: Option<[f64; 3]>,
    },
    /// Generate random positions in box
    #[serde(rename = "random")]
    Random {
        /// Number of atoms
        n_atoms: usize,
        /// Minimum distance between atoms
        min_distance: f64,
    },
}

/// Velocity configuration options
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum VelocityConfig {
    /// Explicit velocities
    #[serde(rename = "explicit")]
    Explicit { velocities: Vec<[f64; 3]> },
    /// Maxwell-Boltzmann distribution at given temperature
    #[serde(rename = "maxwell_boltzmann")]
    MaxwellBoltzmann {
        temperature: f64,
        /// Optional random seed
        seed: Option<u64>,
    },
    /// Zero initial velocities
    #[serde(rename = "zero")]
    Zero,
}

/// Mass configuration options
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MassConfig {
    /// Single mass for all atoms
    Uniform(f64),
    /// Individual masses per atom
    Individual(Vec<f64>),
}

/// Simulation box configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BoxConfig {
    /// Initial box lengths [x, y, z]
    pub lengths: [f64; 3],
    /// Whether to use periodic boundary conditions
    #[serde(default = "default_periodic")]
    pub periodic: bool,
}

/// Simulation configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SimulationConfig {
    /// Ensemble type and parameters
    pub ensemble: EnsembleConfig,
    /// Integration parameters
    pub integration: IntegrationConfig,
    /// Temperature schedule (optional)
    pub temperature_schedule: Option<TemperatureSchedule>,
}

/// Ensemble configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum EnsembleConfig {
    /// Canonical (NVT) ensemble
    #[serde(rename = "nvt")]
    NVT {
        /// Target temperature
        target_temperature: f64,
        /// Thermostat coupling parameter Q_t
        thermostat_coupling: f64,
        /// Boltzmann constant (default: 1.0 for reduced units)
        #[serde(default = "default_kb")]
        k_boltzmann: f64,
    },
    /// Isothermal-isobaric (NPT) ensemble
    #[serde(rename = "npt")]
    NPT {
        /// Target temperature
        target_temperature: f64,
        /// Target pressure
        target_pressure: f64,
        /// Thermostat coupling parameter Q_t
        thermostat_coupling: f64,
        /// Barostat coupling parameter Q_p
        barostat_coupling: f64,
        /// Boltzmann constant (default: 1.0 for reduced units)
        #[serde(default = "default_kb")]
        k_boltzmann: f64,
    },
}

/// Integration parameters
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct IntegrationConfig {
    /// Time step
    pub time_step: f64,
    /// Total number of steps
    pub total_steps: usize,
}

/// Temperature schedule for ramping
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TemperatureSchedule {
    /// Initial temperature
    pub initial_temperature: f64,
    /// Final temperature
    pub final_temperature: f64,
    /// Number of steps for ramping
    pub ramp_steps: usize,
}

/// Potential energy configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum PotentialConfig {
    /// Lennard-Jones potential
    #[serde(rename = "lennard_jones")]
    LennardJones {
        /// Well depth parameter ε
        epsilon: f64,
        /// Collision diameter σ
        sigma: f64,
        /// Cutoff distance (default: 2.5σ)
        #[serde(default = "default_lj_cutoff")]
        cutoff: f64,
    },
}

/// Output configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OutputConfig {
    /// Output interval for main results
    pub output_interval: usize,
    /// Analysis interval for detailed analysis
    #[serde(default = "default_analysis_interval")]
    pub analysis_interval: usize,
    /// Whether to save trajectory
    #[serde(default = "default_save_trajectory")]
    pub save_trajectory: bool,
    /// Trajectory output interval
    #[serde(default = "default_trajectory_interval")]
    pub trajectory_interval: usize,
}

// Default value functions
fn default_periodic() -> bool {
    true
}
fn default_kb() -> f64 {
    1.0
}
fn default_lj_cutoff() -> f64 {
    2.5
}
fn default_analysis_interval() -> usize {
    200
}
fn default_save_trajectory() -> bool {
    false
}
fn default_trajectory_interval() -> usize {
    1000
}

impl MdConfig {
    /// Load configuration from YAML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: MdConfig = serde_yaml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to YAML file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_yaml::to_string(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        // Validate time step
        if self.simulation.integration.time_step <= 0.0 {
            return Err("Time step must be positive".to_string());
        }

        // Validate total steps
        if self.simulation.integration.total_steps == 0 {
            return Err("Total steps must be positive".to_string());
        }

        // Validate ensemble parameters
        match &self.simulation.ensemble {
            EnsembleConfig::NVT {
                target_temperature,
                thermostat_coupling,
                ..
            } => {
                if *target_temperature <= 0.0 {
                    return Err("Target temperature must be positive".to_string());
                }
                if *thermostat_coupling <= 0.0 {
                    return Err("Thermostat coupling must be positive".to_string());
                }
            }
            EnsembleConfig::NPT {
                target_temperature,
                target_pressure,
                thermostat_coupling,
                barostat_coupling,
                ..
            } => {
                if *target_temperature <= 0.0 {
                    return Err("Target temperature must be positive".to_string());
                }
                if *target_pressure <= 0.0 {
                    return Err("Target pressure must be positive".to_string());
                }
                if *thermostat_coupling <= 0.0 {
                    return Err("Thermostat coupling must be positive".to_string());
                }
                if *barostat_coupling <= 0.0 {
                    return Err("Barostat coupling must be positive".to_string());
                }
            }
        }

        // Validate potential parameters
        match &self.potential {
            PotentialConfig::LennardJones {
                epsilon,
                sigma,
                cutoff,
            } => {
                if *epsilon <= 0.0 {
                    return Err("LJ epsilon must be positive".to_string());
                }
                if *sigma <= 0.0 {
                    return Err("LJ sigma must be positive".to_string());
                }
                if *cutoff <= 0.0 {
                    return Err("LJ cutoff must be positive".to_string());
                }
            }
        }

        // Validate box dimensions
        for &length in &self.system.box_setup.lengths {
            if length <= 0.0 {
                return Err("Box lengths must be positive".to_string());
            }
        }

        // Validate output intervals
        if self.output.output_interval == 0 {
            return Err("Output interval must be positive".to_string());
        }
        if self.output.analysis_interval == 0 {
            return Err("Analysis interval must be positive".to_string());
        }
        if self.output.trajectory_interval == 0 {
            return Err("Trajectory interval must be positive".to_string());
        }

        Ok(())
    }

    /// Generate positions based on configuration
    pub fn generate_positions(&self) -> Result<Vec<Vector3<f64>>, String> {
        match &self.system.positions {
            PositionConfig::Explicit { coords } => Ok(coords
                .iter()
                .map(|&c| Vector3::new(c[0], c[1], c[2]))
                .collect()),
            PositionConfig::CubicLattice {
                n_per_side,
                spacing,
                offset,
            } => {
                let mut positions = Vec::new();
                let offset = offset.unwrap_or([0.0, 0.0, 0.0]);

                for i in 0..*n_per_side {
                    for j in 0..*n_per_side {
                        for k in 0..*n_per_side {
                            let pos = Vector3::new(
                                offset[0] + i as f64 * spacing,
                                offset[1] + j as f64 * spacing,
                                offset[2] + k as f64 * spacing,
                            );
                            positions.push(pos);
                        }
                    }
                }
                Ok(positions)
            }
            PositionConfig::Random {
                n_atoms,
                min_distance,
            } => {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let mut positions = Vec::new();
                let box_lengths = Vector3::new(
                    self.system.box_setup.lengths[0],
                    self.system.box_setup.lengths[1],
                    self.system.box_setup.lengths[2],
                );

                for _ in 0..*n_atoms {
                    let mut attempts = 0;
                    let max_attempts = 10000;

                    loop {
                        attempts += 1;
                        if attempts > max_attempts {
                            return Err("Could not generate random positions with minimum distance constraint".to_string());
                        }

                        let candidate = nalgebra::Vector3::<f64>::new(
                            rng.gen::<f64>() * box_lengths.x,
                            rng.gen::<f64>() * box_lengths.y,
                            rng.gen::<f64>() * box_lengths.z,
                        );

                        // Check minimum distance constraint
                        let mut valid = true;
                        for existing in &positions {
                            let diff = candidate - existing;
                            let dist: f64 = nalgebra::Vector3::<f64>::norm(&diff);
                            if dist < *min_distance {
                                valid = false;
                                break;
                            }
                        }

                        if valid {
                            positions.push(candidate);
                            break;
                        }
                    }
                }
                Ok(positions)
            }
        }
    }

    /// Generate velocities based on configuration
    pub fn generate_velocities(&self, n_atoms: usize) -> Result<Vec<Vector3<f64>>, String> {
        match &self.system.velocities {
            VelocityConfig::Explicit { velocities } => {
                if velocities.len() != n_atoms {
                    return Err(format!(
                        "Number of explicit velocities ({}) doesn't match number of atoms ({})",
                        velocities.len(),
                        n_atoms
                    ));
                }
                Ok(velocities
                    .iter()
                    .map(|&v| Vector3::new(v[0], v[1], v[2]))
                    .collect())
            }
            VelocityConfig::MaxwellBoltzmann { temperature, seed } => {
                use rand::SeedableRng;
                use rand_distr::{Distribution, StandardNormal};

                let mut rng = if let Some(seed) = seed {
                    rand::rngs::StdRng::seed_from_u64(*seed)
                } else {
                    rand::rngs::StdRng::from_entropy()
                };

                let mut velocities = Vec::with_capacity(n_atoms);
                let normal = StandardNormal;

                // Sample individual velocities
                for _ in 0..n_atoms {
                    let v = Vector3::new(
                        normal.sample(&mut rng),
                        normal.sample(&mut rng),
                        normal.sample(&mut rng),
                    );
                    velocities.push(v * temperature.sqrt());
                }

                // Remove center-of-mass motion
                let v_cm: Vector3<f64> = velocities.iter().sum::<Vector3<f64>>() / n_atoms as f64;
                for v in &mut velocities {
                    *v -= v_cm;
                }

                // Scale to exact target temperature if we have more than one atom
                if n_atoms > 1 {
                    let current_temp =
                        velocities.iter().map(|v| v.dot(v)).sum::<f64>() / (3.0 * n_atoms as f64);

                    if current_temp > 0.0 {
                        let scale_factor = (temperature / current_temp).sqrt();
                        for v in &mut velocities {
                            *v *= scale_factor;
                        }
                    }
                }

                Ok(velocities)
            }
            VelocityConfig::Zero => Ok(vec![Vector3::zeros(); n_atoms]),
        }
    }

    /// Generate masses based on configuration
    pub fn generate_masses(&self, n_atoms: usize) -> Result<Vec<f64>, String> {
        match &self.system.masses {
            MassConfig::Uniform(mass) => {
                if *mass <= 0.0 {
                    return Err("Mass must be positive".to_string());
                }
                Ok(vec![*mass; n_atoms])
            }
            MassConfig::Individual(masses) => {
                if masses.len() != n_atoms {
                    return Err(format!(
                        "Number of masses ({}) doesn't match number of atoms ({})",
                        masses.len(),
                        n_atoms
                    ));
                }
                for &mass in masses {
                    if mass <= 0.0 {
                        return Err("All masses must be positive".to_string());
                    }
                }
                Ok(masses.clone())
            }
        }
    }

    /// Get current target temperature, potentially from schedule
    pub fn get_target_temperature(&self, current_step: usize) -> f64 {
        if let Some(schedule) = &self.simulation.temperature_schedule {
            if current_step < schedule.ramp_steps {
                let progress = current_step as f64 / schedule.ramp_steps as f64;
                schedule.initial_temperature
                    + (schedule.final_temperature - schedule.initial_temperature) * progress
            } else {
                schedule.final_temperature
            }
        } else {
            match &self.simulation.ensemble {
                EnsembleConfig::NVT {
                    target_temperature, ..
                } => *target_temperature,
                EnsembleConfig::NPT {
                    target_temperature, ..
                } => *target_temperature,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_validation() {
        let mut config = create_test_config();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid time step
        config.simulation.integration.time_step = -0.1;
        assert!(config.validate().is_err());
        config.simulation.integration.time_step = 0.01; // Reset

        // Invalid temperature
        if let EnsembleConfig::NVT {
            ref mut target_temperature,
            ..
        } = config.simulation.ensemble
        {
            *target_temperature = -100.0;
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn test_position_generation() {
        let config = create_test_config();
        let positions = config.generate_positions().unwrap();
        assert_eq!(positions.len(), 8); // 2^3 = 8 atoms

        // Check lattice spacing along x-direction independent of generation order
        let expected_spacing = 1.2;
        let mut xs: Vec<f64> = positions.iter().map(|p| p.x).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        assert!(xs.len() >= 2);
        assert!((xs[1] - xs[0] - expected_spacing).abs() < 1e-10);
    }

    #[test]
    fn test_velocity_generation() {
        let config = create_test_config();
        let velocities = config.generate_velocities(8).unwrap();
        assert_eq!(velocities.len(), 8);

        // Center of mass velocity should be ~0
        let v_cm: Vector3<f64> = velocities.iter().sum::<Vector3<f64>>() / velocities.len() as f64;
        assert!(v_cm.norm() < 1e-10);
    }

    #[test]
    fn test_mass_generation() {
        let config = create_test_config();
        let masses = config.generate_masses(8).unwrap();
        assert_eq!(masses.len(), 8);
        assert_eq!(masses[0], 1.0);
    }

    #[test]
    fn test_yaml_serialization() {
        let config = create_test_config();
        let yaml = serde_yaml::to_string(&config).unwrap();

        // Should be able to deserialize back
        let deserialized: MdConfig = serde_yaml::from_str(&yaml).unwrap();
        assert!(deserialized.validate().is_ok());
    }

    #[test]
    fn test_file_io() {
        let config = create_test_config();

        // Test saving and loading
        let mut temp_file = NamedTempFile::new().unwrap();
        config.to_file(temp_file.path()).unwrap();

        let loaded_config = MdConfig::from_file(temp_file.path()).unwrap();
        assert!(loaded_config.validate().is_ok());
    }

    fn create_test_config() -> MdConfig {
        MdConfig {
            system: SystemConfig {
                positions: PositionConfig::CubicLattice {
                    n_per_side: 2,
                    spacing: 1.2,
                    offset: None,
                },
                velocities: VelocityConfig::MaxwellBoltzmann {
                    temperature: 0.8,
                    seed: Some(42),
                },
                masses: MassConfig::Uniform(1.0),
                box_setup: BoxConfig {
                    lengths: [5.0, 5.0, 5.0],
                    periodic: true,
                },
            },
            simulation: SimulationConfig {
                ensemble: EnsembleConfig::NVT {
                    target_temperature: 0.8,
                    thermostat_coupling: 100.0,
                    k_boltzmann: 1.0,
                },
                integration: IntegrationConfig {
                    time_step: 0.002,
                    total_steps: 10000,
                },
                temperature_schedule: None,
            },
            potential: PotentialConfig::LennardJones {
                epsilon: 1.0,
                sigma: 1.0,
                cutoff: 2.5,
            },
            output: OutputConfig {
                output_interval: 400,
                analysis_interval: 200,
                save_trajectory: false,
                trajectory_interval: 1000,
            },
        }
    }
}
