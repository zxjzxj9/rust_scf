//! Configuration management for SCF calculations
//!
//! This module provides structures for parsing YAML configuration files
//! and command-line arguments for SCF calculations.

use clap::Parser;
use serde::{Deserialize, Serialize};

/// Command-line arguments for SCF calculations
#[derive(Parser, Debug)]
#[command(name = "scf")]
#[command(about = "Self-Consistent Field quantum chemistry calculator", long_about = None)]
pub struct Args {
    /// Path to the YAML configuration file
    #[arg(short, long, default_value = "scf/example/h2o.yaml")]
    pub config_file: String,

    /// Output file path (optional)
    #[arg(short, long)]
    pub output: Option<String>,

    /// Molecular charge (overrides config file)
    #[arg(long)]
    pub charge: Option<i32>,

    /// Spin multiplicity (overrides config file)
    #[arg(long)]
    pub multiplicity: Option<usize>,

    /// Force spin-polarized calculation (UHF)
    #[arg(long)]
    pub spin_polarized: bool,

    /// Density mixing factor (0.0 to 1.0, overrides config file)
    #[arg(long)]
    pub density_mixing: Option<f64>,

    /// Maximum number of SCF cycles (overrides config file)
    #[arg(long)]
    pub max_cycle: Option<usize>,

    /// Enable geometry optimization
    #[arg(long)]
    pub optimize: bool,

    /// Optimization algorithm (cg, sd)
    #[arg(long)]
    pub opt_algorithm: Option<String>,

    /// Maximum optimization iterations
    #[arg(long)]
    pub opt_max_iterations: Option<usize>,

    /// Optimization convergence threshold
    #[arg(long)]
    pub opt_convergence: Option<f64>,

    /// Optimization step size
    #[arg(long)]
    pub opt_step_size: Option<f64>,
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Molecular geometry
    pub geometry: Vec<AtomConfig>,

    /// Basis sets for each element
    #[serde(default)]
    pub basis_sets: std::collections::HashMap<String, String>,

    /// SCF calculation parameters
    #[serde(default)]
    pub scf_params: ScfParams,

    /// Molecular charge (optional)
    #[serde(default)]
    pub charge: Option<i32>,

    /// Spin multiplicity (optional)
    #[serde(default)]
    pub multiplicity: Option<usize>,

    /// Geometry optimization parameters (optional)
    #[serde(default)]
    pub optimization: Option<OptimizationConfig>,
}

/// Atom configuration in the molecular geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomConfig {
    /// Element symbol (e.g., "H", "O", "C")
    pub element: String,

    /// Atomic coordinates [x, y, z] in Angstroms
    pub coords: [f64; 3],
}

/// SCF calculation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScfParams {
    /// Density mixing factor for simple mixing (0.0 to 1.0)
    /// Higher values give more weight to the new density
    #[serde(default)]
    pub density_mixing: Option<f64>,

    /// Maximum number of SCF cycles
    #[serde(default)]
    pub max_cycle: Option<usize>,

    /// Energy convergence threshold in Hartree
    #[serde(default)]
    pub convergence_threshold: Option<f64>,

    /// DIIS subspace size (number of Fock/error matrices to keep)
    /// Set to 0 to disable DIIS acceleration
    /// Typical values: 6-12 for good convergence
    #[serde(default)]
    pub diis_subspace_size: Option<usize>,
}

impl Default for ScfParams {
    fn default() -> Self {
        ScfParams {
            density_mixing: Some(0.5),
            max_cycle: Some(100),
            convergence_threshold: Some(1e-6),
            diis_subspace_size: Some(8), // Enable DIIS by default
        }
    }
}

impl ScfParams {
    /// Apply default values to any missing fields
    pub fn with_defaults(mut self) -> Self {
        let defaults = ScfParams::default();
        if self.density_mixing.is_none() {
            self.density_mixing = defaults.density_mixing;
        }
        if self.max_cycle.is_none() {
            self.max_cycle = defaults.max_cycle;
        }
        if self.convergence_threshold.is_none() {
            self.convergence_threshold = defaults.convergence_threshold;
        }
        if self.diis_subspace_size.is_none() {
            self.diis_subspace_size = defaults.diis_subspace_size;
        }
        self
    }
}

/// Geometry optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationConfig {
    /// Whether optimization is enabled
    #[serde(default)]
    pub enabled: Option<bool>,

    /// Optimization algorithm: "cg" (conjugate gradient) or "sd" (steepest descent)
    #[serde(default)]
    pub algorithm: Option<String>,

    /// Maximum number of optimization iterations
    #[serde(default)]
    pub max_iterations: Option<usize>,

    /// Force convergence threshold (Hartree/Bohr)
    #[serde(default)]
    pub convergence_threshold: Option<f64>,

    /// Step size for line search
    #[serde(default)]
    pub step_size: Option<f64>,
}

impl OptimizationConfig {
    /// Apply default values to any missing fields
    pub fn with_defaults(mut self) -> Self {
        if self.enabled.is_none() {
            self.enabled = Some(false);
        }
        if self.algorithm.is_none() {
            self.algorithm = Some("cg".to_string());
        }
        if self.max_iterations.is_none() {
            self.max_iterations = Some(50);
        }
        if self.convergence_threshold.is_none() {
            self.convergence_threshold = Some(1e-4);
        }
        if self.step_size.is_none() {
            self.step_size = Some(0.1);
        }
        self
    }
}

impl Config {
    /// Apply default values to any missing configuration fields
    pub fn with_defaults(mut self) -> Self {
        self.scf_params = self.scf_params.with_defaults();
        if let Some(ref mut opt) = self.optimization {
            *opt = opt.clone().with_defaults();
        }
        self
    }

    /// Check if DIIS acceleration should be enabled
    pub fn is_diis_enabled(&self) -> bool {
        self.scf_params
            .diis_subspace_size
            .map(|size| size > 0)
            .unwrap_or(false)
    }

    /// Get DIIS subspace size (returns 0 if DIIS is disabled)
    pub fn diis_subspace_size(&self) -> usize {
        self.scf_params.diis_subspace_size.unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = Config {
            geometry: vec![],
            basis_sets: std::collections::HashMap::new(),
            scf_params: ScfParams::default(),
            charge: None,
            multiplicity: None,
            optimization: None,
        }
        .with_defaults();

        assert_eq!(config.scf_params.density_mixing, Some(0.5));
        assert_eq!(config.scf_params.max_cycle, Some(100));
        assert_eq!(config.scf_params.convergence_threshold, Some(1e-6));
        assert_eq!(config.scf_params.diis_subspace_size, Some(8));
    }

    #[test]
    fn test_diis_enabled() {
        let mut config = Config {
            geometry: vec![],
            basis_sets: std::collections::HashMap::new(),
            scf_params: ScfParams {
                density_mixing: Some(0.5),
                max_cycle: Some(100),
                convergence_threshold: Some(1e-6),
                diis_subspace_size: Some(8),
            },
            charge: None,
            multiplicity: None,
            optimization: None,
        };

        assert!(config.is_diis_enabled());
        assert_eq!(config.diis_subspace_size(), 8);

        // Test with DIIS disabled
        config.scf_params.diis_subspace_size = Some(0);
        assert!(!config.is_diis_enabled());
        assert_eq!(config.diis_subspace_size(), 0);
    }

    #[test]
    fn test_yaml_parsing() {
        let yaml = r#"
geometry:
  - element: O
    coords: [0.0, 0.0, 0.0]
  - element: H
    coords: [0.757, 0.586, 0.0]

basis_sets:
  O: "6-31g"
  H: "6-31g"

scf_params:
  density_mixing: 0.5
  max_cycle: 100
  diis_subspace_size: 8
  convergence_threshold: 1e-6
"#;

        let config: Config = serde_yml::from_str(yaml).unwrap();
        assert_eq!(config.geometry.len(), 2);
        assert_eq!(config.geometry[0].element, "O");
        assert_eq!(config.scf_params.diis_subspace_size, Some(8));
        assert!(config.is_diis_enabled());
    }
}
