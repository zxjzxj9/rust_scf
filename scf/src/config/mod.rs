//! Configuration management for SCF calculations
//!
//! This module handles configuration structures, defaults, and validation
//! for SCF calculations and geometry optimizations.

mod args;

pub use args::Args;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main configuration structure for SCF calculations
#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub geometry: Vec<Atom>,
    pub basis_sets: HashMap<String, String>,
    pub scf_params: ScfParams,
    pub optimization: Option<OptimizationParams>,
    pub mp2: Option<Mp2Params>,
    pub ccsd: Option<CcsdParams>,
    pub ci: Option<CiParams>,
    pub charge: Option<i32>,
    pub multiplicity: Option<usize>,
}

/// Atomic position configuration
#[derive(Debug, Deserialize, Serialize)]
pub struct Atom {
    pub element: String,
    pub coords: [f64; 3],
}

/// SCF-specific parameters
#[derive(Debug, Deserialize, Serialize)]
pub struct ScfParams {
    pub density_mixing: Option<f64>,
    pub max_cycle: Option<usize>,
    pub diis_subspace_size: Option<usize>,
    pub convergence_threshold: Option<f64>,
}

impl Default for ScfParams {
    fn default() -> Self {
        ScfParams {
            density_mixing: Some(0.5),
            max_cycle: Some(100),
            diis_subspace_size: Some(8),
            convergence_threshold: Some(1e-6),
        }
    }
}

impl ScfParams {
    /// Apply default values to any missing parameters
    pub fn with_defaults(mut self) -> Self {
        let defaults = Self::default();
        if self.density_mixing.is_none() {
            self.density_mixing = defaults.density_mixing;
        }
        if self.max_cycle.is_none() {
            self.max_cycle = defaults.max_cycle;
        }
        if self.diis_subspace_size.is_none() {
            self.diis_subspace_size = defaults.diis_subspace_size;
        }
        if self.convergence_threshold.is_none() {
            self.convergence_threshold = defaults.convergence_threshold;
        }
        self
    }
}

/// Geometry optimization parameters
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OptimizationParams {
    pub enabled: Option<bool>,
    pub algorithm: Option<String>,
    pub max_iterations: Option<usize>,
    pub convergence_threshold: Option<f64>,
    pub step_size: Option<f64>,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        OptimizationParams {
            enabled: Some(false),
            algorithm: Some("cg".to_string()),
            max_iterations: Some(50),
            convergence_threshold: Some(1e-4),
            step_size: Some(0.1),
        }
    }
}

impl OptimizationParams {
    /// Apply default values to any missing parameters
    pub fn with_defaults(mut self) -> Self {
        let defaults = Self::default();
        if self.enabled.is_none() {
            self.enabled = defaults.enabled;
        }
        if self.algorithm.is_none() {
            self.algorithm = defaults.algorithm;
        }
        if self.max_iterations.is_none() {
            self.max_iterations = defaults.max_iterations;
        }
        if self.convergence_threshold.is_none() {
            self.convergence_threshold = defaults.convergence_threshold;
        }
        if self.step_size.is_none() {
            self.step_size = defaults.step_size;
        }
        self
    }
}

/// MP2 calculation parameters
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Mp2Params {
    pub enabled: Option<bool>,
    pub algorithm: Option<String>, // "direct" or "optimized"
}

impl Default for Mp2Params {
    fn default() -> Self {
        Mp2Params {
            enabled: Some(false),
            algorithm: Some("optimized".to_string()),
        }
    }
}

impl Mp2Params {
    /// Apply default values to any missing parameters
    pub fn with_defaults(mut self) -> Self {
        let defaults = Self::default();
        if self.enabled.is_none() {
            self.enabled = defaults.enabled;
        }
        if self.algorithm.is_none() {
            self.algorithm = defaults.algorithm;
        }
        self
    }
}

/// CCSD calculation parameters
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CcsdParams {
    pub enabled: Option<bool>,
    pub max_iterations: Option<usize>,
    pub convergence_threshold: Option<f64>,
}

impl Default for CcsdParams {
    fn default() -> Self {
        CcsdParams {
            enabled: Some(false),
            max_iterations: Some(50),
            convergence_threshold: Some(1e-7),
        }
    }
}

impl CcsdParams {
    /// Apply default values to any missing parameters
    pub fn with_defaults(mut self) -> Self {
        let defaults = Self::default();
        if self.enabled.is_none() {
            self.enabled = defaults.enabled;
        }
        if self.max_iterations.is_none() {
            self.max_iterations = defaults.max_iterations;
        }
        if self.convergence_threshold.is_none() {
            self.convergence_threshold = defaults.convergence_threshold;
        }
        self
    }
}

/// CI (Configuration Interaction) calculation parameters
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CiParams {
    pub enabled: Option<bool>,
    pub method: Option<String>,    // "cis" or "cisd"
    pub max_states: Option<usize>, // Number of states to compute
    pub convergence_threshold: Option<f64>,
}

impl Default for CiParams {
    fn default() -> Self {
        CiParams {
            enabled: Some(false),
            method: Some("cisd".to_string()),
            max_states: Some(5),
            convergence_threshold: Some(1e-6),
        }
    }
}

impl CiParams {
    /// Apply default values to any missing parameters
    pub fn with_defaults(mut self) -> Self {
        let defaults = Self::default();
        if self.enabled.is_none() {
            self.enabled = defaults.enabled;
        }
        if self.method.is_none() {
            self.method = defaults.method;
        }
        if self.max_states.is_none() {
            self.max_states = defaults.max_states;
        }
        if self.convergence_threshold.is_none() {
            self.convergence_threshold = defaults.convergence_threshold;
        }
        self
    }
}

impl Config {
    /// Apply defaults to all configuration sections
    pub fn with_defaults(mut self) -> Self {
        self.scf_params = self.scf_params.with_defaults();
        if let Some(opt_params) = self.optimization.take() {
            self.optimization = Some(opt_params.with_defaults());
        }
        if let Some(mp2_params) = self.mp2.take() {
            self.mp2 = Some(mp2_params.with_defaults());
        }
        if let Some(ccsd_params) = self.ccsd.take() {
            self.ccsd = Some(ccsd_params.with_defaults());
        }
        if let Some(ci_params) = self.ci.take() {
            self.ci = Some(ci_params.with_defaults());
        }
        self
    }

    /// Check if DIIS acceleration is enabled
    pub fn is_diis_enabled(&self) -> bool {
        self.scf_params.diis_subspace_size.unwrap_or(0) > 0
    }

    /// Get the DIIS subspace size
    pub fn diis_subspace_size(&self) -> usize {
        self.scf_params.diis_subspace_size.unwrap_or(8)
    }

    /// Check if MP2 calculation is enabled
    pub fn is_mp2_enabled(&self) -> bool {
        self.mp2.as_ref().and_then(|m| m.enabled).unwrap_or(false)
    }

    /// Get the MP2 algorithm
    pub fn mp2_algorithm(&self) -> String {
        self.mp2
            .as_ref()
            .and_then(|m| m.algorithm.clone())
            .unwrap_or_else(|| "optimized".to_string())
    }

    /// Check if CCSD calculation is enabled
    pub fn is_ccsd_enabled(&self) -> bool {
        self.ccsd.as_ref().and_then(|c| c.enabled).unwrap_or(false)
    }

    /// Get the CCSD max iterations
    pub fn ccsd_max_iterations(&self) -> usize {
        self.ccsd
            .as_ref()
            .and_then(|c| c.max_iterations)
            .unwrap_or(50)
    }

    /// Get the CCSD convergence threshold
    pub fn ccsd_convergence_threshold(&self) -> f64 {
        self.ccsd
            .as_ref()
            .and_then(|c| c.convergence_threshold)
            .unwrap_or(1e-7)
    }

    /// Check if CI calculation is enabled
    pub fn is_ci_enabled(&self) -> bool {
        self.ci.as_ref().and_then(|c| c.enabled).unwrap_or(false)
    }

    /// Get the CI method (CIS or CISD)
    pub fn ci_method(&self) -> String {
        self.ci
            .as_ref()
            .and_then(|c| c.method.clone())
            .unwrap_or_else(|| "cisd".to_string())
    }

    /// Get the CI max states
    pub fn ci_max_states(&self) -> usize {
        self.ci.as_ref().and_then(|c| c.max_states).unwrap_or(5)
    }

    /// Get the CI convergence threshold
    pub fn ci_convergence_threshold(&self) -> f64 {
        self.ci
            .as_ref()
            .and_then(|c| c.convergence_threshold)
            .unwrap_or(1e-6)
    }
}
