//! Command-line argument parsing for SCF calculations

use clap::Parser;

/// Simple SCF calculation with YAML configuration
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the YAML configuration file
    #[arg(short, long, default_value = "config.yaml")]
    pub config_file: String,

    /// Override density mixing parameter
    #[arg(long)]
    pub density_mixing: Option<f64>,

    /// Override maximum SCF cycles
    #[arg(long)]
    pub max_cycle: Option<usize>,

    /// Override DIIS subspace size
    #[arg(long)]
    pub diis_subspace_size: Option<usize>,

    /// Override convergence threshold
    #[arg(long)]
    pub convergence_threshold: Option<f64>,

    /// Override output file: (default stdout)
    #[arg(short, long)]
    pub output: Option<String>,

    /// Enable geometry optimization
    #[arg(long)]
    pub optimize: bool,

    /// Optimization algorithm (cg or sd)
    #[arg(long, default_value = "cg")]
    pub opt_algorithm: Option<String>,

    /// Maximum optimization iterations
    #[arg(long)]
    pub opt_max_iterations: Option<usize>,

    /// Optimization convergence threshold
    #[arg(long)]
    pub opt_convergence: Option<f64>,

    /// Step size for optimization
    #[arg(long)]
    pub opt_step_size: Option<f64>,

    /// Molecular charge (default: 0 for neutral)
    #[arg(long)]
    pub charge: Option<i32>,

    /// Spin multiplicity (2S+1, default: 1 for singlet)
    #[arg(long)]
    pub multiplicity: Option<usize>,

    /// Use spin-polarized SCF (UHF) instead of restricted SCF
    #[arg(long)]
    pub spin_polarized: bool,
}
