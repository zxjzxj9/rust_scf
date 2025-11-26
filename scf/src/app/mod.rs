mod basis;
mod geometry;
mod report;
mod runner;
mod tasks;
mod workspace;

pub use basis::{Basis631GLoader, BasisRegistry};
pub use geometry::{build_geometry, Geometry};
pub use runner::{run_restricted, run_spin};
pub use workspace::CalculationWorkspace;

use self::report::{report_restricted_summary, report_spin_summary};
use self::tasks::{run_restricted_followups, run_spin_followups};
use crate::config::{Args, Config};
use crate::io::setup_output;
use ::basis::cgto::Basis631G;
use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use std::fs;
use tracing::info;

pub struct ScfApplication {
    args: Args,
    config: Config,
}

impl ScfApplication {
    pub fn from_cli() -> Result<Self> {
        let args = Args::parse();
        let config = load_config(&args)?;
        Ok(Self { args, config })
    }

    pub fn run(mut self) -> Result<()> {
        setup_output(self.args.output.as_ref());

        let geometry = build_geometry(&self.config)?;
        let mut basis_registry = BasisRegistry::<Basis631G, _>::new(Basis631GLoader);
        let basis_map = basis_registry.load_for_elements(&self.config, &geometry.elements)?;
        let workspace = CalculationWorkspace::new(geometry.elements, geometry.coords, basis_map);

        match SolverSelection::determine(&self.args, &self.config) {
            SolverSelection::Restricted => {
                info!("Using restricted SCF (RHF) for neutral singlet systems");
                let mut scf = run_restricted(&workspace, &self.args, &self.config)?;
                report_restricted_summary(&scf);
                run_restricted_followups(&mut scf, &self.args, &self.config, &workspace)?;
            }
            SolverSelection::Spin {
                charge,
                multiplicity,
            } => {
                info!(
                    "Using spin-polarized SCF (UHF) with charge={}, multiplicity={}",
                    charge, multiplicity
                );
                let mut scf = run_spin(&workspace, &self.args, &self.config, charge, multiplicity)?;
                report_spin_summary(&scf);
                run_spin_followups(&mut scf, &self.args, &self.config, &workspace)?;
            }
        }

        Ok(())
    }
}

fn load_config(args: &Args) -> Result<Config> {
    let config_content = fs::read_to_string(&args.config_file)
        .wrap_err_with(|| format!("Unable to read configuration file: {}", args.config_file))?;

    let config = serde_yml::from_str::<Config>(&config_content)
        .wrap_err("Failed to parse configuration file")?
        .with_defaults();

    Ok(config)
}

enum SolverSelection {
    Restricted,
    Spin { charge: i32, multiplicity: usize },
}

impl SolverSelection {
    fn determine(args: &Args, config: &Config) -> Self {
        let charge = args.charge.or(config.charge).unwrap_or(0);
        let multiplicity = args.multiplicity.or(config.multiplicity).unwrap_or(1);
        let use_spin = args.spin_polarized || multiplicity > 1 || charge != 0;

        if use_spin {
            SolverSelection::Spin {
                charge,
                multiplicity,
            }
        } else {
            SolverSelection::Restricted
        }
    }
}
