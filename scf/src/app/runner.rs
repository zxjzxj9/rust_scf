use crate::app::workspace::CalculationWorkspace;
use crate::config::{Args, Config};
use crate::{SimpleSCF, SpinSCF, SCF};
use ::basis::basis::AOBasis;
use ::basis::cgto::Basis631G;
use color_eyre::eyre::Result;
use tracing::info;

pub fn run_restricted(
    workspace: &CalculationWorkspace<Basis631G>,
    args: &Args,
    config: &Config,
) -> Result<SimpleSCF<Basis631G>> {
    let mut scf = SimpleSCF::<Basis631G>::new();
    configure_restricted(&mut scf, args, config);
    initialize_and_run_scf(&mut scf, workspace);
    Ok(scf)
}

pub fn run_spin(
    workspace: &CalculationWorkspace<Basis631G>,
    args: &Args,
    config: &Config,
    charge: i32,
    multiplicity: usize,
) -> Result<SpinSCF<Basis631G>> {
    let mut scf = SpinSCF::<Basis631G>::new();
    configure_spin(&mut scf, args, config, charge, multiplicity);
    initialize_and_run_scf(&mut scf, workspace);
    Ok(scf)
}

fn configure_restricted(scf: &mut SimpleSCF<Basis631G>, args: &Args, config: &Config) {
    scf.density_mixing = args
        .density_mixing
        .or(config.scf_params.density_mixing)
        .unwrap();
    scf.max_cycle = args.max_cycle.or(config.scf_params.max_cycle).unwrap();
    let convergence = args
        .convergence_threshold
        .or(config.scf_params.convergence_threshold)
        .unwrap();
    scf.set_convergence_threshold(convergence);

    if let Some(diis_size) = resolve_diis_size(args, config) {
        info!(
            "Enabling DIIS acceleration with subspace size {}",
            diis_size
        );
        scf.enable_diis(diis_size);
    } else {
        info!("DIIS acceleration disabled");
    }
}

fn configure_spin(
    scf: &mut SpinSCF<Basis631G>,
    args: &Args,
    config: &Config,
    charge: i32,
    multiplicity: usize,
) {
    scf.density_mixing = args
        .density_mixing
        .or(config.scf_params.density_mixing)
        .unwrap_or(scf.density_mixing);
    scf.max_cycle = args
        .max_cycle
        .or(config.scf_params.max_cycle)
        .unwrap_or(scf.max_cycle);
    let convergence = args
        .convergence_threshold
        .or(config.scf_params.convergence_threshold)
        .unwrap_or(scf.convergence_threshold);
    scf.set_convergence_threshold(convergence);
    scf.set_charge(charge);
    scf.set_multiplicity(multiplicity);

    if let Some(diis_size) = resolve_diis_size(args, config) {
        info!(
            "Enabling DIIS acceleration with subspace size {}",
            diis_size
        );
        scf.enable_diis(diis_size);
    } else {
        info!("DIIS acceleration disabled");
    }
}

pub fn initialize_and_run_scf<B: AOBasis + 'static>(
    scf: &mut impl SCF<BasisType = B>,
    workspace: &CalculationWorkspace<B>,
) {
    info!("\nInitializing SCF calculation...");
    scf.init_basis(&workspace.elements, workspace.basis_map());
    scf.init_geometry(&workspace.coords, &workspace.elements);
    scf.init_density_matrix();
    scf.init_fock_matrix();

    info!("\nStarting SCF cycle...\n");
    scf.scf_cycle();
}

fn resolve_diis_size(args: &Args, config: &Config) -> Option<usize> {
    let size = args
        .diis_subspace_size
        .or(config.scf_params.diis_subspace_size)
        .unwrap_or(0);
    if size > 0 {
        Some(size)
    } else {
        None
    }
}
