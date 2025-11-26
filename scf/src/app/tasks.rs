use crate::app::workspace::CalculationWorkspace;
use crate::config::{Args, Config};
use crate::io::print_optimized_geometry;
use crate::{CGOptimizer, GeometryOptimizer, SimpleSCF, SpinSCF, SteepestDescentOptimizer, SCF};
use ::basis::cgto::Basis631G;
use color_eyre::eyre::Result;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use std::fs::File;
use tracing::info;

pub fn run_restricted_followups(
    scf: &mut SimpleSCF<Basis631G>,
    args: &Args,
    config: &Config,
    workspace: &CalculationWorkspace<Basis631G>,
) -> Result<()> {
    if config.is_mp2_enabled() {
        run_mp2(scf, config);
    }

    if config.is_ccsd_enabled() {
        run_ccsd(scf, config);
    }

    if config.is_ci_enabled() {
        run_ci(scf, config);
    }

    maybe_run_geometry_optimization(scf, args, config, workspace)?;
    Ok(())
}

pub fn run_spin_followups(
    scf: &mut SpinSCF<Basis631G>,
    args: &Args,
    config: &Config,
    workspace: &CalculationWorkspace<Basis631G>,
) -> Result<()> {
    maybe_run_geometry_optimization(scf, args, config, workspace)?;
    Ok(())
}

fn run_mp2(scf: &SimpleSCF<Basis631G>, config: &Config) {
    info!("\n===========================================");
    info!("       Starting MP2 Calculation");
    info!("===========================================");

    let mut mp2 = scf.create_mp2();
    let algorithm = config.mp2_algorithm();

    let correlation_energy = match algorithm.to_lowercase().as_str() {
        "direct" => {
            info!("Using direct MP2 algorithm");
            mp2.calculate_mp2_energy_direct()
        }
        _ => {
            info!("Using optimized MP2 algorithm");
            mp2.calculate_mp2_energy()
        }
    };

    let total_mp2_energy = scf.calculate_total_energy() + correlation_energy;

    info!("\n===========================================");
    info!("        MP2 Results Summary");
    info!("===========================================");
    info!(
        "Hartree-Fock energy:       {:.10} au",
        scf.calculate_total_energy()
    );
    info!("MP2 correlation energy:    {:.10} au", correlation_energy);
    info!("Total MP2 energy:          {:.10} au", total_mp2_energy);
    info!("===========================================\n");
}

fn run_ccsd(scf: &SimpleSCF<Basis631G>, config: &Config) {
    info!("\n===========================================");
    info!("       Starting CCSD Calculation");
    info!("===========================================");

    let max_iterations = config.ccsd_max_iterations();
    let convergence_threshold = config.ccsd_convergence_threshold();

    let mut ccsd = scf.create_ccsd(max_iterations, convergence_threshold);
    let correlation_energy = ccsd.solve();

    let total_ccsd_energy = scf.calculate_total_energy() + correlation_energy;

    info!("\n===========================================");
    info!("        CCSD Results Summary");
    info!("===========================================");
    info!(
        "Hartree-Fock energy:       {:.10} au",
        scf.calculate_total_energy()
    );
    info!("CCSD correlation energy:   {:.10} au", correlation_energy);
    info!("Total CCSD energy:         {:.10} au", total_ccsd_energy);
    let t1_diag = ccsd.t1_diagnostic();
    info!("\nT1 diagnostic:             {:.6}", t1_diag);
    if t1_diag > 0.02 {
        info!("WARNING: T1 diagnostic > 0.02 suggests significant multireference character");
        info!("         CCSD may not be appropriate for this system");
    } else {
        info!("T1 diagnostic indicates single-reference character (good for CCSD)");
    }
    info!("===========================================\n");
}

fn run_ci(scf: &SimpleSCF<Basis631G>, config: &Config) {
    info!("\n===========================================");
    info!("       Starting CI Calculation");
    info!("===========================================");

    let max_states = config.ci_max_states();
    let convergence_threshold = config.ci_convergence_threshold();
    let method = config.ci_method();

    let mut ci = scf.create_ci(max_states, convergence_threshold);
    let hf_energy = scf.calculate_total_energy();

    match method.to_lowercase().as_str() {
        "cis" => {
            info!("Running CIS (Configuration Interaction Singles) for excited states");
            let excitation_energies = ci.calculate_cis_energies(max_states);

            info!("\n===========================================");
            info!("        CIS Results Summary");
            info!("===========================================");
            info!("Hartree-Fock ground state: {:.10} au", hf_energy);
            info!("\nExcited States:");
            for (i, &exc_energy) in excitation_energies.iter().enumerate() {
                let total_energy = hf_energy + exc_energy;
                info!(
                    "  State {}: Excitation = {:.6} au ({:.2} eV), Total = {:.10} au",
                    i + 1,
                    exc_energy,
                    exc_energy * 27.2114,
                    total_energy
                );
            }
            info!("===========================================\n");
        }
        "cisd" => {
            info!("Running CISD (Configuration Interaction Singles and Doubles)");
            let correlation_energy = ci.calculate_cisd_energy();
            let total_cisd_energy = hf_energy + correlation_energy;

            info!("\n===========================================");
            info!("        CISD Results Summary");
            info!("===========================================");
            info!("Hartree-Fock energy:       {:.10} au", hf_energy);
            info!("CISD correlation energy:   {:.10} au", correlation_energy);
            info!("Total CISD energy:         {:.10} au", total_cisd_energy);
            info!("===========================================\n");
        }
        _ => {
            info!("Unknown CI method: {}. Using CISD.", method);
            let correlation_energy = ci.calculate_cisd_energy();
            let total_cisd_energy = hf_energy + correlation_energy;

            info!("\n===========================================");
            info!("        CISD Results Summary");
            info!("===========================================");
            info!("Hartree-Fock energy:       {:.10} au", hf_energy);
            info!("CISD correlation energy:   {:.10} au", correlation_energy);
            info!("Total CISD energy:         {:.10} au", total_cisd_energy);
            info!("===========================================\n");
        }
    }
}

fn maybe_run_geometry_optimization<S>(
    scf: &mut S,
    args: &Args,
    config: &Config,
    workspace: &CalculationWorkspace<Basis631G>,
) -> Result<()>
where
    S: SCF + Clone,
{
    if !should_optimize(args, config) {
        return Ok(());
    }

    run_optimization(
        scf,
        args,
        config,
        workspace.coords.clone(),
        workspace.elements.clone(),
    )
}

fn should_optimize(args: &Args, config: &Config) -> bool {
    args.optimize
        || config
            .optimization
            .as_ref()
            .and_then(|o| o.enabled)
            .unwrap_or(false)
}

fn run_optimization<S: SCF + Clone>(
    scf: &mut S,
    args: &Args,
    config: &Config,
    coords: Vec<Vector3<f64>>,
    elements: Vec<Element>,
) -> Result<()> {
    info!("\nStarting geometry optimization...");

    let opt_params = config
        .optimization
        .as_ref()
        .cloned()
        .unwrap_or_default()
        .with_defaults();

    let algorithm = args
        .opt_algorithm
        .clone()
        .or(opt_params.algorithm)
        .unwrap_or_else(|| "cg".to_string());
    let max_iterations = args
        .opt_max_iterations
        .or(opt_params.max_iterations)
        .unwrap_or(50);
    let convergence = args
        .opt_convergence
        .or(opt_params.convergence_threshold)
        .unwrap_or(1e-4);
    let step_size = args.opt_step_size.or(opt_params.step_size).unwrap_or(0.1);

    info!("Optimization parameters:");
    info!("  Algorithm: {}", algorithm);
    info!("  Max iterations: {}", max_iterations);
    info!("  Convergence threshold: {:.6e}", convergence);
    info!("  Step size: {:.4}", step_size);

    let (optimized_coords, final_energy) = match algorithm.to_lowercase().as_str() {
        "cg" => {
            let mut optimizer = CGOptimizer::new(scf, max_iterations, convergence);
            optimizer.set_step_size(step_size);
            optimizer.init(coords.clone(), elements.clone());
            optimizer.optimize()
        }
        "sd" => {
            let mut optimizer = SteepestDescentOptimizer::new(scf, max_iterations, convergence);
            optimizer.set_step_size(step_size);
            optimizer.init(coords.clone(), elements.clone());
            optimizer.optimize()
        }
        _ => {
            info!(
                "Unknown optimization algorithm: {}, defaulting to CG",
                algorithm
            );
            let mut optimizer = CGOptimizer::new(scf, max_iterations, convergence);
            optimizer.set_step_size(step_size);
            optimizer.init(coords.clone(), elements.clone());
            optimizer.optimize()
        }
    };

    info!("\nOptimized geometry:");
    for (i, (coord, elem)) in optimized_coords.iter().zip(elements.iter()).enumerate() {
        info!(
            "  Atom {}: {} at [{:.6}, {:.6}, {:.6}]",
            i + 1,
            elem.get_symbol(),
            coord.x,
            coord.y,
            coord.z
        );
    }
    info!("Final energy: {:.10} au", final_energy);

    if let Some(ref output_file) = args.output {
        let mut file = File::create(output_file)?;
        print_optimized_geometry(&mut file, &optimized_coords, &elements, final_energy)?;
    }

    Ok(())
}
