//! SCF Calculation Command-Line Interface
//!
//! This is the main entry point for running SCF calculations with YAML configuration.

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use std::fs;
use tracing::info;

use scf::{
    config::{Args, Config},
    io::{fetch_basis, print_optimized_geometry, setup_output},
    CGOptimizer, GeometryOptimizer, SteepestDescentOptimizer,
    SimpleSCF, SpinSCF, SCF,
};
use basis::basis::AOBasis;

fn main() -> Result<()> {
    color_eyre::install()?;
    
    let args = Args::parse();
    setup_output(args.output.as_ref());

    // Load and parse configuration
    info!("Reading configuration from: {}", args.config_file);
    let config_content = fs::read_to_string(&args.config_file)
        .wrap_err_with(|| format!("Unable to read configuration file: {}", args.config_file))?;

    let config: Config = serde_yml::from_str::<Config>(&config_content)
        .wrap_err("Failed to parse configuration file")?
        .with_defaults();

    info!("Configuration loaded:\n{:?}", config);

    // Determine calculation type
    let charge = args.charge.or(config.charge).unwrap_or(0);
    let multiplicity = args.multiplicity.or(config.multiplicity).unwrap_or(1);
    let use_spin_polarized = args.spin_polarized || multiplicity > 1 || charge != 0;

    if use_spin_polarized {
        info!("Using spin-polarized SCF (UHF) with charge={}, multiplicity={}", charge, multiplicity);
        run_spin_scf_calculation(config, args, charge, multiplicity)
    } else {
        info!("Using restricted SCF (RHF) for neutral singlet");
        run_simple_scf_calculation(config, args)
    }
}

/// Initialize and run SCF cycle (works with Basis631G)
fn initialize_and_run_scf<B: AOBasis>(
    scf: &mut impl SCF<BasisType = B>,
    elements: &Vec<Element>,
    coords_vec: &Vec<Vector3<f64>>,
    basis_map: HashMap<&str, &B>,
) where
    B: 'static,
{
    info!("\nInitializing SCF calculation...");
    scf.init_basis(elements, basis_map);
    scf.init_geometry(coords_vec, elements);
    scf.init_density_matrix();
    scf.init_fock_matrix();

    info!("\nStarting SCF cycle...\n");
    scf.scf_cycle();
}

/// Run a spin-polarized SCF calculation
fn run_spin_scf_calculation(
    config: Config,
    args: Args,
    charge: i32,
    multiplicity: usize,
) -> Result<()> {
    use basis::cgto::Basis631G;
    
    let (elements, coords_vec) = prepare_geometry(&config)?;
    let basis_map = prepare_basis_sets(&elements)?;

    let mut spin_scf = SpinSCF::<Basis631G>::new();
    
    // Apply configuration parameters
    spin_scf.density_mixing = args.density_mixing
        .or(config.scf_params.density_mixing)
        .unwrap();
    spin_scf.max_cycle = args.max_cycle
        .or(config.scf_params.max_cycle)
        .unwrap();
    spin_scf.set_convergence_threshold(
        config.scf_params.convergence_threshold.unwrap()
    );
    spin_scf.set_charge(charge);
    spin_scf.set_multiplicity(multiplicity);

    // Enable DIIS if configured
    if config.is_diis_enabled() {
        let diis_size = config.diis_subspace_size();
        info!("Enabling DIIS acceleration with subspace size {}", diis_size);
        spin_scf.enable_diis(diis_size);
    } else {
        info!("DIIS acceleration disabled");
    }

    // Initialize and run SCF
    initialize_and_run_scf(&mut spin_scf, &elements, &coords_vec, basis_map.clone());

    // Report results
    info!("\nSpinSCF calculation finished.");
    info!("\nFinal Energy Levels:");
    info!("  Alpha electrons:");
    for (i, energy) in spin_scf.e_level_alpha.iter().enumerate() {
        info!("    Level {}: {:.8} au", i + 1, energy);
    }
    info!("  Beta electrons:");
    for (i, energy) in spin_scf.e_level_beta.iter().enumerate() {
        info!("    Level {}: {:.8} au", i + 1, energy);
    }

    let final_energy = spin_scf.calculate_total_energy();
    info!("\nTotal energy: {:.10} au", final_energy);

    // Run geometry optimization if requested
    if should_optimize(&args, &config) {
        run_optimization(&mut spin_scf, &args, &config, coords_vec, elements)?;
    }

    Ok(())
}

/// Run a simple (restricted) SCF calculation
fn run_simple_scf_calculation(config: Config, args: Args) -> Result<()> {
    use basis::cgto::Basis631G;
    
    let (elements, coords_vec) = prepare_geometry(&config)?;
    let basis_map = prepare_basis_sets(&elements)?;

    let mut scf = SimpleSCF::<Basis631G>::new();

    // Apply configuration parameters (args override config)
    scf.density_mixing = args.density_mixing
        .or(config.scf_params.density_mixing)
        .unwrap();
    scf.max_cycle = args.max_cycle
        .or(config.scf_params.max_cycle)
        .unwrap();
    scf.set_convergence_threshold(
        config.scf_params.convergence_threshold.unwrap()
    );

    // Enable DIIS if configured
    if config.is_diis_enabled() {
        let diis_size = config.diis_subspace_size();
        info!("Enabling DIIS acceleration with subspace size {}", diis_size);
        scf.enable_diis(diis_size);
    } else {
        info!("DIIS acceleration disabled");
    }

    // Initialize and run SCF
    initialize_and_run_scf(&mut scf, &elements, &coords_vec, basis_map);

    // Report results
    info!("\nSCF calculation finished.");
    let final_energy = scf.calculate_total_energy();
    info!("\nFinal Energy Levels:");
    for (i, energy) in scf.e_level.iter().enumerate() {
        info!("  Level {}: {:.8} au", i + 1, energy);
    }
    info!("\nHartree-Fock Total Energy: {:.10} au", final_energy);

    // Run MP2 calculation if requested
    if config.is_mp2_enabled() {
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
        
        let total_mp2_energy = final_energy + correlation_energy;
        
        info!("\n===========================================");
        info!("        MP2 Results Summary");
        info!("===========================================");
        info!("Hartree-Fock energy:       {:.10} au", final_energy);
        info!("MP2 correlation energy:    {:.10} au", correlation_energy);
        info!("Total MP2 energy:          {:.10} au", total_mp2_energy);
        info!("===========================================\n");
    }

    // Run CCSD calculation if requested
    if config.is_ccsd_enabled() {
        info!("\n===========================================");
        info!("       Starting CCSD Calculation");
        info!("===========================================");
        
        let max_iterations = config.ccsd_max_iterations();
        let convergence_threshold = config.ccsd_convergence_threshold();
        
        let mut ccsd = scf.create_ccsd(max_iterations, convergence_threshold);
        let correlation_energy = ccsd.solve();
        
        let total_ccsd_energy = final_energy + correlation_energy;
        
        info!("\n===========================================");
        info!("        CCSD Results Summary");
        info!("===========================================");
        info!("Hartree-Fock energy:       {:.10} au", final_energy);
        info!("CCSD correlation energy:   {:.10} au", correlation_energy);
        info!("Total CCSD energy:         {:.10} au", total_ccsd_energy);
        
        // Calculate and report T1 diagnostic
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

    // Run CI calculation if requested
    if config.is_ci_enabled() {
        info!("\n===========================================");
        info!("       Starting CI Calculation");
        info!("===========================================");
        
        let max_states = config.ci_max_states();
        let convergence_threshold = config.ci_convergence_threshold();
        let method = config.ci_method();
        
        let mut ci = scf.create_ci(max_states, convergence_threshold);
        
        match method.to_lowercase().as_str() {
            "cis" => {
                info!("Running CIS (Configuration Interaction Singles) for excited states");
                let excitation_energies = ci.calculate_cis_energies(max_states);
                
                info!("\n===========================================");
                info!("        CIS Results Summary");
                info!("===========================================");
                info!("Hartree-Fock ground state: {:.10} au", final_energy);
                info!("\nExcited States:");
                for (i, &exc_energy) in excitation_energies.iter().enumerate() {
                    let total_energy = final_energy + exc_energy;
                    info!("  State {}: Excitation = {:.6} au ({:.2} eV), Total = {:.10} au",
                          i + 1, exc_energy, exc_energy * 27.2114, total_energy);
                }
                info!("===========================================\n");
            }
            "cisd" => {
                info!("Running CISD (Configuration Interaction Singles and Doubles)");
                let correlation_energy = ci.calculate_cisd_energy();
                let total_cisd_energy = final_energy + correlation_energy;
                
                info!("\n===========================================");
                info!("        CISD Results Summary");
                info!("===========================================");
                info!("Hartree-Fock energy:       {:.10} au", final_energy);
                info!("CISD correlation energy:   {:.10} au", correlation_energy);
                info!("Total CISD energy:         {:.10} au", total_cisd_energy);
                info!("===========================================\n");
            }
            _ => {
                info!("Unknown CI method: {}. Using CISD.", method);
                let correlation_energy = ci.calculate_cisd_energy();
                let total_cisd_energy = final_energy + correlation_energy;
                
                info!("\n===========================================");
                info!("        CISD Results Summary");
                info!("===========================================");
                info!("Hartree-Fock energy:       {:.10} au", final_energy);
                info!("CISD correlation energy:   {:.10} au", correlation_energy);
                info!("Total CISD energy:         {:.10} au", total_cisd_energy);
                info!("===========================================\n");
            }
        }
    }

    // Run geometry optimization if requested
    if should_optimize(&args, &config) {
        run_optimization(&mut scf, &args, &config, coords_vec, elements)?;
    }

    Ok(())
}

/// Prepare molecular geometry from configuration
fn prepare_geometry(config: &Config) -> Result<(Vec<Element>, Vec<Vector3<f64>>)> {
    info!("\nPreparing geometry...");
    let mut elements = Vec::new();
    let mut coords_vec = Vec::new();

    for atom_config in &config.geometry {
        let element = Element::from_symbol(&atom_config.element)
            .ok_or_else(|| color_eyre::eyre::eyre!("Invalid element symbol: {}", atom_config.element))?;
        let coords = Vector3::new(
            atom_config.coords[0],
            atom_config.coords[1],
            atom_config.coords[2],
        );
        elements.push(element);
        coords_vec.push(coords);
    }

    Ok((elements, coords_vec))
}

/// Prepare basis sets for all elements
fn prepare_basis_sets(
    elements: &[Element],
) -> Result<HashMap<&str, &'static basis::cgto::Basis631G>> {
    use basis::cgto::Basis631G;
    
    info!("\nPreparing basis sets...");
    
    let mut basis_map: HashMap<&str, &Basis631G> = HashMap::new();
    for elem in elements {
        let symbol = elem.get_symbol();
        if basis_map.contains_key(symbol) {
            continue;
        }

        let basis = fetch_basis(symbol)?;
        // Allocate on the heap to ensure a stable memory address
        let basis_ref: &Basis631G = Box::leak(Box::new(basis));
        basis_map.insert(symbol, basis_ref);
    }

    Ok(basis_map)
}

/// Check if geometry optimization should be performed
fn should_optimize(args: &Args, config: &Config) -> bool {
    args.optimize || config.optimization.as_ref().and_then(|o| o.enabled).unwrap_or(false)
}

/// Run geometry optimization
fn run_optimization<S: SCF + Clone>(
    scf: &mut S,
    args: &Args,
    config: &Config,
    coords: Vec<Vector3<f64>>,
    elements: Vec<Element>,
) -> Result<()> {
    use std::fs::File;
    
    info!("\nStarting geometry optimization...");

    // Get optimization parameters
    let opt_params = config
        .optimization
        .as_ref()
        .cloned()
        .unwrap_or_default()
        .with_defaults();
    
    let algorithm = args.opt_algorithm
        .clone()
        .or(opt_params.algorithm)
        .unwrap_or_else(|| "cg".to_string());
    let max_iterations = args.opt_max_iterations
        .or(opt_params.max_iterations)
        .unwrap_or(50);
    let convergence = args.opt_convergence
        .or(opt_params.convergence_threshold)
        .unwrap_or(1e-4);
    let step_size = args.opt_step_size
        .or(opt_params.step_size)
        .unwrap_or(0.1);

    info!("Optimization parameters:");
    info!("  Algorithm: {}", algorithm);
    info!("  Max iterations: {}", max_iterations);
    info!("  Convergence threshold: {:.6e}", convergence);
    info!("  Step size: {:.4}", step_size);

    // Run optimization with the selected algorithm
    let (optimized_coords, final_energy) = match algorithm.to_lowercase().as_str() {
        "cg" => {
            let mut optimizer = CGOptimizer::new(scf, max_iterations, convergence);
            optimizer.set_step_size(step_size);
            optimizer.init(coords, elements.clone());
            optimizer.optimize()
        }
        "sd" => {
            let mut optimizer = SteepestDescentOptimizer::new(scf, max_iterations, convergence);
            optimizer.set_step_size(step_size);
            optimizer.init(coords, elements.clone());
            optimizer.optimize()
        }
        _ => {
            info!("Unknown optimization algorithm: {}, defaulting to CG", algorithm);
            let mut optimizer = CGOptimizer::new(scf, max_iterations, convergence);
            optimizer.set_step_size(step_size);
            optimizer.init(coords, elements.clone());
            optimizer.optimize()
        }
    };

    // Print optimized geometry
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

    // Save optimized geometry if output file is specified
    if let Some(ref output_file) = args.output {
        let mut file = File::create(output_file)?;
        print_optimized_geometry(&mut file, &optimized_coords, &elements, final_energy)?;
    }

    Ok(())
}
