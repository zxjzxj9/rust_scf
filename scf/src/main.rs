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

mod config;
mod io;
mod optim_impl;
mod scf_impl;

use config::{Args, Config};
use io::{fetch_basis, print_optimized_geometry, setup_output};
use optim_impl::{CGOptimizer, GeometryOptimizer, SteepestDescentOptimizer};
use scf_impl::{SimpleSCF, SpinSCF, SCF};

fn main() -> Result<()> {
    println!("DEBUG: Starting main function");
    color_eyre::install()?;
    println!("DEBUG: color_eyre installed");
    
    let args = Args::parse();
    println!("DEBUG: Args parsed: {:?}", args.config_file);
    setup_output(args.output.as_ref());
    println!("DEBUG: Output setup complete");

    // Load and parse configuration
    info!("Reading configuration from: {}", args.config_file);
    println!("DEBUG: About to read config file");
    let config_content = fs::read_to_string(&args.config_file)
        .wrap_err_with(|| format!("Unable to read configuration file: {}", args.config_file))?;
    println!("DEBUG: Config file read successfully");

    let config: Config = serde_yml::from_str::<Config>(&config_content)
        .wrap_err("Failed to parse configuration file")?
        .with_defaults();
    println!("DEBUG: Config parsed successfully");

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
    println!("DEBUG: SpinSCF created");
    
    // Apply configuration parameters
    spin_scf.density_mixing = config.scf_params.density_mixing.unwrap();
    spin_scf.max_cycle = config.scf_params.max_cycle.unwrap();
    spin_scf.set_charge(charge);
    spin_scf.set_multiplicity(multiplicity);

    // Override with command-line arguments if provided
    if let Some(dm) = args.density_mixing {
        info!("Overriding density_mixing with: {}", dm);
        spin_scf.density_mixing = dm;
    }
    if let Some(mc) = args.max_cycle {
        info!("Overriding max_cycle with: {}", mc);
        spin_scf.max_cycle = mc;
    }

    // Initialize and run SCF
    info!("\nInitializing SpinSCF calculation...");
    println!("DEBUG: About to init basis");
    spin_scf.init_basis(&elements, basis_map.clone());
    println!("DEBUG: About to init geometry");
    spin_scf.init_geometry(&coords_vec, &elements);
    println!("DEBUG: About to init density matrix");
    spin_scf.init_density_matrix();
    println!("DEBUG: About to init fock matrix");
    spin_scf.init_fock_matrix();
    println!("DEBUG: SpinSCF initialization complete");

    info!("\nStarting SpinSCF cycle...\n");
    println!("DEBUG: About to start SpinSCF cycle");
    spin_scf.scf_cycle();
    println!("DEBUG: SpinSCF cycle complete");

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
    println!("DEBUG: SimpleSCF created");

    // Apply configuration parameters
    scf.density_mixing = config.scf_params.density_mixing.unwrap();
    scf.max_cycle = config.scf_params.max_cycle.unwrap();

    // Override with command-line arguments if provided
    if let Some(dm) = args.density_mixing {
        info!("Overriding density_mixing with: {}", dm);
        scf.density_mixing = dm;
    }
    if let Some(mc) = args.max_cycle {
        info!("Overriding max_cycle with: {}", mc);
        scf.max_cycle = mc;
    }

    // Initialize and run SCF
    info!("\nInitializing SCF calculation...");
    println!("DEBUG: About to init basis");
    scf.init_basis(&elements, basis_map);
    println!("DEBUG: About to init geometry");
    scf.init_geometry(&coords_vec, &elements);
    println!("DEBUG: About to init density matrix");
    scf.init_density_matrix();
    println!("DEBUG: About to init fock matrix");
    scf.init_fock_matrix();
    println!("DEBUG: SCF initialization complete");

    info!("\nStarting SCF cycle...\n");
    println!("DEBUG: About to start SCF cycle");
    scf.scf_cycle();
    println!("DEBUG: SCF cycle complete");

    // Report results
    info!("\nSCF calculation finished.");
    info!("\nFinal Energy Levels:");
    for (i, energy) in scf.e_level.iter().enumerate() {
        info!("  Level {}: {:.8} au", i + 1, energy);
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
    println!("DEBUG: Starting basis set preparation");
    
    let mut basis_map: HashMap<&str, &Basis631G> = HashMap::new();
    for elem in elements {
        let symbol = elem.get_symbol();
        if basis_map.contains_key(symbol) {
            continue;
        }

        let basis = fetch_basis(symbol)?;
        println!("DEBUG: Fetched basis for {}", symbol);
        // Allocate on the heap to ensure a stable memory address
        let basis_ref: &Basis631G = Box::leak(Box::new(basis));
        basis_map.insert(symbol, basis_ref);
        println!("DEBUG: Added {} to basis_map", symbol);
    }
    println!("DEBUG: Basis set preparation complete");

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
