use basis::cgto::Basis631G;
use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::fs;

mod scf;
use crate::optim::{CGOptimizer, GeometryOptimizer, SteepestDescentOptimizer};
mod optim;
mod simple;
use crate::scf::SCF;
use crate::simple::SimpleSCF;
use tracing::info;
use tracing_subscriber::{fmt::layer, layer::SubscriberExt, util::SubscriberInitExt, Registry};

#[derive(Debug, Deserialize, Serialize)]
struct Config {
    geometry: Vec<Atom>,
    basis_sets: HashMap<String, String>,
    scf_params: ScfParams,
    optimization: Option<OptimizationParams>,
}

#[derive(Debug, Deserialize, Serialize)]
struct OptimizationParams {
    enabled: Option<bool>,
    algorithm: Option<String>,
    max_iterations: Option<usize>,
    convergence_threshold: Option<f64>,
    step_size: Option<f64>,
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

fn fetch_basis(atomic_symbol: &str) -> Result<Basis631G> {
    println!("DEBUG: Attempting to fetch basis set for {}", atomic_symbol);
    let url = format!(
        "https://www.basissetexchange.org/api/basis/6-31g/format/nwchem?elements={}",
        atomic_symbol
    );
    println!("DEBUG: URL: {}", url);
    let response = reqwest::blocking::get(&url)
        .wrap_err_with(|| format!("Failed to fetch basis set for {}", atomic_symbol))?;
    println!("DEBUG: HTTP request successful");
    let basis_str = response
        .text()
        .wrap_err("Failed to get response text from basis set API")?;
    println!("DEBUG: Got response text, length: {}", basis_str.len());
    Ok(Basis631G::parse_nwchem(&basis_str))
}

#[derive(Debug, Deserialize, Serialize)]
struct Atom {
    element: String,
    coords: [f64; 3],
}

#[derive(Debug, Deserialize, Serialize)]
struct ScfParams {
    density_mixing: Option<f64>,
    max_cycle: Option<usize>,
    diis_subspace_size: Option<usize>,
    convergence_threshold: Option<f64>,
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

/// Simple SCF calculation with YAML configuration
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the YAML configuration file
    #[arg(short, long, default_value = "config.yaml")]
    config_file: String,

    /// Override density mixing parameter
    #[arg(long)]
    density_mixing: Option<f64>,

    /// Override maximum SCF cycles
    #[arg(long)]
    max_cycle: Option<usize>,

    /// Override DIIS subspace size
    #[arg(long)]
    diis_subspace_size: Option<usize>,

    /// Override convergence threshold
    #[arg(long)]
    convergence_threshold: Option<f64>,

    /// Override output file: (default stdout)
    #[arg(short, long)]
    output: Option<String>,

    /// Enable geometry optimization
    #[arg(long)]
    optimize: bool,

    /// Optimization algorithm (cg or sd)
    #[arg(long, default_value = "cg")]
    opt_algorithm: Option<String>,

    /// Maximum optimization iterations
    #[arg(long)]
    opt_max_iterations: Option<usize>,

    /// Optimization convergence threshold
    #[arg(long)]
    opt_convergence: Option<f64>,

    /// Step size for optimization
    #[arg(long)]
    opt_step_size: Option<f64>,
}

fn setup_output(output_path: Option<&String>) {
    match output_path {
        Some(path) => {
            info!("Output will be written to: {}", path);
            if let Ok(log) = File::create(path) {
                let file_layer = layer()
                    .with_writer(log);
                Registry::default().with(file_layer).init();
            } else {
                eprintln!("Could not create output file: {}", path);
            }
        }
        None => {
            // Initialize tracing for stdout
            let stdout_layer = layer().with_writer(std::io::stdout);
            Registry::default().with(stdout_layer).init();
            info!("Output will be printed to stdout");
        }
    }
}

fn print_optimized_geometry<W: Write>(
    writer: &mut W,
    coords: &[Vector3<f64>],
    elements: &[Element],
    energy: f64,
) -> Result<()> {
    writeln!(writer, "Optimized geometry:")?;
    for (i, (coord, elem)) in coords.iter().zip(elements.iter()).enumerate() {
        writeln!(
            writer,
            "  Atom {}: {} at [{:.6}, {:.6}, {:.6}]",
            i + 1,
            elem.get_symbol(),
            coord.x,
            coord.y,
            coord.z
        )?;
    }
    writeln!(writer, "Final energy: {:.10} au", energy)?;
    Ok(())
}

fn main() -> Result<()> {
    println!("DEBUG: Starting main function");
    color_eyre::install()?;
    println!("DEBUG: color_eyre installed");
    let args = Args::parse();
    println!("DEBUG: Args parsed: {:?}", args.config_file);
    setup_output(args.output.as_ref());
    println!("DEBUG: Output setup complete");

    // 1. Read YAML configuration file
    info!("Reading configuration from: {}", args.config_file);
    println!("DEBUG: About to read config file");
    let config_file_content = fs::read_to_string(&args.config_file)
        .wrap_err_with(|| format!("Unable to read configuration file: {}", args.config_file))?;
    println!("DEBUG: Config file read successfully");

    let mut config: Config = serde_yml::from_str(&config_file_content)
        .wrap_err("Failed to parse configuration file")?;
    println!("DEBUG: Config parsed successfully");

    // Apply defaults to missing values but don't overwrite existing ones
    let default_params = ScfParams::default();
    if config.scf_params.density_mixing.is_none() {
        config.scf_params.density_mixing = default_params.density_mixing;
    }
    if config.scf_params.max_cycle.is_none() {
        config.scf_params.max_cycle = default_params.max_cycle;
    }
    if config.scf_params.diis_subspace_size.is_none() {
        config.scf_params.diis_subspace_size = default_params.diis_subspace_size;
    }
    if config.scf_params.convergence_threshold.is_none() {
        config.scf_params.convergence_threshold = default_params.convergence_threshold;
    }

    info!("Configuration loaded:\n{:?}", config);
    println!("DEBUG: About to create SimpleSCF");

    // Init SCF with params from config
    let mut scf = SimpleSCF::new();
    println!("DEBUG: SimpleSCF created");
    scf.density_mixing = config.scf_params.density_mixing.unwrap_or(0.5);
    scf.max_cycle = config.scf_params.max_cycle.unwrap_or(100);

    // 2. Override parameters from command line if provided
    if let Some(dm) = args.density_mixing {
        info!("Overriding density_mixing with: {}", dm);
        scf.density_mixing = dm;
    }
    if let Some(mc) = args.max_cycle {
        info!("Overriding max_cycle with: {}", mc);
        scf.max_cycle = mc;
    }

    // 3. Prepare Geometry
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

    // 4. Prepare Basis Sets
    info!("\nPreparing basis sets...");
    println!("DEBUG: Starting basis set preparation");
    let mut basis_map: HashMap<&str, &Basis631G> = HashMap::new();
    for elem in &elements {
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

    // 5. Initialize and run SCF
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

    info!("\nSCF calculation finished.");
    info!("\nFinal Energy Levels:");
    for (i, energy) in scf.e_level.iter().enumerate() {
        info!("  Level {}: {:.8} au", i + 1, energy);
    }

    // 6. Run geometry optimization if requested
    let should_optimize = args.optimize ||
        config.optimization.as_ref().and_then(|o| o.enabled).unwrap_or(false);

    if should_optimize {
        info!("\nStarting geometry optimization...");

        // Get optimization parameters
        let opt_params = config.optimization.unwrap_or_default();
        let algorithm = args.opt_algorithm
            .or_else(|| opt_params.algorithm)
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
                let mut optimizer = CGOptimizer::new(&mut scf, max_iterations, convergence);
                optimizer.set_step_size(step_size);
                optimizer.init(coords_vec.clone(), elements.clone());
                optimizer.optimize()
            },
            "sd" => {
                let mut optimizer = SteepestDescentOptimizer::new(&mut scf, max_iterations, convergence);
                optimizer.set_step_size(step_size);
                optimizer.init(coords_vec.clone(), elements.clone());
                optimizer.optimize()
            },
            _ => {
                info!("Unknown optimization algorithm: {}, defaulting to CG", algorithm);
                let mut optimizer = CGOptimizer::new(&mut scf, max_iterations, convergence);
                optimizer.set_step_size(step_size);
                optimizer.init(coords_vec.clone(), elements.clone());
                optimizer.optimize()
            }
        };

        // Print optimized geometry
        info!("\nOptimized geometry:");
        for (i, (coord, elem)) in optimized_coords.iter().zip(elements.iter()).enumerate() {
            info!("  Atom {}: {} at [{:.6}, {:.6}, {:.6}]",
                i + 1, elem.get_symbol(), coord.x, coord.y, coord.z);
        }
        info!("Final energy: {:.10} au", final_energy);

        // Save optimized geometry if needed
        if let Some(ref output_file) = args.output {
            let mut file = File::create(output_file)?;
            print_optimized_geometry(&mut file, &optimized_coords, &elements, final_energy)?;
        }
    }

    Ok(())
}