use basis::basis::{AOBasis, Basis};
use basis::cgto::Basis631G;
use clap::Parser;
use color_eyre::eyre::Result;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::process::{id, Command};
use std::rc::Rc;
use std::{fs, io};

mod scf;
use crate::optim::{GeometryOptimizer, CGOptimizer, SteepestDescentOptimizer};
mod optim;  // Make sure this is included with your other mod declarations
mod simple;
use crate::scf::SCF;
use crate::simple::SimpleSCF;
use tracing::{event, info, span, Level};
use tracing_subscriber::{fmt::layer, layer::SubscriberExt, util::SubscriberInitExt, Registry};

// Define a configuration struct to hold YAML data
#[derive(Debug, Deserialize, Serialize)]
struct Config {
    geometry: Vec<Atom>,
    basis_sets: HashMap<String, String>, // Element symbol -> basis set name (string for now)
    scf_params: ScfParams,
    optimization: Option<OptimizationParams>, // Add this field
}

#[derive(Debug, Deserialize, Serialize)]
struct OptimizationParams {
    enabled: Option<bool>,
    algorithm: Option<String>,  // "cg" or "sd" for steepest descent
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

fn fetch_basis(atomic_symbol: &str) -> Basis631G {
    let url = format!(
        "https://www.basissetexchange.org/api/basis/6-31g/format/nwchem?elements={}",
        atomic_symbol
    );
    let basis_str = reqwest::blocking::get(url).unwrap().text().unwrap();
    Basis631G::parse_nwchem(&basis_str)
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

// set default values for ScfParams
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
    #[arg(short, long, default_value = "output.txt")]
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

fn main() -> Result<()> {
    color_eyre::install()?;

    let args = Args::parse();

    // Choose the writer based on the presence of the output file path.
    match args.output {
        Some(ref path) => {
            info!("Output will be written to: {}", path);
            let log = File::create(path).expect("Could not create file");
            let file_layer = layer().with_writer(log);
            // install writer to log, using tracing
            Registry::default().with(file_layer).init()
        }
        None => {
            info!("Output will be printed to stdout");
        }
    };

    // 1. Read YAML configuration file
    info!("Reading configuration from: {}", args.config_file);
    let config_file_content =
        fs::read_to_string(&args.config_file).expect("Unable to read configuration file");

    let mut config: Config =
        serde_yaml::from_str(&config_file_content).expect("Unable to parse configuration file");

    config.scf_params = ScfParams::default();
    info!("Configuration loaded:\n{:?}", config);
    let mut scf = SimpleSCF::new();

    // 2. Override parameters from command line if provided
    if let Some(dm) = args.density_mixing {
        info!("Overriding density_mixing with: {}", dm);
        config.scf_params.density_mixing = Some(dm);
        scf.density_mixing = dm;
    }
    if let Some(mc) = args.max_cycle {
        info!("Overriding max_cycle with: {}", mc);
        config.scf_params.max_cycle = Some(mc);
        scf.max_cycle = mc;
    }

    // 3. Prepare Basis Sets (This part needs to be adapted to your basis library)
    info!("\nPreparing basis sets...");

    // // 4. Prepare Geometry
    info!("\nPreparing geometry...");
    let mut elements = Vec::new();
    let mut coords_vec = Vec::new();
    for atom_config in &config.geometry {
        let element = Element::from_symbol(&atom_config.element)
            .expect(&format!("Invalid element symbol: {}", atom_config.element));
        let coords = Vector3::new(
            atom_config.coords[0],
            atom_config.coords[1],
            atom_config.coords[2],
        );
        elements.push(element);
        coords_vec.push(coords);
    }

    // HashMap from symbol to a reference of the stored basis.
    let mut basis_map: HashMap<&str, &Basis631G> = HashMap::new();

    for elem in &elements {
        let symbol = elem.get_symbol();
        if basis_map.contains_key(symbol) {
            continue;
        }

        let basis = fetch_basis(symbol);
        // Allocate on the heap to ensure a stable memory address.
        let basis_ref: &Basis631G = Box::leak(Box::new(basis));
        basis_map.insert(symbol, basis_ref);
    }

    // 5. Initialize and run SCF
    info!("\nInitializing SCF calculation...");

    scf.init_basis(&elements, basis_map);
    scf.init_geometry(&coords_vec, &elements);
    scf.init_density_matrix();
    scf.init_fock_matrix();

    info!("\nStarting SCF cycle...\n");
    scf.scf_cycle();

    info!("\nSCF calculation finished.");
    // make it print the final energy levels, print it prettier
    info!("\nFinal Energy Levels:");
    // info!("{:?}", scf.e_level);
    for (i, energy) in scf.e_level.iter().enumerate() {
        info!("  Level {}: {:.8} au", i + 1, energy);
    }

    // Add after SCF calculation
    if args.optimize || config.optimization.as_ref().and_then(|o| o.enabled).unwrap_or(false) {
        info!("\nStarting geometry optimization...");

        // Get optimization parameters
        let opt_params = config.optimization.unwrap_or_default();
        let algorithm = args.opt_algorithm
            .clone()
            .or_else(|| opt_params.algorithm.clone())
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

        // Create optimizer
        if algorithm.to_lowercase() == "cg" {
            // Create CG optimizer
            let mut optimizer = CGOptimizer::new(&mut scf, max_iterations, convergence);
            optimizer.set_step_size(step_size);

            // Initialize with current geometry
            optimizer.init(coords_vec.clone(), elements.clone());

            // Run optimization
            let (optimized_coords, final_energy) = optimizer.optimize();

            // Print optimized geometry
            info!("\nOptimized geometry:");
            for (i, (coord, elem)) in optimized_coords.iter().zip(elements.iter()).enumerate() {
                info!("  Atom {}: {} at [{:.6}, {:.6}, {:.6}]",
                i + 1, elem.get_symbol(), coord.x, coord.y, coord.z);
            }
            info!("Final energy: {:.10} au", final_energy);

            // Save optimized geometry if needed
            if let Some(output_file) = args.output {
                let mut file = File::create(output_file)?;
                writeln!(file, "Optimized geometry:")?;
                for (i, (coord, elem)) in optimized_coords.iter().zip(elements.iter()).enumerate() {
                    writeln!(file, "  Atom {}: {} at [{:.6}, {:.6}, {:.6}]",
                    i + 1, elem.get_symbol(), coord.x, coord.y, coord.z)?;
                }
                writeln!(file, "Final energy: {:.10} au", final_energy)?;
            }
        } else if algorithm.to_lowercase() == "sd" {
            // Similar code for steepest descent optimizer
            info!("Using Steepest Descent optimizer");
            let mut optimizer: SteepestDescentOptimizer<SimpleSCF<Basis631G>> = GeometryOptimizer::new(&mut scf, max_iterations, convergence);
            optimizer.set_step_size(step_size);
            optimizer.init(coords_vec.clone(), elements.clone());
            let (optimized_coords, final_energy) = optimizer.optimize();
            info!("\nOptimized geometry:");
            for (i, (coord, elem)) in optimized_coords.iter().zip(elements.iter()).enumerate() {
                info!("  Atom {}: {} at [{:.6}, {:.6}, {:.6}]",
                i + 1, elem.get_symbol(), coord.x, coord.y, coord.z);
            }
            info!("Final energy: {:.10} au", final_energy);
        } else {
            info!("Unknown optimization algorithm: {}", algorithm);
        }
    }

    Ok(())
}
