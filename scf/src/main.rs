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
use crate::optim::{GeometryOptimizer, CGOptimizer};
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

    // if let Some(diis_size) = args.diis_subspace_size {
    //     info!("Overriding diis_subspace_size with: {}", diis_size);
    //     config.scf_params.diis_subspace_size = Some(diis_size);
    //     scf.diis_subspace_size = diis_size;
    // }
    // if let Some(conv_thresh) = args.convergence_threshold {
    //     info!("Overriding convergence_threshold with: {}", conv_thresh);
    //     config.scf_params.convergence_threshold = Some(conv_thresh);
    //     scf.convergence_threshold = conv_thresh;
    // }

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

    Ok(())
}
