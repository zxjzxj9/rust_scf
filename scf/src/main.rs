use basis::basis::{AOBasis, Basis};
use periodic_table_on_an_enum::Element;
use nalgebra::Vector3;
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use clap::Parser;
use std::fs;
use basis::cgto::Basis631G;

mod scf;
mod simple;
use crate::scf::SCF;
use crate::simple::SimpleSCF;

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
}

fn main() {
    let args = Args::parse();

    // 1. Read YAML configuration file
    println!("Reading configuration from: {}", args.config_file);
    let config_file_content = fs::read_to_string(&args.config_file)
        .expect("Unable to read configuration file");

    let mut config: Config = serde_yaml::from_str(&config_file_content)
        .expect("Unable to parse configuration file");

    println!("Configuration loaded:\n{:?}", config);

    // 2. Override parameters from command line if provided
    if let Some(dm) = args.density_mixing {
        println!("Overriding density_mixing with: {}", dm);
        config.scf_params.density_mixing = Some(dm);
    }
    if let Some(mc) = args.max_cycle {
        println!("Overriding max_cycle with: {}", mc);
        config.scf_params.max_cycle = Some(mc);
    }
    if let Some(diis_size) = args.diis_subspace_size {
        println!("Overriding diis_subspace_size with: {}", diis_size);
        config.scf_params.diis_subspace_size = Some(diis_size);
    }
    if let Some(conv_thresh) = args.convergence_threshold {
        println!("Overriding convergence_threshold with: {}", conv_thresh);
        config.scf_params.convergence_threshold = Some(conv_thresh);
    }

    // 3. Prepare Basis Sets (This part needs to be adapted to your basis library)
    println!("\nPreparing basis sets...");


    // // 4. Prepare Geometry
    println!("\nPreparing geometry...");
    let mut elements = Vec::new();
    let mut coords_vec = Vec::new();
    for atom_config in &config.geometry {
        let element =  Element::from_symbol(&atom_config.element)
            .expect(&format!("Invalid element symbol: {}", atom_config.element));
        let coords = Vector3::new(
            atom_config.coords[0],
            atom_config.coords[1],
            atom_config.coords[2],
        );
        elements.push(element);
        coords_vec.push(coords);
    }

    let mut basis_map: HashMap<&str, &'static Basis631G> = HashMap::new();

    for elem in &elements {
        let symbol = elem.get_symbol();
        if basis_map.contains_key(symbol) {
            continue;
        }

        let basis = fetch_basis(symbol);

        // mutable borrow of basis_storage
        basis_map.insert(symbol, &basis);
    }


    // 5. Initialize and run SCF
    println!("\nInitializing SCF calculation...");
    let mut scf = SimpleSCF::new();

    scf.init_basis(&elements, basis_map);
    scf.init_geometry(&coords_vec, &elements);
    scf.init_density_matrix();
    scf.init_fock_matrix();

    println!("\nStarting SCF cycle...\n");
    scf.scf_cycle();

    println!("\nSCF calculation finished.");
    println!("Final Energy Levels:\n{:?}", scf.e_level);
    // You can add code here to print other results like total energy if you implement it in SimpleSCF
}