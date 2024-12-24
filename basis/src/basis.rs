use std::fs::File;
use std::io::{Read, Write};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use serde_pickle;
use crate::gto::GTO;


#[derive(Debug, Serialize, Deserialize)]
pub struct ContractedGTO {
    pub primitives: Vec<GTO>,
    pub coefficients: Vec<f64>,
    // shell_type: 1s, 2s, 2px, 2py, 2pz, ...
    pub shell_type: str,
    pub n: i32,
    pub l: i32,
    pub m: i32,
    pub s: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Basis631G {
    // define of the basis set
    pub core: ContractedGTO,
    pub valence_inner: ContractedGTO,
    pub valence_outer: GTO,
}

impl Basis631G {
    // Serialize to pickle format
    pub fn to_pickle(&self) -> Result<Vec<u8>, serde_pickle::Error> {
        let options = serde_pickle::SerOptions::new();
        serde_pickle::to_vec(self, options)
    }

    // Deserialize from pickle format
    pub fn from_pickle(bytes: &[u8]) -> Result<Self, serde_pickle::Error> {
        let options = serde_pickle::DeOptions::new();
        serde_pickle::from_slice(bytes, options)
    }

    // Save to file in pickle format
    pub fn save_to_file(&self, filename: &str) -> std::io::Result<()> {
        let serialized = self.to_pickle()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = File::create(filename)?;
        file.write_all(&serialized)
    }

    // Load from file in pickle format
    pub fn load_from_file(filename: &str) -> std::io::Result<Self> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        Self::from_pickle(&buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
