use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use crate::basis::gto::GTO;

#[derive(Debug, Serialize, Deserialize)]
pub enum ShellType {
    S,  // l = 0
    P,  // l = 1
    D,  // l = 2
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContractedGTO {
    pub primitives: Vec<GTO>,
    pub coefficients: Vec<f64>,
    pub shell_type: ShellType,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Basis631G {
    pub core: ContractedGTO,
    pub valence_inner: ContractedGTO,
    pub valence_outer: GTO,
}

