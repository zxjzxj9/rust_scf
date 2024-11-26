// import necessary libraries
use std::f64;
// define a struct for Gaussian-type orbital
#[derive(Debug)]
pub struct GTO {
    pub alpha: f64,
    pub l_xyz: (i32, i32, i32),
    pub xyz: (f64, f64, f64),
    pub norm: f64,
}
