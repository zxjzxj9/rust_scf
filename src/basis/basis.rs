use nalgebra::Vector3;

trait Basis {
    fn evaluate(&self, r: &Vector3<f64>) -> f64;
    fn overlap(&self, other: &Self) -> f64;
    fn kinetic(&self, other: &Self) -> f64;
    fn potential(&self, other: &Self, R: &Vector3<f64>) -> f64;
    fn two_electron(&self, other: &Self) -> f64;
}