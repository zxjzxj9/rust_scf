use crate::app::basis::BasisMap;
use ::basis::basis::AOBasis;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;

/// In-memory representation of everything needed to run an SCF calculation.
#[derive(Clone)]
pub struct CalculationWorkspace<B: AOBasis + 'static> {
    pub elements: Vec<Element>,
    pub coords: Vec<Vector3<f64>>,
    pub basis: BasisMap<B>,
}

impl<B: AOBasis> CalculationWorkspace<B> {
    pub fn new(elements: Vec<Element>, coords: Vec<Vector3<f64>>, basis: BasisMap<B>) -> Self {
        Self {
            elements,
            coords,
            basis,
        }
    }

    pub fn basis_map(&self) -> BasisMap<B> {
        self.basis.clone()
    }
}
