extern crate nalgebra as na;
use na::Vector3;
use periodic_table_on_an_enum::Element;

trait SCF {
    fn init_geometry(&self, coords: &Vec<Vector3<f64>>, elems: &Vec<Element>);
    fn init_density_matrix(&self);
    fn init_fock_matrix(&self);
}
