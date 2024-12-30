use basis::cgto::Basis631G;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
// use basis;

// #[pyfunction]
// fn load_basis_from_py_str(input: &str) -> Basis631G {
//     Basis631G::parse_nwchem(input)
// }

// #[pyfunction]
// fn dump_basis_to_pkl(basis: &Basis631G, filename: &str) -> std::io::Result<()> {
//     basis.save_to_file(filename)
// }
//
// #[pyfunction]
// fn load_basis_from_pkl(filename: &str) -> Basis631G{
//     Basis631G::load_from_file(filename).unwrap()
// }

#[pyfunction]
fn load_basis_from_py_str(input: &str, fname: &str) {
    Basis631G::parse_nwchem(input).save_to_file(fname).unwrap();
}

#[pymodule]
fn basis_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add the `add` function to the Python module
    m.add_function(wrap_pyfunction!(load_basis_from_py_str, m)?)?;
    Ok(())
}