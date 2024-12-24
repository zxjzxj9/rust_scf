use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
// use basis;

#[pyfunction]
fn double(x: i32) -> i32 {
    x * 2
}

#[pymodule]
fn basis_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add the `add` function to the Python module
    m.add_function(wrap_pyfunction!(double, m)?)?;
    Ok(())
}