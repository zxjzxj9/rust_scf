use crate::app::{
    build_geometry, run_restricted, run_spin, Basis631GLoader, BasisRegistry, CalculationWorkspace,
};
use crate::config::{Args, Config};
use crate::{SimpleSCF, SpinSCF, SCF};
use basis::cgto::Basis631G;
use nalgebra::DMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyclass(name = "ScfResult")]
pub struct PyScfResult {
    #[pyo3(get)]
    hf_energy: f64,
    #[pyo3(get)]
    orbital_energies_alpha: Vec<f64>,
    #[pyo3(get)]
    orbital_energies_beta: Vec<f64>,
    #[pyo3(get)]
    density_alpha: Vec<Vec<f64>>,
    #[pyo3(get)]
    density_beta: Vec<Vec<f64>>,
    #[pyo3(get)]
    calculation_type: String,
    #[pyo3(get)]
    charge: i32,
    #[pyo3(get)]
    multiplicity: usize,
}

#[pymethods]
impl PyScfResult {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "<ScfResult type={}, hf_energy={:.8} au>",
            self.calculation_type, self.hf_energy
        ))
    }
}

impl PyScfResult {
    fn from_restricted(scf: &SimpleSCF<Basis631G>) -> Self {
        let orbital_energies: Vec<f64> = scf.e_level.iter().cloned().collect();
        let density = matrix_to_vecvec(&scf.density_matrix);
        Self {
            hf_energy: scf.calculate_total_energy(),
            orbital_energies_alpha: orbital_energies.clone(),
            orbital_energies_beta: orbital_energies,
            density_alpha: density.clone(),
            density_beta: density,
            calculation_type: "restricted".to_string(),
            charge: 0,
            multiplicity: 1,
        }
    }

    fn from_spin(scf: &SpinSCF<Basis631G>) -> Self {
        let alpha_energies: Vec<f64> = scf.e_level_alpha.iter().cloned().collect();
        let beta_energies: Vec<f64> = scf.e_level_beta.iter().cloned().collect();
        let density_alpha = matrix_to_vecvec(scf.get_density_matrix_alpha());
        let density_beta = matrix_to_vecvec(scf.get_density_matrix_beta());

        Self {
            hf_energy: scf.calculate_total_energy(),
            orbital_energies_alpha: alpha_energies,
            orbital_energies_beta: beta_energies,
            density_alpha,
            density_beta,
            calculation_type: "spin".to_string(),
            charge: scf.charge,
            multiplicity: scf.multiplicity,
        }
    }
}

#[pyfunction]
pub fn run_restricted_scf(config_yaml: &str) -> PyResult<PyScfResult> {
    let config = parse_config(config_yaml)?;
    let workspace = prepare_workspace(&config)?;
    let args = default_args();
    let scf = run_restricted(&workspace, &args, &config).map_err(to_py_err)?;
    Ok(PyScfResult::from_restricted(&scf))
}

#[pyfunction]
#[pyo3(signature = (config_yaml, charge=0, multiplicity=1))]
pub fn run_spin_scf(config_yaml: &str, charge: i32, multiplicity: usize) -> PyResult<PyScfResult> {
    let config = parse_config(config_yaml)?;
    let workspace = prepare_workspace(&config)?;
    let args = spin_args(charge, multiplicity);
    let scf = run_spin(&workspace, &args, &config, charge, multiplicity).map_err(to_py_err)?;
    Ok(PyScfResult::from_spin(&scf))
}

#[pymodule]
pub fn scf(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScfResult>()?;
    m.add_function(wrap_pyfunction!(run_restricted_scf, m)?)?;
    m.add_function(wrap_pyfunction!(run_spin_scf, m)?)?;
    Ok(())
}

fn parse_config(yaml: &str) -> PyResult<Config> {
    serde_yml::from_str::<Config>(yaml)
        .map(Config::with_defaults)
        .map_err(|err| PyValueError::new_err(format!("Failed to parse configuration: {err}")))
}

fn prepare_workspace(config: &Config) -> PyResult<CalculationWorkspace<Basis631G>> {
    let geometry = build_geometry(config).map_err(to_py_err)?;
    let mut basis_registry = BasisRegistry::<Basis631G, _>::new(Basis631GLoader);
    let basis_map = basis_registry
        .load_for_elements(config, &geometry.elements)
        .map_err(to_py_err)?;
    Ok(CalculationWorkspace::new(
        geometry.elements,
        geometry.coords,
        basis_map,
    ))
}

fn default_args() -> Args {
    Args {
        config_file: "<python>".to_string(),
        density_mixing: None,
        max_cycle: None,
        diis_subspace_size: None,
        convergence_threshold: None,
        output: None,
        optimize: false,
        opt_algorithm: None,
        opt_max_iterations: None,
        opt_convergence: None,
        opt_step_size: None,
        charge: None,
        multiplicity: None,
        spin_polarized: false,
    }
}

fn spin_args(charge: i32, multiplicity: usize) -> Args {
    Args {
        spin_polarized: true,
        charge: Some(charge),
        multiplicity: Some(multiplicity),
        ..default_args()
    }
}

fn matrix_to_vecvec(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|i| {
            (0..matrix.ncols())
                .map(|j| matrix[(i, j)])
                .collect::<Vec<f64>>()
        })
        .collect()
}

fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}
