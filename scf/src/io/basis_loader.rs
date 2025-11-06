//! Basis set loading utilities

use basis::cgto::Basis631G;
use color_eyre::eyre::{Result, WrapErr};

/// Fetch basis set from online database
pub fn fetch_basis(atomic_symbol: &str) -> Result<Basis631G> {
    println!("DEBUG: Attempting to fetch basis set for {}", atomic_symbol);
    let url = format!(
        "https://www.basissetexchange.org/api/basis/6-31g/format/nwchem?elements={}",
        atomic_symbol
    );
    println!("DEBUG: URL: {}", url);
    let response = reqwest::blocking::get(&url)
        .wrap_err_with(|| format!("Failed to fetch basis set for {}", atomic_symbol))?;
    println!("DEBUG: HTTP request successful");
    let basis_str = response
        .text()
        .wrap_err("Failed to get response text from basis set API")?;
    println!("DEBUG: Got response text, length: {}", basis_str.len());
    Ok(Basis631G::parse_nwchem(&basis_str))
}

