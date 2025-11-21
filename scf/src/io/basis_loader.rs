//! Basis set loading utilities

use basis::cgto::Basis631G;
use color_eyre::eyre::{Result, WrapErr};
use std::fs;
use std::path::Path;

/// Fetch basis set from local file or online database
pub fn fetch_basis(atomic_symbol: &str) -> Result<Basis631G> {
    println!("DEBUG: Attempting to load basis set for {}", atomic_symbol);
    
    // First try to load from local file
    let local_path = format!("tests/basis_sets/6-31g.{}.nwchem", atomic_symbol.to_lowercase());
    if Path::new(&local_path).exists() {
        println!("DEBUG: Loading from local file: {}", local_path);
        let basis_str = fs::read_to_string(&local_path)
            .wrap_err_with(|| format!("Failed to read local basis set file: {}", local_path))?;
        return Ok(Basis631G::parse_nwchem(&basis_str));
    }
    
    // Fall back to fetching from web
    println!("DEBUG: Local file not found, fetching from web");
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

