//! Input/Output operations for SCF calculations

use basis::cgto::Basis631G;
use color_eyre::eyre::Result;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use std::io::Write;
use tracing_subscriber;

/// Setup output configuration (logging, etc.)
pub fn setup_output(output_file: Option<&String>) {
    // Initialize tracing subscriber for logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    if let Some(file) = output_file {
        tracing::info!("Output will be written to: {}", file);
    }
}

/// Fetch basis set for a given element symbol from Basis Set Exchange
///
/// Fetches 6-31G basis set in NWChem format from the online API
pub fn fetch_basis(atomic_symbol: &str) -> Result<Basis631G> {
    use color_eyre::eyre::WrapErr;

    tracing::info!("Fetching basis set for {}", atomic_symbol);

    // Use HTTP instead of HTTPS to avoid redirect issues
    let url = format!(
        "http://www.basissetexchange.org/api/basis/6-31g/format/nwchem/?elements={}",
        atomic_symbol
    );

    tracing::info!("URL: {}", url);

    let response = reqwest::blocking::get(&url)
        .wrap_err_with(|| format!("Failed to fetch basis set for {}", atomic_symbol))?;

    let basis_str = response
        .text()
        .wrap_err("Failed to get response text from basis set API")?;

    tracing::info!("Received {} characters from API", basis_str.len());
    tracing::debug!("Basis string:\n{}", &basis_str[..basis_str.len().min(500)]);

    Ok(Basis631G::parse_nwchem(&basis_str))
}

/// Print optimized geometry to output file
pub fn print_optimized_geometry<W: Write>(
    writer: &mut W,
    coords: &[Vector3<f64>],
    elements: &[Element],
    energy: f64,
) -> Result<()> {
    writeln!(writer, "Optimized Geometry")?;
    writeln!(writer, "==================\n")?;
    writeln!(writer, "Final Energy: {:.10} au\n", energy)?;
    writeln!(writer, "Atomic Coordinates (Angstroms):")?;
    writeln!(writer, "{:>4} {:>8} {:>12} {:>12} {:>12}", "Atom", "Element", "X", "Y", "Z")?;
    writeln!(writer, "{}", "-".repeat(52))?;

    for (i, (coord, elem)) in coords.iter().zip(elements.iter()).enumerate() {
        writeln!(
            writer,
            "{:>4} {:>8} {:>12.6} {:>12.6} {:>12.6}",
            i + 1,
            elem.get_symbol(),
            coord.x,
            coord.y,
            coord.z
        )?;
    }

    Ok(())
}
