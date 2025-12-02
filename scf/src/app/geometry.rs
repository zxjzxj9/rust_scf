use crate::config::Config;
use color_eyre::eyre::{eyre, Result};
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use tracing::info;

/// Geometry information (elements and coordinates) prepared from the user
/// configuration.
pub struct Geometry {
    pub elements: Vec<Element>,
    pub coords: Vec<Vector3<f64>>,
}

/// Build the molecular geometry defined in the YAML configuration.
pub fn build_geometry(config: &Config) -> Result<Geometry> {
    info!("\nPreparing geometry...");

    let mut elements = Vec::with_capacity(config.geometry.len());
    let mut coords = Vec::with_capacity(config.geometry.len());

    for atom in &config.geometry {
        let element = Element::from_symbol(&atom.element)
            .ok_or_else(|| eyre!("Invalid element symbol: {}", atom.element))?;
        let vector = Vector3::new(atom.coords[0], atom.coords[1], atom.coords[2]);
        elements.push(element);
        coords.push(vector);
    }

    Ok(Geometry { elements, coords })
}


