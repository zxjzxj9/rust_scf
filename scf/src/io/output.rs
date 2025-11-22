//! Output formatting and logging utilities

use color_eyre::eyre::Result;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::time::SystemTime as StdSystemTime;
use tracing::info;
use tracing_subscriber::{
    fmt::format::Writer, fmt::layer, fmt::time::FormatTime, layer::SubscriberExt,
    util::SubscriberInitExt, Registry,
};

/// Custom time formatter that shows only seconds
struct SecondPrecisionTimer;

impl FormatTime for SecondPrecisionTimer {
    fn format_time(&self, w: &mut Writer<'_>) -> fmt::Result {
        let now = StdSystemTime::now();
        let duration = now
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();

        // Format as HH:MM:SS (only seconds precision)
        let total_seconds = duration.as_secs();
        let hours = (total_seconds / 3600) % 24;
        let minutes = (total_seconds / 60) % 60;
        let seconds = total_seconds % 60;

        write!(w, "{:02}:{:02}:{:02}", hours, minutes, seconds)
    }
}

/// Setup output logging to file or stdout
pub fn setup_output(output_path: Option<&String>) {
    match output_path {
        Some(path) => {
            info!("Output will be written to: {}", path);
            if let Ok(log) = File::create(path) {
                let file_layer = layer()
                    .with_writer(log)
                    .with_timer(SecondPrecisionTimer)
                    .with_ansi(false);
                Registry::default().with(file_layer).init();
            } else {
                eprintln!("Could not create output file: {}", path);
            }
        }
        None => {
            // Initialize tracing for stdout
            let stdout_layer = layer()
                .with_writer(std::io::stdout)
                .with_timer(SecondPrecisionTimer)
                .with_ansi(true);
            Registry::default().with(stdout_layer).init();
            info!("Output will be printed to stdout");
        }
    }
}

/// Print optimized geometry to a writer
pub fn print_optimized_geometry<W: Write>(
    writer: &mut W,
    coords: &[Vector3<f64>],
    elements: &[Element],
    energy: f64,
) -> Result<()> {
    writeln!(writer, "Optimized geometry:")?;
    for (i, (coord, elem)) in coords.iter().zip(elements.iter()).enumerate() {
        writeln!(
            writer,
            "  Atom {}: {} at [{:.6}, {:.6}, {:.6}]",
            i + 1,
            elem.get_symbol(),
            coord.x,
            coord.y,
            coord.z
        )?;
    }
    writeln!(writer, "Final energy: {:.10} au", energy)?;
    Ok(())
}
