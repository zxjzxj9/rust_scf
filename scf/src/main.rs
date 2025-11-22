//! SCF Calculation Command-Line Interface
//!
//! Delegates to the modular application layer defined in `scf::app`.

use color_eyre::eyre::Result;
use scf::app::ScfApplication;

fn main() -> Result<()> {
    color_eyre::install()?;
    ScfApplication::from_cli()?.run()
}
