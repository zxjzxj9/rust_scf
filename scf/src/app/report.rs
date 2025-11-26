use crate::{SimpleSCF, SpinSCF, SCF};
use basis::cgto::Basis631G;
use tracing::info;

pub fn report_restricted_summary(scf: &SimpleSCF<Basis631G>) {
    info!("\nSCF calculation finished.");
    let final_energy = scf.calculate_total_energy();

    info!("\nFinal Energy Levels:");
    for (i, energy) in scf.e_level.iter().enumerate() {
        info!("  Level {}: {:.8} au", i + 1, energy);
    }

    info!("\nHartree-Fock Total Energy: {:.10} au", final_energy);
}

pub fn report_spin_summary(scf: &SpinSCF<Basis631G>) {
    info!("\nSpinSCF calculation finished.");

    info!("\nFinal Energy Levels:");
    info!("  Alpha electrons:");
    for (i, energy) in scf.e_level_alpha.iter().enumerate() {
        info!("    Level {}: {:.8} au", i + 1, energy);
    }
    info!("  Beta electrons:");
    for (i, energy) in scf.e_level_beta.iter().enumerate() {
        info!("    Level {}: {:.8} au", i + 1, energy);
    }

    let final_energy = scf.calculate_total_energy();
    info!("\nTotal energy: {:.10} au", final_energy);
}
