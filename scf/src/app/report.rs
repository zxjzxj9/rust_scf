use crate::{SimpleSCF, SpinSCF, SCF};
use basis::cgto::Basis631G;
use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use tracing::info;

pub fn report_restricted_summary(scf: &SimpleSCF<Basis631G>) {
    info!("\nSCF calculation finished.");
    let final_energy = scf.calculate_total_energy();

    info!("\nFinal Energy Levels:");
    for (i, energy) in scf.e_level.iter().enumerate() {
        info!("  Level {}: {:.8} au", i + 1, energy);
    }

    info!("\nHartree-Fock Total Energy: {:.10} au", final_energy);
    report_forces("Restricted", &scf.elems, &scf.calculate_forces());
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
    report_forces("Spin-polarized", &scf.elems, &scf.calculate_forces());
}

fn report_forces(kind: &str, elems: &[Element], forces: &[Vector3<f64>]) {
    if forces.is_empty() {
        return;
    }

    info!("\n{kind} SCF forces (Hartree/Bohr):");
    for (idx, (elem, force)) in elems.iter().zip(forces.iter()).enumerate() {
        let magnitude = force.norm();
        info!(
            "  Atom {:>2} {:>2}: [{:+.6}, {:+.6}, {:+.6}] |F| = {:.6}",
            idx + 1,
            elem.get_symbol(),
            force.x,
            force.y,
            force.z,
            magnitude
        );
    }
}
