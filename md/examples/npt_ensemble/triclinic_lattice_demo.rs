// Example demonstrating arbitrary lattice structures with Lennard-Jones potential
// This example shows how to use triclinic (non-orthogonal) boxes for simulations

use md::lj_pot::LennardJones;
use md::run_md::ForceProvider;
use nalgebra::{Matrix3, Vector3};

fn main() {
    println!("=== Triclinic Lattice Demo ===\n");

    // Example 1: Orthogonal box (traditional)
    println!("1. Orthogonal (Cubic) Box:");
    let box_lengths = Vector3::new(10.0, 10.0, 10.0);
    let _lj_ortho = LennardJones::new(1.0, 1.0, box_lengths);
    println!(
        "   Box dimensions: {} x {} x {}",
        box_lengths.x, box_lengths.y, box_lengths.z
    );
    println!(
        "   Volume: {:.2}\n",
        box_lengths.x * box_lengths.y * box_lengths.z
    );

    // Example 2: Triclinic box with 60-degree angles (hexagonal)
    println!("2. Hexagonal (60-degree) Lattice:");
    let a = Vector3::new(10.0, 0.0, 0.0);
    let b = Vector3::new(5.0, 8.66025404, 0.0); // 60 degrees from a
    let c = Vector3::new(0.0, 0.0, 10.0);
    let lattice_hex = Matrix3::from_columns(&[a, b, c]);
    let lj_hex = LennardJones::from_lattice(1.0, 1.0, lattice_hex);

    let volume_hex = lattice_hex.determinant().abs();
    println!("   Lattice vectors:");
    println!("     a = [{:.2}, {:.2}, {:.2}]", a.x, a.y, a.z);
    println!("     b = [{:.2}, {:.2}, {:.2}]", b.x, b.y, b.z);
    println!("     c = [{:.2}, {:.2}, {:.2}]", c.x, c.y, c.z);
    println!("   Volume: {:.2}", volume_hex);
    println!("   Angle between a and b: 60°\n");

    // Example 3: Fully triclinic box
    println!("3. Fully Triclinic Lattice:");
    let a = Vector3::new(8.0, 0.0, 0.0);
    let b = Vector3::new(2.0, 7.75, 0.0);
    let c = Vector3::new(1.0, 2.0, 9.0);
    let lattice_tri = Matrix3::from_columns(&[a, b, c]);
    let _lj_tri = LennardJones::from_lattice(1.0, 1.0, lattice_tri);

    let volume_tri = lattice_tri.determinant().abs();
    println!("   Lattice vectors:");
    println!("     a = [{:.2}, {:.2}, {:.2}]", a.x, a.y, a.z);
    println!("     b = [{:.2}, {:.2}, {:.2}]", b.x, b.y, b.z);
    println!("     c = [{:.2}, {:.2}, {:.2}]", c.x, c.y, c.z);
    println!("   Volume: {:.2}\n", volume_tri);

    // Demonstrate periodic boundary conditions
    println!("=== PBC with Triclinic Box ===\n");

    // Two atoms near opposite edges - should wrap through PBC
    let positions = vec![
        Vector3::new(0.5, 0.5, 0.5),
        Vector3::new(9.5, 8.0, 0.5), // Near opposite corner in hexagonal box
    ];

    println!("Atom positions:");
    println!(
        "  Atom 1: [{:.2}, {:.2}, {:.2}]",
        positions[0].x, positions[0].y, positions[0].z
    );
    println!(
        "  Atom 2: [{:.2}, {:.2}, {:.2}]",
        positions[1].x, positions[1].y, positions[1].z
    );

    let direct_distance = (positions[1] - positions[0]).norm();
    println!("\nDirect distance: {:.2}", direct_distance);

    // Apply minimum image convention
    let rij = lj_hex.minimum_image(positions[1] - positions[0]);
    let mic_distance = rij.norm();
    println!("Distance with PBC (minimum image): {:.2}", mic_distance);
    println!("Wrapped vector: [{:.2}, {:.2}, {:.2}]", rij.x, rij.y, rij.z);

    // Calculate forces and energy
    let forces = lj_hex.compute_forces(&positions);
    let energy = lj_hex.compute_potential_energy(&positions);

    println!("\nEnergy: {:.6} ε", energy);
    println!("Forces:");
    println!(
        "  Atom 1: [{:.4}, {:.4}, {:.4}]",
        forces[0].x, forces[0].y, forces[0].z
    );
    println!(
        "  Atom 2: [{:.4}, {:.4}, {:.4}]",
        forces[1].x, forces[1].y, forces[1].z
    );
    println!(
        "  Total: [{:.4}, {:.4}, {:.4}] (should be ~0)",
        (forces[0] + forces[1]).x,
        (forces[0] + forces[1]).y,
        (forces[0] + forces[1]).z
    );

    // Example 4: Dynamic lattice (NPT simulation)
    println!("\n=== Dynamic Lattice Update (NPT) ===\n");

    let mut lj_npt = LennardJones::new(1.0, 1.0, Vector3::new(10.0, 10.0, 10.0));
    let initial_volume = lj_npt.lattice.determinant().abs();
    println!("Initial cubic box:");
    println!("  Volume: {:.2}", initial_volume);

    // Simulate box deformation (e.g., due to pressure)
    let new_a = Vector3::new(9.5, 0.0, 0.0);
    let new_b = Vector3::new(0.5, 10.2, 0.0);
    let new_c = Vector3::new(0.0, 0.0, 10.3);
    let new_lattice = Matrix3::from_columns(&[new_a, new_b, new_c]);

    lj_npt.set_lattice(new_lattice);
    let new_volume = lj_npt.lattice.determinant().abs();

    println!("\nAfter NPT step (box deformed):");
    println!("  Volume: {:.2}", new_volume);
    println!(
        "  ΔV: {:.2} ({:.1}%)",
        new_volume - initial_volume,
        100.0 * (new_volume - initial_volume) / initial_volume
    );
    println!("  New lattice vectors:");
    println!("    a = [{:.2}, {:.2}, {:.2}]", new_a.x, new_a.y, new_a.z);
    println!("    b = [{:.2}, {:.2}, {:.2}]", new_b.x, new_b.y, new_b.z);
    println!("    c = [{:.2}, {:.2}, {:.2}]", new_c.x, new_c.y, new_c.z);

    // Crystal lattices
    println!("\n=== Common Crystal Lattices ===\n");

    // FCC lattice
    println!("Face-Centered Cubic (FCC):");
    let a_fcc = 5.0;
    let fcc_a = Vector3::new(0.0, a_fcc / 2.0, a_fcc / 2.0);
    let fcc_b = Vector3::new(a_fcc / 2.0, 0.0, a_fcc / 2.0);
    let fcc_c = Vector3::new(a_fcc / 2.0, a_fcc / 2.0, 0.0);
    let lattice_fcc = Matrix3::from_columns(&[fcc_a, fcc_b, fcc_c]);
    let _lj_fcc = LennardJones::from_lattice(1.0, 1.0, lattice_fcc);
    println!("  Lattice parameter: {:.2}", a_fcc);
    println!(
        "  Primitive cell volume: {:.2}",
        lattice_fcc.determinant().abs()
    );

    // BCC lattice
    println!("\nBody-Centered Cubic (BCC):");
    let a_bcc = 5.0;
    let bcc_a = Vector3::new(-a_bcc / 2.0, a_bcc / 2.0, a_bcc / 2.0);
    let bcc_b = Vector3::new(a_bcc / 2.0, -a_bcc / 2.0, a_bcc / 2.0);
    let bcc_c = Vector3::new(a_bcc / 2.0, a_bcc / 2.0, -a_bcc / 2.0);
    let lattice_bcc = Matrix3::from_columns(&[bcc_a, bcc_b, bcc_c]);
    let _lj_bcc = LennardJones::from_lattice(1.0, 1.0, lattice_bcc);
    println!("  Lattice parameter: {:.2}", a_bcc);
    println!(
        "  Primitive cell volume: {:.2}",
        lattice_bcc.determinant().abs()
    );

    println!("\n=== Summary ===");
    println!("The LennardJones potential now supports:");
    println!("  ✓ Arbitrary triclinic lattices");
    println!("  ✓ Proper minimum image convention for non-orthogonal boxes");
    println!("  ✓ Dynamic lattice updates for NPT simulations");
    println!("  ✓ Common crystal structures (FCC, BCC, hexagonal, etc.)");
    println!("  ✓ Full backward compatibility with orthogonal boxes");
}
