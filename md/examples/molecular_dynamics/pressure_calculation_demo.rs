// Example demonstrating pressure calculations with Lennard-Jones potential
// This shows how to compute pressure, virial, and pressure tensor for MD simulations

use md::lj_pot::LennardJones;
use nalgebra::{Matrix3, Vector3};

fn main() {
    println!("=== Pressure Calculation Demo ===\n");

    // Setup: Argon-like system in reduced units
    // ε = 120 K * k_B, σ = 3.4 Å, m = 40 amu
    let epsilon = 1.0; // Reduced units
    let sigma = 1.0;
    let mass = 1.0;

    // Example 1: Single particle (ideal gas limit)
    println!("1. Ideal Gas Limit (Single Particle)");
    let box_lengths = Vector3::new(10.0, 10.0, 10.0);
    let lj = LennardJones::new(epsilon, sigma, box_lengths);

    let positions = vec![Vector3::new(5.0, 5.0, 5.0)];
    let velocities = vec![Vector3::new(1.0, 1.0, 1.0)];

    let ke = LennardJones::kinetic_energy(&velocities, mass);
    let temp = LennardJones::temperature_from_kinetic_energy(ke, positions.len());
    let pressure = lj.compute_pressure(&positions, ke);

    println!("   N = {}", positions.len());
    println!("   V = {:.2}", lj.volume());
    println!("   T = {:.4} (reduced)", temp);
    println!("   K = {:.4}", ke);
    println!("   P = {:.6} (reduced)", pressure);
    println!(
        "   Ideal gas P = N*k*T/V = {:.6}",
        positions.len() as f64 * temp / lj.volume()
    );
    println!(
        "   ΔP = {:.2e}\n",
        (pressure - positions.len() as f64 * temp / lj.volume()).abs()
    );

    // Example 2: Two particles - attractive regime
    println!("2. Two Particles - Attractive Regime");
    let r_min = 2_f64.powf(1.0 / 6.0);
    let r_attractive = 1.5 * r_min;

    let positions = vec![
        Vector3::new(5.0, 5.0, 5.0),
        Vector3::new(5.0 + r_attractive, 5.0, 5.0),
    ];
    let velocities = vec![Vector3::new(1.0, 0.5, 0.0), Vector3::new(-1.0, -0.5, 0.0)];

    let ke = LennardJones::kinetic_energy(&velocities, mass);
    let pe = lj.compute_potential_energy(&positions);
    let virial = lj.compute_virial(&positions);
    let temp = LennardJones::temperature_from_kinetic_energy(ke, positions.len());
    let pressure = lj.compute_pressure(&positions, ke);

    println!("   Separation r = {:.4} σ", r_attractive);
    println!("   T = {:.4}", temp);
    println!("   K = {:.4}", ke);
    println!("   U = {:.4}", pe);
    println!("   E_total = {:.4}", ke + pe);
    println!("   Virial W = {:.4}", virial);
    println!("   P = {:.6}", pressure);
    println!(
        "   P (ideal) = {:.6}",
        2.0 * positions.len() as f64 * temp / lj.volume()
    );
    println!(
        "   P (correction) = {:.6} (attractive)\n",
        pressure - 2.0 * positions.len() as f64 * temp / lj.volume()
    );

    // Example 3: Two particles - repulsive regime
    println!("3. Two Particles - Repulsive Regime");
    let r_repulsive = 0.9; // Less than σ

    let positions = vec![
        Vector3::new(5.0, 5.0, 5.0),
        Vector3::new(5.0 + r_repulsive, 5.0, 5.0),
    ];

    let ke = LennardJones::kinetic_energy(&velocities, mass);
    let pe = lj.compute_potential_energy(&positions);
    let virial = lj.compute_virial(&positions);
    let temp = LennardJones::temperature_from_kinetic_energy(ke, positions.len());
    let pressure = lj.compute_pressure(&positions, ke);

    println!("   Separation r = {:.4} σ", r_repulsive);
    println!("   T = {:.4}", temp);
    println!("   K = {:.4}", ke);
    println!("   U = {:.4} (repulsive!)", pe);
    println!("   E_total = {:.4}", ke + pe);
    println!("   Virial W = {:.4} (positive)", virial);
    println!("   P = {:.6} (high pressure!)", pressure);
    println!(
        "   P (ideal) = {:.6}",
        2.0 * positions.len() as f64 * temp / lj.volume()
    );
    println!(
        "   P (correction) = {:.6} (repulsive)\n",
        pressure - 2.0 * positions.len() as f64 * temp / lj.volume()
    );

    // Example 4: Many particles - liquid-like density
    println!("4. Liquid-like System (108 particles)");

    // Create FCC lattice with 108 atoms (3x3x4 unit cells)
    let lattice_constant = 1.5 * sigma;
    let mut positions = Vec::new();

    for ix in 0..3 {
        for iy in 0..3 {
            for iz in 0..4 {
                let base = Vector3::new(
                    ix as f64 * lattice_constant,
                    iy as f64 * lattice_constant,
                    iz as f64 * lattice_constant,
                );

                // FCC positions in unit cell
                positions.push(base);
                positions
                    .push(base + Vector3::new(0.5 * lattice_constant, 0.5 * lattice_constant, 0.0));
                positions
                    .push(base + Vector3::new(0.5 * lattice_constant, 0.0, 0.5 * lattice_constant));

                // Only add 4th atom for first 3 z-layers to get exactly 108
                if positions.len() < 108 {
                    positions.push(
                        base + Vector3::new(0.0, 0.5 * lattice_constant, 0.5 * lattice_constant),
                    );
                }
            }
        }
    }
    positions.truncate(108);

    let box_size = 3.0 * lattice_constant;
    let lj_liquid = LennardJones::new(
        epsilon,
        sigma,
        Vector3::new(box_size, box_size, 1.5 * box_size),
    );

    // Initialize velocities (Maxwell-Boltzmann-like at T ≈ 1.0)
    let mut velocities = Vec::new();
    for i in 0..positions.len() {
        let phase = i as f64 * 2.0 * std::f64::consts::PI / positions.len() as f64;
        velocities.push(Vector3::new(
            (phase).cos(),
            (phase + 1.0).sin(),
            (phase + 2.0).cos(),
        ));
    }

    let ke = LennardJones::kinetic_energy(&velocities, mass);
    let pe = lj_liquid.compute_potential_energy(&positions);
    let virial = lj_liquid.compute_virial(&positions);
    let temp = LennardJones::temperature_from_kinetic_energy(ke, positions.len());
    let pressure = lj_liquid.compute_pressure(&positions, ke);

    let density = positions.len() as f64 * sigma.powi(3) / lj_liquid.volume();

    println!("   N = {}", positions.len());
    println!("   V = {:.2} σ³", lj_liquid.volume());
    println!("   ρ = {:.4} σ⁻³", density);
    println!("   T = {:.4}", temp);
    println!("   K = {:.4}", ke);
    println!("   U = {:.4}", pe);
    println!("   U/N = {:.4} ε/particle", pe / positions.len() as f64);
    println!("   E_total = {:.4}", ke + pe);
    println!("   Virial W = {:.4}", virial);
    println!("   P = {:.6} ε/σ³", pressure);
    println!(
        "   P (reduced) = {:.4}\n",
        pressure * sigma.powi(3) / epsilon
    );

    // Example 5: Pressure tensor (anisotropic system)
    println!("5. Pressure Tensor - Anisotropic System");

    // Create system with particles aligned along x-axis
    let positions_aniso = vec![
        Vector3::new(4.0, 5.0, 5.0),
        Vector3::new(6.0, 5.0, 5.0),
        Vector3::new(8.0, 5.0, 5.0),
    ];
    let velocities_aniso = vec![
        Vector3::new(2.0, 0.1, 0.1),
        Vector3::new(-1.0, 0.1, -0.1),
        Vector3::new(-1.0, -0.1, 0.1),
    ];

    let lj_aniso = LennardJones::new(epsilon, sigma, Vector3::new(15.0, 10.0, 10.0));

    let config_tensor = lj_aniso.compute_pressure_tensor(&positions_aniso);
    let pressure_scalar =
        lj_aniso.compute_pressure_from_tensor(&positions_aniso, &velocities_aniso, mass);

    // Compute full pressure tensor
    let volume = lj_aniso.volume();
    let mut kinetic_tensor = Matrix3::zeros();
    for i in 0..positions_aniso.len() {
        for alpha in 0..3 {
            for beta in 0..3 {
                kinetic_tensor[(alpha, beta)] +=
                    mass * velocities_aniso[i][alpha] * velocities_aniso[i][beta];
            }
        }
    }
    let pressure_tensor = (kinetic_tensor + config_tensor) / volume;

    println!("   System: 3 particles along x-axis");
    println!("   Box: [{:.1}, {:.1}, {:.1}]", 15.0, 10.0, 10.0);
    println!("\n   Pressure Tensor (ε/σ³):");
    println!("   ┌                                      ┐");
    println!(
        "   │ {:8.4}  {:8.4}  {:8.4} │",
        pressure_tensor[(0, 0)],
        pressure_tensor[(0, 1)],
        pressure_tensor[(0, 2)]
    );
    println!(
        "   │ {:8.4}  {:8.4}  {:8.4} │",
        pressure_tensor[(1, 0)],
        pressure_tensor[(1, 1)],
        pressure_tensor[(1, 2)]
    );
    println!(
        "   │ {:8.4}  {:8.4}  {:8.4} │",
        pressure_tensor[(2, 0)],
        pressure_tensor[(2, 1)],
        pressure_tensor[(2, 2)]
    );
    println!("   └                                      ┘");
    println!("\n   Diagonal elements:");
    println!(
        "   P_xx = {:.6} (largest - particles/velocity along x)",
        pressure_tensor[(0, 0)]
    );
    println!("   P_yy = {:.6}", pressure_tensor[(1, 1)]);
    println!("   P_zz = {:.6}", pressure_tensor[(2, 2)]);
    println!(
        "\n   Scalar pressure: P = (P_xx + P_yy + P_zz)/3 = {:.6}\n",
        pressure_scalar
    );

    // Example 6: Pressure in triclinic box
    println!("6. Pressure in Triclinic (Hexagonal) Lattice");

    let a = Vector3::new(8.0, 0.0, 0.0);
    let b = Vector3::new(4.0, 6.928, 0.0); // 60 degrees
    let c = Vector3::new(0.0, 0.0, 8.0);
    let lattice = Matrix3::from_columns(&[a, b, c]);

    let lj_hex = LennardJones::from_lattice(epsilon, sigma, lattice);

    let positions_hex = vec![
        Vector3::new(2.0, 2.0, 2.0),
        Vector3::new(6.0, 4.0, 6.0),
        Vector3::new(4.0, 5.0, 4.0),
    ];
    let velocities_hex = vec![
        Vector3::new(1.0, 0.5, 0.2),
        Vector3::new(-0.5, -1.0, -0.3),
        Vector3::new(-0.5, 0.5, 0.1),
    ];

    let ke = LennardJones::kinetic_energy(&velocities_hex, mass);
    let temp = LennardJones::temperature_from_kinetic_energy(ke, positions_hex.len());
    let pressure = lj_hex.compute_pressure(&positions_hex, ke);

    println!("   Hexagonal lattice (60° angle)");
    println!("   Volume = {:.2} σ³", lj_hex.volume());
    println!("   N = {}", positions_hex.len());
    println!("   T = {:.4}", temp);
    println!("   P = {:.6} ε/σ³", pressure);
    println!("   (Pressure calculation works for any lattice geometry!)\n");

    // Summary
    println!("=== Summary ===");
    println!("Pressure calculation methods:");
    println!("  ✓ compute_virial() - Virial contribution W = Σ r·F");
    println!("  ✓ compute_pressure() - Scalar pressure P = (2K + W)/(3V)");
    println!("  ✓ compute_pressure_tensor() - Full 3×3 pressure tensor");
    println!("  ✓ compute_pressure_from_tensor() - Scalar P from tensor");
    println!("\nUtility methods:");
    println!("  ✓ kinetic_energy() - Calculate K from velocities");
    println!("  ✓ temperature_from_kinetic_energy() - Extract T from K");
    println!("  ✓ volume() - Get simulation box volume");
    println!("\nFeatures:");
    println!("  ✓ Works with orthogonal and triclinic boxes");
    println!("  ✓ Parallel computation (uses Rayon)");
    println!("  ✓ Proper PBC with minimum image convention");
    println!("  ✓ Essential for NPT ensemble simulations");
}
