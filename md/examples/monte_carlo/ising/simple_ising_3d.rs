use md::ising::{analysis, IsingModel3D};

/// Simple 3D Ising Model demonstration
///
/// This is a minimal example showing how to use the 3D Ising model
/// for basic simulations and analysis.

fn main() {
    println!("🔬 Simple 3D Ising Model Example");
    println!("==================================\n");

    // Create a small 3D Ising lattice
    let lattice_size = 8;
    let temperature = 4.0; // Below critical temperature

    println!(
        "Setting up {}×{}×{} 3D Ising lattice at T = {:.2}",
        lattice_size, lattice_size, lattice_size, temperature
    );

    let mut ising = IsingModel3D::new(lattice_size, temperature);

    // Show initial random configuration
    println!("\nInitial random configuration (middle slice):");
    ising.print_slice(lattice_size / 2);

    println!("\nInitial properties:");
    println!("- Energy per site: {:.4}", ising.energy_per_site());
    println!(
        "- Magnetization per site: {:.4}",
        ising.magnetization_per_site()
    );
    println!(
        "- Absolute magnetization: {:.4}",
        ising.abs_magnetization_per_site()
    );

    // Run some Monte Carlo steps
    println!("\nRunning 1000 Monte Carlo steps...");
    for _ in 0..1000 {
        ising.monte_carlo_step();
    }

    // Show equilibrated configuration
    println!("\nAfter equilibration (middle slice):");
    ising.print_slice(lattice_size / 2);

    println!("\nFinal properties:");
    println!("- Energy per site: {:.4}", ising.energy_per_site());
    println!(
        "- Magnetization per site: {:.4}",
        ising.magnetization_per_site()
    );
    println!(
        "- Absolute magnetization: {:.4}",
        ising.abs_magnetization_per_site()
    );

    // Temperature comparison
    println!("\n🌡️ Critical Temperature Information:");
    println!(
        "- 2D Critical Temperature: {:.4}",
        analysis::critical_temperature_2d()
    );
    println!(
        "- 3D Critical Temperature: {:.4}",
        analysis::critical_temperature_3d()
    );
    println!("- Current Temperature: {:.4}", temperature);
    println!(
        "- Current T/T_c(3D): {:.4}",
        temperature / analysis::critical_temperature_3d()
    );

    if temperature < analysis::critical_temperature_3d() {
        println!("- System is in the ORDERED phase (below T_c)");
    } else {
        println!("- System is in the DISORDERED phase (above T_c)");
    }

    // Show multiple slices for 3D visualization
    println!("\n📚 Multiple slices through 3D lattice:");
    ising.print_multiple_slices(3);

    // Quick temperature sweep
    println!("\n🌡️ Quick temperature sweep:");
    let temperatures = [2.0, 4.0, 4.511, 6.0];

    for &temp in &temperatures {
        let mut test_ising = IsingModel3D::new(6, temp);

        // Quick equilibration
        for _ in 0..500 {
            test_ising.monte_carlo_step();
        }

        println!(
            "T = {:.3}: E = {:6.3}, |M| = {:5.3}, Status: {}",
            temp,
            test_ising.energy_per_site(),
            test_ising.abs_magnetization_per_site(),
            if temp < analysis::critical_temperature_3d() {
                "Ordered"
            } else {
                "Disordered"
            }
        );
    }

    println!("\n✅ Simple 3D Ising model example complete!");
    println!("\nKey differences from 2D:");
    println!("- Each spin has 6 neighbors instead of 4");
    println!("- Higher critical temperature (4.511 vs 2.269)");
    println!("- Stronger ordering tendency");
    println!("- More computationally intensive");
}
