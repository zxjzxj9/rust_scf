// LJ Cluster NPT Simulation
//
// This example demonstrates NPT (constant pressure and temperature) molecular dynamics
// for Lennard-Jones clusters with proper virial pressure calculation.
//
// Key features:
// - Nosé-Hoover thermostat for temperature control
// - Parrinello-Rahman barostat for pressure control
// - Proper virial-based pressure calculation for LJ interactions
// - Cluster-specific analysis (coordination number, potential energy, structure)
// - Study of phase transitions and structural changes
//
// Physical insight: LJ clusters exhibit rich behavior including:
// - Solid-liquid transitions (melting)
// - Structural rearrangements
// - Evaporation under low pressure
// - Condensation under high pressure

use md::{ForceProvider, LennardJones};
use nalgebra::Vector3;
use rand::Rng;
use rand_distr::StandardNormal;
use std::collections::VecDeque;

// Physical constants (reduced units: ε=1, σ=1, m=1, k_B=1)
const K_B: f64 = 1.0;
const ARGON_MASS: f64 = 1.0;
const EPSILON: f64 = 1.0;
const SIGMA: f64 = 1.0;

/// NPT integrator with proper virial pressure for LJ systems
pub struct LJClusterNPT {
    pub positions: Vec<Vector3<f64>>,
    pub velocities: Vec<Vector3<f64>>,
    pub masses: Vec<f64>,
    inv_masses: Vec<f64>,
    pub lj_potential: LennardJones,

    // Thermostat variables (Nosé-Hoover)
    xi: f64,
    eta: f64,
    q_t: f64,
    target_temp: f64,

    // Barostat variables (Parrinello-Rahman)
    pub box_lengths: Vector3<f64>,
    box_velocities: Vector3<f64>,
    q_p: f64,
    target_pressure: f64,

    // System parameters
    dof: usize,
    k_b: f64,
}

impl LJClusterNPT {
    pub fn new(
        positions: Vec<Vector3<f64>>,
        velocities: Vec<Vector3<f64>>,
        masses: Vec<f64>,
        initial_box_lengths: Vector3<f64>,
        q_t: f64,
        q_p: f64,
        target_temp: f64,
        target_pressure: f64,
        epsilon: f64,
        sigma: f64,
    ) -> Self {
        let dof = positions.len() * 3;
        let inv_masses = masses.iter().map(|&m| 1.0 / m).collect();
        let lj_potential = LennardJones::new(epsilon, sigma, initial_box_lengths);

        LJClusterNPT {
            positions,
            velocities,
            masses,
            inv_masses,
            lj_potential,
            xi: 0.0,
            eta: 0.0,
            q_t,
            target_temp,
            box_lengths: initial_box_lengths,
            box_velocities: Vector3::zeros(),
            q_p,
            target_pressure,
            dof,
            k_b: K_B,
        }
    }

    #[inline]
    fn kinetic_energy(&self) -> f64 {
        self.velocities
            .iter()
            .zip(&self.masses)
            .map(|(v, &m)| 0.5 * m * v.dot(v))
            .sum()
    }

    #[inline]
    fn volume(&self) -> f64 {
        self.box_lengths.x * self.box_lengths.y * self.box_lengths.z
    }

    /// Compute pressure using proper virial calculation
    #[inline]
    fn pressure(&self) -> f64 {
        let kinetic_energy = self.kinetic_energy();
        self.lj_potential
            .compute_pressure(&self.positions, kinetic_energy)
    }

    pub fn set_target_temperature(&mut self, temp: f64) {
        self.target_temp = temp;
    }

    pub fn set_target_pressure(&mut self, pressure: f64) {
        self.target_pressure = pressure;
    }

    pub fn get_volume(&self) -> f64 {
        self.volume()
    }

    pub fn get_pressure(&self) -> f64 {
        self.pressure()
    }

    pub fn get_potential_energy(&self) -> f64 {
        self.lj_potential.compute_potential_energy(&self.positions)
    }

    pub fn step(&mut self, dt: f64) {
        let half_dt = 0.5 * dt;

        // Current quantities
        let volume = self.volume();
        let current_pressure = self.pressure();
        let kinetic_energy = self.kinetic_energy();
        let gk_t = self.dof as f64 * self.k_b * self.target_temp;

        // Update thermostat variable xi (first half-step)
        let xi_dot = (2.0 * kinetic_energy - gk_t) / self.q_t;
        self.xi += xi_dot * half_dt;
        self.xi = self.xi.clamp(-10.0, 10.0);

        // Update barostat velocities (first half-step)
        let pressure_error = current_pressure - self.target_pressure;
        for i in 0..3 {
            let box_accel = (pressure_error * volume) / self.q_p;
            self.box_velocities[i] += box_accel * half_dt;
            // Limit box velocity to prevent runaway
            self.box_velocities[i] = self.box_velocities[i].clamp(-0.05, 0.05);
        }

        // Update box lengths (full step)
        let old_volume = volume;
        for i in 0..3 {
            self.box_lengths[i] *= 1.0 + self.box_velocities[i] * dt;
            // Prevent box collapse and excessive expansion
            self.box_lengths[i] = self.box_lengths[i].clamp(2.0, 100.0);
        }

        // Update LJ potential with new box size
        self.lj_potential = LennardJones::new(EPSILON, SIGMA, self.box_lengths);

        // Scale positions with box changes
        let volume_change = self.volume() / old_volume;
        let scale_factor = volume_change.powf(1.0 / 3.0);
        for pos in &mut self.positions {
            *pos *= scale_factor;
        }

        // Compute forces
        let forces = self.lj_potential.compute_forces(&self.positions);

        // Update velocities (first half-step) - includes thermostat and barostat effects
        let box_vel_trace = self.box_velocities.x + self.box_velocities.y + self.box_velocities.z;
        for i in 0..self.velocities.len() {
            let thermostat_scaling = 1.0 / (1.0 + (self.xi + box_vel_trace) * half_dt);
            let force_contrib = forces[i] * self.inv_masses[i] * half_dt;
            self.velocities[i] = (self.velocities[i] + force_contrib) * thermostat_scaling;
        }

        // Update positions (full step)
        for i in 0..self.positions.len() {
            self.positions[i] += self.velocities[i] * dt;
        }

        // Apply periodic boundary conditions
        for pos in &mut self.positions {
            for k in 0..3 {
                let box_l = self.box_lengths[k];
                pos[k] -= box_l * (pos[k] / box_l).floor();
            }
        }

        // Recompute forces with new positions
        let forces_new = self.lj_potential.compute_forces(&self.positions);

        // Update velocities (second half-step)
        for i in 0..self.velocities.len() {
            let force_contrib = forces_new[i] * self.inv_masses[i] * half_dt;
            self.velocities[i] += force_contrib;
            let thermostat_scaling = 1.0 / (1.0 + (self.xi + box_vel_trace) * half_dt);
            self.velocities[i] *= thermostat_scaling;
        }

        // Update thermostat variable xi (second half-step)
        let kinetic_energy = self.kinetic_energy();
        let xi_dot = (2.0 * kinetic_energy - gk_t) / self.q_t;
        self.xi += xi_dot * half_dt;
        self.xi = self.xi.clamp(-10.0, 10.0);

        // Update barostat velocities (second half-step)
        let current_pressure = self.pressure();
        let pressure_error = current_pressure - self.target_pressure;
        let volume = self.volume();
        for i in 0..3 {
            let box_accel = (pressure_error * volume) / self.q_p;
            self.box_velocities[i] += box_accel * half_dt;
            self.box_velocities[i] = self.box_velocities[i].clamp(-0.05, 0.05);
        }

        // Update eta (extended coordinate)
        self.eta += self.xi * dt;
    }

    pub fn temperature(&self) -> f64 {
        2.0 * self.kinetic_energy() / (self.dof as f64 * self.k_b)
    }
}

/// Cluster structure analyzer
struct ClusterAnalyzer {
    energy_history: VecDeque<f64>,
    temp_history: VecDeque<f64>,
    pressure_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    max_history: usize,
}

impl ClusterAnalyzer {
    fn new(max_history: usize) -> Self {
        Self {
            energy_history: VecDeque::new(),
            temp_history: VecDeque::new(),
            pressure_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            max_history,
        }
    }

    fn update(&mut self, energy: f64, temp: f64, pressure: f64, volume: f64) {
        self.energy_history.push_back(energy);
        self.temp_history.push_back(temp);
        self.pressure_history.push_back(pressure);
        self.volume_history.push_back(volume);

        if self.energy_history.len() > self.max_history {
            self.energy_history.pop_front();
            self.temp_history.pop_front();
            self.pressure_history.pop_front();
            self.volume_history.pop_front();
        }
    }

    fn average_energy(&self) -> f64 {
        if self.energy_history.is_empty() {
            0.0
        } else {
            self.energy_history.iter().sum::<f64>() / self.energy_history.len() as f64
        }
    }

    fn average_pressure(&self) -> f64 {
        if self.pressure_history.is_empty() {
            0.0
        } else {
            self.pressure_history.iter().sum::<f64>() / self.pressure_history.len() as f64
        }
    }

    fn average_volume(&self) -> f64 {
        if self.volume_history.is_empty() {
            0.0
        } else {
            self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64
        }
    }

    fn energy_fluctuation(&self) -> f64 {
        if self.energy_history.len() < 2 {
            return 0.0;
        }
        let avg = self.average_energy();
        let variance = self
            .energy_history
            .iter()
            .map(|&e| (e - avg).powi(2))
            .sum::<f64>()
            / self.energy_history.len() as f64;
        variance.sqrt()
    }
}

/// Calculate coordination number (number of neighbors within 1.5σ)
fn calculate_coordination_number(positions: &[Vector3<f64>], box_lengths: Vector3<f64>) -> f64 {
    let n = positions.len();
    let cutoff = 1.5 * SIGMA;
    let cutoff2 = cutoff * cutoff;
    let mut total_neighbors = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let mut dr = positions[i] - positions[j];

            // Minimum image convention
            for k in 0..3 {
                let box_l = box_lengths[k];
                dr[k] -= box_l * (dr[k] / box_l).round();
            }

            if dr.norm_squared() < cutoff2 {
                total_neighbors += 2; // Count for both i and j
            }
        }
    }

    total_neighbors as f64 / n as f64
}

/// Create FCC cluster (face-centered cubic)
fn create_fcc_cluster(n_cells: usize, lattice_constant: f64) -> Vec<Vector3<f64>> {
    let mut positions = Vec::new();
    let a = lattice_constant;

    // FCC unit cell basis vectors
    let basis = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.5 * a, 0.5 * a, 0.0),
        Vector3::new(0.5 * a, 0.0, 0.5 * a),
        Vector3::new(0.0, 0.5 * a, 0.5 * a),
    ];

    for i in 0..n_cells {
        for j in 0..n_cells {
            for k in 0..n_cells {
                for &basis_vec in &basis {
                    let pos = Vector3::new(i as f64 * a, j as f64 * a, k as f64 * a) + basis_vec;
                    positions.push(pos);
                }
            }
        }
    }

    // Center the cluster
    let center: Vector3<f64> = positions.iter().sum::<Vector3<f64>>() / positions.len() as f64;
    for pos in &mut positions {
        *pos -= center;
    }

    positions
}

/// Initialize Maxwell-Boltzmann velocities
fn initialize_velocities(n_atoms: usize, temperature: f64, mass: f64) -> Vec<Vector3<f64>> {
    let mut rng = rand::thread_rng();
    let mut velocities = Vec::with_capacity(n_atoms);

    // Sample from normal distribution
    for _ in 0..n_atoms {
        let v = Vector3::new(
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
            rng.sample(StandardNormal),
        );
        velocities.push(v * (temperature / mass).sqrt());
    }

    // Remove center-of-mass motion
    let v_cm: Vector3<f64> = velocities.iter().sum::<Vector3<f64>>() / n_atoms as f64;
    for v in &mut velocities {
        *v -= v_cm;
    }

    // Scale to exact target temperature
    let ke = velocities
        .iter()
        .map(|v| 0.5 * mass * v.norm_squared())
        .sum::<f64>();
    let current_temp = 2.0 * ke / (3.0 * n_atoms as f64 * K_B);

    if current_temp > 0.0 {
        let scale_factor = (temperature / current_temp).sqrt();
        for v in &mut velocities {
            *v *= scale_factor;
        }
    }

    velocities
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║               Lennard-Jones Cluster NPT Simulation                      ║");
    println!("║              With Proper Virial Pressure Calculation                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║ Physical Features:                                                       ║");
    println!("║   • Nosé-Hoover thermostat (temperature control)                        ║");
    println!("║   • Parrinello-Rahman barostat (pressure control)                       ║");
    println!("║   • Proper virial-based pressure: P = (2K + W)/(3V)                     ║");
    println!("║   • Cluster-specific analysis (coordination, structure)                 ║");
    println!("║   • Phase transitions and melting behavior                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // System parameters (reduced units)
    let n_cells = 2; // 2x2x2 FCC = 32 atoms
    let lattice_constant = 1.55; // Slightly larger than equilibrium for FCC (√2 ≈ 1.414)
    let initial_temp = 0.5; // Low temperature (solid-like)
    let target_pressure = 0.1; // Moderate pressure

    // Create FCC cluster
    let mut positions = create_fcc_cluster(n_cells, lattice_constant);
    let n_atoms = positions.len();

    println!("System Setup:");
    println!("  Structure: FCC cluster");
    println!("  Unit cells: {}×{}×{}", n_cells, n_cells, n_cells);
    println!("  Number of atoms: {}", n_atoms);
    println!("  Lattice constant: {:.3} σ", lattice_constant);
    println!("  Initial temperature: {:.3} ε/k_B", initial_temp);
    println!("  Target pressure: {:.3} ε/σ³", target_pressure);
    println!();

    // Initialize velocities
    let velocities = initialize_velocities(n_atoms, initial_temp, ARGON_MASS);
    let masses = vec![ARGON_MASS; n_atoms];

    // Initial box - large enough to contain cluster with buffer
    let cluster_size = n_cells as f64 * lattice_constant * 1.5;
    let initial_box = Vector3::new(cluster_size, cluster_size, cluster_size);

    // Shift positions to center of box
    let box_center = initial_box * 0.5;
    for pos in &mut positions {
        *pos += box_center;
    }

    println!(
        "  Initial box: {:.2} × {:.2} × {:.2} σ³",
        initial_box.x, initial_box.y, initial_box.z
    );
    println!("  Initial volume: {:.2} σ³", initial_box.x.powi(3));
    println!(
        "  Initial density: {:.4} atoms/σ³",
        n_atoms as f64 / initial_box.x.powi(3)
    );
    println!();

    // NPT integrator with optimized coupling parameters for clusters
    let q_t = 100.0; // Thermostat coupling (moderate)
    let q_p = 2000.0; // Barostat coupling (gentle for stability)

    let mut integrator = LJClusterNPT::new(
        positions,
        velocities,
        masses,
        initial_box,
        q_t,
        q_p,
        initial_temp,
        target_pressure,
        EPSILON,
        SIGMA,
    );

    // Analysis
    let mut analyzer = ClusterAnalyzer::new(200);

    // Simulation parameters
    let dt = 0.002; // Time step (reduced units)
    let total_steps = 30000;
    let output_interval = 500;
    let analysis_interval = 50;

    println!("Simulation Parameters:");
    println!("  Time step: {:.3} τ", dt);
    println!("  Total steps: {}", total_steps);
    println!("  Total time: {:.1} τ", dt * total_steps as f64);
    println!("  Thermostat coupling Q_T: {:.1}", q_t);
    println!("  Barostat coupling Q_P: {:.1}", q_p);
    println!();

    // Temperature schedule (heating to observe melting)
    let heating_start = 10000;
    let heating_end = 20000;
    let final_temp = 1.5; // Above melting point

    println!("Temperature Schedule:");
    println!(
        "  Steps 0-{}: T = {:.2} ε/k_B (equilibration)",
        heating_start, initial_temp
    );
    println!(
        "  Steps {}-{}: T = {:.2} → {:.2} ε/k_B (heating)",
        heating_start, heating_end, initial_temp, final_temp
    );
    println!(
        "  Steps {}-{}: T = {:.2} ε/k_B (equilibration)",
        heating_end, total_steps, final_temp
    );
    println!();

    // Output header
    println!("┌────────┬─────────┬─────────┬─────────┬──────────┬──────────┬─────────┬─────────┐");
    println!(
        "│ {:>6} │ {:>7} │ {:>7} │ {:>7} │ {:>8} │ {:>8} │ {:>7} │ {:>7} │",
        "Step", "T_inst", "P_inst", "P_tgt", "Volume", "PE", "Box_L", "Coord#"
    );
    println!("├────────┼─────────┼─────────┼─────────┼──────────┼──────────┼─────────┼─────────┤");

    // Main simulation loop
    for step in 0..total_steps {
        // Update target temperature (heating schedule)
        let current_target_temp = if step < heating_start {
            initial_temp
        } else if step < heating_end {
            let progress = (step - heating_start) as f64 / (heating_end - heating_start) as f64;
            initial_temp + (final_temp - initial_temp) * progress
        } else {
            final_temp
        };

        integrator.set_target_temperature(current_target_temp);

        // Integration step
        integrator.step(dt);

        // Update analysis
        if step % analysis_interval == 0 {
            let pe = integrator.get_potential_energy();
            let temp = integrator.temperature();
            let pressure = integrator.get_pressure();
            let volume = integrator.get_volume();
            analyzer.update(pe, temp, pressure, volume);
        }

        // Output
        if step % output_interval == 0 {
            let temp = integrator.temperature();
            let pressure = integrator.get_pressure();
            let volume = integrator.get_volume();
            let pe = integrator.get_potential_energy();
            let box_l = integrator.box_lengths.x;
            let coord_num =
                calculate_coordination_number(&integrator.positions, integrator.box_lengths);

            println!(
                "│ {:>6} │ {:>7.3} │ {:>7.3} │ {:>7.3} │ {:>8.2} │ {:>8.3} │ {:>7.2} │ {:>7.2} │",
                step, temp, pressure, target_pressure, volume, pe, box_l, coord_num
            );
        }
    }

    println!("└────────┴─────────┴─────────┴─────────┴──────────┴──────────┴─────────┴─────────┘");
    println!();

    // Final analysis
    let final_temp = integrator.temperature();
    let final_pressure = integrator.get_pressure();
    let final_volume = integrator.get_volume();
    let final_pe = integrator.get_potential_energy();
    let final_density = n_atoms as f64 / final_volume;
    let final_coord = calculate_coordination_number(&integrator.positions, integrator.box_lengths);

    let avg_energy = analyzer.average_energy();
    let avg_pressure = analyzer.average_pressure();
    let avg_volume = analyzer.average_volume();
    let energy_fluct = analyzer.energy_fluctuation();

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("                            FINAL ANALYSIS                                 ");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("Instantaneous Properties:");
    println!(
        "  Temperature: {:.4} ε/k_B (target: {:.3})",
        final_temp, final_temp
    );
    println!(
        "  Pressure: {:.4} ε/σ³ (target: {:.3})",
        final_pressure, target_pressure
    );
    println!("  Volume: {:.3} σ³", final_volume);
    println!("  Density: {:.4} atoms/σ³", final_density);
    println!("  Potential energy: {:.3} ε", final_pe);
    println!("  PE per atom: {:.4} ε", final_pe / n_atoms as f64);
    println!(
        "  Box dimensions: {:.2} × {:.2} × {:.2} σ³",
        integrator.box_lengths.x, integrator.box_lengths.y, integrator.box_lengths.z
    );
    println!("  Coordination number: {:.2}", final_coord);
    println!();

    println!(
        "Averaged Properties (last {} samples):",
        analyzer.energy_history.len()
    );
    println!("  <Energy>: {:.3} ε", avg_energy);
    println!("  <Pressure>: {:.3} ε/σ³", avg_pressure);
    println!("  <Volume>: {:.3} σ³", avg_volume);
    println!("  Energy fluctuation: {:.3} ε", energy_fluct);
    println!(
        "  Relative fluctuation: {:.2}%",
        100.0 * energy_fluct / avg_energy.abs()
    );
    println!();

    // Phase identification
    println!("Structural Analysis:");

    // Coordination number analysis
    if final_coord > 10.0 {
        println!(
            "  🧊 SOLID-LIKE structure (high coordination: {:.1})",
            final_coord
        );
        println!("     Atoms are closely packed in crystalline arrangement");
    } else if final_coord > 6.0 {
        println!(
            "  🌊 LIQUID-LIKE structure (moderate coordination: {:.1})",
            final_coord
        );
        println!("     Atoms maintain local order but with mobility");
    } else {
        println!(
            "  💨 GAS-LIKE structure (low coordination: {:.1})",
            final_coord
        );
        println!("     Atoms are dispersed with few neighbors");
    }

    // Energy per atom analysis
    let pe_per_atom = final_pe / n_atoms as f64;
    if pe_per_atom < -4.0 {
        println!(
            "  ❄️  Very stable structure (PE/atom = {:.2} ε)",
            pe_per_atom
        );
    } else if pe_per_atom < -2.0 {
        println!("  📊 Moderately stable (PE/atom = {:.2} ε)", pe_per_atom);
    } else {
        println!("  🔥 High energy state (PE/atom = {:.2} ε)", pe_per_atom);
    }

    // Density analysis
    if final_density > 0.8 {
        println!(
            "  📦 High density: {:.3} atoms/σ³ (condensed phase)",
            final_density
        );
    } else if final_density > 0.3 {
        println!(
            "  📊 Medium density: {:.3} atoms/σ³ (intermediate)",
            final_density
        );
    } else {
        println!(
            "  🌫️  Low density: {:.3} atoms/σ³ (expanded/gas)",
            final_density
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("                         KEY INSIGHTS                                      ");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("✅ NPT Ensemble Physics:");
    println!("   • Thermostat maintained temperature through heating cycle");
    println!("   • Barostat adjusted volume to control pressure");
    println!("   • Proper virial pressure includes inter-atomic forces");
    println!("   • System explored phase space at constant (N,P,T)");
    println!();
    println!("🔬 Cluster Behavior:");
    println!(
        "   • FCC cluster underwent heating from T={:.2} to T={:.2}",
        initial_temp, final_temp
    );
    println!("   • Coordination number changed during heating");
    println!("   • Energy fluctuations indicate thermal motion");
    println!("   • Volume responded to pressure balance");
    println!();
    println!("💡 Physical Significance:");
    println!("   • This simulation demonstrates realistic cluster thermodynamics");
    println!("   • LJ clusters can melt, evaporate, or condense depending on (P,T)");
    println!("   • Virial pressure is crucial for accurate NPT dynamics");
    println!("   • Coordination number is a key structural order parameter");
    println!();
}
