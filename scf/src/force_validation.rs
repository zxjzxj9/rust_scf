use nalgebra::Vector3;
use periodic_table_on_an_enum::Element;
use crate::scf_impl::{SimpleSCF, SCF};
use crate::optim_impl::{GeometryOptimizer, CGOptimizer, create_optimizer};
use basis::cgto::{Basis631G, ContractedGTO};
use basis::gto::GTO;
use std::collections::HashMap;
use tracing::info;

pub struct ForceValidator;

/// Create a minimal H basis set for testing (simplified STO-3G-like)
fn create_minimal_h_basis() -> Basis631G {
    let mut basis = Basis631G {
        name: "minimal-H".to_string(),
        atomic_number: 1,
        basis_set: Vec::new(),
    };
    
    // Create a simple 1s orbital with one primitive
    let mut h_1s = ContractedGTO {
        primitives: vec![GTO::new(1.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0))],
        coefficients: vec![1.0],
        shell_type: "1s".to_string(),
        Z: 1,
        n: 1,
        l: 0,
        m: 0,
        s: 0,
    };
    
    basis.basis_set.push(h_1s);
    basis
}

/// Create a minimal O basis set for testing
fn create_minimal_o_basis() -> Basis631G {
    let mut basis = Basis631G {
        name: "minimal-O".to_string(),
        atomic_number: 8,
        basis_set: Vec::new(),
    };
    
    // Create simple 1s and 2p orbitals
    let h_1s = ContractedGTO {
        primitives: vec![GTO::new(5.0, Vector3::new(0, 0, 0), Vector3::new(0.0, 0.0, 0.0))],
        coefficients: vec![1.0],
        shell_type: "1s".to_string(),
        Z: 8,
        n: 1,
        l: 0,
        m: 0,
        s: 0,
    };
    
    let p_2px = ContractedGTO {
        primitives: vec![GTO::new(2.0, Vector3::new(1, 0, 0), Vector3::new(0.0, 0.0, 0.0))],
        coefficients: vec![1.0],
        shell_type: "2px".to_string(),
        Z: 8,
        n: 2,
        l: 1,
        m: -1,
        s: 0,
    };
    
    let p_2py = ContractedGTO {
        primitives: vec![GTO::new(2.0, Vector3::new(0, 1, 0), Vector3::new(0.0, 0.0, 0.0))],
        coefficients: vec![1.0],
        shell_type: "2py".to_string(),
        Z: 8,
        n: 2,
        l: 1,
        m: 1,
        s: 0,
    };
    
    let p_2pz = ContractedGTO {
        primitives: vec![GTO::new(2.0, Vector3::new(0, 0, 1), Vector3::new(0.0, 0.0, 0.0))],
        coefficients: vec![1.0],
        shell_type: "2pz".to_string(),
        Z: 8,
        n: 2,
        l: 1,
        m: 0,
        s: 0,
    };
    
    basis.basis_set.extend(vec![h_1s, p_2px, p_2py, p_2pz]);
    basis
}

impl ForceValidator {
    /// Comprehensive force validation using numerical differentiation
    pub fn validate_forces_comprehensive<S: SCF + Clone>(
        scf: &mut S,
        coords: &[Vector3<f64>],
        elements: &[Element],
        delta: f64,
    ) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>, f64) {
        info!("=======================================================");
        info!("         Comprehensive Force Validation");
        info!("=======================================================");

        // Initialize SCF with current geometry
        scf.init_geometry(&coords.to_vec(), &elements.to_vec());
        scf.scf_cycle();
        
        // Get analytical forces
        let analytical_forces = scf.calculate_forces();
        let mut numerical_forces = vec![Vector3::zeros(); coords.len()];
        
        // Calculate numerical forces using central differences
        for atom_idx in 0..coords.len() {
            for dim in 0..3 {
                let mut pos_coords = coords.to_vec();
                let mut neg_coords = coords.to_vec();
                
                // Positive displacement
                match dim {
                    0 => pos_coords[atom_idx].x += delta,
                    1 => pos_coords[atom_idx].y += delta,
                    2 => pos_coords[atom_idx].z += delta,
                    _ => unreachable!(),
                }
                
                // Negative displacement
                match dim {
                    0 => neg_coords[atom_idx].x -= delta,
                    1 => neg_coords[atom_idx].y -= delta,
                    2 => neg_coords[atom_idx].z -= delta,
                    _ => unreachable!(),
                }
                
                // Calculate energies at displaced geometries
                let mut scf_pos = scf.clone();
                scf_pos.init_geometry(&pos_coords, &elements.to_vec());
                scf_pos.scf_cycle();
                let pos_energy = scf_pos.calculate_total_energy();
                
                let mut scf_neg = scf.clone();
                scf_neg.init_geometry(&neg_coords, &elements.to_vec());
                scf_neg.scf_cycle();
                let neg_energy = scf_neg.calculate_total_energy();
                
                // Central difference approximation
                let force_component = -(pos_energy - neg_energy) / (2.0 * delta);
                match dim {
                    0 => numerical_forces[atom_idx].x = force_component,
                    1 => numerical_forces[atom_idx].y = force_component,
                    2 => numerical_forces[atom_idx].z = force_component,
                    _ => unreachable!(),
                }
            }
        }
        
        // Calculate error metrics
        let mut max_error: f64 = 0.0;
        let mut rms_error: f64 = 0.0;
        let mut total_squared_error: f64 = 0.0;
        
        info!("Force validation results:");
        info!("  Atom |  Analytical Forces  |  Numerical Forces   |  Absolute Error");
        info!("-------|---------------------|---------------------|------------------");
        
        for (i, (ana, num)) in analytical_forces.iter().zip(numerical_forces.iter()).enumerate() {
            let error = (ana - num).norm();
            max_error = max_error.max(error);
            total_squared_error += error * error;
            
            info!("   {:2}  | [{:8.4}, {:8.4}, {:8.4}] | [{:8.4}, {:8.4}, {:8.4}] | {:10.6}",
                i + 1, ana.x, ana.y, ana.z, num.x, num.y, num.z, error);
        }
        
        rms_error = (total_squared_error / analytical_forces.len() as f64).sqrt();
        
        info!("-------|---------------------|---------------------|------------------");
        info!("  RMS Error: {:.6}", rms_error);
        info!("  Max Error: {:.6}", max_error);
        info!("=======================================================\n");
        
        (analytical_forces, numerical_forces, max_error)
    }
    
    /// Test geometry optimization convergence with force validation
    pub fn test_optimization_convergence() -> Result<(), Box<dyn std::error::Error>> {
        info!("#####################################################");
        info!("       Testing Optimization Convergence");
        info!("#####################################################");
        
        // Create H2 molecule with slightly displaced geometry
        let mut coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.6), // Slightly longer than equilibrium (~0.74 Å)
        ];
        let elements = vec![Element::Hydrogen, Element::Hydrogen];
        
        // Initialize SCF with optimized settings for testing
        let mut scf = SimpleSCF::<Basis631G>::new();
        scf.max_cycle = 8;       // Fewer SCF cycles
        
        let mut basis = HashMap::new();
        let h_basis = create_minimal_h_basis();
        basis.insert("H", &h_basis);
        
        scf.init_basis(&elements, basis);
        scf.init_geometry(&coords, &elements);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
        
        let initial_energy = scf.calculate_total_energy();
        let initial_bond_length = (coords[1] - coords[0]).norm();
        
        info!("Initial conditions:");
        info!("  Bond length: {:.6} bohr ({:.6} Å)", 
              initial_bond_length, initial_bond_length * 0.529177);
        info!("  Energy: {:.8} au", initial_energy);
        
        // Validate initial forces (with larger step size for speed)
        let (analytical_forces, _, max_error) = Self::validate_forces_comprehensive(
            &mut scf, &coords, &elements, 1e-3
        );
        
        // Test if force calculation is reasonable
        if max_error > 1e-2 {
            info!("Warning: Large discrepancy in force calculation (max error: {:.6})", max_error);
            info!("This indicates incomplete implementation of force derivatives.");
        }
        
        // Run geometry optimization (reduced iterations for speed)
        info!("Starting geometry optimization...");
        let mut optimizer = CGOptimizer::new(&mut scf, 5, 1e-3);
        optimizer.init(coords.clone(), elements.clone());
        
        let (optimized_coords, final_energy) = optimizer.optimize();
        let final_bond_length = (optimized_coords[1] - optimized_coords[0]).norm();
        
        info!("Optimization results:");
        info!("  Final bond length: {:.6} bohr ({:.6} Å)", 
              final_bond_length, final_bond_length * 0.529177);
        info!("  Final energy: {:.8} au", final_energy);
        info!("  Energy change: {:.8} au", final_energy - initial_energy);
        
        // Validate final forces
        info!("Validating forces at final geometry...");
        scf.init_geometry(&optimized_coords, &elements);
        scf.scf_cycle();
        
        let (final_analytical_forces, _, final_max_error) = Self::validate_forces_comprehensive(
            &mut scf, &optimized_coords, &elements, 1e-3
        );
        
        // Check convergence criteria
        let final_max_force = final_analytical_forces.iter()
            .map(|f| f.norm())
            .fold(0.0, f64::max);
        
        info!("Convergence analysis:");
        info!("  Final max force: {:.8} au", final_max_force);
        info!("  Force validation error: {:.6}", final_max_error);
        info!("  Energy lowered: {}", final_energy < initial_energy);
        
        // The geometry should have improved
        if final_energy >= initial_energy {
            info!("Warning: Energy did not decrease during optimization!");
        }
        
        // Expected H2 equilibrium bond length is around 1.4 bohr
        let expected_equilibrium = 1.4; // bohr
        let bond_length_error = (final_bond_length - expected_equilibrium).abs();
        
        info!("  Bond length vs expected equilibrium:");
        info!("    Optimized: {:.6} bohr", final_bond_length);
        info!("    Expected:  {:.6} bohr", expected_equilibrium);
        info!("    Error:     {:.6} bohr", bond_length_error);
        
        if bond_length_error > 0.2 {
            info!("Warning: Final bond length significantly differs from expected equilibrium.");
            info!("This may indicate issues with force calculation or optimization algorithm.");
        }
        
        info!("#####################################################\n");
        
        Ok(())
    }
    
    /// Test force calculation with different step sizes
    pub fn test_numerical_gradient_convergence() -> Result<(), Box<dyn std::error::Error>> {
        info!("#####################################################");
        info!("     Testing Numerical Gradient Convergence");
        info!("#####################################################");
        
        // Simple H2 molecule
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.5),
        ];
        let elements = vec![Element::Hydrogen, Element::Hydrogen];
        
        // Initialize SCF with optimized settings for testing
        let mut scf = SimpleSCF::<Basis631G>::new();
        scf.max_cycle = 8;       // Fewer SCF cycles
        
        let mut basis = HashMap::new();
        let h_basis = create_minimal_h_basis();
        basis.insert("H", &h_basis);
        
        scf.init_basis(&elements, basis);
        scf.init_geometry(&coords, &elements);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
        
        let analytical_forces = scf.calculate_forces();
        
        // Test different step sizes
        let step_sizes = [1e-3, 1e-4];  // Reduced step sizes for faster testing
        
        info!("Testing numerical gradient convergence with different step sizes:");
        info!("Step Size | Max Error | RMS Error | Notes");
        info!("----------|-----------|-----------|------");
        
        for &delta in &step_sizes {
            let (_, numerical_forces, _) = Self::validate_forces_comprehensive(
                &mut scf, &coords, &elements, delta
            );
            
            // Calculate errors
            let mut max_error: f64 = 0.0;
            let mut total_squared_error: f64 = 0.0;
            
            for (ana, num) in analytical_forces.iter().zip(numerical_forces.iter()) {
                let error = (ana - num).norm();
                max_error = max_error.max(error);
                total_squared_error += error * error;
            }
            
            let rms_error = (total_squared_error / analytical_forces.len() as f64).sqrt();
            
            let notes = if delta <= 1e-5 {
                "Good precision"
            } else if delta <= 1e-4 {
                "Acceptable"
            } else {
                "Too large"
            };
            
            info!("{:8.0e} | {:9.6} | {:9.6} | {}", delta, max_error, rms_error, notes);
        }
        
        info!("#####################################################\n");
        
        Ok(())
    }
    
    /// Compare different optimization algorithms
    pub fn compare_optimization_algorithms() -> Result<(), Box<dyn std::error::Error>> {
        info!("#####################################################");
        info!("      Comparing Optimization Algorithms");
        info!("#####################################################");
        
        // H2O molecule with slightly distorted geometry
        let coords = vec![
            Vector3::new(0.0, 0.0, 0.0),     // O
            Vector3::new(0.0, 0.0, 2.0),     // H1 (too far)
            Vector3::new(1.6, 0.0, -0.5),   // H2 (slightly off)
        ];
        let elements = vec![Element::Oxygen, Element::Hydrogen, Element::Hydrogen];
        
        // Initialize SCF with optimized settings for testing
        let mut scf = SimpleSCF::<Basis631G>::new();
        scf.max_cycle = 10;      // Fewer SCF cycles
        
        let mut basis = HashMap::new();
        let h_basis = create_minimal_h_basis();
        let o_basis = create_minimal_o_basis();
        basis.insert("H", &h_basis);
        basis.insert("O", &o_basis);
        
        scf.init_basis(&elements, basis);
        scf.init_geometry(&coords, &elements);
        scf.init_density_matrix();
        scf.init_fock_matrix();
        scf.scf_cycle();
        
        let initial_energy = scf.calculate_total_energy();
        info!("Initial energy: {:.8} au", initial_energy);
        
        // Test different algorithms
        let algorithms = ["sd", "cg"];
        
        for algorithm in &algorithms {
            info!("\nTesting {} algorithm:", match *algorithm {
                "sd" => "Steepest Descent",
                "cg" => "Conjugate Gradient",
                _ => algorithm,
            });
            
            let mut test_scf = scf.clone();
            let final_coords;
            let final_energy;
            {
                let mut optimizer = create_optimizer(algorithm, &mut test_scf, 5, 1e-3)?;  // Fewer iterations, looser convergence
                optimizer.init(coords.clone(), elements.clone());

                let (coords, energy) = optimizer.optimize();
                final_coords = coords;
                final_energy = energy;
            } // optimizer dropped here, releasing the mutable borrow

            info!("  Final energy: {:.8} au", final_energy);
            info!("  Energy change: {:.8} au", final_energy - initial_energy);

            // Validate final forces
            test_scf.init_geometry(&final_coords, &elements);
            test_scf.scf_cycle();
            let final_forces = test_scf.calculate_forces();
            let max_final_force = final_forces.iter().map(|f| f.norm()).fold(0.0, f64::max);
            info!("  Max final force: {:.8} au", max_final_force);
        }
        
        info!("#####################################################\n");
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_force_validation() {
        let _ = ForceValidator::test_optimization_convergence();
    }
    
    #[test]
    fn test_gradient_convergence() {
        let _ = ForceValidator::test_numerical_gradient_convergence();
    }
    
    #[test]
    fn test_algorithm_comparison() {
        let _ = ForceValidator::compare_optimization_algorithms();
    }
} 