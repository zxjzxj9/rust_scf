#!/usr/bin/env python3

try:
    from pyscf import gto, scf
    import numpy as np
    
    # H2O molecule with same geometry as Rust test
    mol = gto.Mole()
    mol.atom = [
        ['O', (0.000000,  0.000000,  0.000000)],
        ['H', (0.757000,  0.586000,  0.000000)],
        ['H', (-0.757000, 0.586000,  0.000000)]
    ]
    mol.basis = '6-31g'
    mol.unit = 'bohr'  # Important: specify bohr units
    mol.charge = 0
    mol.spin = 0  # Singlet state
    mol.build()
    
    # Perform RHF calculation
    print("=" * 60)
    print("PySCF H2O Calculation with 6-31G basis")
    print("=" * 60)
    print(f"Number of electrons: {mol.nelectron}")
    print(f"Nuclear repulsion energy: {mol.energy_nuc():.12f} au")
    print()
    
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-8  # Match Rust convergence threshold
    energy = mf.kernel()
    
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total SCF energy:       {energy:.12f} au")
    print(f"Nuclear repulsion:      {mol.energy_nuc():.12f} au")
    print(f"Electronic energy:      {energy - mol.energy_nuc():.12f} au")
    print("=" * 60)
    
except ImportError:
    print("PySCF not available. Please install it with:")
    print("pip install pyscf")

