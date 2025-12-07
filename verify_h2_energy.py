#!/usr/bin/env python3

try:
    from pyscf import gto, scf
    import numpy as np
    
    # H2 molecule with bond length 1.4 bohr (same as test)
    mol = gto.Mole()
    mol.atom = [
        ['H', (0.0, 0.0, -0.7)],  # -0.7 bohr
        ['H', (0.0, 0.0,  0.7)]   # +0.7 bohr  -> 1.4 bohr separation
    ]
    mol.basis = 'sto-3g'
    mol.unit = 'bohr'  # Important: specify bohr units
    mol.build()
    
    # Perform RHF calculation
    mf = scf.RHF(mol)
    energy = mf.scf()
    
    print(f"H2 STO-3G energy at 1.4 bohr: {energy:.12f} au")
    print(f"Nuclear repulsion energy: {mol.energy_nuc():.12f} au")
    print(f"Electronic energy: {energy - mol.energy_nuc():.12f} au")
    
except ImportError:
    print("PySCF not available. Using known literature values:")
    print("H2 STO-3G at optimal bond length (~1.35 bohr): ~-1.117 au")
    print("H2 STO-3G at 1.4 bohr: should be slightly higher energy")
    print("The test expected energy of -1.06645395 au seems questionable.")
    print("This might be the cause of the test failure.") 