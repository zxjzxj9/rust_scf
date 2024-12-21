#! /usr/bin/env python

import requests
import json
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

# Match Rust enums and structs
class ShellType(Enum):
    S = 0  # l = 0
    P = 1  # l = 1
    D = 2  # l = 2
    F = 3  # l = 3

@dataclass
class GTO:
    exponent: float
    coefficients: List[float]  # Changed to handle multiple coefficients
    angular_momentum: List[int]  # Added to store l values

@dataclass
class BasisSet:
    shells: List[GTO]
    element: int  # Added to store atomic number

def parse_basis_set(json_data: Dict, element_number: str) -> BasisSet:
    shells = []

    # Get the electron shells for the specific element
    element_data = json_data['elements'][element_number]['electron_shells']

    for shell in element_data:
        # Convert exponents to floats
        exponents = [float(exp) for exp in shell['exponents']]

        # Get angular momentum values
        angular_momentum = shell['angular_momentum']

        # Get coefficients for all angular momentum components
        coefficient_sets = shell['coefficients']

        # For each exponent, create a GTO with all its coefficients
        for i in range(len(exponents)):
            coeffs = [float(coeff_set[i]) for coeff_set in coefficient_sets]
            gto = GTO(
                exponent=exponents[i],
                coefficients=coeffs,
                angular_momentum=angular_momentum
            )
            shells.append(gto)

    return BasisSet(
        shells=shells,
        element=int(element_number)
    )

if __name__ == "__main__":

    base_url = "https://www"

    json_data = requests.get("https://www.basissetexchange.org/download_basis/basis/6-31g/format/json/?version=1&elements=20").json()

    # Parse all elements in the basis set
    basis_sets = {}
    for element_number in json_data['elements'].keys():
        basis_sets[element_number] = parse_basis_set(json_data, element_number)

    # Serialize to pickle
    with open('basis_sets.pkl', 'wb') as f:
        pickle.dump(basis_sets, f, protocol=4)

    print("Pickle file created successfully!")

    # Print some information about what was saved
    for element_num, basis in basis_sets.items():
        print(f"\nElement {element_num}:")
        print(f"Number of shells: {len(basis.shells)}")
        for i, shell in enumerate(basis.shells):
            print(f"  Shell {i}: angular momentum {shell.angular_momentum}, "
                  f"{len(shell.coefficients)} coefficients")