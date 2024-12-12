#! /usr/bin/env python

import sympy as sp
from sympy import symbols, exp, diff, integrate, simplify, codegen
from sympy.abc import alpha


def calculate_gto1d_kinetic_integral():
    # Define symbolic variables
    x, y, z = symbols('x y z', real=True)
    alpha1, alpha2 = symbols('α₁ α₂', positive=True, real=True)
    Ax, Ay, Az = symbols('Ax Ay Az', positive=True, real=True)  # Center of first GTO
    Bx, By, Bz = symbols('Bx By Bz', positive=True, real=True)  # Center of second GTO

    # Define the GTOs centered at points A and B
    gto1 = exp(-alpha1*((x-Ax)**2))
    gto2 = exp(-alpha2*((x-Bx)**2))

    # Calculate the Laplacian of gto2 (∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²)
    laplacian_gto2 = diff(gto2, x, 2)
    integrand = gto1 * (-0.5 * laplacian_gto2)

    # Calculate the triple integral over all space
    integral = simplify(integrate(integrand, (x, -sp.oo, sp.oo)))

    return integral

def calculate_gto_kinetic_integral():
    # Define symbolic variables
    x, y, z = symbols('x y z', real=True)
    alpha1, alpha2 = symbols('α₁ α₂', positive=True, real=True)
    Ax, Ay, Az = symbols('Ax Ay Az', positive=True, real=True)  # Center of first GTO
    Bx, By, Bz = symbols('Bx By Bz', positive=True, real=True)  # Center of second GTO    # Define the GTOs centered at points A and B
    gto1 = exp(-alpha1*((x-Ax)**2 + (y-Ay)**2 + (z-Az)**2))
    gto2 = exp(-alpha2*((x-Bx)**2 + (y-By)**2 + (z-Bz)**2))

    # Calculate the Laplacian of gto2 (∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²)
    laplacian_gto2 = (diff(gto2, x, 2) +
                      diff(gto2, y, 2) +
                      diff(gto2, z, 2))

    # Construct the integrand for kinetic energy (-1/2 ∇²)
    integrand = gto1 * (-0.5 * laplacian_gto2)

    # Calculate the triple integral over all space
    integral = integrate(
            integrate(
                integrate(integrand, (x, -sp.oo, sp.oo)),
                            (y, -sp.oo, sp.oo)),
                         (z, -sp.oo, sp.oo))

    return sp.simplify(integral)

if __name__ == "__main__":
    # Calculate and display the result
    result = calculate_gto1d_kinetic_integral()
    print("Kinetic energy integral between two s-type GTO1ds:")
    print(result)