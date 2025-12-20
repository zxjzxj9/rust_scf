#!/usr/bin/env python3
"""
Generate pointwise TPSS reference values using PySCF/libxc for Rust unit tests.

This avoids relying on total molecular energies (grid-dependent) and instead validates the
functional itself at fixed (rho, sigma, tau) points.

Usage (from repo root):
  `/Users/victor/Programs/rust_scf/venv/bin/python scf/tools/gen_tpss_libxc_ref.py`
"""

from __future__ import annotations

import numpy as np
from pyscf.dft.libxc import eval_xc


def main() -> None:
    # Deterministic sample points: (rho, sigma=|grad rho|^2, tau)
    pts = [
        (1e-4, 0.0, 1e-5),
        (5e-4, 1e-8, 2e-4),
        (1e-3, 5e-7, 5e-4),
        (5e-3, 1e-5, 1e-3),
        (1e-2, 2e-5, 2e-3),
        (5e-2, 2e-4, 1e-2),
        (1e-1, 5e-4, 2e-2),
        (2e-1, 1e-3, 5e-2),
        (5e-1, 2e-3, 1e-1),
        (1.0, 5e-3, 2e-1),
    ]

    ng = len(pts)
    rho = np.zeros((6, ng))
    rho[4] = 0.0  # lapl (unused here)
    for i, (r, sig, tau) in enumerate(pts):
        rho[0, i] = r
        g = np.sqrt(sig)
        rho[1, i] = g
        rho[2, i] = 0.0
        rho[3, i] = 0.0
        rho[5, i] = tau

    exc, vxc, fxc, kxc = eval_xc("tpss", rho, spin=0, deriv=1)
    vrho, vsigma, vlapl, vtau = vxc

    print("PTS = [")
    for i, (r, sig, tau) in enumerate(pts):
        e = r * exc[i]  # energy density per volume
        print(
            "  (%.16e, %.16e, %.16e, %.16e, %.16e, %.16e),"
            % (r, sig, tau, e, vrho[i], vtau[i])
        )
    print("]")

    print("VSIGMA = [")
    for i in range(ng):
        print("  %.16e," % vsigma[i])
    print("]")


if __name__ == "__main__":
    main()



