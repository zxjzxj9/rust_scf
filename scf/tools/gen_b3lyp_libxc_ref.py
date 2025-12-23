#!/usr/bin/env python3
"""
Generate pointwise B3LYP component reference values using PySCF/libxc for Rust unit tests.

We validate the functional definitions at fixed (rho, sigma) points to avoid grid-dependent
total-energy comparisons.

Usage (from repo root):
  `/Users/victor/Programs/rust_scf/venv/bin/python scf/tools/gen_b3lyp_libxc_ref.py`
"""

from __future__ import annotations

import numpy as np
from pyscf.dft.libxc import eval_xc


def build_gga_rho_array(pts: list[tuple[float, float]]) -> np.ndarray:
    """Return GGA-shaped rho array (4, N) with gradient along x so sigma=|grad|^2."""
    ng = len(pts)
    rho = np.zeros((4, ng))
    for i, (r, sig) in enumerate(pts):
        rho[0, i] = r
        g = np.sqrt(sig)
        rho[1, i] = g
        rho[2, i] = 0.0
        rho[3, i] = 0.0
    return rho


def dump_gga(name: str, pts: list[tuple[float, float]]) -> None:
    rho_arr = build_gga_rho_array(pts)
    exc, vxc, _, _ = eval_xc(name, rho_arr, spin=0, deriv=1)
    vrho, vsigma = vxc
    print(f"{name.upper().replace('-', '_').replace(' ', '_')} = [")
    for i, (r, sig) in enumerate(pts):
        e = r * exc[i]
        print(
            "  (%.16e, %.16e, %.16e, %.16e, %.16e),"
            % (r, sig, e, vrho[i], vsigma[i])
        )
    print("]\n")


def dump_lda(name: str, pts_rho: list[float]) -> None:
    rho_arr = np.zeros((1, len(pts_rho)))
    rho_arr[0, :] = np.array(pts_rho)
    exc, vxc, _, _ = eval_xc(name, rho_arr, spin=0, deriv=1)
    (vrho,) = vxc
    print(f"{name.upper().replace('-', '_').replace(' ', '_')} = [")
    for i, r in enumerate(pts_rho):
        e = r * exc[i]
        print("  (%.16e, %.16e, %.16e)," % (r, e, vrho[i]))
    print("]\n")


def main() -> None:
    pts = [
        (1e-4, 0.0),
        (5e-4, 1e-8),
        (1e-3, 5e-7),
        (5e-3, 1e-5),
        (1e-2, 2e-5),
        (5e-2, 2e-4),
        (1e-1, 5e-4),
        (2e-1, 1e-3),
        (5e-1, 2e-3),
        (1.0, 5e-3),
    ]

    dump_gga("gga_x_b88", pts)
    dump_gga("gga_c_lyp", pts)
    dump_lda("lda_c_vwn_rpa", [r for (r, _) in pts])
    dump_gga("b3lyp", pts)  # semilocal part only (HF exchange handled separately)


if __name__ == "__main__":
    main()


