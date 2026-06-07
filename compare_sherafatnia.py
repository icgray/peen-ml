#!/usr/bin/env python3
"""
compare_sherafatnia.py
======================
Compare the Sherafatnia/Poozesh analytical formula against the
Shen & Atluri (2006) model already implemented in impact_sim.py.

Key difference:
  Sherafatnia — computes Hertzian ELASTIC LOADING stresses (no unloading).
  Shen & Atluri — full elastic-plastic cycle: loading + elastic unloading
                  → gives POST-IMPACT RESIDUAL stresses.

FEA comparison context:
  True FEA residual stresses match Shen & Atluri much more closely because
  FEA also applies the full loading–unloading cycle. Sherafatnia predicts
  only the transient stress state during impact, not the permanent state.

Usage:
    python compare_sherafatnia.py
    python compare_sherafatnia.py --V 40 --D 0.0008 --sy 800e6
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- path setup ----
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE / "src" / "peen-ml"
for _p in [str(_SRC), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from impact_sim import ShotPeenParams, compute_contact_params, compute_stress_field


# ---------------------------------------------------------------------------
# Sherafatnia / Poozesh formula (from Sherafatnia.py)
# ---------------------------------------------------------------------------

def sherafatnia_stress_profile(
    E_s: float, nu_s: float,
    E_b: float, nu_b: float,
    sy: float,
    D: float, V: float,
    rho_s: float = 7800.0,
    k: float = 0.8,
    phi: float = math.pi / 2,
    n_points: int = 3000,
) -> dict:
    """Compute Sherafatnia stress profile for one impact.

    Returns dict with:
      Z         : (n,) depth from surface (m)
      sigma_x   : (n,) lateral stress during loading (Pa)
      sigma_z   : (n,) normal stress during loading (Pa)
      sigma_eq  : (n,) equivalent (von Mises) stress (Pa)
      a_e       : elastic contact radius (m)
      p0        : peak contact pressure (Pa)
    """
    # Equivalent modulus
    E_eq = 1.0 / ((1 - nu_s**2) / E_s + (1 - nu_b**2) / E_b)
    R = D / 2.0
    V_n = V * math.sin(phi)

    # Sherafatnia peak pressure (energy-based)
    p0 = (1.0 / math.pi) * (40.0 * math.pi * rho_s * V_n**2 * k * E_eq**4) ** (1.0 / 5.0)

    # Elastic contact radius (from Hertz: a_e = π p0 R / (2 E_eq))
    a_e = (math.pi * p0 * R) / (2.0 * E_eq)

    Z = np.linspace(0, 3 * a_e, n_points)
    Z_bar = Z / a_e

    # Hertzian stress coefficients
    A = 1.0 / (1.0 + Z_bar**2)
    B = 1.0 - Z_bar * np.arctan2(1.0, Z_bar)

    sigma_x = -p0 * (-A / 2.0 + (1.0 + nu_b) * B)
    sigma_z = -p0 * A
    sigma_eq = sigma_x - sigma_z   # Sherafatnia equivalent (note: not von Mises)

    return {
        "Z":        Z,
        "sigma_x":  sigma_x,
        "sigma_z":  sigma_z,
        "sigma_eq": sigma_eq,
        "a_e":      a_e,
        "p0":       p0,
    }


# ---------------------------------------------------------------------------
# Shen & Atluri residual stress profile (from impact_sim.py)
# ---------------------------------------------------------------------------

def shen_atluri_residual_profile(
    E_s: float, nu_s: float,
    E_b: float, nu_b: float,
    sy: float, c: float,
    D: float, V: float,
    rho_s: float = 7800.0,
    phi: float = math.pi / 2,
    n_depth: int = 3000,
) -> dict:
    """Compute Shen & Atluri post-impact RESIDUAL stress profile."""
    p = ShotPeenParams(
        E_s=E_s, nu_s=nu_s,
        E_b=E_b, nu_b=nu_b,
        sigma_yield=sy, c=c,
        D=D, V=V, rho_s=rho_s, phi=phi,
        n_depth=n_depth,
    )
    contact = compute_contact_params(p)
    sf      = compute_stress_field(contact, p)

    return {
        "Z":           sf["Z"],
        "sR":          sf["sR"],           # post-impact biaxial residual stress
        "sigma_xe":    sf["sigma_xe"],     # elastic loading stress (for comparison)
        "sigma_ze":    sf["sigma_ze"],
        "sigma_eqe":   sf["sigma_eqe"],
        "ae":          contact["ae"],
        "p0":          contact["p0"],
    }


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(
    E_s: float = 210e9, nu_s: float = 0.28,
    E_b: float = 205e9, nu_b: float = 0.29,
    sy:  float = 1511e6, c: float = 3.0e9,
    D:   float = 1e-3, V: float = 50.0,
    rho_s: float = 7800.0,
    save_path: str = "sherafatnia_vs_shen_atluri.png",
):
    print("=" * 60)
    print("Sherafatnia vs Shen & Atluri comparison")
    print(f"  E_s={E_s/1e9:.0f} GPa  E_b={E_b/1e9:.0f} GPa  sy={sy/1e6:.0f} MPa")
    print(f"  D={D*1e3:.1f} mm  V={V:.0f} m/s  rho_s={rho_s:.0f} kg/m³")
    print("=" * 60)

    sh  = sherafatnia_stress_profile(E_s, nu_s, E_b, nu_b, sy, D, V, rho_s)
    sa  = shen_atluri_residual_profile(E_s, nu_s, E_b, nu_b, sy, c, D, V, rho_s)

    print(f"\nSherafatnia:   p0={sh['p0']/1e6:.1f} MPa  a_e={sh['a_e']*1e6:.1f} µm")
    print(f"Shen & Atluri: p0={sa['p0']/1e6:.1f} MPa  ae={sa['ae']*1e6:.1f} µm")

    print("\nNote: Sherafatnia computes LOADING stresses (transient during impact).")
    print("      Shen & Atluri computes POST-IMPACT RESIDUAL stresses.")
    print("      FEA compares best with Shen & Atluri residual results.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: loading (elastic) stress comparison ----
    ax = axes[0]
    Z_sh_mm = sh["Z"] * 1e3
    Z_sa_mm = sa["Z"] * 1e3
    ax.plot(Z_sh_mm, sh["sigma_x"] / 1e6,
            label=r"Sherafatnia $\sigma_x$ (loading)", color="tab:blue", lw=1.5)
    ax.plot(Z_sh_mm, sh["sigma_z"] / 1e6,
            label=r"Sherafatnia $\sigma_z$ (loading)", color="tab:blue",
            lw=1.5, linestyle="--")
    ax.plot(Z_sa_mm, sa["sigma_xe"] / 1e6,
            label=r"Shen & Atluri $\sigma_x^e$ (loading)", color="tab:orange", lw=1.5)
    ax.plot(Z_sa_mm, sa["sigma_ze"] / 1e6,
            label=r"Shen & Atluri $\sigma_z^e$ (loading)", color="tab:orange",
            lw=1.5, linestyle="--")
    ax.axhline(-sy / 1e6, color="gray", lw=0.8, linestyle=":", label=f"$σ_y$={sy/1e6:.0f} MPa")
    ax.set_xlabel("Depth from surface (mm)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("Elastic loading stress — Sherafatnia vs Shen & Atluri")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # ---- Right: residual stress ----
    ax = axes[1]
    ax.plot(Z_sa_mm, sa["sR"] / 1e6,
            label=r"Shen & Atluri $\sigma_R$ (post-impact residual)", color="tab:red", lw=2)
    ax.axhline(0, color="k", lw=0.5, linestyle="--")
    ax.axhline(-sy / 1e6, color="gray", lw=0.8, linestyle=":",
               label=f"$σ_y$={sy/1e6:.0f} MPa")
    ax.fill_between(Z_sa_mm, sa["sR"] / 1e6, 0,
                    where=sa["sR"] < 0, alpha=0.15, color="tab:red",
                    label="Compressive zone")
    ax.set_xlabel("Depth from surface (mm)")
    ax.set_ylabel("Residual Stress (MPa)")
    ax.set_title("Post-impact residual stress — Shen & Atluri\n"
                 "(Sherafatnia has no unloading → no residual stress estimate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig.suptitle(
        f"Sherafatnia formula vs Shen & Atluri (2006) | "
        f"D={D*1e3:.1f}mm  V={V:.0f}m/s  σ_y={sy/1e6:.0f}MPa",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--V",    type=float, default=50.0,    help="Impact velocity m/s")
    parser.add_argument("--D",    type=float, default=1e-3,    help="Shot diameter m")
    parser.add_argument("--sy",   type=float, default=1511e6,  help="Yield stress Pa")
    parser.add_argument("--Eb",   type=float, default=205e9,   help="Target Young's modulus Pa")
    parser.add_argument("--Es",   type=float, default=210e9,   help="Shot Young's modulus Pa")
    parser.add_argument("--rho",  type=float, default=7800.0,  help="Shot density kg/m³")
    parser.add_argument("--out",  default="sherafatnia_vs_shen_atluri.png")
    args = parser.parse_args()

    compare(
        E_s=args.Es, nu_s=0.28,
        E_b=args.Eb, nu_b=0.29,
        sy=args.sy, c=3.0e9,
        D=args.D, V=args.V, rho_s=args.rho,
        save_path=args.out,
    )


if __name__ == "__main__":
    main()
