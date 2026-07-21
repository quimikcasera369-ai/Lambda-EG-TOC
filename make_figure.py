#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figure.py  --  Letter v3, Figure 1.  Corrected 2026-07-21.

FIXES vs the v2 figure:
  F7  Omega_m was 0.315 while eq.(10) of the letter uses 0.3. Unified to 0.3.
  F8  the caption claimed the three survivors "become distinguishable at z>~2",
      but the +-0.010 dex band drawn in the figure ENVELOPES the tanh curve at
      every redshift. The separations are now annotated with their actual
      values in units of the band, and the text says what the figure shows.
  F9  new left panel: the excluded (c2 != 0) interpolants, whose flow collapses
      to Phi = 0 at finite lambda. That is the actual exclusion argument and it
      was not plotted anywhere.

Outputs sigma_Phi_discriminator.{pdf,png}
"""
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({
    "font.family": "serif", "font.size": 10.5,
    "axes.labelsize": 11.5, "axes.titlesize": 11,
    "legend.fontsize": 8.6, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.8, "figure.dpi": 150,
})

OMEGA_M, SIGMA0, BAND = 0.3, 0.07, 0.010   # SIGMA0 = Psi0, logarithmic (dex)
PHI0 = 10**SIGMA0 - 1                      # = 0.175, eq.(12): the flow lives in Phi
E = lambda z: np.sqrt(OMEGA_M*(1+z)**3 + 1 - OMEGA_M)
z_of_E = lambda e: ((e**2 - (1-OMEGA_M))/OMEGA_M)**(1/3.) - 1

x = sp.Symbol('x', positive=True)
EXCLUDED = [(r"Simple  $x/(1+x)$",              x/(1+x),                   "#2a6cad", "--"),
            (r"TeVeS toy",                     2*x/(1+2*x+sp.sqrt(1+4*x)),"#a83232", "-."),
            (r"Exponential  $1-e^{-x}$",       1-sp.exp(-x),              "#3f7d3f", ":")]
SURVIVORS = [(r"Standard  $\mu=x/\sqrt{1+x^{2}}$",              -1.0,   "#1f2733", "-"),
             (r"$\hat\mu$ entropic $\sqrt{1+(2x)^{-2}}-(2x)^{-1}$", -2.0, "#2a6cad", "--"),
             (r"$\tanh$  $\mu=\tanh(x)$",                      -2/3,   "#a83232", ":")]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4))

# ---------------------------------------------------------------- left panel
# z_* are 1.90 (simple), 1.03 (TeVeS), 4.58 (exp): the first two are close in x,
# so they are separated vertically instead of horizontally.
LBL_OFFSET = {0: (0, 21), 1: (0, 7), 2: (0, 7)}
for i_ex, (lbl, mu, col, ls) in enumerate(EXCLUDED):
    mp = sp.diff(mu, x)
    bf = sp.lambdify(x, -x*(mu - x*mp)/(mu*(mu + x*mp)), "numpy")

    def rhs(L, y):
        P = max(y[0], 0.0)
        return [float(bf(np.sqrt(P*(2+P)))) if P > 1e-14 else 0.0]

    ev = lambda L, y: y[0] - 1e-9
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(rhs, [0, 6], [PHI0], rtol=1e-11, atol=1e-15,
                    events=ev, dense_output=True)
    lstar = sol.t_events[0][0]
    zs = np.linspace(0, z_of_E(np.exp(lstar)), 400)
    axL.plot(zs, [max(sol.sol(np.log(E(zz)))[0], 0) for zz in zs],
             ls=ls, color=col, label=lbl)
    axL.plot([z_of_E(np.exp(lstar))], [0], 'o', color=col, ms=5.5, zorder=5)
    axL.annotate(rf"$z_*={z_of_E(np.exp(lstar)):.2f}$",
                 xy=(z_of_E(np.exp(lstar)), 0), xytext=LBL_OFFSET[i_ex],
                 textcoords="offset points", ha="center", fontsize=8.4, color=col)

mu_std = x/sp.sqrt(1+x**2)
mp = sp.diff(mu_std, x)
bfs = sp.lambdify(x, -x*(mu_std - x*mp)/(mu_std*(mu_std + x*mp)), "numpy")
sol = solve_ivp(lambda L, y: [float(bfs(np.sqrt(max(y[0], 0)*(2+max(y[0], 0)))))],
                [0, 6], [PHI0], rtol=1e-11, atol=1e-15, dense_output=True)
zg = np.linspace(0, 5, 400)
axL.plot(zg, [sol.sol(np.log(E(zz)))[0] for zg_, zz in [(0, z) for z in zg]],
         color="#1f2733", lw=2.2, label=r"Standard ($c_2=0$): never reaches $0$")
axL.set_xlabel(r"redshift  $z$"); axL.set_ylabel(r"BTFR departure  $\Phi(z)$")
axL.set_title(r"Excluded interpolants: $c_2\neq0$ $\Rightarrow$ collapse at finite $\lambda$")
axL.set_xlim(0, 5); axL.set_ylim(-0.010, 0.19)
axL.axhline(0, color="k", lw=0.6, alpha=0.5)
axL.grid(alpha=0.3, lw=0.5); axL.legend(loc="upper right", framealpha=0.95, edgecolor="0.7")

# ---------------------------------------------------------------- right panel
zg = np.linspace(0, 5, 500)
curves = {}
for lbl, bp, col, ls in SURVIVORS:
    s = SIGMA0*E(zg)**bp
    curves[lbl] = s
    axR.plot(zg, s, ls=ls, color=col, lw=2.0 if ls != "-" else 2.2, label=lbl)

std = curves[SURVIVORS[0][0]]
axR.fill_between(zg, std-BAND, std+BAND, color="#1f2733", alpha=0.13,
                 label=rf"forecast band ($\pm{BAND:.3f}$ dex per $z$-bin)")
axR.errorbar([0], [SIGMA0], yerr=[0.007], fmt='o', color='#1f2733', ms=5.5,
             capsize=3.5, elinewidth=1.2, label=r"SPARC anchor at $z=0$")

# annotate the ACTUAL maximum separations
for bp, col, off, ha in [(-2.0, "#2a6cad", (-8, -4), "right"),
                         (-2/3, "#a83232", (9, 2), "left")]:
    d = np.abs(SIGMA0*E(zg)**-1 - SIGMA0*E(zg)**bp)
    i = d.argmax()
    axR.annotate("", xy=(zg[i], SIGMA0*E(zg[i])**-1), xytext=(zg[i], SIGMA0*E(zg[i])**bp),
                 arrowprops=dict(arrowstyle="<->", color=col, lw=1.0, alpha=0.85))
    axR.annotate(rf"max {d[i]/BAND:.2f}$\,\times$band" "\n" rf"at $z={zg[i]:.1f}$",
                 xy=(zg[i], np.sqrt(SIGMA0*E(zg[i])**-1 * SIGMA0*E(zg[i])**bp)),
                 xytext=off, textcoords="offset points", fontsize=7.8, color=col, ha=ha)

axR.set_yscale("log"); axR.set_xlim(0, 5); axR.set_ylim(5e-4, 0.12)
axR.set_xlabel(r"redshift  $z$")
axR.set_ylabel(r"intrinsic BTFR scatter  $\sigma_\Phi(z)$  [dex]")
axR.set_title(r"Surviving interpolants ($c_2=0$): $\sigma_\Phi\propto(H/H_0)^{\beta'(0)}$")
axR.grid(True, which="major", alpha=0.33, lw=0.5)
axR.grid(True, which="minor", alpha=0.14, lw=0.4)
axR.legend(loc="lower left", framealpha=0.96, edgecolor="0.7", fancybox=False)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(f"sigma_Phi_discriminator.{ext}", bbox_inches="tight",
                **({"dpi": 200} if ext == "png" else {}))
print("Saved sigma_Phi_discriminator.{pdf,png}")

# ---------------------------------------------------------------- numbers
print(f"\nOmega_m = {OMEGA_M}, sigma_Phi(0) = {SIGMA0} dex, band = +-{BAND} dex\n")
print(f"{'z':>5s} {'H/H0':>7s} {'std':>9s} {'mu-hat':>9s} {'tanh':>9s}"
      f" {'|std-muhat|/band':>17s} {'|std-tanh|/band':>16s}")
for zv in [0, 0.5, 1, 1.5, 2, 3, 4, 5]:
    e = E(zv); a, b, c = SIGMA0*e**-1, SIGMA0*e**-2, SIGMA0*e**(-2/3)
    print(f"{zv:5.1f} {e:7.3f} {a:9.4f} {b:9.4f} {c:9.4f}"
          f" {abs(a-b)/BAND:17.2f} {abs(a-c)/BAND:16.2f}")
zf = np.linspace(0.01, 10, 20000)
for lab, bp in [("std vs mu-hat", -2.0), ("std vs tanh  ", -2/3)]:
    d = np.abs(SIGMA0*E(zf)**-1 - SIGMA0*E(zf)**bp); i = d.argmax()
    print(f"\n  {lab}: max separation {d[i]:.4f} dex = {d[i]/BAND:.2f} band widths at z = {zf[i]:.2f}")
print("\n  Separation is NOT monotonic in z: all survivors tend to zero scatter,")
print("  so a fixed absolute error budget resolves them best at z ~ 1-2.")
