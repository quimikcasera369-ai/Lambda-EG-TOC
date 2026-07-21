#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flow_coeffs.py  --  Letter v3.  Replaces flow_coeffs_refined.py (v2).
Corrected 2026-07-21.

FIXES vs flow_coeffs_refined.py:
  F4  the coefficient extractor used .coeff(Phi,1) on an UNEXPANDED expression,
      so for tanh it printed  "coef Phi = -3*Phi/5 - 2/3"  and  "kappa = 0"
      instead of beta'(0) = -2/3, kappa = +3/5. Now extracted from the u-series
      via u^2 = 2Phi + Phi^2, u^4 = 4Phi^2 (exact), and cross-checked.
  F5  the printed summary claimed tanh gives a NON-ANALYTIC flow, contradicting
      both its own computation (a1 = 0) and Table 1. Summary rewritten.
  F6  added: numerical integration of the EXACT flow (no series), which is what
      establishes finite-lambda collapse and the values of z_*.

Convention:  beta(Phi) = beta'(0) Phi - kappa Phi^2 + O(Phi^3)
"""
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp

x = sp.Symbol('x', positive=True)
OMEGA_M = 0.3          # consistent with eq.(10) of the letter
PHI0 = 10**0.07 - 1     # = 0.175; see eq.(12): Psi0=0.07 dex -> Phi0=10^Psi0-1

MUS = [("Standard  x/sqrt(1+x^2)", x/sp.sqrt(1+x**2)),
       ("mu-hat entropic",         sp.sqrt(1+1/(4*x**2)) - 1/(2*x)),
       ("tanh(x)",                 sp.tanh(x)),
       ("Simple    x/(1+x)",       x/(1+x)),
       ("TeVeS toy",               2*x/(1+2*x+sp.sqrt(1+4*x))),
       ("Exponential 1-e^-x",      1-sp.exp(-x))]


def coeffs(mu):
    """Return (c2, c3, beta'(0), kappa) or (c2, c3, None, None) if c2 != 0."""
    t = sp.series(mu, x, 0, 4).removeO()
    c2, c3 = sp.simplify(t.coeff(x, 2)), sp.simplify(t.coeff(x, 3))
    mp = sp.diff(mu, x)
    b = sp.expand(sp.series(-x*(mu - x*mp)/(mu*(mu + x*mp)), x, 0, 5).removeO())
    a1, a2, a4 = b.coeff(x, 1), b.coeff(x, 2), b.coeff(x, 4)
    if sp.simplify(a1) != 0:                      # non-Lipschitz branch
        return c2, c3, None, None
    # u^2 = 2Phi + Phi^2 and u^4 = 4Phi^2 + O(Phi^3)  =>  exact through Phi^2
    return c2, c3, sp.simplify(2*a2), sp.simplify(-(a2 + 4*a4))


def collapse(mu, Phi0=PHI0):
    """Integrate the EXACT flow dPhi/dlambda = beta(Phi). Return lambda* or None."""
    mp = sp.diff(mu, x)
    bf = sp.lambdify(x, -x*(mu - x*mp)/(mu*(mu + x*mp)), "numpy")

    def rhs(L, y):
        P = max(y[0], 0.0)
        return [float(bf(np.sqrt(P*(2+P)))) if P > 1e-14 else 0.0]

    ev = lambda L, y: y[0] - 1e-9
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(rhs, [0, 12], [Phi0], rtol=1e-11, atol=1e-15, events=ev)
    return (sol.t_events[0][0] if len(sol.t_events[0]) else None), sol


def z_of_lambda(lam):
    E = np.exp(lam)
    return ((E**2 - (1 - OMEGA_M))/OMEGA_M)**(1/3.) - 1


print("=" * 78)
print("FLOW COEFFICIENTS AND WELL-POSEDNESS   [beta = beta'(0) Phi - kappa Phi^2]")
print("=" * 78)
print(f"{'mu(x)':26s} {'c2':>6s} {'c3':>7s} {'beta_p(0)':>10s} {'kappa':>7s} "
      f"{'lambda*':>9s} {'z*':>7s}")
rows = []
for name, mu in MUS:
    c2, c3, bp, kap = coeffs(mu)
    lam, _ = collapse(mu)
    zs = z_of_lambda(lam) if lam is not None else None
    rows.append((name, c2, c3, bp, kap, lam, zs))
    print(f"{name:26s} {str(c2):>6s} {str(c3):>7s} "
          f"{(str(bp) if bp is not None else '---'):>10s} "
          f"{(str(kap) if kap is not None else '---'):>7s} "
          f"{(f'{lam:.3f}' if lam else '---'):>9s} "
          f"{(f'{zs:.2f}' if zs else '---'):>7s}")

print(f"\n  (Phi0 = {PHI0}, Omega_m = {OMEGA_M})")
print("  c2 = 0  -> analytic flow, exponential relaxation, Phi > 0 for all lambda.")
print("  c2 != 0 -> beta ~ (c2/sqrt2) sqrt(Phi): NOT Lipschitz at Phi=0;")
print("             Phi reaches zero at FINITE lambda* (Peano non-uniqueness),")
print("             so a population with a spread of Phi splits into an ATOM at")
print("             Phi=0 (already collapsed) plus a tail: BIMODALITY, not narrowing.")

print("\n" + "=" * 78)
print("SENSITIVITY OF z* TO THE ASSUMED Phi0")
print("=" * 78)
alt = [("Phi0 = 10^0.07-1 = 0.175 (adopted)", 10**0.07 - 1),
       ("Phi0 = 0.07 (dex read as fraction)", 0.07)]
print(f"{'mu(x)':26s} " + " ".join(f"{lab:>32s}" for lab, _ in alt))
for name, mu in MUS[3:]:
    out = []
    for _, P0 in alt:
        lam, _ = collapse(mu, P0)
        out.append(f"z* = {z_of_lambda(lam):.2f}" if lam else "---")
    print(f"{name:26s} " + " ".join(f"{o:>32s}" for o in out))
print("\n  Adopted: Psi0 = 0.07 dex is a LOGARITHMIC scatter; since the non-linear")
print("  flow does not commute with log <-> fraction, it must be converted:")
print("  Phi0 = 10^Psi0 - 1 = 0.175. This is not a new parameter, it is the same")
print("  quantity in the variable where the flow has a square root.")

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print("""
  PROJECTION GATES (letter, section 3):
    T (tilt)  -> mu = x/sqrt(1+x^2)  both MOND limits, analytic.  SURVIVES.
    M (magn.) -> mu = min(x,1)       flow vanishes identically.   FAILS.
    L (long.) -> mu = 0 for x<=1     no deep-MOND limit.          FAILS.
    P (perp.) -> mu = 0 for x<1      no deep-MOND limit.          FAILS.

  PUBLISHED INTERPOLATING FUNCTIONS, by the c2 = 0 criterion:
    PASS (analytic flow, linear attractor):
      Standard x/sqrt(1+x^2)      beta'(0) = -1,    kappa = +1/2
      mu-hat entropic             beta'(0) = -2,    kappa = -3
      tanh(x)                     beta'(0) = -2/3,  kappa = +3/5
    FAIL (non-Lipschitz flow, finite-lambda collapse):
      Simple x/(1+x), TeVeS toy, Exponential 1-e^-x

  What the failure means: NOT that these functions are bad fits to rotation
  curves. Only that they are incompatible with the additional hypothesis that
  the BTFR is a linear attractor of the cosmological flow.
""")
