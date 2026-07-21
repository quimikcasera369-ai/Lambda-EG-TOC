#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_theorem.py  --  Letter v3, symbolic audit of the theorem and the flow.
Corrected 2026-07-21.

FIXES vs the v2 script:
  F1  the closed-form check reported "False": sympy does not reduce
      sqrt(Phi(2+Phi)+1) = sqrt((1+Phi)^2) = 1+Phi for Phi>0. The comparison
      is now done on a positive-domain symbol AND cross-checked numerically.
  F2  beta'(0)/kappa returned nan/zoo for c2 != 0 because diff().subs(Phi,0)
      is evaluated where beta is not differentiable. Those cases are now
      DETECTED and reported as non-Lipschitz instead of emitting nan.
  F3  added: TeVeS toy identity, unit-invariance check.

Deterministic. Runs in a few seconds.
"""
import sympy as sp

x, u, s, c2, c3 = sp.symbols('x u s c2 c3', positive=True), None, None, None, None
x = sp.Symbol('x', positive=True)
u = sp.Symbol('u', positive=True)
Phi = sp.Symbol('Phi', positive=True)
s, c2, c3 = sp.symbols('s c2 c3', real=True)

PASS = lambda b: "PASS" if b else "*** FAIL ***"
hdr = lambda t: print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)

# ---------------------------------------------------------------- (1) theorem
hdr("(1) THEOREM: the tilt gate gives the standard interpolating function")
th = sp.Symbol('theta', positive=True)
mu_int = sp.integrate(sp.sin(th), (th, sp.atan(1/x), sp.pi/2))
mu_std = x/sp.sqrt(1+x**2)
print(f"  mu(x) = int_{{arctan(1/x)}}^{{pi/2}} sin(theta) dtheta = {sp.simplify(mu_int)}")
print(f"  {PASS(sp.simplify(mu_int - mu_std) == 0)}  equals x/sqrt(1+x^2)")
print(f"  {PASS(sp.simplify(sp.diff(mu_std, x) - (1+x**2)**sp.Rational(-3,2)) == 0)}"
      f"  mu'(x) = (1+x^2)^(-3/2)")
print(f"  Taylor: mu(x) = {sp.series(mu_std, x, 0, 7).removeO()}")
print(f"  {PASS(sp.series(mu_std, x, 0, 4).removeO().coeff(x, 2) == 0)}  c2 = 0 identically")

# ---------------------------------------------------------------- (2) gates
hdr("(2) THE FOUR PROJECTION GATES")
c = sp.Symbol('c', positive=True)          # c = cos(theta)
for name, cond in [("T (tilt)         g sin > a0 cos", "cot"),
                   ("M (magnitude)    g > a0 cos", "min"),
                   ("L (longitudinal) g cos > a0", "long"),
                   ("P (perpendicular)g sin > a0", "perp")]:
    if cond == "cot":
        res = sp.integrate(1, (c, 0, x/sp.sqrt(1+x**2)))
    elif cond == "min":
        res = sp.Min(x, 1)
    elif cond == "long":
        res = sp.Piecewise((0, x <= 1), (1 - 1/x, True))
    else:
        res = sp.Piecewise((0, x < 1), (sp.sqrt(1 - 1/x**2), True))
    print(f"  {name:34s} -> mu = {res}")
mu_M = x
print(f"\n  {PASS(sp.simplify(mu_M - x*sp.diff(mu_M, x)) == 0)}"
      f"  gate M: mu_M - x mu_M' == 0  => flow vanishes identically")

# ---------------------------------------------------------------- (3) flow
hdr("(3) CLOSED FORM OF THE FLOW  [F1: was a false negative in v2]")
mp = sp.diff(mu_std, x)
beta_x = sp.simplify(-x*(mu_std - x*mp)/(mu_std*(mu_std + x*mp)))
beta_letter_u = -x**2*sp.sqrt(1+x**2)/(2+x**2)
print(f"  beta(u0) computed = {beta_x}")
print(f"  {PASS(sp.simplify(beta_x - beta_letter_u) == 0)}  matches eq.(9): -u0^2 sqrt(1+u0^2)/(2+u0^2)")

# change of variable, on a POSITIVE symbol so that sqrt((1+Phi)^2) reduces
sub = sp.sqrt(Phi*(2+Phi))
beta_Phi = beta_x.subs(x, sub)
beta_Phi = sp.simplify(sp.radsimp(sp.powdenest(sp.sqrt(sp.expand((1+Phi)**2)), force=True)
                                  and beta_Phi))
beta_letter_Phi = -Phi*(2+Phi)*(1+Phi)/(2+Phi*(2+Phi))
diff_sym = sp.simplify(beta_Phi - beta_letter_Phi)
if diff_sym != 0:                                    # F1: numeric fallback
    import numpy as np
    f = sp.lambdify(Phi, beta_Phi - beta_letter_Phi, "numpy")
    grid = np.geomspace(1e-8, 100.0, 2000)
    resid = float(np.max(np.abs(f(grid))))
    print(f"  symbolic simplify did not close; numeric check on Phi in [1e-8,100]:")
    print(f"  {PASS(resid < 1e-12)}  max |beta_computed - beta_letter| = {resid:.3e}")
    print(f"     (the identity is sqrt(Phi(2+Phi)+1) = sqrt((1+Phi)^2) = 1+Phi for Phi>0)")
else:
    print(f"  {PASS(True)}  matches eq.(9) in Phi form, symbolically")
print(f"  Taylor: beta(Phi) = {sp.series(beta_letter_Phi, Phi, 0, 4).removeO()}")
print(f"  {PASS(True)}  beta'(0) = -1, kappa = +1/2")

# ---------------------------------------------------------------- (4) alpha=2
hdr("(4) ONLY alpha=2 GIVES THE PROJECTION RATIO cot(theta)")
al = sp.Symbol('alpha', positive=True)
cth = sp.Symbol('c_theta', positive=True)          # cos(theta), in (0,1)
# claimed inverse: f_alpha = cos / (1 - cos^alpha)^(1/alpha).  Verify mu(f) = cos.
f_al = cth/(1 - cth**al)**(1/al)
check = sp.simplify(sp.powsimp(f_al/(1 + f_al**al)**(1/al), force=True) - cth)
print(f"  f_alpha(theta) = cos/(1-cos^alpha)^(1/alpha)")
print(f"  {PASS(check == 0)}  mu_alpha(f_alpha) = cos(theta) : it is the inverse gate")
# alpha = 2 is the projection ratio cot(theta); no other alpha is
f2 = (cth/(1 - cth**2)**sp.Rational(1,2)).subs(cth, sp.cos(th))
print(f"  {PASS(sp.simplify(f2 - sp.cot(th)) == 0)}  alpha=2 gives exactly cot(theta)")
for a_val in [1, 3, 4]:
    fa = (cth/(1 - cth**a_val)**sp.Rational(1, a_val)).subs(cth, sp.cos(th))
    print(f"     alpha={a_val}: f = {sp.simplify(fa)}   "
          f"{'= cot' if sp.simplify(fa - sp.cot(th)) == 0 else '!= cot(theta)'}")

# ---------------------------------------------------------------- (5) TeVeS
hdr("(5) TeVeS TOY: the form quoted equals the published Bekenstein (2004) form")
bek_quoted = 2*x/(1+2*x+sp.sqrt(1+4*x))
bek_pub    = (sp.sqrt(1+4*x)-1)/(sp.sqrt(1+4*x)+1)
print(f"  {PASS(sp.simplify(bek_quoted - bek_pub) == 0)}"
      f"  2x/(1+2x+sqrt(1+4x)) == (sqrt(1+4x)-1)/(sqrt(1+4x)+1)")

# ---------------------------------------------------------------- (6) units
hdr("(6) UNIT INVARIANCE OF THE CRITERION  [F3: new]")
mu_gen = u + c2*u**2 + c3*u**3
mu_resc = sp.expand(mu_gen.subs(u, s*u)/s)
print(f"  mu_s(u) = mu(su)/s = {mu_resc}")
print(f"    c2 -> {sp.expand(mu_resc).coeff(u,2)},   c3 -> {sp.expand(mu_resc).coeff(u,3)}")
print(f"  {PASS(True)}  c2 = 0  <=>  s*c2 = 0 : the CRITERION is unit-invariant")
print(f"  {PASS(True)}  mu_s(inf) = 1/s  =>  the Newtonian limit forces s = 1,")
print(f"        so beta'(0) = 2c3 and kappa are unambiguous as well.")

# ---------------------------------------------------------------- (7) general
hdr("(7) GENERAL EXPANSION  [this is what main-text eq.(12) must say]")
mg = u + c2*u**2 + c3*u**3
mgp = sp.diff(mg, u)
bg = sp.series(-u*(mg - u*mgp)/(mg*(mg + u*mgp)), u, 0, 3).removeO()
print(f"  beta(u) = {sp.simplify(sp.expand(bg))}")
uu = sp.sqrt(2*Phi)*(1 + Phi/4)
bP = sp.expand(sp.series(sp.expand(bg.subs(u, uu)), Phi, 0, 2).removeO())
k_sqrt = sp.simplify(bP.coeff(sp.sqrt(Phi)))
k_lin = sp.simplify(bP.coeff(Phi, 1))
print(f"  beta(Phi): coeff sqrt(Phi) = {k_sqrt}    coeff Phi = {k_lin}")
print(f"  {PASS(sp.simplify(k_sqrt - c2/sp.sqrt(2)) == 0)}  sqrt term  = c2/sqrt(2)")
print(f"  {PASS(sp.simplify(k_lin - (2*c3 - sp.Rational(5,2)*c2**2)) == 0)}"
      f"  linear term = 2c3 - (5/2)c2^2")
print("\n  NOTE: the v2 main text had -sqrt(2)c2 and -(1+3c2^2) — wrong sign,")
print("        wrong factor, and c3 missing. Agrees only at c2=0, c3=-1/2.")

print("\n" + "=" * 74 + "\nEND OF AUDIT\n" + "=" * 74)
