"""
Extract model parameters from infinite dilution activity coefficients (γ∞)
using the EXACT model equations shown on the slide.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Two-constant Margules (alternative form from slide):
   ln γ1 = [a12 + 2(a21 - a12) x1] x2²
   ln γ2 = [a21 + 2(a12 - a21) x2] x1²

   At x1 → 0  (γ1∞):  ln γ1∞ = a12 · 1²  →  a12 = ln γ1∞
   At x2 → 0  (γ2∞):  ln γ2∞ = a21 · 1²  →  a21 = ln γ2∞

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. Van Laar (slide form):
   ln γ1 = α / [1 + (α/β)(x1/x2)]²
   ln γ2 = β / [1 + (β/α)(x2/x1)]²

   At x1 → 0  (γ1∞):  denominator → 1  →  ln γ1∞ = α  →  α = ln γ1∞
   At x2 → 0  (γ2∞):  denominator → 1  →  ln γ2∞ = β  →  β = ln γ2∞

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. Wilson (slide form):
   ln γ1 = -ln(x1 + x2·Λ12) + x2 [ Λ12/(x1 + x2·Λ12) - Λ21/(x1·Λ21 + x2) ]
   ln γ2 = -ln(x2 + x1·Λ21) - x1 [ Λ12/(x1 + x2·Λ12) - Λ21/(x1·Λ21 + x2) ]

   At x1 → 0 (x2 → 1):
     ln γ1∞ = -ln(Λ12) + [1 - Λ21/1·... ]
     Let's derive carefully:
       x1→0, x2→1:
       term1 = -ln(0 + 1·Λ12) = -ln(Λ12)
       bracket = Λ12/(0 + 1·Λ12) - Λ21/(0·Λ21 + 1)
               = 1 - Λ21
       → ln γ1∞ = -ln(Λ12) + 1·(1 - Λ21) = -ln(Λ12) + 1 - Λ21

   At x2 → 0 (x1 → 1):
     ln γ2∞:
       term1 = -ln(0 + 1·Λ21) = -ln(Λ21)
       bracket at x1=1, x2=0:
              = Λ12/(1 + 0·Λ12) - Λ21/(1·Λ21 + 0)
              = Λ12 - 1
       → ln γ2∞ = -ln(Λ21) - 1·(Λ12 - 1) = -ln(Λ21) + 1 - Λ12

   So the Wilson infinite dilution equations are:
     ln γ1∞ = -ln(Λ12) + 1 - Λ21   ... (same as before)
     ln γ2∞ = -ln(Λ21) + 1 - Λ12   ... (same as before)
   → Wilson equations confirmed unchanged. ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from scipy.optimize import fsolve

# ── Experimental γ∞ data ─────────────────────────────────────────────────────
systems = {
    "EtOH – Methyl Acetate": {
        "comp1": "Ethanol",
        "comp2": "Methyl Acetate",
        "g1": 3.108,
        "g2": 3.096,
    },
    "Methyl Acetate – Ethyl Acetate": {
        "comp1": "Methyl Acetate",
        "comp2": "Ethyl Acetate",
        "g1": 1.196,
        "g2": 1.003,
    },
    "Ethyl Acetate – EtOH": {
        "comp1": "Ethyl Acetate",
        "comp2": "Ethanol",
        "g1": 3.896,
        "g2": 2.250,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Model functions (for verification across full composition range)
# ─────────────────────────────────────────────────────────────────────────────

def margules(x1, a12, a21):
    """Returns (ln γ1, ln γ2) using 2-constant Margules (slide form)."""
    x2 = 1 - x1
    lng1 = (a12 + 2*(a21 - a12)*x1) * x2**2
    lng2 = (a21 + 2*(a12 - a21)*x2) * x1**2
    return lng1, lng2


def vanlaar(x1, alpha, beta):
    """Returns (ln γ1, ln γ2) using van Laar (slide form)."""
    x2 = 1 - x1
    eps = 1e-15  # guard against x1=0 or x2=0
    denom1 = (1 + (alpha/beta) * (x1 / (x2 + eps)))**2
    denom2 = (1 + (beta/alpha) * (x2 / (x1 + eps)))**2
    lng1 = alpha / denom1
    lng2 = beta  / denom2
    return lng1, lng2


def wilson(x1, L12, L21):
    """Returns (ln γ1, ln γ2) using Wilson (slide form)."""
    x2 = 1 - x1
    eps = 1e-15
    A = x1 + x2*L12 + eps
    B = x1*L21 + x2 + eps
    bracket = L12/A - L21/B
    lng1 = -np.log(A) + x2 * bracket
    lng2 = -np.log(B) - x1 * bracket
    return lng1, lng2


# ─────────────────────────────────────────────────────────────────────────────
# Parameter extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_margules(g1, g2):
    """a12 = ln γ1∞,  a21 = ln γ2∞"""
    return np.log(g1), np.log(g2)


def extract_vanlaar(g1, g2):
    """α = ln γ1∞,  β = ln γ2∞"""
    return np.log(g1), np.log(g2)


def extract_wilson(g1, g2):
    """
    Solve simultaneously:
      -ln(Λ12) + 1 - Λ21 = ln γ1∞
      -ln(Λ21) + 1 - Λ12 = ln γ2∞
    Requires Λ12 > 0 and Λ21 > 0.
    """
    lnG1 = np.log(g1)
    lnG2 = np.log(g2)

    def eqs(p):
        L12, L21 = p
        return [
            -np.log(L12) + 1 - L21 - lnG1,
            -np.log(L21) + 1 - L12 - lnG2,
        ]

    best = None
    for L12_0, L21_0 in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3),
                          (0.1, 0.9), (0.9, 0.1), (0.2, 0.2),
                          (0.4, 1.5), (1.5, 0.4)]:
        try:
            sol = fsolve(eqs, [L12_0, L21_0], full_output=True)
            x, _, ier, _ = sol
            L12, L21 = x
            res = np.max(np.abs(eqs(x)))
            if ier == 1 and L12 > 1e-9 and L21 > 1e-9 and res < 1e-10:
                if best is None or res < best[2]:
                    best = (L12, L21, res)
        except Exception:
            pass

    if best is None:
        return None, None
    return best[0], best[1]


# ─────────────────────────────────────────────────────────────────────────────
# Verification helper: evaluate model at x1=ε and x1=1-ε, compare to γ∞
# ─────────────────────────────────────────────────────────────────────────────

def verify_model(model_fn, params, g1, g2, label):
    eps = 1e-9
    lng1_check, _ = model_fn(eps,    *params)   # x1→0
    _, lng2_check = model_fn(1-eps,  *params)   # x2→0
    err1 = abs(lng1_check - np.log(g1))
    err2 = abs(lng2_check - np.log(g2))
    ok = "✓" if max(err1, err2) < 1e-7 else "✗"
    return ok, err1, err2


# ─────────────────────────────────────────────────────────────────────────────
# Main output
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 74
SEP2 = "─" * 74

print(SEP)
print("  PARAMETER EXTRACTION  |  56 °C  |  EXPERIMENTAL γ∞ VALUES")
print("  Model equations: exactly as shown on slide")
print(SEP)

results = {}

for name, d in systems.items():
    c1, c2 = d["comp1"], d["comp2"]
    g1, g2 = d["g1"], d["g2"]

    a12, a21 = extract_margules(g1, g2)
    alpha, beta = extract_vanlaar(g1, g2)
    L12, L21 = extract_wilson(g1, g2)

    results[name] = dict(a12=a12, a21=a21, alpha=alpha, beta=beta, L12=L12, L21=L21)

    ok_m, e1m, e2m = verify_model(margules, (a12, a21),     g1, g2, "Margules")
    ok_v, e1v, e2v = verify_model(vanlaar,  (alpha, beta),  g1, g2, "van Laar")
    if L12 is not None:
        ok_w, e1w, e2w = verify_model(wilson, (L12, L21),   g1, g2, "Wilson")
    else:
        ok_w, e1w, e2w = "✗", float('nan'), float('nan')

    print(f"\n{SEP2}")
    print(f"  {name}")
    print(f"  (1) {c1}  |  (2) {c2}")
    print(f"  γ∞₁ = {g1}   →   ln γ∞₁ = {np.log(g1):.6f}")
    print(f"  γ∞₂ = {g2}   →   ln γ∞₂ = {np.log(g2):.6f}")
    print(f"{SEP2}")

    print(f"\n  ① 2-Constant Margules  (slide: alt form)  {ok_m}")
    print(f"     a₁₂ = ln γ∞₁ = {a12:.6f}")
    print(f"     a₂₁ = ln γ∞₂ = {a21:.6f}")
    print(f"     Residuals: |err₁|={e1m:.2e}  |err₂|={e2m:.2e}")

    print(f"\n  ② Van Laar  {ok_v}")
    print(f"     α   = ln γ∞₁ = {alpha:.6f}")
    print(f"     β   = ln γ∞₂ = {beta:.6f}")
    print(f"     Residuals: |err₁|={e1v:.2e}  |err₂|={e2v:.2e}")

    print(f"\n  ③ Wilson  (solved numerically)  {ok_w}")
    if L12 is not None:
        print(f"     Λ₁₂ = {L12:.6f}")
        print(f"     Λ₂₁ = {L21:.6f}")
        print(f"     Verification:  -ln(Λ₁₂)+1-Λ₂₁ = {-np.log(L12)+1-L21:.6f}  (ln γ∞₁ = {np.log(g1):.6f})")
        print(f"     Verification:  -ln(Λ₂₁)+1-Λ₁₂ = {-np.log(L21)+1-L12:.6f}  (ln γ∞₂ = {np.log(g2):.6f})")
        print(f"     Residuals: |err₁|={e1w:.2e}  |err₂|={e2w:.2e}")
    else:
        print(f"     ⚠  No physical solution (Λ > 0) found.")

print(f"\n{SEP}")
print("  SUMMARY TABLE")
print(SEP)
print(f"  {'System':<38} {'a12 / α':<10} {'a21 / β':<10}  {'Λ12':<10} {'Λ21':<10}")
print(f"  {'':─<38} {'Margules/vL':─<21}  {'Wilson':─<21}")
for name, r in results.items():
    short = name[:37]
    L12s = f"{r['L12']:.5f}" if r['L12'] is not None else "N/A"
    L21s = f"{r['L21']:.5f}" if r['L21'] is not None else "N/A"
    print(f"  {short:<38} {r['a12']:<10.5f} {r['a21']:<10.5f}  {L12s:<10} {L21s:<10}")

print(SEP)
