"""Grid-density scaling of the calibration stability constant (thesis Appendix A.2.11).

Black--Scholes call prices C(K, T; sigma, r) with spot S0 are observed on uniform
n_K x n_T grids over Omega = [K_min, K_max] x [T_min, T_max]. For the parameter
vector phi = (sigma, r) (p = 2) the sensitivity (Jacobian) matrix of the
observation operator is assembled with quadrature weights w = |Omega| / N, so that
J_delta^T J_delta is a Riemann sum of the continuous Gram matrix

    G = int_Omega grad_phi C(K,T) grad_phi C(K,T)^T dK dT .

The price-only stability constant is C_stab = 1 / sigma_min(J_delta). The
Greek-augmented operator appends the parameter sensitivities of the strike
derivatives dC/dK and d2C/dK2, scaled by ell and ell^2 with a characteristic
strike length ell (so that every row carries price units):

    J_full = [ J_price ; ell * J_dK ; ell^2 * J_dKK ] .

The continuous limit C_star = 1 / sigma_min(J) is evaluated with a 200 x 200
Gauss--Legendre rule. Usage:

    python scripts/stability_scaling.py            # table + fits (LaTeX rows)
    python scripts/stability_scaling.py --ell 10   # other strike length scale
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import norm

S0, SIGMA, R = 100.0, 0.20, 0.05
K_MIN, K_MAX, T_MIN, T_MAX = 80.0, 120.0, 0.1, 1.0
AREA = (K_MAX - K_MIN) * (T_MAX - T_MIN)
GRIDS = [(2, 2), (3, 2), (3, 3), (4, 4), (5, 5), (7, 7), (10, 10), (15, 15), (20, 20)]


def bs_surfaces(K, T, sigma, r):
    """Call price and its first/second strike derivatives (no dividends)."""
    sq = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sq
    d2 = d1 - sq
    disc = np.exp(-r * T)
    price = S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
    dK = -disc * norm.cdf(d2)
    dKK = disc * norm.pdf(d2) / (K * sq)
    return price, dK, dKK


def sensitivities(K, T, h=1e-6):
    """Central-difference derivatives w.r.t. (sigma, r) of (price, dK, dKK)."""
    out = []
    for dsig, dr in ((h, 0.0), (0.0, h)):
        up = bs_surfaces(K, T, SIGMA + dsig, R + dr)
        dn = bs_surfaces(K, T, SIGMA - dsig, R - dr)
        out.append([(u - d) / (2 * h) for u, d in zip(up, dn)])
    # out[j][k]: derivative w.r.t. parameter j of surface k -> stack as (N, p) per k
    return [np.stack([out[0][k], out[1][k]], axis=-1) for k in range(3)]


def stability_constants(K, T, w, ell):
    """(C_price, C_full) for nodes K, T with quadrature weights w."""
    Jp, Jd, Jg = sensitivities(K, T)
    sw = np.sqrt(w)[:, None]
    J_price = sw * Jp
    J_full = np.vstack([J_price, ell * sw * Jd, ell**2 * sw * Jg])
    c_price = 1.0 / np.linalg.svd(J_price, compute_uv=False)[-1]
    c_full = 1.0 / np.linalg.svd(J_full, compute_uv=False)[-1]
    return c_price, c_full


def uniform_grid(n_k, n_t):
    K, T = np.meshgrid(np.linspace(K_MIN, K_MAX, n_k), np.linspace(T_MIN, T_MAX, n_t), indexing="ij")
    K, T = K.ravel(), T.ravel()
    return K, T, np.full(K.size, AREA / K.size)


def gauss_grid(n=200):
    x, wx = np.polynomial.legendre.leggauss(n)
    k = 0.5 * (K_MAX - K_MIN) * x + 0.5 * (K_MAX + K_MIN)
    t = 0.5 * (T_MAX - T_MIN) * x + 0.5 * (T_MAX + T_MIN)
    wk = 0.5 * (K_MAX - K_MIN) * wx
    wt = 0.5 * (T_MAX - T_MIN) * wx
    K, T = np.meshgrid(k, t, indexing="ij")
    W = np.outer(wk, wt)
    return K.ravel(), T.ravel(), W.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ell", type=float, default=20.0, help="strike length scale for Greek rows")
    args = ap.parse_args()

    c_star_p, c_star_f = stability_constants(*gauss_grid(), ell=args.ell)
    print(f"ell = {args.ell}; continuous limits: C*_price = {c_star_p:.4f}, C*_full = {c_star_f:.4f}")
    rows = []
    for n_k, n_t in GRIDS:
        c_p, c_f = stability_constants(*uniform_grid(n_k, n_t), ell=args.ell)
        delta = 2.0 / ((n_k - 1) + (n_t - 1))
        rows.append((n_k, n_t, n_k * n_t, delta, c_p, c_f))
    print(f"{'grid':>8} {'N':>4} {'d/d0':>6} {'C_price':>9} {'C_full':>9} {'C_p/C*p':>8} {'C_f/C*f':>8} {'C_f/C_p':>8}")
    for n_k, n_t, N, d, c_p, c_f in rows:
        print(f"{n_k:>3}x{n_t:<4} {N:>4} {d:6.3f} {c_p:9.4f} {c_f:9.4f} {c_p / c_star_p:8.3f} {c_f / c_star_f:8.3f} {c_f / c_p:8.3f}")

    # power-law fit of |C_delta - C_star| ~ A delta^beta over delta/delta0 in [0.1, 0.5]
    for label, idx, c_star in (("price", 4, c_star_p), ("full", 5, c_star_f)):
        sel = [r for r in rows if 0.1 - 1e-9 <= r[3] <= 0.5 + 1e-9]
        dev = np.array([abs(r[idx] - c_star) for r in sel])
        dl = np.array([r[3] for r in sel])
        signs = {np.sign(r[idx] - c_star) for r in sel}
        A = np.vstack([np.log(dl), np.ones_like(dl)]).T
        coef, res, *_ = np.linalg.lstsq(A, np.log(dev), rcond=None)
        resid = np.log(dev) - A @ coef
        se = np.sqrt(resid @ resid / max(len(dl) - 2, 1) / np.sum((np.log(dl) - np.log(dl).mean()) ** 2))
        print(f"fit {label:5s}: beta = {coef[0]:.2f} +/- {se:.2f} (points {len(dl)}, sign(C_delta - C*) = {sorted(signs)})")

    # asymptotic rate of the price-only constant on fine square grids
    print("\nlocal slopes of log|C_delta - C*| vs log(delta), price-only, square grids:")
    prev = None
    for n in (20, 40, 80, 160, 320):
        c_p, _ = stability_constants(*uniform_grid(n, n), ell=args.ell)
        d, dev = 1.0 / (n - 1), abs(c_star_p - c_p)
        if prev is not None:
            print(f"  n = {prev[2]:>3} -> {n:>3}: slope {np.log(prev[1] / dev) / np.log(prev[0] / d):.3f}")
        prev = (d, dev, n)

    print("\nLaTeX rows (constants in units of 1e-3):")
    for n_k, n_t, N, d, c_p, c_f in rows:
        print(f"${n_k} \\times {n_t}$ & {N} & {d:.2f} & {1e3 * c_p:.2f} & {1e3 * c_f:.2f} "
              f"& {c_p / c_star_p:.2f} & {c_f / c_star_f:.2f} & {c_f / c_p:.2f} \\\\")
    print(f"limits: C*_price = {1e3 * c_star_p:.2f}e-3, C*_full = {1e3 * c_star_f:.2f}e-3")


if __name__ == "__main__":
    main()
