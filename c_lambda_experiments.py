"""
c_lambda_experiments.py — data generation for the inflation-constant c_lambda study.

Standalone driver: imports the estimator/theory building blocks and writes .pkl
data for two kinds of figures (plotted separately in make_figures.py):

  1. LINE DATA  (compute_c_lambda_line_data)
     c_lambda vs lambda, theory + empirics (mean & s.e.m. over ntrials draws),
     for several k_l values at a fixed (gamma, rho). Feeds plots 1 & 2.

  2. HEATMAP DATA  (compute_c_lambda_heatmap_data)
     theory c_lambda over a (lambda, k_l) grid at a fixed (gamma, rho). Feeds plot 3.

Empirical c_lambda uses the validated closed form
    c_lambda = 1 + (lambda * beta * q1) / (1 + beta * q2),
    beta = k_l * D / n^2,  q1 = y^T K_x A^-1 y,  q2 = y^T K_x A^-1 K_x y,
computing one pseudo-inverse per (trial, lambda) and deriving every k_l from the
two shared quadratic forms q1, q2 (so extra k_l values are nearly free). This is
the same formula unit_tests.py checks against the library estimators.

Run `python c_lambda_experiments.py` to regenerate all four .pkl files.
"""

import os
import pickle as pkl

import numpy as np

from kernel_ridge_regression import (
    generate_spiked_covariance,
    generate_random_alignment_vector,
    generate_data,
)
from stieltjes_asymptotics import compute_spiked_covariance_intermediate_quantities


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "c_lambda_data")

# ---- experiment settings ---------------------------------------------------
LINE_CONFIGS = [(0.0, 0.0), (10.0, 0.6)]        # (gamma, rho) for plots 1 & 2
HEATMAP_CONFIGS = [(0.0, 0.0), (10.0, 0.6)]     # (gamma, rho) for plot 3 (one each)
K_LS = [0.0, 1.0, 10.0]


def default_config():
    return dict(D=1000, n=500, sigma=0.5, ntrials=100, seed=0,
                lambda_max=2.0, lambda_step=0.01,            # line plots
                heat_lambda_max=1.0, heat_lambda_step=0.01,  # heatmap
                heat_kl_max=10.0, heat_kl_step=0.1)


# ---- shared helpers --------------------------------------------------------
def _make_instance(D, gamma, rho, seed):
    """Fixed spike direction v, covariance Sigma, and target w_star."""
    np.random.seed(seed)
    v = np.random.randn(D)
    v /= np.linalg.norm(v)
    Sigma = generate_spiked_covariance(D, v, gamma) if gamma > 0 else np.eye(D)
    w_star = generate_random_alignment_vector(v, rho)
    return Sigma, w_star


def _quadratic_forms(X, y, lam, D, n):
    """Return (q1, q2) = (y^T K_x A^-1 y, y^T K_x A^-1 K_x y) for one dataset."""
    K_x = (1.0 / D) * (X.T @ X)
    A = K_x + lam * np.eye(n)
    A_inv = np.linalg.pinv(A)
    Ky = K_x @ y
    q1 = float(np.dot(y, K_x @ (A_inv @ y)))
    q2 = float(np.dot(Ky, A_inv @ Ky))
    return q1, q2


def _c_lambda_from_forms(lam, k_l, q1, q2, D, n):
    beta = k_l * D / n**2
    return 1.0 + (lam * beta * q1) / (1.0 + beta * q2)


def _c_lambda_theory(n, D, k_l, lam, gamma, rho, sigma):
    return float(np.real(compute_spiked_covariance_intermediate_quantities(
        n, D, k_l, lam, gamma, rho, sigma)["c_lambda"]))


# ---- (1) line data: c_lambda vs lambda, several k_l ------------------------
def compute_c_lambda_line_data(gamma, rho, cfg, k_ls=K_LS, out_dir=OUT_DIR):
    D, n = cfg["D"], cfg["n"]
    sigma, ntrials, seed = cfg["sigma"], cfg["ntrials"], cfg["seed"]
    lambdas = np.round(np.arange(0.0, cfg["lambda_max"] + 1e-9, cfg["lambda_step"]), 6)
    nkl, nlam = len(k_ls), len(lambdas)

    Sigma, w_star = _make_instance(D, gamma, rho, seed)

    # theory (dense, cheap)
    c_theory = np.zeros((nkl, nlam))
    for a, k_l in enumerate(k_ls):
        for j, lam in enumerate(lambdas):
            c_theory[a, j] = _c_lambda_theory(n, D, k_l, lam, gamma, rho, sigma)

    # empirics: one pinv per (trial, lambda); all k_l share q1, q2
    c_trials = np.zeros((ntrials, nkl, nlam))
    print(f"[lines gamma={gamma:g} rho={rho:g}] {ntrials} trials x {nlam} lambdas ...",
          flush=True)
    for t in range(ntrials):
        trial_seed = seed + 1 + t
        np.random.seed(trial_seed)
        X, y, _ = generate_data(D, n, Sigma, noise_std=sigma, w_star=w_star, rng=trial_seed)
        for j, lam in enumerate(lambdas):
            q1, q2 = _quadratic_forms(X, y, lam, D, n)
            for a, k_l in enumerate(k_ls):
                c_trials[t, a, j] = _c_lambda_from_forms(lam, k_l, q1, q2, D, n)

    c_emp_mean = c_trials.mean(axis=0)
    c_emp_sem = c_trials.std(axis=0, ddof=1) / np.sqrt(ntrials)

    results = dict(
        kind="c_lambda_lines",
        D=D, n=n, psi=n / D, sigma=sigma, gamma=gamma, rho=rho,
        ntrials=ntrials, seed=seed,
        k_ls=np.asarray(k_ls, dtype=float), lambdas=lambdas,
        c_theory=c_theory, c_emp_mean=c_emp_mean, c_emp_sem=c_emp_sem,
    )
    os.makedirs(out_dir, exist_ok=True)
    fname = (f"c_lambda_lines_gamma={gamma:g}_rho={rho:g}_D={D}_n={n}"
             f"_sigma={sigma:g}_ntrials={ntrials}.pkl")
    path = os.path.join(out_dir, fname)
    with open(path, "wb") as f:
        pkl.dump(results, f)
    print(f"[lines gamma={gamma:g} rho={rho:g}] saved -> {path}", flush=True)
    return path


# ---- (2) heatmap data: theory c_lambda over (lambda, k_l) ------------------
def compute_c_lambda_heatmap_data(gamma, rho, cfg, out_dir=OUT_DIR):
    D, n, sigma = cfg["D"], cfg["n"], cfg["sigma"]
    lambdas = np.round(np.arange(0.0, cfg["heat_lambda_max"] + 1e-9, cfg["heat_lambda_step"]), 6)
    k_ls = np.round(np.arange(0.0, cfg["heat_kl_max"] + 1e-9, cfg["heat_kl_step"]), 6)

    # rows = k_l, cols = lambda
    C = np.zeros((len(k_ls), len(lambdas)))
    for i, k_l in enumerate(k_ls):
        for j, lam in enumerate(lambdas):
            C[i, j] = _c_lambda_theory(n, D, k_l, lam, gamma, rho, sigma)

    results = dict(
        kind="c_lambda_heatmap",
        D=D, n=n, psi=n / D, sigma=sigma, gamma=gamma, rho=rho,
        lambdas=lambdas, k_ls=k_ls, c_theory=C,
    )
    os.makedirs(out_dir, exist_ok=True)
    fname = (f"c_lambda_heatmap_gamma={gamma:g}_rho={rho:g}_D={D}_n={n}"
             f"_sigma={sigma:g}.pkl")
    path = os.path.join(out_dir, fname)
    with open(path, "wb") as f:
        pkl.dump(results, f)
    print(f"[heatmap gamma={gamma:g} rho={rho:g}] saved -> {path}", flush=True)
    return path


def main():
    cfg = default_config()
    for gamma, rho in LINE_CONFIGS:
        compute_c_lambda_line_data(gamma, rho, cfg)
    for gamma, rho in HEATMAP_CONFIGS:
        compute_c_lambda_heatmap_data(gamma, rho, cfg)


if __name__ == "__main__":
    main()
