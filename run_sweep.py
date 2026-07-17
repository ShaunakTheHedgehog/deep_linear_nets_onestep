"""
run_sweep.py — one (gamma, rho) spiked-covariance experiment per invocation.

Designed for a SLURM job array: each array task runs a SINGLE (gamma, rho)
combination and writes its own .pkl, so the whole sweep runs in parallel.
Aggregation/plotting happens separately (make_figures.py).

This is a standalone driver. It imports the estimation/theory functions from
the existing pipeline as building blocks and does NOT modify them.

Fixed sweep (index order is row-major over GAMMAS x RHOS):
    gamma in {0, 1, 5}   x   rho in {0, 0.2, 0.4, 0.6, 0.8, 1.0}   -> 18 tasks

Usage
-----
Single combination by array index (what SLURM uses):
    python run_sweep.py --index $SLURM_ARRAY_TASK_ID

Single combination explicitly:
    python run_sweep.py --gamma 5 --rho 0.8

Whole sweep serially (local smoke test):
    python run_sweep.py --all --ntrials 5
"""

import os
import argparse
import pickle as pkl

import numpy as np

from kernel_ridge_regression import (
    generate_spiked_covariance,
    generate_random_alignment_vector,
    generate_data,
    compute_gen_error,
)
from stieltjes_asymptotics import compute_spiked_covariance_model_bias_and_variance


# ---- the sweep grid (task index maps into this, row-major) ------------------
GAMMAS = [0.0, 1.0, 5.0]
RHOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COMBOS = [(g, r) for g in GAMMAS for r in RHOS]   # len == 18


def default_config():
    return dict(D=1000, n=500, sigma=0.5, k_l=10.0, ntrials=50, seed=0,
                lambda_step=0.08, lambda_max=1.0, theory_step=0.01,
                out_dir="spiked_sweep")


def out_filename(cfg, gamma, rho):
    return (f"spiked_gamma={gamma:g}_rho={rho:g}_D={cfg['D']}_n={cfg['n']}"
            f"_sigma={cfg['sigma']:g}_kl={cfg['k_l']:g}_ntrials={cfg['ntrials']}.pkl")


def run_one(gamma, rho, cfg):
    """Run one (gamma, rho) combination; save a results dict; return its path."""
    D, n = cfg["D"], cfg["n"]
    sigma, k_l = cfg["sigma"], cfg["k_l"]
    ntrials, seed = cfg["ntrials"], cfg["seed"]
    psi = n / D

    # empirical lambda grid (coarse: markers) and theory grid (fine: smooth curve)
    lambdas = np.round(np.append(np.arange(0.0, cfg["lambda_max"], cfg["lambda_step"]),
                                 cfg["lambda_max"]), 6)
    lambdas_theory = np.round(np.arange(0.0, cfg["lambda_max"] + 1e-9, cfg["theory_step"]), 6)

    # --- fixed problem instance (spike direction v, target w_star) ---
    # Seed the global RNG: generate_random_alignment_vector and generate_data's
    # noise term both draw from np.random, so this pins them for reproducibility.
    np.random.seed(seed)
    v = np.random.randn(D)
    v /= np.linalg.norm(v)
    Sigma = generate_spiked_covariance(D, v, gamma) if gamma > 0 else np.eye(D)
    w_star = generate_random_alignment_vector(v, rho)

    # --- empirics: per-trial generalization error (retained for s.e.m.) ---
    init_trials = np.zeros((ntrials, len(lambdas)))
    feat_trials = np.zeros((ntrials, len(lambdas)))
    print(f"[gamma={gamma:g} rho={rho:g}] running {ntrials} trials "
          f"x {len(lambdas)} lambdas ...", flush=True)
    for t in range(ntrials):
        trial_seed = seed + 1 + t
        np.random.seed(trial_seed)   # pins the noise term inside generate_data
        X, y, _ = generate_data(D, n, Sigma, noise_std=sigma, w_star=w_star, rng=trial_seed)
        for j, lam in enumerate(lambdas):
            init_trials[t, j] = compute_gen_error(w_star, Sigma, X, y, k_l=0., ridge_lambda=lam)
            feat_trials[t, j] = compute_gen_error(w_star, Sigma, X, y, k_l=k_l, ridge_lambda=lam)

    init_mean, feat_mean = init_trials.mean(0), feat_trials.mean(0)
    init_sem = init_trials.std(0, ddof=1) / np.sqrt(ntrials)
    feat_sem = feat_trials.std(0, ddof=1) / np.sqrt(ntrials)

    # --- theory: dense grid ---
    init_B = np.zeros_like(lambdas_theory); init_V = np.zeros_like(lambdas_theory)
    init_G = np.zeros_like(lambdas_theory)
    feat_B = np.zeros_like(lambdas_theory); feat_V = np.zeros_like(lambdas_theory)
    feat_G = np.zeros_like(lambdas_theory)
    for j, lam in enumerate(lambdas_theory):
        init_B[j], init_V[j], init_G[j] = compute_spiked_covariance_model_bias_and_variance(
            n, D, 0., lam, gamma, rho, sigma)
        feat_B[j], feat_V[j], feat_G[j] = compute_spiked_covariance_model_bias_and_variance(
            n, D, k_l, lam, gamma, rho, sigma)

    results = dict(
        D=D, n=n, psi=psi, rho=rho, spike_strength=gamma, noise_std=sigma,
        k_l=k_l, ntrials=ntrials, seed=seed,
        lambdas=lambdas, lambdas_theory=lambdas_theory,
        init_gen_errors=init_mean, feat_gen_errors=feat_mean,
        init_gen_errors_sem=init_sem, feat_gen_errors_sem=feat_sem,
        init_gen_errors_trials=init_trials, feat_gen_errors_trials=feat_trials,
        init_gen_error_theory=init_G, feat_gen_error_theory=feat_G,
        init_bias_theory=init_B, init_variance_theory=init_V,
        feat_bias_theory=feat_B, feat_variance_theory=feat_V,
    )

    os.makedirs(cfg["out_dir"], exist_ok=True)
    path = os.path.join(cfg["out_dir"], out_filename(cfg, gamma, rho))
    with open(path, "wb") as f:
        pkl.dump(results, f)
    print(f"[gamma={gamma:g} rho={rho:g}] saved -> {path}", flush=True)
    return path


def build_config(args):
    cfg = default_config()
    for key in ("D", "n", "sigma", "k_l", "ntrials", "seed",
                "lambda_step", "lambda_max", "theory_step", "out_dir"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=int, default=None,
                   help="task index into the (gamma, rho) grid; "
                        "defaults to $SLURM_ARRAY_TASK_ID if set")
    p.add_argument("--gamma", type=float, default=None, help="override: run this gamma")
    p.add_argument("--rho", type=float, default=None, help="override: run this rho")
    p.add_argument("--all", action="store_true", help="run every combo serially (local test)")
    p.add_argument("--D", type=int, default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--k_l", type=float, default=None)
    p.add_argument("--ntrials", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lambda-step", dest="lambda_step", type=float, default=None)
    p.add_argument("--lambda-max", dest="lambda_max", type=float, default=None)
    p.add_argument("--theory-step", dest="theory_step", type=float, default=None)
    p.add_argument("--out-dir", dest="out_dir", type=str, default=None)
    args = p.parse_args()

    cfg = build_config(args)

    if args.all:
        for g, r in COMBOS:
            run_one(g, r, cfg)
        return

    if args.gamma is not None and args.rho is not None:
        run_one(args.gamma, args.rho, cfg)
        return

    index = args.index
    if index is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if index is None:
        p.error("provide --index, or --gamma and --rho, or --all "
                "(or run under SLURM with $SLURM_ARRAY_TASK_ID set)")
    if not (0 <= index < len(COMBOS)):
        p.error(f"--index must be in [0, {len(COMBOS) - 1}]; got {index}")

    gamma, rho = COMBOS[index]
    run_one(gamma, rho, cfg)


if __name__ == "__main__":
    main()
