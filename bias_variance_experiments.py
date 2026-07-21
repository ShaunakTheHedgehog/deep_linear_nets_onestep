"""
bias_variance_experiments.py — bias/variance/gen-error data for the B-V-G figures.

The saved sweep pkls only store *empirical generalization error*; empirical bias
and variance require an outer expectation over datasets (a function of the
trial-averaged estimator), so they must be generated here.

For each (gamma, rho) config this driver, at each lambda:
  * draws `ntrials` datasets (same seed=0 as run_sweep, so empirical G reproduces
    the saved sweep values -- a built-in cross-check),
  * forms w_init (k_l=0) and w_feat (k_l) on each,
  * accumulates running sums to get, WITHOUT storing every estimator:
        B = (1/D)(w_bar - w*)^T S (w_bar - w*)
        V = (1/D)[ mean_t w_t^T S w_t  -  w_bar^T S w_bar ]
    (the exact formulas from unit_tests.py; here S = Sigma = I + gamma v v^T,
     so S-quadratics use  u^T S u = u.u + gamma (v.u)^2  -- no dense Sigma needed),
  * keeps per-trial G_t for a proper mean +/- s.e.m. on generalization error.

Point-estimate identity B + V = mean_t(G_t) is asserted. Theory B, V, G come from
compute_spiked_covariance_model_bias_and_variance.

Run `python bias_variance_experiments.py` to (re)generate both .pkl files.
"""

import os
import pickle as pkl

import numpy as np

from kernel_ridge_regression import (
    generate_spiked_covariance,
    generate_random_alignment_vector,
    generate_data,
)
from stieltjes_asymptotics import compute_spiked_covariance_model_bias_and_variance


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "bias_variance_data")

CONFIGS = [(0.0, 0.0), (10.0, 0.6)]     # (gamma, rho)


def default_config():
    return dict(D=1000, n=500, sigma=0.5, k_l=10.0, ntrials=100, seed=0,
                lambda_max=1.0, lambda_step=0.01)


def _make_instance(D, gamma, rho, seed):
    """Spike direction v and target w_star; RNG sequence matches run_sweep."""
    np.random.seed(seed)
    v = np.random.randn(D)
    v /= np.linalg.norm(v)
    # generate_spiked_covariance / np.eye draw no randomness, so w_star matches run_sweep
    _ = generate_spiked_covariance(D, v, gamma) if gamma > 0 else None
    w_star = generate_random_alignment_vector(v, rho)
    return v, w_star


def _sigma_quad(u, v, gamma):
    """u^T Sigma u for Sigma = I + gamma v v^T."""
    return float(u @ u + gamma * (v @ u) ** 2)


def _estimators(X, y, k_l, lam, D, n):
    """(w_init, w_feat) from a single A^{-1} (matches compute_w_init/compute_w_feat)."""
    K_x = (1.0 / D) * (X.T @ X)
    A = K_x + lam * np.eye(n)
    A_inv = np.linalg.pinv(A)
    w_init = (1.0 / np.sqrt(D)) * X @ (A_inv @ y)
    beta = k_l * D / n**2
    Ky = K_x @ y
    num = lam * beta * np.dot(y, K_x @ (A_inv @ y))
    denom = 1.0 + beta * np.dot(Ky, A_inv @ Ky)
    w_feat = (1.0 + num / denom) * w_init
    return w_init, w_feat


def run_one(gamma, rho, cfg, out_dir=OUT_DIR):
    D, n = cfg["D"], cfg["n"]
    sigma, k_l = cfg["sigma"], cfg["k_l"]
    ntrials, seed = cfg["ntrials"], cfg["seed"]
    lambdas = np.round(np.arange(0.0, cfg["lambda_max"] + 1e-9, cfg["lambda_step"]), 6)
    nlam = len(lambdas)

    v, w_star = _make_instance(D, gamma, rho, seed)
    Sigma = generate_spiked_covariance(D, v, gamma) if gamma > 0 else np.eye(D)

    # running accumulators (no per-trial estimator storage)
    S1_init = np.zeros((nlam, D)); S1_feat = np.zeros((nlam, D))   # sum_t w
    S2_init = np.zeros(nlam);      S2_feat = np.zeros(nlam)        # sum_t w^T S w
    G_init_tr = np.zeros((ntrials, nlam)); G_feat_tr = np.zeros((ntrials, nlam))

    print(f"[BV gamma={gamma:g} rho={rho:g}] {ntrials} trials x {nlam} lambdas ...", flush=True)
    for t in range(ntrials):
        trial_seed = seed + 1 + t
        np.random.seed(trial_seed)
        X, y, _ = generate_data(D, n, Sigma, noise_std=sigma,
                                w_star=w_star, rng=trial_seed)
        for j, lam in enumerate(lambdas):
            wi, wf = _estimators(X, y, k_l, lam, D, n)
            S1_init[j] += wi; S1_feat[j] += wf
            S2_init[j] += _sigma_quad(wi, v, gamma)
            S2_feat[j] += _sigma_quad(wf, v, gamma)
            G_init_tr[t, j] = _sigma_quad(wi - w_star, v, gamma) / D
            G_feat_tr[t, j] = _sigma_quad(wf - w_star, v, gamma) / D

    def emp_bias_var(S1, S2):
        B = np.empty(nlam); V = np.empty(nlam)
        for j in range(nlam):
            w_bar = S1[j] / ntrials
            B[j] = _sigma_quad(w_bar - w_star, v, gamma) / D
            V[j] = (S2[j] / ntrials - _sigma_quad(w_bar, v, gamma)) / D
        return B, V

    init_B_emp, init_V_emp = emp_bias_var(S1_init, S2_init)
    feat_B_emp, feat_V_emp = emp_bias_var(S1_feat, S2_feat)
    init_G_emp = G_init_tr.mean(0); init_G_sem = G_init_tr.std(0, ddof=1) / np.sqrt(ntrials)
    feat_G_emp = G_feat_tr.mean(0); feat_G_sem = G_feat_tr.std(0, ddof=1) / np.sqrt(ntrials)

    # point-estimate identity check: B + V == mean(G_t)
    assert np.allclose(init_B_emp + init_V_emp, init_G_emp, atol=1e-9)
    assert np.allclose(feat_B_emp + feat_V_emp, feat_G_emp, atol=1e-9)

    # theory
    init_B_th = np.empty(nlam); init_V_th = np.empty(nlam); init_G_th = np.empty(nlam)
    feat_B_th = np.empty(nlam); feat_V_th = np.empty(nlam); feat_G_th = np.empty(nlam)
    for j, lam in enumerate(lambdas):
        init_B_th[j], init_V_th[j], init_G_th[j] = compute_spiked_covariance_model_bias_and_variance(
            n, D, 0.0, lam, gamma, rho, sigma)
        feat_B_th[j], feat_V_th[j], feat_G_th[j] = compute_spiked_covariance_model_bias_and_variance(
            n, D, k_l, lam, gamma, rho, sigma)

    results = dict(
        kind="bias_variance", D=D, n=n, psi=n / D, sigma=sigma, gamma=gamma, rho=rho,
        k_l=k_l, ntrials=ntrials, seed=seed, lambdas=lambdas,
        # empirical
        init_bias_emp=init_B_emp, init_variance_emp=init_V_emp,
        init_gen_error_emp=init_G_emp, init_gen_error_sem=init_G_sem,
        feat_bias_emp=feat_B_emp, feat_variance_emp=feat_V_emp,
        feat_gen_error_emp=feat_G_emp, feat_gen_error_sem=feat_G_sem,
        # theory
        init_bias_theory=init_B_th, init_variance_theory=init_V_th, init_gen_error_theory=init_G_th,
        feat_bias_theory=feat_B_th, feat_variance_theory=feat_V_th, feat_gen_error_theory=feat_G_th,
    )
    os.makedirs(out_dir, exist_ok=True)
    fname = (f"bias_variance_gamma={gamma:g}_rho={rho:g}_D={D}_n={n}"
             f"_sigma={sigma:g}_kl={k_l:g}_ntrials={ntrials}.pkl")
    path = os.path.join(out_dir, fname)
    with open(path, "wb") as f:
        pkl.dump(results, f)
    print(f"[BV gamma={gamma:g} rho={rho:g}] saved -> {path}", flush=True)
    return path


def main():
    cfg = default_config()
    for gamma, rho in CONFIGS:
        run_one(gamma, rho, cfg)


if __name__ == "__main__":
    main()
