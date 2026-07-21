"""
unit_tests.py — empirics-vs-theory validation suite for the spiked-covariance
one-step feature-learning model.

Two layers of checks:

  (A) LIBRARY vs REFERENCE FORMULA
      The estimator functions in kernel_ridge_regression.py (compute_w_init,
      compute_w_feat, compute_gen_error) reproduce the intended closed forms
      to machine precision on sampled data. Catches typos/regressions in the code.

  (B) EMPIRICS vs THEORY  (averaged over many datasets)
      The Stieltjes-transform predictions in stieltjes_asymptotics.py match the
      trial-averaged empirics for
          - the inflation constant c_lambda
          - generalization error G   (baseline and feature-learning)
          - the bias/variance decomposition B, V
      plus the exact identities  B + V = G  and the  lambda = 0  sanity checks,
      and a theory-vs-theory cross-check that the isotropic formulas agree with
      the spiked ones at gamma = 0.

Run:
    python unit_tests.py            # full suite (~30 s)
    python unit_tests.py --quick    # fewer trials, looser tolerances (fast smoke)

Depends only on numpy. Exits non-zero if any check fails.
"""

import sys
import numpy as np

from kernel_ridge_regression import (
    generate_spiked_covariance, generate_random_alignment_vector, generate_data,
    compute_w_init, compute_w_feat, compute_gen_error,
)
from stieltjes_asymptotics import (
    compute_spiked_covariance_model_bias_and_variance,
    compute_spiked_covariance_intermediate_quantities,
    compute_linear_model_bias_and_variance,
    compute_feature_learning_model_bias_and_variance,
)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
QUICK = "--quick" in sys.argv

D, n = 1000, 500
SIGMA = 0.5
K_L = 10.0
NTRIALS = 60 if QUICK else 300           # bias/variance needs many draws to stabilise
LAMBDAS = np.array([0.0, 0.1, 0.3, 0.7])
# (gamma, rho) settings tiling the phase diagram: isotropic, mild spike, strong spike
CONFIGS = [(0.0, 0.5), (1.0, 0.5), (5.0, 0.8)]
BASE_SEED = 0

# statistical tolerances
NSIGMA = 6.0            # accept empirics within this many s.e.m. of theory
ATOL_C = 3e-3          # abs floor for c_lambda
ATOL_G = 6e-3          # abs floor for G
RTOL_BV = 0.08 if not QUICK else 0.15   # rel tol for bias/variance (finite-D bias)
ATOL_BV = 0.012        # abs floor for bias/variance
ATOL_EXACT = 1e-8      # machine-precision identities / library-vs-reference


# ---------------------------------------------------------------------------
# tiny test harness
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name, emp, theory, sem=None, atol=0.0, rtol=0.0, nsigma=NSIGMA):
        """Pass if emp is within atol, rtol*|theory|, or nsigma*sem of theory."""
        diff = abs(emp - theory)
        thresh = max(atol, rtol * abs(theory))
        if sem is not None and np.isfinite(sem):
            thresh = max(thresh, nsigma * sem)
        ok = diff <= thresh
        self.passed += ok
        self.failed += (not ok)
        tag = "PASS" if ok else "FAIL"
        sem_str = f" sem={sem:.1e} ({diff/sem:+.2f}s)" if (sem and sem > 0) else ""
        print(f"  [{tag}] {name:42s} emp={emp:+.5f} th={theory:+.5f} "
              f"|d|={diff:.1e} tol={thresh:.1e}{sem_str}")
        return ok

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 70)
        print(f"  {self.passed}/{total} checks passed, {self.failed} failed")
        print("=" * 70)
        return self.failed == 0


# ---------------------------------------------------------------------------
# reference (spec) empirical formulas + simulation
# ---------------------------------------------------------------------------
def reference_estimators(X, y, k_l, lam):
    """
    Reference implementation of the empirical estimators, straight from the spec.
    Uses a single A^{-1} and returns (w_init, w_feat, c_lambda).
    """
    Dloc, nloc = X.shape
    K_x = (1.0 / Dloc) * (X.T @ X)
    A = K_x + lam * np.eye(nloc)
    A_inv = np.linalg.pinv(A)
    w_init = (1.0 / np.sqrt(Dloc)) * X @ (A_inv @ y)

    beta = k_l * Dloc / nloc**2
    Ky = K_x @ y
    num = lam * beta * np.dot(y, K_x @ (A_inv @ y))     # lambda * beta * y^T K_x A^-1 y
    denom = 1.0 + beta * np.dot(Ky, A_inv @ Ky)         # 1 + beta * y^T K_x A^-1 K_x y
    c_lambda = 1.0 + num / denom
    w_feat = c_lambda * w_init
    return w_init, w_feat, c_lambda


def make_instance(gamma, rho, seed):
    np.random.seed(seed)
    v = np.random.randn(D)
    v /= np.linalg.norm(v)
    Sigma = generate_spiked_covariance(D, v, gamma) if gamma > 0 else np.eye(D)
    w_star = generate_random_alignment_vector(v, rho)
    return Sigma, w_star


def simulate(gamma, rho, seed=BASE_SEED, ntrials=NTRIALS):
    """Run ntrials datasets; return per-trial c_lambda, G, and stored estimators."""
    Sigma, w_star = make_instance(gamma, rho, seed)
    L = len(LAMBDAS)
    c_emp = np.zeros((ntrials, L))
    G_init = np.zeros((ntrials, L))
    G_feat = np.zeros((ntrials, L))
    W_init = np.zeros((ntrials, L, D))
    W_feat = np.zeros((ntrials, L, D))
    for t in range(ntrials):
        np.random.seed(seed + 1 + t)
        X, y, _ = generate_data(D, n, Sigma, noise_std=SIGMA, w_star=w_star, rng=seed + 1 + t)
        for j, lam in enumerate(LAMBDAS):
            wi, wf, c = reference_estimators(X, y, K_L, lam)
            W_init[t, j] = wi
            W_feat[t, j] = wf
            c_emp[t, j] = c
            di = w_star - wi
            df = w_star - wf
            G_init[t, j] = di @ (Sigma @ di) / D
            G_feat[t, j] = df @ (Sigma @ df) / D
    return dict(Sigma=Sigma, w_star=w_star, c_emp=c_emp,
                G_init=G_init, G_feat=G_feat, W_init=W_init, W_feat=W_feat)


def empirical_bias_variance(W, w_star, Sigma):
    """B = (w_bar-w*)^T S (w_bar-w*)/D ,  V = E[(w-w_bar)^T S (w-w_bar)]/D ."""
    w_bar = W.mean(axis=0)
    d = w_bar - w_star
    B = d @ (Sigma @ d) / D
    dev = W - w_bar
    quad = np.sum(dev * (dev @ Sigma), axis=1)   # dev_t^T Sigma dev_t
    V = quad.mean() / D
    return B, V


def sem(col):
    return col.std(ddof=1) / np.sqrt(len(col))


# ---------------------------------------------------------------------------
# (A) library reproduces the reference formulas exactly
# ---------------------------------------------------------------------------
def test_library_matches_reference(run):
    print("\n[A] library (kernel_ridge_regression) vs reference formula")
    gamma, rho = 5.0, 0.8
    Sigma, w_star = make_instance(gamma, rho, seed=BASE_SEED)
    worst = dict(w_init=0.0, w_feat=0.0, c=0.0, gen=0.0, k0=0.0)
    for t in range(5):
        np.random.seed(100 + t)
        X, y, _ = generate_data(D, n, Sigma, noise_std=SIGMA, w_star=w_star, rng=100 + t)
        for lam in LAMBDAS:
            wi_ref, wf_ref, c_ref = reference_estimators(X, y, K_L, lam)
            wi_lib = compute_w_init(X, y, lam)
            wf_lib = compute_w_feat(X, y, K_L, lam)
            worst["w_init"] = max(worst["w_init"], np.max(np.abs(wi_lib - wi_ref)))
            worst["w_feat"] = max(worst["w_feat"], np.max(np.abs(wf_lib - wf_ref)))
            # c_lambda recovered from the library as ||ratio|| of w_feat to w_init
            c_lib = np.dot(wf_lib, wi_lib) / np.dot(wi_lib, wi_lib)
            worst["c"] = max(worst["c"], abs(c_lib - c_ref))
            # compute_gen_error must equal the quadratic form of the reference w_feat
            g_lib = compute_gen_error(w_star, Sigma, X, y, k_l=K_L, ridge_lambda=lam)
            df = w_star - wf_ref
            worst["gen"] = max(worst["gen"], abs(g_lib - df @ (Sigma @ df) / D))
            # compute_w_feat with k_l=0 must reduce to w_init
            wf0 = compute_w_feat(X, y, 0.0, lam)
            worst["k0"] = max(worst["k0"], np.max(np.abs(wf0 - wi_lib)))
    run.check("compute_w_init == reference", worst["w_init"], 0.0, atol=ATOL_EXACT)
    run.check("compute_w_feat == reference", worst["w_feat"], 0.0, atol=ATOL_EXACT)
    run.check("library c_lambda == reference", worst["c"], 0.0, atol=ATOL_EXACT)
    run.check("compute_gen_error == quad form", worst["gen"], 0.0, atol=ATOL_EXACT)
    run.check("compute_w_feat(k_l=0) == w_init", worst["k0"], 0.0, atol=ATOL_EXACT)


# ---------------------------------------------------------------------------
# (B) empirics vs theory, per (gamma, rho)
# ---------------------------------------------------------------------------
def test_empirics_vs_theory(run, gamma, rho):
    print(f"\n[B] empirics vs theory  (gamma={gamma:g}, rho={rho:g}, "
          f"D={D}, n={n}, sigma={SIGMA}, k_l={K_L}, {NTRIALS} draws)")
    sim = simulate(gamma, rho)
    Sigma, w_star = sim["Sigma"], sim["w_star"]

    for j, lam in enumerate(LAMBDAS):
        # ---- theory ----
        c_th = float(np.real(compute_spiked_covariance_intermediate_quantities(
            n, D, K_L, lam, gamma, rho, SIGMA)["c_lambda"]))
        Bi_th, Vi_th, Gi_th = compute_spiked_covariance_model_bias_and_variance(
            n, D, 0.0, lam, gamma, rho, SIGMA)
        Bf_th, Vf_th, Gf_th = compute_spiked_covariance_model_bias_and_variance(
            n, D, K_L, lam, gamma, rho, SIGMA)

        # ---- empirics ----
        c_emp = sim["c_emp"][:, j].mean()
        Gi_emp = sim["G_init"][:, j].mean()
        Gf_emp = sim["G_feat"][:, j].mean()
        Bi_emp, Vi_emp = empirical_bias_variance(sim["W_init"][:, j, :], w_star, Sigma)
        Bf_emp, Vf_emp = empirical_bias_variance(sim["W_feat"][:, j, :], w_star, Sigma)

        print(f"  -- lambda = {lam:g} --")
        run.check("c_lambda", c_emp, c_th, sem=sem(sim["c_emp"][:, j]), atol=ATOL_C)
        run.check("G_init", Gi_emp, Gi_th, sem=sem(sim["G_init"][:, j]), atol=ATOL_G)
        run.check("G_feat", Gf_emp, Gf_th, sem=sem(sim["G_feat"][:, j]), atol=ATOL_G)
        run.check("bias_init", Bi_emp, Bi_th, atol=ATOL_BV, rtol=RTOL_BV)
        run.check("var_init", Vi_emp, Vi_th, atol=ATOL_BV, rtol=RTOL_BV)
        run.check("bias_feat", Bf_emp, Bf_th, atol=ATOL_BV, rtol=RTOL_BV)
        run.check("var_feat", Vf_emp, Vf_th, atol=ATOL_BV, rtol=RTOL_BV)

        # exact identities B + V = G (theory to machine precision; empirics exactly)
        run.check("theory  B+V == G (init)", Bi_th + Vi_th, Gi_th, atol=ATOL_EXACT)
        run.check("theory  B+V == G (feat)", Bf_th + Vf_th, Gf_th, atol=ATOL_EXACT)
        run.check("emp     B+V == mean(G) (feat)", Bf_emp + Vf_emp, Gf_emp, atol=1e-7)


# ---------------------------------------------------------------------------
# (C) lambda = 0 sanity: c_lambda = 1  =>  feat == init
# ---------------------------------------------------------------------------
def test_lambda_zero(run):
    print("\n[C] lambda = 0 sanity (c_lambda = 1, so feat == init)")
    for gamma, rho in CONFIGS:
        c_th = float(np.real(compute_spiked_covariance_intermediate_quantities(
            n, D, K_L, 0.0, gamma, rho, SIGMA)["c_lambda"]))
        _, _, Gi = compute_spiked_covariance_model_bias_and_variance(n, D, 0.0, 0.0, gamma, rho, SIGMA)
        _, _, Gf = compute_spiked_covariance_model_bias_and_variance(n, D, K_L, 0.0, gamma, rho, SIGMA)
        run.check(f"c_lambda(0)=1  [g={gamma:g},r={rho:g}]", c_th, 1.0, atol=ATOL_EXACT)
        run.check(f"G_feat==G_init [g={gamma:g},r={rho:g}]", Gf, Gi, atol=ATOL_EXACT)


# ---------------------------------------------------------------------------
# (D) theory-vs-theory: isotropic formulas == spiked formulas at gamma = 0
# ---------------------------------------------------------------------------
def test_isotropic_consistency(run):
    print("\n[D] isotropic theory == spiked theory at gamma = 0")
    for lam in LAMBDAS:
        Bi_iso, Vi_iso, Gi_iso = compute_linear_model_bias_and_variance(n, D, lam, SIGMA)
        Bf_iso, Vf_iso, Gf_iso = compute_feature_learning_model_bias_and_variance(n, D, K_L, lam, SIGMA)
        # spiked at gamma=0, rho=0 (isotropic target); baseline (k_l=0) and feature (k_l)
        Bi_sp, Vi_sp, Gi_sp = compute_spiked_covariance_model_bias_and_variance(n, D, 0.0, lam, 0.0, 0.0, SIGMA)
        Bf_sp, Vf_sp, Gf_sp = compute_spiked_covariance_model_bias_and_variance(n, D, K_L, lam, 0.0, 0.0, SIGMA)
        run.check(f"G_init iso==spiked [lam={lam:g}]", Gi_iso, Gi_sp, atol=1e-6)
        run.check(f"G_feat iso==spiked [lam={lam:g}]", Gf_iso, Gf_sp, atol=1e-6)


# ---------------------------------------------------------------------------
def main():
    print(f"unit_tests.py {'(quick)' if QUICK else '(full)'}  "
          f"D={D} n={n} sigma={SIGMA} k_l={K_L} ntrials={NTRIALS}")
    run = Runner()
    test_library_matches_reference(run)
    for gamma, rho in CONFIGS:
        test_empirics_vs_theory(run, gamma, rho)
    test_lambda_zero(run)
    test_isotropic_consistency(run)
    ok = run.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
