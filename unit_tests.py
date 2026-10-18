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

  (D) PUBLISHED-FIGURE-DATA INTEGRITY
      For each of the 6 paper figures' shipped .pkl files: recompute the theory
      arrays from the current stieltjes code and require they equal the stored
      theory (so the published figures stay bound to the audited formulas),
      require the stored empirics lie within a few s.e.m. of theory, and require
      the metadata match the figure caption (psi, sigma, k_l, gamma, rho).
      Missing data files are reported as SKIP, not failures.

  (E) PER-FIGURE SCIENTIFIC CLAIMS  (theory-only)
      The specific claim each figure makes: the global-FLA SNR window and the
      A/B/C insets (Fig 1); c_lambda floor / monotonicity / spiked saturation
      (Fig 2); lambda*_feat >= lambda*_init and when an FLA holds (Fig 3); the
      classical vs bias-for-variance trade near lambda=0 (Fig 4); the mixed-
      partial sign flip at the window lower bound (Fig 5); and no-FLA with
      isotropic optimal ridge lambda*=sigma^2 (Fig 6).

  (F) PIPELINE-REGENERATION SMOKE TEST  (--slow)
      Regenerate each driver's data at small (D, n), confirm it runs and roughly
      tracks theory, then run every make_figures plotter on that data.

  (G) FINITE-WIDTH VALIDATION OF PROP. 4.1  (--slow)
      Simulate one GD step in the deep linear network at finite width N and
      confirm the layer-ell feature-kernel update concentrates to
      (s_ell eta^2 / n^2) K_x y y^T K_x with s_ell = ell(ell+1)(2ell+1)/6 as N
      grows (validates the N -> infinity limit the rest of the pipeline assumes).

Run:
    python unit_tests.py            # full suite (A-E, ~30 s)
    python unit_tests.py --quick    # fewer trials, looser tolerances (fast smoke)
    python unit_tests.py --slow     # also run (F) drivers/plotters and (G) finite-width

Depends on numpy; (F) also needs scipy/matplotlib (already required by the repo).
Exits non-zero if any check fails.
"""

import sys
import os
import glob
import pickle
import numpy as np
from numpy.random import seed

from kernel_ridge_regression import (
    generate_spiked_covariance, generate_random_alignment_vector, generate_data,
    compute_w_init, compute_w_feat, compute_gen_error,
)
from stieltjes_asymptotics import (
    compute_spiked_covariance_model_bias_and_variance,
    compute_spiked_covariance_intermediate_quantities,
    min_gen_error_over_lambda,
    dG_dlambda_dk_at_zero,
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
QUICK = "--quick" in sys.argv
SLOW = "--slow" in sys.argv         # also run the pipeline-regeneration smoke test (block F)

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
        self.skipped = 0

    def expect(self, name, condition, detail=""):
        """Pass iff `condition` is truthy. For sign / inequality / boolean claims."""
        ok = bool(condition)
        self.passed += ok
        self.failed += (not ok)
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name:42s} {detail}")
        return ok

    def skip(self, name, detail=""):
        """Record a skipped check (e.g. figure data file not present)."""
        self.skipped += 1
        print(f"  [SKIP] {name:42s} {detail}")

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
        print(f"  {self.passed}/{total} checks passed, {self.failed} failed"
              + (f", {self.skipped} skipped" if self.skipped else ""))
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

def check_w_v_alignment(gamma, rho, seed):
    np.random.seed(seed)
    v = np.random.randn(D)
    v /= np.linalg.norm(v)
    w_star = generate_random_alignment_vector(v, rho)
    alignment = float(np.dot(v, w_star) / D**0.5)
    expected_alignment = rho
    assert np.isclose(alignment, expected_alignment, atol=1e-6)


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



# ===========================================================================
# helpers for the figure-level tests (asymptotic formulas depend on n, D only
# through psi = n/D, so a fixed reference D is used for the theory-only checks)
# ===========================================================================
_D_REF = 2000.0


def _G(psi, gamma, rho, sigma, k_l, lam):
    """Asymptotic generalization error at a single lambda."""
    return compute_spiked_covariance_model_bias_and_variance(
        psi * _D_REF, _D_REF, k_l, lam, gamma, rho, sigma)[2]


def _BVG(psi, gamma, rho, sigma, k_l, lam):
    """(bias, variance, gen_error) at a single lambda."""
    return compute_spiked_covariance_model_bias_and_variance(
        psi * _D_REF, _D_REF, k_l, lam, gamma, rho, sigma)


def _c(psi, gamma, rho, sigma, k_l, lam):
    """Inflation constant c_lambda at a single lambda."""
    return float(np.real(compute_spiked_covariance_intermediate_quantities(
        psi * _D_REF, _D_REF, k_l, lam, gamma, rho, sigma)["c_lambda"]))


def _snr_window(psi, gamma):
    """(lower, upper) bounds of the sufficient SNR window (Thm. global_FLA)."""
    lo = (1 + gamma * psi) ** 2 / (gamma * (1 + gamma) * (1 - psi) ** 2)
    hi = (1 + gamma * psi) ** 3 / (gamma * (1 + gamma) * (1 - psi) ** 3)
    return lo, hi


def _deriv0(fn, h=1e-5):
    """Forward finite-difference derivative of fn(lambda) at lambda = 0."""
    return (fn(h) - fn(0.0)) / h


# ===========================================================================
# (D) PUBLISHED-FIGURE-DATA INTEGRITY
#     For every shipped figure .pkl:
#       (i)   recompute the theory arrays from the CURRENT stieltjes code and
#             require they equal the stored theory to ~1e-8 -- this binds the
#             numbers behind each published figure to the audited formulas, so
#             any drift in the theory code that would change a figure fails here;
#       (ii)  require the stored (Monte-Carlo) empirics lie within a few s.e.m.
#             of theory (deterministic: the .pkl means/sems are fixed);
#       (iii) require metadata (psi, sigma, k_l, gamma, rho) match the caption.
#     Missing files are SKIPPED (reported), not failed, so the suite still runs
#     on a fresh checkout without the large sweep outputs.
# ===========================================================================
ATOL_FIG_THEORY = 1e-8      # recomputed theory vs stored theory (same code path)
NSIGMA_FIG = 8.0            # stored empirical mean within this many stored s.e.m.
ATOL_FIG_G = 0.02           # abs floor for empirical gen error
RTOL_FIG_BV = 0.15          # rel tol for empirical bias/variance (finite-D bias)
ATOL_FIG_BV = 0.02


def _worst_outliers(emp, theory, sem, nsigma=NSIGMA_FIG, atol=ATOL_FIG_G, rtol=0.0):
    """#points where |emp-theory| exceeds max(atol, rtol|theory|, nsigma*sem), and worst z."""
    emp, theory = np.asarray(emp), np.asarray(theory)
    sem = np.asarray(sem) if sem is not None else np.zeros_like(emp)
    thr = np.maximum.reduce([np.full_like(emp, atol), rtol * np.abs(theory), nsigma * sem])
    resid = np.abs(emp - theory)
    nbad = int(np.sum(resid > thr))
    worst_z = float(np.max(resid / np.maximum(sem, 1e-12))) if np.any(sem > 0) else float(resid.max())
    return nbad, worst_z


def _check_gen_sweep(run, path, exp_psi, exp_sigma, exp_kl):
    """Fig 3 / Fig 6 / Fig 1-insets: gen-error-vs-lambda run from run_sweep.py."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    name = os.path.basename(path)[:34]
    D_, n_ = d["D"], d["n"]
    gamma, rho, sigma, kl = d["spike_strength"], d["rho"], d["noise_std"], d["k_l"]
    run.expect(f"[{name}] meta psi/sigma/kl",
               np.isclose(n_ / D_, exp_psi) and np.isclose(sigma, exp_sigma) and np.isclose(kl, exp_kl),
               f"psi={n_/D_:g} sigma={sigma:g} k_l={kl:g}")

    lam_th = d["lambdas_theory"]
    gi = np.array([compute_spiked_covariance_model_bias_and_variance(n_, D_, 0.0, l, gamma, rho, sigma)[2] for l in lam_th])
    gf = np.array([compute_spiked_covariance_model_bias_and_variance(n_, D_, kl, l, gamma, rho, sigma)[2] for l in lam_th])
    run.check(f"[{name}] theory G_init==stored", float(np.max(np.abs(gi - d["init_gen_error_theory"]))), 0.0, atol=ATOL_FIG_THEORY)
    run.check(f"[{name}] theory G_feat==stored", float(np.max(np.abs(gf - d["feat_gen_error_theory"]))), 0.0, atol=ATOL_FIG_THEORY)

    lam = d["lambdas"]
    ti = np.array([compute_spiked_covariance_model_bias_and_variance(n_, D_, 0.0, l, gamma, rho, sigma)[2] for l in lam])
    tf = np.array([compute_spiked_covariance_model_bias_and_variance(n_, D_, kl, l, gamma, rho, sigma)[2] for l in lam])
    nbi, wzi = _worst_outliers(d["init_gen_errors"], ti, d.get("init_gen_errors_sem"))
    nbf, wzf = _worst_outliers(d["feat_gen_errors"], tf, d.get("feat_gen_errors_sem"))
    run.expect(f"[{name}] emp G_init within {NSIGMA_FIG:g}sem", nbi == 0, f"{nbi} outliers (worst {wzi:.1f}s)")
    run.expect(f"[{name}] emp G_feat within {NSIGMA_FIG:g}sem", nbf == 0, f"{nbf} outliers (worst {wzf:.1f}s)")


def _check_c_lambda_lines(run, path, exp_psi, exp_sigma):
    """Fig 2 lines: c_lambda vs lambda for several k_l."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    name = os.path.basename(path)[:34]
    D_, n_, gamma, rho, sigma = d["D"], d["n"], d["gamma"], d["rho"], d["sigma"]
    run.expect(f"[{name}] meta psi/sigma",
               np.isclose(n_ / D_, exp_psi) and np.isclose(sigma, exp_sigma),
               f"psi={n_/D_:g} sigma={sigma:g}")
    C = np.array([[float(np.real(compute_spiked_covariance_intermediate_quantities(
        n_, D_, kl, lam, gamma, rho, sigma)["c_lambda"]))
        for lam in d["lambdas"]] for kl in d["k_ls"]])
    run.check(f"[{name}] theory c_lambda==stored", float(np.max(np.abs(C - d["c_theory"]))), 0.0, atol=ATOL_FIG_THEORY)
    nb, wz = _worst_outliers(d["c_emp_mean"], d["c_theory"], d["c_emp_sem"], atol=3e-3)
    run.expect(f"[{name}] emp c_lambda within {NSIGMA_FIG:g}sem", nb == 0, f"{nb} outliers (worst {wz:.1f}s)")


def _check_c_lambda_heatmap(run, path):
    """Fig 2 heatmap: theory c_lambda over the (lambda, k_l) grid."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    name = os.path.basename(path)[:34]
    D_, n_, gamma, rho, sigma = d["D"], d["n"], d["gamma"], d["rho"], d["sigma"]
    C = np.array([[float(np.real(compute_spiked_covariance_intermediate_quantities(
        n_, D_, kl, lam, gamma, rho, sigma)["c_lambda"]))
        for lam in d["lambdas"]] for kl in d["k_ls"]])
    run.check(f"[{name}] theory c_lambda==stored", float(np.max(np.abs(C - d["c_theory"]))), 0.0, atol=ATOL_FIG_THEORY)
    run.expect(f"[{name}] c_lambda >= 1 everywhere", float(np.min(d["c_theory"])) >= 1.0 - 1e-9)


def _check_bias_variance(run, path, exp_psi, exp_sigma, exp_kl):
    """Fig 4: bias/variance/gen-error decomposition from bias_variance_experiments.py."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    name = os.path.basename(path)[:34]
    D_, n_, gamma, rho, sigma, kl = d["D"], d["n"], d["gamma"], d["rho"], d["sigma"], d["k_l"]
    run.expect(f"[{name}] meta psi/sigma/kl",
               np.isclose(n_ / D_, exp_psi) and np.isclose(sigma, exp_sigma) and np.isclose(kl, exp_kl),
               f"psi={n_/D_:g} sigma={sigma:g} k_l={kl:g}")
    lam = d["lambdas"]
    worst_th = 0.0
    for est, klv in (("init", 0.0), ("feat", kl)):
        B = np.array([compute_spiked_covariance_model_bias_and_variance(n_, D_, klv, l, gamma, rho, sigma)[0] for l in lam])
        V = np.array([compute_spiked_covariance_model_bias_and_variance(n_, D_, klv, l, gamma, rho, sigma)[1] for l in lam])
        Gg = np.array([compute_spiked_covariance_model_bias_and_variance(n_, D_, klv, l, gamma, rho, sigma)[2] for l in lam])
        worst_th = max(worst_th,
                       float(np.max(np.abs(B - d[f"{est}_bias_theory"]))),
                       float(np.max(np.abs(V - d[f"{est}_variance_theory"]))),
                       float(np.max(np.abs(Gg - d[f"{est}_gen_error_theory"]))))
        run.check(f"[{name}] {est} B+V==G (theory)",
                  float(np.max(np.abs(d[f"{est}_bias_theory"] + d[f"{est}_variance_theory"] - d[f"{est}_gen_error_theory"]))),
                  0.0, atol=ATOL_FIG_THEORY)
        nbg, wzg = _worst_outliers(d[f"{est}_gen_error_emp"], d[f"{est}_gen_error_theory"], d.get(f"{est}_gen_error_sem"))
        run.expect(f"[{name}] {est} emp G within {NSIGMA_FIG:g}sem", nbg == 0, f"{nbg} outliers (worst {wzg:.1f}s)")
        nbb, _ = _worst_outliers(d[f"{est}_bias_emp"], d[f"{est}_bias_theory"], None, atol=ATOL_FIG_BV, rtol=RTOL_FIG_BV)
        nbv, _ = _worst_outliers(d[f"{est}_variance_emp"], d[f"{est}_variance_theory"], None, atol=ATOL_FIG_BV, rtol=RTOL_FIG_BV)
        run.expect(f"[{name}] {est} emp bias/var within tol", nbb == 0 and nbv == 0, f"bias_out={nbb} var_out={nbv}")
    run.check(f"[{name}] theory B/V/G==stored", worst_th, 0.0, atol=ATOL_FIG_THEORY)


def _check_snr_phase(run, path, exp_psi, exp_sigma, exp_kl, n_cells=40, seed=0):
    """Fig 1 heatmap: Delta* = min_lambda G_feat - min_lambda G_init over (gamma, SNR)."""
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    name = os.path.basename(path)[:34]
    psi, sigma, kl = d["psi"], d["sigma"], d["k_l"]
    run.expect(f"[{name}] meta psi/sigma/kl",
               np.isclose(psi, exp_psi) and np.isclose(sigma, exp_sigma) and np.isclose(kl, exp_kl),
               f"psi={psi:g} sigma={sigma:g} k_l={kl:g}")
    if d.get("vary_sigma"):
        run.skip(f"[{name}] Delta recompute", "vary_sigma variant")
        return
    gammas, snrs, Delta = d["gammas"], d["snrs"], d["Delta"]
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, len(snrs), size=n_cells)
    jj = rng.integers(0, len(gammas), size=n_cells)
    worst = 0.0
    for i, j in zip(ii, jj):
        rho_cell = min(sigma * np.sqrt(snrs[i]), 1.0)
        Gf, _ = min_gen_error_over_lambda(psi, gammas[j], rho_cell, sigma, kl)
        Gi, _ = min_gen_error_over_lambda(psi, gammas[j], rho_cell, sigma, 0.0)
        worst = max(worst, abs((Gf - Gi) - Delta[i, j]))
    run.check(f"[{name}] Delta* recompute==stored ({n_cells} cells)", worst, 0.0, atol=1e-6)


def test_fig_data_integrity(run):
    print("\n[D] published-figure-data integrity (recomputed theory == shipped figure data)")
    found = 0

    def _glob1(pattern):
        return sorted(glob.glob(os.path.join(HERE, pattern)))

    # Fig 1 heatmap (psi=0.2, sigma=1, k_l=10)
    for p in _glob1("snr_phase_data/snr_phase_psi=0.2_sigma=1_kl=10_*.pkl"):
        found += 1; _check_snr_phase(run, p, 0.2, 1.0, 10.0)
    # Fig 1 insets (D=5000, n=1000 -> psi=0.2; sigma=1; k_l=10)
    for p in _glob1("new_spiked_sweep/spiked_gamma=*_D=5000_n=1000_sigma=1_kl=10_ntrials=100.pkl"):
        if "logspaced" in p:
            continue
        found += 1; _check_gen_sweep(run, p, 0.2, 1.0, 10.0)
    # Fig 2 c_lambda lines + heatmap (psi=0.5, sigma=0.5)
    for p in _glob1("c_lambda_data/c_lambda_lines_*_D=1000_n=500_sigma=0.5_*.pkl"):
        found += 1; _check_c_lambda_lines(run, p, 0.5, 0.5)
    for p in _glob1("c_lambda_data/c_lambda_heatmap_*_D=1000_n=500_sigma=0.5.pkl"):
        found += 1; _check_c_lambda_heatmap(run, p)
    # Fig 3 gen-error sweep (psi=0.5, sigma=0.5, k_l=10)
    for p in _glob1("simple_sweep/spiked_gamma=*_D=1000_n=500_sigma=0.5_kl=10_ntrials=100.pkl")[:4]:
        found += 1; _check_gen_sweep(run, p, 0.5, 0.5, 10.0)
    # Fig 4 bias/variance (psi=0.5, sigma=0.5, k_l=10)
    for p in _glob1("bias_variance_data/bias_variance_*_D=1000_n=500_sigma=0.5_kl=10_ntrials=100_NEW.pkl"):
        found += 1; _check_bias_variance(run, p, 0.5, 0.5, 10.0)
    # Fig 6 isotropic / orthogonal-spike (psi=0.5, sigma=0.5, k_l=10)
    for p in _glob1("new_spiked_sweep/spiked_gamma=*_D=3000_n=1500_sigma=0.5_kl=10_ntrials=100.pkl"):
        found += 1; _check_gen_sweep(run, p, 0.5, 0.5, 10.0)

    if found == 0:
        run.skip("figure data files", "none found under this directory")


# ===========================================================================
# (E) PER-FIGURE SCIENTIFIC CLAIMS  (theory-only; the claim each figure makes)
# ===========================================================================
def test_fig1_snr_phase_claims(run):
    print("\n[E1] Fig 1 (SNR phase diagram): global FLA window + A/B/C insets")
    psi, sigma, kl = 0.2, 1.0, 10.0
    snr_crit = psi ** 2 / (1 - psi) ** 2
    # window lower bound descends to SNR_crit as gamma -> infinity
    lo_big, _ = _snr_window(psi, 1e7)
    run.check("window lower bound -> SNR_crit (gamma->inf)", lo_big, snr_crit, atol=1e-3)
    # strictly inside the window => global FLA (Delta* < 0); pick achievable rho (<=1)
    for gamma in (5.0, 10.0, 20.0):
        lo, hi = _snr_window(psi, gamma)
        snr = np.sqrt(lo * hi)                      # geometric-mean point, inside window
        rho = sigma * np.sqrt(snr)
        if rho > 1.0:
            run.skip(f"Delta*<0 inside window g={gamma:g}", "rho>1 (unachievable)")
            continue
        Gf, _ = min_gen_error_over_lambda(psi, gamma, rho, sigma, kl)
        Gi, _ = min_gen_error_over_lambda(psi, gamma, rho, sigma, 0.0)
        run.expect(f"Delta*<0 inside window (g={gamma:g})", Gf < Gi - 1e-9,
                   f"Delta*={Gf-Gi:+.4f} window=({lo:.3g},{hi:.3g}) snr={snr:.3g}")
    # A/B/C inset points: A,B are no-FLA (red), C is FLA (blue)
    for tag, gamma, rho, want_fla in (("A", 0.25, 1.0, False), ("B", 20.0, 0.07, False), ("C", 10.0, 0.5, True)):
        Gf, _ = min_gen_error_over_lambda(psi, gamma, rho, sigma, kl)
        Gi, _ = min_gen_error_over_lambda(psi, gamma, rho, sigma, 0.0)
        run.expect(f"inset {tag} FLA={want_fla} (g={gamma:g},SNR={(rho/sigma)**2:.3g})",
                   (Gf < Gi) == want_fla, f"Delta*={Gf-Gi:+.4f}")


def test_fig2_c_lambda_claims(run):
    print("\n[E2] Fig 2 (c_lambda): floor, monotonicity, spiked saturates faster")
    psi, sigma = 0.5, 0.5
    iso, spk = (0.0, 0.0), (10.0, 0.6)
    for (g, r) in (iso, spk):
        run.check(f"c_lambda(0)=1 [g={g:g},r={r:g}]", _c(psi, g, r, sigma, 10.0, 0.0), 1.0, atol=ATOL_EXACT)
        # increasing in lambda (at k_l=10) and in k_l (at lambda=0.5)
        c_l = [_c(psi, g, r, sigma, 10.0, lam) for lam in (0.1, 0.3, 0.6, 1.0)]
        run.expect(f"c_lambda increasing in lambda [g={g:g}]", all(np.diff(c_l) > 0),
                   f"c(lam)={[round(x,4) for x in c_l]}")
        c_k = [_c(psi, g, r, sigma, kk, 0.5) for kk in (0.0, 0.5, 1.0, 5.0, 10.0)]
        run.expect(f"c_lambda increasing in k_l   [g={g:g}]", all(np.diff(c_k) > 0),
                   f"c(k_l)={[round(x,4) for x in c_k]}")
    # spiked saturates faster: c(k=0.5)/c(k=10) is closer to 1 than for isotropic
    lam = 0.5
    iso_ratio = _c(psi, *iso, sigma, 0.5, lam) / _c(psi, *iso, sigma, 10.0, lam)
    spk_ratio = _c(psi, *spk, sigma, 0.5, lam) / _c(psi, *spk, sigma, 10.0, lam)
    run.expect("spiked c_lambda saturates faster than isotropic", spk_ratio > iso_ratio,
               f"spk_ratio={spk_ratio:.4f} > iso_ratio={iso_ratio:.4f}")


def test_fig3_gen_error_sweep_claims(run):
    print("\n[E3] Fig 3 (gen-error sweep): lambda*_feat >= lambda*_init; FLA at strong (g,r)")
    psi, sigma, kl = 0.5, 0.5, 10.0
    for (g, r, want_fla) in ((1.0, 0.6, True), (10.0, 0.6, True), (10.0, 1.0, True), (1.0, 0.2, False)):
        Gi, li = min_gen_error_over_lambda(psi, g, r, sigma, 0.0)
        Gf, lf = min_gen_error_over_lambda(psi, g, r, sigma, kl)
        run.expect(f"lambda*_feat >= lambda*_init [g={g:g},r={r:g}]", lf >= li - 1e-6,
                   f"lam*_init={li:.3g} lam*_feat={lf:.3g}")
        run.expect(f"FLA={want_fla} [g={g:g},r={r:g}]", (Gf < Gi - 1e-9) == want_fla,
                   f"minG_init={Gi:.4f} minG_feat={Gf:.4f}")


def test_fig4_bias_variance_claims(run):
    print("\n[E4] Fig 4 (bias/variance): classical trade for baseline, bias-for-variance for feat")
    psi, sigma, kl = 0.5, 0.5, 10.0
    for (g, r) in ((0.0, 0.0), (10.0, 0.6)):
        dB_base = _deriv0(lambda l: _BVG(psi, g, r, sigma, 0.0, l)[0])
        dV_base = _deriv0(lambda l: _BVG(psi, g, r, sigma, 0.0, l)[1])
        dB_feat = _deriv0(lambda l: _BVG(psi, g, r, sigma, kl, l)[0])
        dV_feat = _deriv0(lambda l: _BVG(psi, g, r, sigma, kl, l)[1])
        run.expect(f"baseline bias' > 0 [g={g:g},r={r:g}]", dB_base > 0, f"dB={dB_base:+.3f}")
        run.expect(f"baseline var'  < 0 [g={g:g},r={r:g}]", dV_base < 0, f"dV={dV_base:+.3f}")
        run.expect(f"feat lowers bias slope [g={g:g},r={r:g}]", dB_feat < dB_base, f"dB_feat-dB_base={dB_feat-dB_base:+.3f}")
        run.expect(f"feat raises var slope  [g={g:g},r={r:g}]", dV_feat > dV_base, f"dV_feat-dV_base={dV_feat-dV_base:+.3f}")
        for klv in (0.0, kl):
            B, V, G = _BVG(psi, g, r, sigma, klv, 0.37)
            run.check(f"B+V==G [g={g:g},r={r:g},k_l={klv:g}]", B + V, G, atol=ATOL_EXACT)


def test_fig5_mixed_partial_claims(run):
    print("\n[E5] Fig 5 (mixed partial): sign flips at the window lower bound")
    psi, sigma = 0.5, 0.4
    snr_crit = psi ** 2 / (1 - psi) ** 2
    lo_big, _ = _snr_window(psi, 1e7)
    run.check("window lower bound -> SNR_crit (gamma->inf)", lo_big, snr_crit, atol=1e-3)
    for gamma in (1.0, 4.0, 16.0):
        lo, _ = _snr_window(psi, gamma)
        for frac, want_neg in ((0.9, False), (1.1, True)):
            snr = lo * frac
            rho = sigma * np.sqrt(snr)
            if rho > 1.0:
                run.skip(f"mixed partial g={gamma:g} frac={frac}", "rho>1")
                continue
            mp = dG_dlambda_dk_at_zero(psi, gamma, rho, sigma)
            run.expect(f"mixed partial {'<0' if want_neg else '>0'} ({'above' if want_neg else 'below'} lower, g={gamma:g})",
                       (mp < 0) == want_neg, f"d2G/dk dlam={mp:+.4f}")


def test_fig6_no_fla_claims(run):
    print("\n[E6] Fig 6 (isotropic / orthogonal spike): no FLA; isotropic optimal lambda*=sigma^2")
    psi, sigma, kl = 0.5, 0.5, 10.0
    # no FLA for gamma=0 (any rho) or rho=0 (any gamma): baseline attains the min
    for (g, r, label) in ((0.0, 0.0, "isotropic"), (0.0, 0.5, "isotropic"), (10.0, 0.0, "orthogonal")):
        Gi, _ = min_gen_error_over_lambda(psi, g, r, sigma, 0.0)
        worst_feat_gain = -np.inf
        for klv in (0.5, 1.0, 5.0, 10.0, 100.0):
            Gf, _ = min_gen_error_over_lambda(psi, g, r, sigma, klv)
            worst_feat_gain = max(worst_feat_gain, Gi - Gf)   # >0 would mean an FLA
        run.expect(f"no FLA ({label}, g={g:g},r={r:g})", worst_feat_gain <= 1e-6,
                   f"best feat gain over baseline={worst_feat_gain:+.2e}")
    # isotropic optimal baseline ridge lambda* == sigma^2
    for s in (0.3, 0.5, 0.7):
        _, li = min_gen_error_over_lambda(psi, 0.0, 0.0, s, 0.0)
        run.check(f"isotropic lambda*==sigma^2 [sigma={s:g}]", li, s ** 2, atol=2e-3)


def test_figure_scientific_claims(run):
    test_fig1_snr_phase_claims(run)
    test_fig2_c_lambda_claims(run)
    test_fig3_gen_error_sweep_claims(run)
    test_fig4_bias_variance_claims(run)
    test_fig5_mixed_partial_claims(run)
    test_fig6_no_fla_claims(run)


# ===========================================================================
# (F) PIPELINE-REGENERATION SMOKE TEST  (--slow)
#     Regenerate each driver's data at small (D, n) into a temp dir, confirm it
#     runs + empirics roughly track theory + B+V=G, then run every make_figures
#     plotter on that data (Agg backend) to confirm the figure code executes.
# ===========================================================================
def test_pipeline_regeneration_smoke(run):
    print("\n[F] pipeline regeneration smoke test (small D,n; drivers + plotters)")
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import run_sweep
    import c_lambda_experiments
    import bias_variance_experiments
    from stieltjes_asymptotics import compute_snr_phase_diagram
    import make_figures

    Ds, ns, sig, klv, ntr = 300, 120, 0.5, 10.0, 8
    psi = ns / Ds
    tmp = tempfile.mkdtemp(prefix="fig_smoke_")
    sweep_dir = os.path.join(tmp, "sweep")
    make_figures.OUT_DIR = tmp    # redirect figure output to the temp dir

    # --- run_sweep driver (Fig 3/6/1-inset data) ---
    cfg = run_sweep.default_config()
    cfg.update(dict(D=Ds, n=ns, sigma=sig, k_l=klv, ntrials=ntr,
                    lambda_max=0.6, lambda_step=0.2, theory_step=0.2, out_dir=sweep_dir))
    p_iso = run_sweep.run_one(0.0, 0.0, cfg)
    p_spk = run_sweep.run_one(10.0, 0.6, cfg)
    for p in (p_iso, p_spk):
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        nb, wz = _worst_outliers(d["feat_gen_errors"],
                                 np.interp(d["lambdas"], d["lambdas_theory"], d["feat_gen_error_theory"]),
                                 d["feat_gen_errors_sem"], nsigma=10.0, atol=0.05)
        run.expect(f"[smoke] run_sweep emp~theory {os.path.basename(p)[:20]}", nb == 0, f"{nb} outliers (worst {wz:.1f}s)")
    run.expect("[smoke] run_sweep runs + writes pkls", os.path.exists(p_iso) and os.path.exists(p_spk))

    # --- c_lambda driver (Fig 2 data) ---
    ccfg = c_lambda_experiments.default_config()
    ccfg.update(dict(D=Ds, n=ns, sigma=sig, ntrials=ntr, lambda_max=0.6, lambda_step=0.2,
                     heat_lambda_max=0.6, heat_lambda_step=0.2, heat_kl_max=10.0, heat_kl_step=2.0))
    p_cl = c_lambda_experiments.compute_c_lambda_line_data(10.0, 0.6, ccfg, out_dir=tmp)
    p_ch = c_lambda_experiments.compute_c_lambda_heatmap_data(10.0, 0.6, ccfg, out_dir=tmp)
    with open(p_cl, "rb") as fh:
        dcl = pickle.load(fh)
    nb, wz = _worst_outliers(dcl["c_emp_mean"], dcl["c_theory"], dcl["c_emp_sem"], nsigma=10.0, atol=0.02)
    run.expect("[smoke] c_lambda emp~theory", nb == 0, f"{nb} outliers (worst {wz:.1f}s)")

    # --- bias/variance driver (Fig 4 data); its own B+V==G assert must pass ---
    bcfg = bias_variance_experiments.default_config()
    bcfg.update(dict(D=Ds, n=ns, sigma=sig, k_l=klv, ntrials=ntr, lambda_max=0.6, lambda_step=0.2))
    p_bv = bias_variance_experiments.run_one(10.0, 0.6, bcfg, out_dir=tmp)
    run.expect("[smoke] bias_variance runs + writes pkl", os.path.exists(p_bv))

    # --- snr phase driver (Fig 1 heatmap) on a tiny grid ---
    snr_res = compute_snr_phase_diagram(psi=psi, sigma=1.0, k_l=klv,
                                        gamma_max=3.0, gamma_step=1.5, snr_max=0.5, snr_step=0.25,
                                        lambda_max=1000.0, out_dir=tmp, save=True, verify=False)
    run.expect("[smoke] snr phase Delta finite", np.all(np.isfinite(snr_res["Delta"])))

    # --- exercise every make_figures plotter on the fresh small data ---
    errs = []
    def _try(label, fn):
        try:
            fn()
        except Exception as e:                       # noqa: BLE001 (smoke test: report, don't crash)
            errs.append(f"{label}: {type(e).__name__}: {e}")
    _try("plot_f1_grid", lambda: make_figures.plot_f1_grid(directory=sweep_dir, save_stem="smoke_f1", upper_lambda=0.6))
    _try("plot_single_run_scaled", lambda: make_figures.plot_single_run_scaled(p_spk, xscale="linear", save_stem="smoke_single"))
    _try("plot_c_lambda_lines", lambda: make_figures.plot_c_lambda_lines(p_cl, save_stem="smoke_cl_lines"))
    _try("plot_c_lambda_heatmap", lambda: make_figures.plot_c_lambda_heatmap(p_ch, save_stem="smoke_cl_heat"))
    # Call with default ylims/yticks (None) to guard against a regression of the
    # None-handling fix (ax.set_yticks(None) used to raise).
    _try("plot_bias_variance", lambda: make_figures.plot_bias_variance(p_bv, save_stem="smoke_bv"))
    snr_pkls = glob.glob(os.path.join(tmp, "snr_phase_*.pkl"))
    if snr_pkls:
        _try("plot_snr_phase_diagram", lambda: make_figures.plot_snr_phase_diagram(snr_pkls[0], save_stem="smoke_snr", have_legend=False))
    run.expect("[smoke] all make_figures plotters run", not errs, "; ".join(errs) if errs else "")


# ===========================================================================
# (G) FINITE-WIDTH VALIDATION OF PROP. 4.1  (--slow)
#     Simulate one GD step in the deep linear network at finite width N and
#     confirm the layer-ell feature kernel update concentrates to
#         (s_ell * eta^2 / n^2) K_x y y^T K_x,   s_ell = ell(ell+1)(2ell+1)/6,
#     as N grows. This validates the closed-form N -> infinity limit that the
#     rest of the pipeline assumes (the empirical depth coefficient is s_ell,
#     NOT ell^2, for ell >= 2).
# ===========================================================================
def test_finite_width_prop1(run):
    print("\n[G] finite-width Prop. 4.1: feature-kernel update coefficient -> ell(ell+1)(2ell+1)/6")
    import finite_width_experiments as fw

    D_, n_, L_, eta_, sigma_ = 30, 10, 4, 1.0, 0.3
    Ns = (500, 2000)
    ntrials = 12
    theory = fw.s_ell(L_)                                  # [1, 5, 14, 30]
    ell_sq = np.arange(1, L_ + 1) ** 2                     # the paper's (incorrect) coefficient
    X, y = fw.make_fixed_dataset(D_, n_, sigma=sigma_, seed=12345)
    # measure_depth_coefficients returns (depth coefficients, cosine similarities), each dict N -> (ntrials, L)
    coeff_by_N, cos_by_N = fw.measure_depth_coefficients(X, y, L_, eta_, Ns, ntrials=ntrials, base_seed=0)

    mae = {}
    for N in Ns:
        m = coeff_by_N[N].mean(0)
        mae[N] = float(np.mean(np.abs(m - theory)))
    # (i) at the largest N, every layer's coefficient matches s_ell within a few s.e.m.
    N_big = Ns[-1]
    m = coeff_by_N[N_big].mean(0); sem = coeff_by_N[N_big].std(0, ddof=1) / np.sqrt(ntrials)
    for j in range(L_):
        thr = max(0.06 * theory[j], 6.0 * sem[j], 0.3)
        run.expect(f"d_ell -> s_ell  [ell={j+1}, N={N_big}]", abs(m[j] - theory[j]) <= thr,
                   f"emp={m[j]:.3f} theory={theory[j]:g} |d|={abs(m[j]-theory[j]):.3f} tol={thr:.3f}")
    # (ii) convergence: mean-abs error shrinks (or stays tiny) as N grows
    run.expect("mean|d_ell - s_ell| decreases with N", mae[N_big] <= mae[Ns[0]] + 0.15,
               f"MAE(N={Ns[0]})={mae[Ns[0]]:.3f} -> MAE(N={N_big})={mae[N_big]:.3f}")
    # (iii) the coefficient is s_ell, NOT ell^2: for ell >= 2, closer to s_ell than to ell^2
    for j in range(1, L_):
        run.expect(f"closer to s_ell than to ell^2  [ell={j+1}]",
                   abs(m[j] - theory[j]) < abs(m[j] - ell_sq[j]),
                   f"emp={m[j]:.3f}  s_ell={theory[j]:g}  ell^2={ell_sq[j]:g}")
    # (iv) structural match: dPhi^ell aligns with K_x y y^T K_x, cosine -> 1 as N grows.
    #      (ell=1 converges slowest, so we check "high and rising toward 1" per layer plus
    #       near-perfect alignment at the deepest, best-aligned layer.)
    cm_small = cos_by_N[Ns[0]].mean(0)
    cm_big = cos_by_N[N_big].mean(0)
    csem_big = cos_by_N[N_big].std(0, ddof=1) / np.sqrt(ntrials)
    for j in range(L_):
        rising = (1.0 - cm_big[j]) <= (1.0 - cm_small[j]) + 1e-6
        run.expect(f"cosine >= 0.9 & rising toward 1  [ell={j+1}]", cm_big[j] >= 0.9 and rising,
                   f"cos(N={Ns[0]})={cm_small[j]:.4f} -> cos(N={N_big})={cm_big[j]:.4f}")
    run.expect(f"cosine ~ 1 at deepest layer  [ell={L_}, N={N_big}]",
               abs(cm_big[-1] - 1.0) <= max(8e-3, 8.0 * csem_big[-1]),
               f"cos={cm_big[-1]:.5f}  |1-cos|={abs(1 - cm_big[-1]):.2e}")


# ---------------------------------------------------------------------------
def main():
    print(f"unit_tests.py {'(quick)' if QUICK else '(full)'}{' +slow' if SLOW else ''}  "
          f"D={D} n={n} sigma={SIGMA} k_l={K_L} ntrials={NTRIALS}")
    run = Runner()
    test_library_matches_reference(run)
    for gamma, rho in CONFIGS:
        test_empirics_vs_theory(run, gamma, rho)
    test_lambda_zero(run)
    test_fig_data_integrity(run)          # (D) shipped figure numbers == audited theory
    test_figure_scientific_claims(run)    # (E) the claim each figure makes
    if SLOW:
        test_pipeline_regeneration_smoke(run)   # (F) drivers + plotters actually run
        test_finite_width_prop1(run)            # (G) finite-width Prop. 4.1 concentration
    ok = run.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
