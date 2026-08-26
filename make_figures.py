"""
make_figures.py — paper-ready figures from saved sweeps.

Standalone plotting module: it only *reads* the .pkl outputs produced by
`kernel_ridge_regression.py` and writes figure files. It does not import or
modify any part of the estimation/theory pipeline.

Style follows FIGURES.md:
  * f_init (baseline)          -> grey
  * f_feat (feature learning)  -> saturated colour
  * theory                     -> solid line
  * simulation                 -> markers (s.e.m. bars unavailable; see note)
  * every caption states psi, gamma, rho, sigma, beta_coeff, D, n, #draws
  * vector (PDF) + raster (PNG) output

Main entry points:
  plot_f1_grid()      F1-style G-vs-lambda small multiples over the (gamma, rho) sweep
  plot_isotropic()    G-vs-lambda for the isotropic (gamma=0) runs, if present

Run `python make_figures.py` to regenerate all figures.
"""

import os
import re
import glob
import pickle

from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from stieltjes_asymptotics import min_gen_error_over_lambda


HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(HERE, "new_spiked_sweep")                  # clean parallel run
SPIKED_DIR = os.path.join(HERE, "spiked_covariance_experiments")  # legacy runs
CLAMBDA_DIR = os.path.join(HERE, "c_lambda_data")                 # c_lambda study data
BV_DIR = os.path.join(HERE, "bias_variance_data")                 # bias/variance study data
SNR_PHASE_DIR = os.path.join(HERE, "snr_phase_data")             # SNR phase-diagram data
JOINT_PHASE_DIR = os.path.join(HERE, "joint_phase_data")         # joint (lambda,k_l) optimum
OUT_DIR = os.path.join(HERE, "paper_figures")

# ---- colour / style semantics (consistent across every figure) -------------
INIT_COLOR = "#E05C5C"   # red  : baseline  f_init
FEAT_COLOR = "#0072B2"   # blue  : feature-learning  f_feat  (colourblind-safe)

# per-k_l colours for the c_lambda line plots (colourblind-safe)
# KL_COLORS = {0.0: "#6E6E6E", 1.0: "#0072B2", 10.0: "#D55E00"}

# make a plt.cm.cool colormap that starts at one end when k_l = 0 and is at the other end when k_l = 10
KL_CMAP = plt.cm.cool
KL_COLORS = {k_l: KL_CMAP(1. - k_l / 10.0) for k_l in [0.0, 0.5, 1.0, 5.0, 10.0]}




def set_paper_style():
    """Global rcParams for column-width ML-theory figures (>= 8pt, vector-friendly)."""
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,   # embed as TrueType so text stays editable in the PDF
        "ps.fonttype": 42,
    })


# ---- data loading ----------------------------------------------------------
_SPIKED_RE = re.compile(r"spike_strength=([0-9.]+)_rho=([0-9.]+)")


def load_spiked_runs(directory=None):
    """
    Return {(gamma, rho): results_dict} for every saved spiked run.

    Prefers the clean parallel sweep in spiked_sweep/ (new schema with a dense
    theory grid + s.e.m.); falls back to the legacy spiked_covariance_experiments/.
    """
    if directory is None:
        directory = SWEEP_DIR if glob.glob(os.path.join(SWEEP_DIR, "spiked_*.pkl")) else SPIKED_DIR
    runs = {}
    for pattern in ("spiked_*.pkl", "asymptotics_*.pkl"):
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            with open(path, "rb") as fh:
                d = pickle.load(fh)
            runs[(float(d["spike_strength"]), float(d["rho"]))] = d
    return runs, directory


def load_isotropic_runs(directory=HERE):
    """Return list of results_dicts for the isotropic (gamma=0) runs."""
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "isotropic_*ntrials*.pkl"))):
        with open(path, "rb") as fh:
            out.append(pickle.load(fh))
    return out


def _lambda_star(lambdas, G):
    """(lambda*, G*) minimiser of a theory curve."""
    i = int(np.nanargmin(G))
    return lambdas[i], G[i]


def _draw_lambda_star_tick(ax, lam, G, color):
    """Small upward caret at the bottom spine marking a curve's minimiser."""
    ax.plot([lam], [-0.02], marker="^", markersize=5, color=color,
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)
    ax.plot([-0.02], [G], marker=">", markersize=5, color=color,
            transform=ax.get_yaxis_transform(), clip_on=False, zorder=6)
    # ax.plot([0.0, lam], [G, G], linestyle='dashed', color=color, zorder=2, lw=1.5)
    # ax.plot([lam, lam], [0.0, G], linestyle='dashed', color=color, zorder=2, lw=1.5)
    ax.scatter(lam, G, color=color, zorder=6, marker='*', s=150, edgecolor='black', clip_on=False)


def _tight_ylim_with_headroom(ax, xlim_max, series, bottom_pad_frac=0.06,
                              top_pad_frac=0.06, headroom_frac=0.22):
    """
    Set a per-panel y-limit tight around the curves actually visible in
    [0, xlim_max], plus reserved headroom at the top for an annotation.

    series: list of (x_array, y_array) pairs (y_array may itself already
    include +/- s.e.m. bounds by passing y +/- yerr as separate series).
    Returns the axes-fraction y-position that sits safely inside the
    reserved headroom band (for placing text with transform=ax.transAxes).
    """
    ys = []
    for x, y in series:
        x = np.asarray(x)
        y = np.asarray(y)
        mask = x <= xlim_max + 1e-9
        if mask.any():
            ys.append(y[mask])
    y_all = np.concatenate(ys)
    y_all = y_all[np.isfinite(y_all)]
    ymin, ymax = float(y_all.min()), float(y_all.max())
    yrange = max(ymax - ymin, 1e-9)

    bottom = ymin - bottom_pad_frac * yrange
    top = ymax + top_pad_frac * yrange + headroom_frac * yrange
    ax.set_ylim(bottom, top)

    total = top - bottom
    # midpoint of the reserved headroom band, in axes-fraction coordinates
    band_bottom_frac = (ymax + top_pad_frac * yrange - bottom) / total
    annot_frac = band_bottom_frac + 0.5 * (1.0 - band_bottom_frac)
    return annot_frac


# ---- F1: G vs lambda, theory vs simulation, across (gamma, rho) -------------
def plot_f1_grid(gammas=None, rhos=None, markevery=8,
                 save_stem="F1_spiked_gen_error", upper_lambda=1.0):
    """
    Small-multiples grid: rows = gamma, cols = rho. Per panel, four series:
    init/feat theory (solid, grey/blue) + init/feat empirics (markers).
    Shared y-axis within each row so cross-panel comparison is honest.
    """
    set_paper_style()
    runs, directory = load_spiked_runs()
    if not runs:
        raise FileNotFoundError(f"No spiked .pkl runs found in {SWEEP_DIR} or {SPIKED_DIR}")

    if gammas is None:
        gammas = sorted({g for (g, _) in runs})
    if rhos is None:
        available = {r for (_, r) in runs}
        preferred = [0.0, 0.2, 0.6, 1.0]
        rhos = [r for r in preferred if r in available] or sorted(available)

    nrow, ncol = len(gammas), len(rhos)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                             sharex=True, sharey=False, squeeze=False)

    # metadata for the caption (assumed constant across the sweep)
    meta = next(iter(runs.values()))
    D, n = meta["D"], meta["n"]
    sigma = meta["noise_std"]
    k_l = meta.get("k_l", meta.get("beta_coeff"))  # new schema uses k_l
    psi = n / D

    for i, g in enumerate(gammas):
        for j, r in enumerate(rhos):
            print(i, j)
            print(g, r)
            ax = axes[i][j]
            d = runs.get((g, r))
            if d is None:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                        transform=ax.transAxes, color="0.6", fontsize=8)
                continue

            lam_th = d.get("lambdas_theory", d["lambdas"])   # dense grid if present
            lam = d["lambdas"]                                # empirical (marker) grid
            # theory: solid lines on the dense grid
            ax.plot(lam_th, d["init_gen_error_theory"], color=INIT_COLOR, zorder=3)
            ax.plot(lam_th, d["feat_gen_error_theory"], color=FEAT_COLOR, zorder=4)
            # simulation: markers, with s.e.m. bars when per-trial data was retained
            init_sem = d.get("init_gen_errors_sem")
            feat_sem = d.get("feat_gen_errors_sem")
            me = markevery if len(lam) > 20 else 1
            ax.errorbar(lam, d["init_gen_errors"], yerr=init_sem, linestyle="none",
                        marker="o", markersize=3.6, markerfacecolor="none",
                        markeredgecolor=INIT_COLOR, markeredgewidth=0.9,
                        ecolor=INIT_COLOR, elinewidth=0.8, capsize=1.5,
                        markevery=me, errorevery=me, zorder=3)
            ax.errorbar(lam, d["feat_gen_errors"], yerr=feat_sem, linestyle="none",
                        marker="s", markersize=3.4, markerfacecolor="none",
                        markeredgecolor=FEAT_COLOR, markeredgewidth=0.9,
                        ecolor=FEAT_COLOR, elinewidth=0.8, capsize=1.5,
                        markevery=me, errorevery=me, zorder=4)

            # minimiser ticks (from the dense theory grid)
            # _draw_lambda_star_tick(ax, _lambda_star(lam_th, d["init_gen_error_theory"])[0], INIT_COLOR)
            # _draw_lambda_star_tick(ax, _lambda_star(lam_th, d["feat_gen_error_theory"])[0], FEAT_COLOR)
            min_G_init, min_lam_init = min_gen_error_over_lambda(psi, g, r, sigma, 0.)
            min_G_feat, min_lam_feat = min_gen_error_over_lambda(psi, g, r, sigma, k_l)

            _draw_lambda_star_tick(ax, min_lam_init, min_G_init, INIT_COLOR)
            _draw_lambda_star_tick(ax, min_lam_feat, min_G_feat, FEAT_COLOR)

            ax.set_xlim(0, upper_lambda)

            # tight, per-panel y-limits around only the curves visible in [0, upper_lambda],
            # with headroom reserved at the top so the SNR annotation never sits on a curve
            init_lo = d["init_gen_errors"] - (init_sem if init_sem is not None else 0.0)
            init_hi = d["init_gen_errors"] + (init_sem if init_sem is not None else 0.0)
            feat_lo = d["feat_gen_errors"] - (feat_sem if feat_sem is not None else 0.0)
            feat_hi = d["feat_gen_errors"] + (feat_sem if feat_sem is not None else 0.0)
            annot_y = _tight_ylim_with_headroom(
                ax, upper_lambda,
                series=[(lam_th, d["init_gen_error_theory"]), (lam_th, d["feat_gen_error_theory"]),
                        (lam, init_lo), (lam, init_hi), (lam, feat_lo), (lam, feat_hi)],
            )

            # SNR annotation (rho^2 / sigma^2), placed in the reserved headroom band
            snr = (r ** 2) / (sigma ** 2)
            ax.text(0.96, annot_y, rf"$\rho^2/\sigma^2={snr:.2g}$",
                    ha="right", va="center", transform=ax.transAxes, fontsize=9.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

            ax.margins(x=0.02)
            if i == 0:
                ax.set_title(rf"$\rho={r:g}$", fontsize=16)
            if j == 0:
                ax.set_ylabel(rf"$\gamma={g:g}$", fontsize=16)
            if i == nrow - 1:
                ax.set_xlabel(r"ridge $\lambda$")

    # single figure-level legend
    handles = [
        Line2D([0], [0], color=INIT_COLOR, label=r"$\hat f_{\mathrm{init}}$ theory"),
        Line2D([0], [0], color=FEAT_COLOR, label=r"$\hat f_{\mathrm{feat}}$ theory"),
        Line2D([0], [0], color=INIT_COLOR, marker="o", markerfacecolor="none",
               linestyle="none", label=r"$\hat f_{\mathrm{init}}$ sim. ($\pm$s.e.m.)"),
        Line2D([0], [0], color=FEAT_COLOR, marker="s", markerfacecolor="none",
               linestyle="none", label=r"$\hat f_{\mathrm{feat}}$ sim. ($\pm$s.e.m.)"),
        Line2D([0], [0], color="0.4", marker="^", linestyle="none",
               label=r"$\lambda^\star$ (theory min.)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=13)

    ntrials = meta.get("ntrials", _infer_ntrials(directory))
    seed = meta.get("seed", "?")
    caption = (
        rf"$G$ vs $\lambda$, theory (solid) vs simulation (markers, $\pm$1 s.e.m.). "
        rf"$\psi={psi:g}$, $D={D}$, $n={n}$, $\sigma={sigma:g}$, "
        rf"$k_\ell={k_l:g}$; {ntrials} dataset draws; seed$={seed}$. "
        rf"Linear $\lambda$ grid."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.4, color="0.25")

    fig.tight_layout(rect=[0, 0.0, 1, 0.98])
    _savefig(fig, save_stem)
    plt.close(fig)


def _infer_ntrials(directory):
    for path in glob.glob(os.path.join(directory, "*ntrials=*.pkl")):
        m = re.search(r"ntrials=(\d+)", path)
        if m:
            return int(m.group(1))
    return "?"


# ---- scale-flexible single run (linear / log / symlog prototypes) ----------
def _log_marker_indices(lam, per_decade=6, lam_lo=None):
    """Indices into `lam` closest to log-uniform targets (~per_decade each)."""
    pos = lam > 0
    lo = float(lam[pos].min()) if lam_lo is None else float(lam_lo)
    hi = float(lam.max())
    ndec = max(np.log10(hi / lo), 1e-9)
    ntarget = max(2, int(np.ceil(ndec * per_decade)) + 1)
    targets = np.geomspace(lo, hi, ntarget)
    idx = sorted({int(np.argmin(np.abs(lam - t))) for t in targets})
    return np.array(idx, dtype=int)


def _theory_on_grid(d, lam_grid):
    """Recompute the two theory curves on an arbitrary lambda grid (closed form)."""
    from stieltjes_asymptotics import compute_spiked_covariance_model_bias_and_variance
    D, n = d["D"], d["n"]
    k_l, gamma, rho, sigma = d.get("k_l"), d["spike_strength"], d["rho"], d["noise_std"]
    gi = np.array([compute_spiked_covariance_model_bias_and_variance(
        n, D, 0.0, l, gamma, rho, sigma)[2] for l in lam_grid])
    gf = np.array([compute_spiked_covariance_model_bias_and_variance(
        n, D, k_l, l, gamma, rho, sigma)[2] for l in lam_grid])
    return gi, gf


def plot_single_run_scaled(pkl_path, xscale="symlog", markers_per_decade=6,
                           marker_lambda_spacing=0.08, upper_lambda=None,
                           ymax=None, show_scale_in_title=True,
                           out_dir=None, save_stem=None, use_me=True):
    """
    G vs lambda for one saved run, with a selectable x-scale:
      'linear' -- as before (uniform marker spacing in lambda)
      'log'    -- log lambda, cut off at the smallest nonzero simulated lambda
      'symlog' -- linear below linthresh (so lambda = 0 is kept), log above

    Theory is recomputed on a dense log-spaced grid so the curves stay smooth at
    small lambda; simulation markers are snapped to log-uniform targets.
    """
    set_paper_style()
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)

    D, n = d["D"], d["n"]
    sigma, k_l = d["noise_std"], d.get("k_l")
    gamma, rho = d["spike_strength"], d["rho"]
    psi = n / D
    already_using_log = d.get("use_log_spaced_lambdas", False)
    print(f'Already using log-spaced lambdas? {already_using_log}')

    lam = d["lambdas"]
    if upper_lambda is None:
        upper_lambda = float(lam.max())
    lam_lo = float(lam[lam > 0].min())          # left cutoff / symlog threshold

    # ---- theory grid ----
    if xscale == "linear":
        lam_th = d.get("lambdas_theory", lam)
        th_i, th_f = d["init_gen_error_theory"], d["feat_gen_error_theory"]
    else:
        lam_th = np.geomspace(lam_lo, upper_lambda, 10_000)
        if xscale == "symlog":                   # add the linear stretch incl. 0
            lam_th = np.unique(np.concatenate([np.linspace(0.0, lam_lo, 1000), lam_th]))
        th_i, th_f = _theory_on_grid(d, lam_th)

    me = 1
    # ---- marker indices ----
    if xscale == "linear":
        step = float(np.median(np.diff(lam))) if len(lam) > 1 else marker_lambda_spacing
        me = max(1, round(marker_lambda_spacing / step))
    else:
        if use_me:
            me = list(_log_marker_indices(lam, markers_per_decade, lam_lo))
            # symlog displays lambda = 0, so mark that simulated point too (pure log
            # cannot show it). At lambda = 0 we have c_lambda = 1 exactly, so the init
            # and feat markers coincide -- which is the point worth showing.
            if xscale == "symlog" and lam[0] == 0.0 and 0 not in me:
                me = [0] + me

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(lam_th, th_i, color=INIT_COLOR, zorder=3)
    ax.plot(lam_th, th_f, color=FEAT_COLOR, zorder=4)

    init_sem, feat_sem = d.get("init_gen_errors_sem"), d.get("feat_gen_errors_sem")
    ax.errorbar(lam, d["init_gen_errors"], yerr=init_sem, linestyle="none",
                marker="o", markersize=4.5, markerfacecolor="none",
                markeredgecolor=INIT_COLOR, markeredgewidth=1.0,
                ecolor=INIT_COLOR, elinewidth=0.9, capsize=2.0,
                markevery=me, errorevery=me, zorder=3)
    ax.errorbar(lam, d["feat_gen_errors"], yerr=feat_sem, linestyle="none",
                marker="s", markersize=4.2, markerfacecolor="none",
                markeredgecolor=FEAT_COLOR, markeredgewidth=1.0,
                ecolor=FEAT_COLOR, elinewidth=0.9, capsize=2.0,
                markevery=me, errorevery=me, zorder=4)

    min_G_init, min_lam_init = min_gen_error_over_lambda(psi, gamma, rho, sigma, 0.)
    min_G_feat, min_lam_feat = min_gen_error_over_lambda(psi, gamma, rho, sigma, k_l)
    _draw_lambda_star_tick(ax, min_lam_init, min_G_init, INIT_COLOR)
    _draw_lambda_star_tick(ax, min_lam_feat, min_G_feat, FEAT_COLOR)

    if xscale == "log":
        ax.set_xscale("log")
        ax.set_xlim(lam_lo, upper_lambda)
        x_lo = lam_lo
    elif xscale == "symlog":
        ax.set_xscale("symlog", linthresh=lam_lo, linscale=0.5)
        ax.set_xlim(0.0, upper_lambda)
        x_lo = 0.0
    else:
        ax.set_xlim(0.0, upper_lambda)
        x_lo = 0.0

    # tight y-limits over only the visible x-range
    def _vis(x, y):
        x, y = np.asarray(x), np.asarray(y)
        m = (x >= x_lo - 1e-12) & (x <= upper_lambda + 1e-9)
        return y[m]
    ys = [_vis(lam_th, th_i), _vis(lam_th, th_f)]
    for arr, s in ((d["init_gen_errors"], init_sem), (d["feat_gen_errors"], feat_sem)):
        s = s if s is not None else 0.0
        ys += [_vis(lam, arr - s), _vis(lam, arr + s)]
    yall = np.concatenate(ys); yall = yall[np.isfinite(yall)]
    lo_y, hi_y = float(yall.min()), float(yall.max())
    if ymax is not None:                       # crop the top; keep the data-driven floor
        hi_y = float(ymax)
    pad = 0.06 * max(hi_y - lo_y, 1e-9)
    ax.set_ylim(lo_y - pad, hi_y if ymax is not None else hi_y + pad)

    ax.set_xlabel(r"Ridge ($\lambda$)")
    ax.set_ylabel("Generalization Error")
    title = rf"$\gamma={gamma:g}$, $\rho={rho:g}$"
    if show_scale_in_title:
        title += rf"  [{xscale}]"
    ax.set_title(title)
    ax.tick_params(axis='both', labelsize=18)

    fig.tight_layout()
    if save_stem is None:
        save_stem = f"G_gamma={gamma:g}_rho={rho:g}_D={D}_n={n}_sigma={sigma:g}_{xscale}"

        if already_using_log:
            save_stem += "_loglam"
    target = out_dir or OUT_DIR
    os.makedirs(target, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(target, f"{save_stem}.{ext}"))
    plt.close(fig)
    print(f"[saved] {os.path.join(target, save_stem)}.pdf / .png")
    return os.path.join(target, f"{save_stem}.pdf")


# ---- single-run: G vs lambda for one saved (gamma, rho) pkl ----------------
def plot_single_run(pkl_path, marker_lambda_spacing=0.08, upper_lambda=None,
                    save_stem=None):
    """
    G vs lambda for a single saved run: init vs feat, theory (solid) vs
    empirics (markers +/- s.e.m.). Marker spacing is given in lambda units
    (not index count) so it's robust to whatever lambda_step the file used.
    """
    set_paper_style()
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)

    D, n = d["D"], d["n"]
    sigma = d["noise_std"]
    k_l = d.get("k_l")
    gamma, rho = d["spike_strength"], d["rho"]
    psi = n / D

    lam_th = d.get("lambdas_theory", d["lambdas"])
    lam = d["lambdas"]
    if upper_lambda is None:
        upper_lambda = float(lam.max())

    # convert the requested lambda-unit spacing into an index stride
    lam_step = float(np.median(np.diff(lam))) if len(lam) > 1 else marker_lambda_spacing
    me = max(1, round(marker_lambda_spacing / lam_step))

    if marker_lambda_spacing == 3:
        indices = np.array([0, 2, 4, 5, 6, 9, 15, 30, 50, 70, 90])
        next_indices = np.arange(90, len(lam), me)
        me = np.concatenate([indices, next_indices])

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    ax.plot(lam_th, d["init_gen_error_theory"], color=INIT_COLOR, zorder=3,
            label=r"$\hat f_{\mathrm{init}}$ theory")
    ax.plot(lam_th, d["feat_gen_error_theory"], color=FEAT_COLOR, zorder=4,
            label=r"$\hat f_{\mathrm{feat}}$ theory")

    init_sem = d.get("init_gen_errors_sem")
    feat_sem = d.get("feat_gen_errors_sem")
    ax.errorbar(lam, d["init_gen_errors"], yerr=init_sem, linestyle="none",
                marker="o", markersize=4.5, markerfacecolor="none",
                markeredgecolor=INIT_COLOR, markeredgewidth=1.0,
                ecolor=INIT_COLOR, elinewidth=0.9, capsize=2.0,
                markevery=me, errorevery=me, zorder=3,
                label=r"$\hat f_{\mathrm{init}}$ sim. ($\pm$s.e.m.)")
    ax.errorbar(lam, d["feat_gen_errors"], yerr=feat_sem, linestyle="none",
                marker="s", markersize=4.2, markerfacecolor="none",
                markeredgecolor=FEAT_COLOR, markeredgewidth=1.0,
                ecolor=FEAT_COLOR, elinewidth=0.9, capsize=2.0,
                markevery=me, errorevery=me, zorder=4,
                label=r"$\hat f_{\mathrm{feat}}$ sim. ($\pm$s.e.m.)")
    
    min_G_init, min_lam_init = min_gen_error_over_lambda(psi, gamma, rho, sigma, 0.)
    min_G_feat, min_lam_feat = min_gen_error_over_lambda(psi, gamma, rho, sigma, k_l)

    # ax.scatter([min_G_init], [min_lam_init], color=INIT_COLOR, zorder=5, marker='*', s=130, edgecolor='black', clip_on=False)

    _draw_lambda_star_tick(ax, min_lam_init, min_G_init, INIT_COLOR)
    _draw_lambda_star_tick(ax, min_lam_feat, min_G_feat, FEAT_COLOR)

    ax.set_xlim(0, upper_lambda)
    init_lo = d["init_gen_errors"] - (init_sem if init_sem is not None else 0.0)
    init_hi = d["init_gen_errors"] + (init_sem if init_sem is not None else 0.0)
    feat_lo = d["feat_gen_errors"] - (feat_sem if feat_sem is not None else 0.0)
    feat_hi = d["feat_gen_errors"] + (feat_sem if feat_sem is not None else 0.0)
    _tight_ylim_with_headroom(
        ax, upper_lambda,
        series=[(lam_th, d["init_gen_error_theory"]), (lam_th, d["feat_gen_error_theory"]),
                (lam, init_lo), (lam, init_hi), (lam, feat_lo), (lam, feat_hi)],
        headroom_frac=0.06,
    )

    ax.set_xlabel(r"Ridge ($\lambda$)")
    ax.set_ylabel("Generalization Error")
    ax.set_title(rf"$\gamma={gamma:g}$, $\rho={rho:g}$")
    ax.tick_params(axis='both', labelsize=18)
    # ax.legend(frameon=False, loc="best")

    # ntrials, seed = d.get("ntrials", "?"), d.get("seed", "?")
    # caption = (
    #     rf"$\psi={psi:g}$, $D={D}$, $n={n}$, $\sigma={sigma:g}$, $k_\ell={k_l:g}$; "
    #     rf"{ntrials} dataset draws; seed$={seed}$. Markers every $\Delta\lambda\approx{marker_lambda_spacing:g}$."
    # )
    # fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=8.0, color="0.25")

    fig.tight_layout()
    if save_stem is None:
        save_stem = f"G_gamma={gamma:g}_rho={rho:g}_D={D}_n={n}_sigma={sigma:g}"
    _savefig(fig, save_stem)
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


def save_single_run_legend(save_stem="single_run_legend"):
    """
    Standalone legend for plot_single_run's four series (theory init/feat,
    sim init/feat), matching its exact colors/markers, as its own figure.
    """
    set_paper_style()
    handles = [
        Line2D([0], [0], color=INIT_COLOR, lw=2.0, label=r"$\hat f_{\mathrm{init}}$ theory"),
        Line2D([0], [0], color=FEAT_COLOR, lw=2.0, label=r"$\hat f_{\mathrm{feat}}$ theory"),
        Line2D([0], [0], color=INIT_COLOR, marker="o", markersize=4.5,
               markerfacecolor="none", markeredgewidth=1.0, linestyle="none",
               label=r"$\hat f_{\mathrm{init}}$ sim. ($\pm$s.e.m.)"),
        Line2D([0], [0], color=FEAT_COLOR, marker="s", markersize=4.2,
               markerfacecolor="none", markeredgewidth=1.0, linestyle="none",
               label=r"$\hat f_{\mathrm{feat}}$ sim. ($\pm$s.e.m.)"),
    ]
    fig = plt.figure(figsize=(3.0, 1.6))
    fig.legend(handles=handles, loc="center", frameon=False,
              alignment="left", fontsize=12)
    _savefig(fig, save_stem)
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


# ---- isotropic (gamma = 0) runs -------------------------------------------
def plot_isotropic(save_stem="isotropic_gen_error"):
    """G vs lambda for the isotropic runs (theory solid, simulation markers)."""
    set_paper_style()
    runs = load_isotropic_runs()
    if not runs:
        print("[plot_isotropic] no isotropic runs found; skipping.")
        return

    # pool empirical means across ntrials variants by trial-weighting when possible
    d = max(runs, key=lambda r: r["init_gen_errors"].size)
    lam = d["lambdas"]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(lam, d["init_gen_error_theory"], color=INIT_COLOR, label=r"$\hat f_{\mathrm{init}}$ theory")
    ax.plot(lam, d["feat_gen_error_theory"], color=FEAT_COLOR, label=r"$\hat f_{\mathrm{feat}}$ theory")
    ax.plot(lam, d["init_gen_errors"], linestyle="none", marker="o", markersize=4,
            markerfacecolor="none", markeredgecolor=INIT_COLOR, label=r"$\hat f_{\mathrm{init}}$ sim.")
    ax.plot(lam, d["feat_gen_errors"], linestyle="none", marker="s", markersize=4,
            markerfacecolor="none", markeredgecolor=FEAT_COLOR, label=r"$\hat f_{\mathrm{feat}}$ sim.")

    psi = d["n"] / d["D"]
    ax.set_xlabel(r"ridge $\lambda$")
    ax.set_ylabel(r"$G(\hat w)$")
    ax.set_title(rf"Isotropic ($\gamma=0$): $\psi={psi:g}$, $\sigma={d['noise_std']:g}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    _savefig(fig, save_stem)
    plt.close(fig)


# ---- c_lambda study: line plots (theory + empirics, several k_l) -----------
def _kl_color(k_l, idx):
    if k_l in KL_COLORS:
        return KL_COLORS[k_l]
    return plt.cm.viridis(0.15 + 0.7 * idx / max(1, len(KL_COLORS)))


def plot_c_lambda_lines(pkl_path, mark_every_lambda=0.1, upper_lambda=None,
                        ylim=None, save_stem=None):
    """
    c_lambda vs lambda for several k_l: theory (solid) + empirics (markers,
    +/- s.e.m.). Marker spacing given in lambda units (toggleable).
    """
    set_paper_style()
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)

    lam = d["lambdas"]
    k_ls = d["k_ls"]
    gamma, rho, sigma = d["gamma"], d["rho"], d["sigma"]
    D, n, psi = d["D"], d["n"], d["psi"]
    ntrials, seed = d.get("ntrials", "?"), d.get("seed", "?")
    if upper_lambda is None:
        upper_lambda = float(lam.max())

    lam_step = float(np.median(np.diff(lam))) if len(lam) > 1 else mark_every_lambda
    me = max(1, round(mark_every_lambda / lam_step))

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    marker_cycle = ["o", "s", "D", "^", "v"]
    for a, k_l in enumerate(k_ls):
        color = _kl_color(float(k_l), a)
        print(d['c_emp_sem'][a])
        ax.plot(lam, d["c_theory"][a], color=color, zorder=3)
        ax.errorbar(lam, d["c_emp_mean"][a], yerr=d["c_emp_sem"][a], linestyle="none",
                    marker=marker_cycle[a % len(marker_cycle)], markersize=4.5,
                    markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.0,
                    ecolor=color, elinewidth=0.9, capsize=2.0,
                    markevery=me, errorevery=me, zorder=5)

    ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)  # c_lambda = 1 floor
    ax.set_xlim(0, upper_lambda)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(r"ridge $\lambda$")
    ax.set_ylabel(r"$c_\lambda$")
    title = rf"$\gamma={gamma:g}$, $\rho={rho:g}$"
    ax.set_title(title)

    ax.tick_params(axis='both', labelsize=16)

    # legend: one colour entry per k_l, plus theory/sim style key
    handles = [Line2D([0], [0], color=_kl_color(float(k_l), a),
                      marker=marker_cycle[a % len(marker_cycle)], markerfacecolor="none",
                      label=rf"$k_\ell={k_l:g}$")
               for a, k_l in enumerate(k_ls)]
    # handles += [
    #     Line2D([0], [0], color="0.2", label="theory"),
    #     Line2D([0], [0], color="0.2", marker="o", markerfacecolor="none",
    #            linestyle="none", label=r"sim. ($\pm$s.e.m.)"),
    # ]
    fig_legend = plt.figure(figsize=(10.4, 1.8))
    fig_legend.legend(handles=handles, loc="center", frameon=True, framealpha=0.92,
                    edgecolor="0.7", fontsize=14, ncol=1)
    _savefig(fig_legend, f"c_lambda_plot_legend")
    plt.close(fig_legend)
    # ax.legend(handles=handles, frameon=False, loc="upper left", ncol=1)

    caption = (rf"$\psi={psi:g}$, $D={D}$, $n={n}$, $\sigma={sigma:g}$; "
               rf"{ntrials} draws; seed$={seed}$. Markers every "
               rf"$\Delta\lambda\approx{mark_every_lambda:g}$.")
    fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=8.0, color="0.25")

    fig.tight_layout()
    if save_stem is None:
        save_stem = f"c_lambda_lines_gamma={gamma:g}_rho={rho:g}"
    _savefig(fig, save_stem)
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


# ---- c_lambda study: theory heatmap over (lambda, k_l) ---------------------
def plot_c_lambda_heatmap(pkl_path, cmap="viridis", vmin=1.0, vmax=None, save_stem=None):
    """Heatmap of theory c_lambda over (lambda, k_l), floor anchored at c=1.

    Pass a shared `vmax` to make multiple heatmaps directly comparable.
    """
    set_paper_style()
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)

    lam, k_ls, C = d["lambdas"], d["k_ls"], d["c_theory"]
    gamma, rho, sigma = d["gamma"], d["rho"], d["sigma"]
    D, n, psi = d["D"], d["n"], d["psi"]

    if vmax is None:
        vmax = float(C.max())
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    pcm = ax.pcolormesh(lam, k_ls, C, cmap=cmap, vmin=vmin, vmax=vmax,
                        shading="gouraud")

    # set ax tick size
    ax.tick_params(axis='both', labelsize=16)
    
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"$c_\lambda$")

    # set tick labels font size
    cbar.ax.tick_params(labelsize=16)

    # allow for at most 4 or 5 ticks on cbar
    cbar.locator = MaxNLocator(nbins=5)

    # set cbar ticks manually
    cbar.set_ticks(np.array([1.0, 1.5, 2.0, 2.5]))
    

    ax.set_xlabel(r"ridge $\lambda$")
    ax.set_ylabel(r"$k_\ell$")
    ax.set_title(rf"$c_\lambda$ theory:  $\gamma={gamma:g}$, $\rho={rho:g}$")
    ax.grid(False)

    caption = rf"$\psi={psi:g}$, $D={D}$, $n={n}$, $\sigma={sigma:g}$."
    fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=8.0, color="0.25")

    fig.tight_layout()
    if save_stem is None:
        save_stem = f"c_lambda_heatmap_gamma={gamma:g}_rho={rho:g}"
    _savefig(fig, save_stem)
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


def _shared_clambda_line_ylim(paths, pad=0.04):
    """Common y-range covering theory + empirics (+/- s.e.m.) across all line runs."""
    lo, hi = np.inf, -np.inf
    for p in paths:
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        for a in (d["c_theory"], d["c_emp_mean"] - d["c_emp_sem"],
                  d["c_emp_mean"] + d["c_emp_sem"]):
            lo = min(lo, float(np.nanmin(a)))
            hi = max(hi, float(np.nanmax(a)))
    rng = max(hi - lo, 1e-9)
    return (lo - pad * rng, hi + pad * rng)


def _shared_clambda_heatmap_vmax(paths):
    """Common colorbar upper bound across all heatmap runs."""
    vmax = -np.inf
    for p in paths:
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        vmax = max(vmax, float(np.nanmax(d["c_theory"])))
    return vmax


def plot_all_c_lambda(directory=CLAMBDA_DIR, mark_every_lambda=0.1, shared=False):
    """
    Render every c_lambda line and heatmap dataset found in `directory`.

    Each plot autoscales to its own data by default. Pass shared=True to make
    all line plots use a common y-range and all heatmaps a common colorbar
    (vmin=1, shared vmax), so the two configs are directly comparable.
    """
    line_paths = sorted(glob.glob(os.path.join(directory, "c_lambda_lines_*.pkl")))
    heat_paths = sorted(glob.glob(os.path.join(directory, "c_lambda_heatmap_*.pkl")))

    ylim = _shared_clambda_line_ylim(line_paths) if (shared and line_paths) else None
    vmax = _shared_clambda_heatmap_vmax(heat_paths) if (shared and heat_paths) else None

    for path in line_paths:
        plot_c_lambda_lines(path, mark_every_lambda=mark_every_lambda, ylim=ylim)
    for path in heat_paths:
        plot_c_lambda_heatmap(path, vmax=vmax)


# ---- bias / variance / gen-error decomposition vs lambda -------------------
# color = estimator (grey init / blue feat); linestyle & marker = quantity
_BV_QUANTITIES = [
    # key_stem, label,          linestyle,     marker
    ("gen_error", "Gen. error", "-",            "o"),
    ("bias",      "Bias",       (0, (5, 2)),    "^"),
    ("variance",  "Variance",   (0, (1, 1.4)),  "D"),
]


def plot_bias_variance(pkl_path, mark_every_lambda=0.1, upper_lambda=None, save_stem=None):
    """
    Bias, variance, and generalization error vs lambda: theory (lines) + empirics
    (markers), for baseline (k_l=0) and feature-learning (k_l) estimators.
    Colour encodes estimator, linestyle/marker encodes quantity. Both theory
    gen-error minima are highlighted with a star + dotted guides to each axis.
    """
    set_paper_style()
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)

    lam = d["lambdas"]
    if upper_lambda is None:
        upper_lambda = float(lam.max())
    lam_step = float(np.median(np.diff(lam))) if len(lam) > 1 else mark_every_lambda
    me = max(1, round(mark_every_lambda / lam_step))

    D, n, psi, sigma = d["D"], d["n"], d["psi"], d["sigma"]
    k_l, gamma, rho = d["k_l"], d["gamma"], d["rho"]
    ntrials, seed = d.get("ntrials", "?"), d.get("seed", "?")

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    estimators = [("init", INIT_COLOR, rf"$k_\ell=0$"),
                  ("feat", FEAT_COLOR, rf"$k_\ell={k_l:g}$")]

    for est, color, _label in estimators:
        for stem, _qlabel, ls, marker in _BV_QUANTITIES:
            y_th = d[f"{est}_{stem}_theory"]
            y_emp = d[f"{est}_{stem}_emp"]
            ax.plot(lam, y_th, color=color, linestyle=ls,
                    lw=2.0 if stem == "gen_error" else 1.6, zorder=3)
            if stem == "gen_error":
                ax.errorbar(lam, y_emp, yerr=d[f"{est}_gen_error_sem"], linestyle="none",
                            marker=marker, markersize=4.3, markerfacecolor="none",
                            markeredgecolor=color, markeredgewidth=1.0,
                            ecolor=color, elinewidth=0.8, capsize=1.5,
                            markevery=me, errorevery=me, zorder=4)
            else:
                ax.plot(lam, y_emp, linestyle="none", marker=marker, markersize=4.3,
                        markerfacecolor="none", markeredgecolor=color,
                        markeredgewidth=1.0, markevery=me, zorder=4)

    ax.set_xlim(0, upper_lambda)
    ax.set_ylim(bottom=0.0)   # anchor at the x-axis so minima guides can reach it

    # highlight theory gen-error minima (both estimators): star + dotted guides
    # drawn to y=0 (the x-axis) and x=0 (the y-axis)
    for est, color, _label in estimators:
        G = d[f"{est}_gen_error_theory"]
        mask = lam <= upper_lambda + 1e-9
        idx = int(np.nanargmin(np.where(mask, G, np.inf)))
        lstar, gstar = lam[idx], G[idx]
        ax.plot([lstar, lstar], [0.0, gstar], color=color, ls=":", lw=1.0, zorder=2)
        ax.plot([0, lstar], [gstar, gstar], color=color, ls=":", lw=1.0, zorder=2)
        ax.plot([lstar], [gstar], marker="*", markersize=12, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=6)

    ax.set_xlabel(r"ridge $\lambda$")

    caption = (rf"$\psi={psi:g}$, $D={D}$, $n={n}$, $\sigma={sigma:g}$, $k_\ell={k_l:g}$; "
               rf"{ntrials} draws; seed$={seed}$. $\star$ = theory $G$ minimum. "
               rf"Markers every $\Delta\lambda\approx{mark_every_lambda:g}$.")
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.6, color="0.25")

    fig.tight_layout()
    if save_stem is None:
        save_stem = f"bias_variance_gamma={gamma:g}_rho={rho:g}"
    _savefig(fig, save_stem)
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


def save_bias_variance_legend(k_l=10.0, save_stem="bias_variance_legend"):
    """
    Standalone legend (no plot) for the bias/variance figures, to drop into
    Illustrator separately. Two groups: colour = estimator, style = quantity.
    """
    set_paper_style()
    est_handles = [Line2D([0], [0], color=INIT_COLOR, lw=2.6, label=r"$k_\ell=0$ (baseline)"),
                   Line2D([0], [0], color=FEAT_COLOR, lw=2.6, label=rf"$k_\ell={k_l:g}$ (feat.)")]
    qty_handles = [Line2D([0], [0], color="0.35", linestyle=ls, marker=marker,
                          markerfacecolor="none", markersize=6, label=qlabel)
                   for (_s, qlabel, ls, marker) in _BV_QUANTITIES]

    fig = plt.figure(figsize=(4.2, 2.2))
    leg1 = fig.legend(handles=est_handles, frameon=False, loc="upper left",
                      bbox_to_anchor=(0.03, 0.95), title="estimator (color)",
                      alignment="left", fontsize=11, title_fontsize=10)
    fig.add_artist(leg1)
    fig.legend(handles=qty_handles, frameon=False, loc="lower left",
               bbox_to_anchor=(0.03, 0.05),
               title="quantity  (theory: lines,  sim.: markers)",
               alignment="left", fontsize=11, title_fontsize=10)
    _savefig(fig, save_stem)
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


def plot_all_bias_variance(directory=BV_DIR, mark_every_lambda=0.1):
    """Render every bias/variance dataset (legend-less) plus one standalone legend."""
    for path in sorted(glob.glob(os.path.join(directory, "bias_variance_*.pkl"))):
        plot_bias_variance(path, mark_every_lambda=mark_every_lambda)
    save_bias_variance_legend()


# ---- SNR phase diagram: Delta* over (gamma, rho^2/sigma^2) ------------------
def plot_snr_phase_diagram(pkl_path, cmap="RdBu_r", show_upper=True, save_stem=None,
                           gamma_range=None, snr_range=None, save=True, have_legend=True):
    """
    Heatmap of Delta* = inf_lambda G_feat - inf_lambda G_init over
    (gamma, rho^2/sigma^2). Diverging colormap on a SYMMETRIC linear scale
    (white exactly at 0, blue negative, red positive, equal both sides).

    Overlays the analytic SNR bound curves. With show_upper=True the window is
    hatched between the lower and upper curves. With show_upper=False (e.g. when
    the upper bound sits above the plotted rho^2/sigma^2 cap) only the lower curve
    is drawn and the whole region above it is hatched.
    """
    from matplotlib.patches import Patch

    set_paper_style()
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)

    gammas, snrs, Delta = d["gammas"], d["snrs"], d["Delta"]
    psi, k_l = d["psi"], d["k_l"]
    vary_sigma = d["vary_sigma"]
    sigma = d["sigma"]
    rho = d["rho"]
    fixed_tag = f'sigma={sigma:g}' if not vary_sigma else f'rho={rho:g}'

    snr_max = float(snrs.max())

    # symmetric scale: white at 0, equal extent on both sides (no distortion)
    M = float(np.nanmax(np.abs(Delta)))
    norm = mpl.colors.Normalize(vmin=-M, vmax=M)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    pcm = ax.pcolormesh(gammas, snrs, Delta, cmap=cmap, norm=norm, shading="gouraud")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, extend="neither")
    # cbar.set_label(r"$\Delta^\star = \inf_\lambda \mathcal{G}_{\mathrm{feat}}"
    #                r" - \inf_\lambda \mathcal{G}_{\mathrm{init}}$")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Generalization Error Difference", fontsize=15)

    # --- analytic SNR bound curves ---
    gc = np.linspace(max(gammas.min(), 1e-4), gammas.max(), 2000)
    upper = (1 + gc * psi) ** 3 / (gc * (1 + gc) * (1 - psi) ** 3)
    lower = (1 + gc * psi**2) / (gc * (1 - psi) ** 2)      # L_glob (corrected)
    lower_plot = np.where(lower <= snr_max, lower, np.nan)

    # find all gamma, SNR pairs where the difference is approximately zero and mark each of them with a small black dot
    # This is to highlight the boundary where feature learning neither helps nor hurts
    ax.contour(gammas, snrs, Delta, levels=[0], colors="silver", linewidths=2., linestyles="dotted", zorder=6)

    ax.scatter([0.25], [1.0], color='red', zorder=8, marker='*', s=200, edgecolor='black', clip_on=False, lw=1.9)
    ax.scatter([20.], [0.0049], color='red', zorder=8, marker='*', s=200, edgecolor='black', clip_on=False, lw=1.9)
    ax.scatter([10.], [0.25], color='darkblue', zorder=8, marker='*', s=200, edgecolor='black', clip_on=False, lw=1.9)
    # hatched "SNR window" (heatmap stays visible: facecolor none).
    # show_upper: hatch between L and U. Otherwise U is above the cap, so the
    # visible window is everything above L up to the top edge.
    lo_c = np.clip(lower, 0, snr_max)
    if show_upper:
        window_top = np.clip(upper, 0, snr_max)
        top_mask = np.isfinite(upper) & (upper > lower)
    else:
        window_top = np.full_like(gc, snr_max)
        top_mask = np.ones_like(gc, dtype=bool)
    mask = np.isfinite(lower) & (lower < snr_max) & top_mask
    ax.fill_between(gc, lo_c, window_top, where=mask, facecolor="none",
                    edgecolor="0.5", hatch="/", linewidth=0.0, zorder=4)

    if show_upper:
        ax.plot(gc, np.where(upper <= snr_max, upper, np.nan),
                color="k", lw=2.0, ls="-", zorder=5)
    ax.plot(gc, lower_plot, color="k", lw=2.0, ls="--", zorder=5)

    if gamma_range is not None:
        ax.set_xlim(gamma_range)
    else:
        ax.set_xlim(gammas.min(), gammas.max())
    
    if snr_range is not None:
        ax.set_ylim(snr_range)
    else:
        ax.set_ylim(0, snr_max)
    ax.set_xlabel(r"Spike Strength ($\gamma$)", fontsize=16)
    ax.set_ylabel(r"Signal-to-Noise Ratio ($\rho^2/\sigma^2$)", fontsize=16)
    ax.grid(False)

    # set tick mark font size
    ax.tick_params(axis="both", which="major", labelsize=14)

    # plot legend in a separate file
    if have_legend:
        handles = []
        if show_upper:
            handles.append(Line2D([0], [0], color="k", lw=2.0, ls="-",
                        label=r"$\rho^2/\sigma^2 = \dfrac{(1+\gamma\psi)^3}{\gamma(1+\gamma)(1-\psi)^3}$ (upper bound)"))
        handles.append(Line2D([0], [0], color="k", lw=2.0, ls="--",
                    label=r"$\rho^2/\sigma^2 =\dfrac{1+\gamma\psi^2}{\gamma(1-\psi)^2}$ (lower bound)"))
        # handles.append(Patch(facecolor="none", edgecolor="0.15", hatch="/-", label="Sufficient SNR Window"))
        plt.rcParams['hatch.linewidth'] = 0.5
        handles.append(Patch(facecolor="none", edgecolor="0.5", hatch="//", label="Sufficient SNR Window"))

        handles.append(Line2D([0], [0], color="silver", lw=2.0, ls="dotted",
                    label="Feature Learning \nAdvantage Threshold"))
        # ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92,
        #           edgecolor="0.7", fontsize=10)
        # make separate file
        fig_legend = plt.figure(figsize=(10.4, 1.8))
        fig_legend.legend(handles=handles, loc="center", frameon=True, framealpha=0.92,
                        edgecolor="0.7", fontsize=14, ncol=2)
        if save:
            _savefig(fig_legend, f"snr_phase_legend")
        plt.close(fig_legend)

    # caption = (rf"$\psi={psi:g}$, $\{fixed_tag}$, $k_\ell={k_l:g}$. "
    #            rf"Blue: feature learning lowers optimal error; red: it raises it.")
    # fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8.5, color="0.25")
    range_tag = ""
    if gamma_range is not None or snr_range is not None:
        range_tag = f"gamma_range={gamma_range}_snr_range={snr_range}"

    fig.tight_layout()
    if save_stem is None:
        save_stem = f"snr_phase_psi={psi:g}_{fixed_tag}_kl={k_l:g}_{range_tag}_NEW"
    if save:
        _savefig(fig, save_stem)
    else:
        plt.show()
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


def plot_all_snr_phase(directory=SNR_PHASE_DIR):
    """Render every SNR phase-diagram dataset found in `directory`."""
    for path in sorted(glob.glob(os.path.join(directory, "snr_phase_*.pkl"))):
        plot_snr_phase_diagram(path)

def plot_all_heatmaps(pkl_path, directory=SNR_PHASE_DIR):
    pkl_path = os.path.join(directory, pkl_path)
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)

    gammas = d["gammas"]
    snrs = d["snrs"]
    Ginits = d["Ginit"]
    Gfeats = d["Gfeat"]

    # now plot the heatmaps for G_init and G_feat
    set_paper_style()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    pcm1 = axs[0].pcolormesh(gammas, snrs, Ginits, cmap="viridis", shading="gouraud")
    cbar1 = fig.colorbar(pcm1, ax=axs[0], pad=0.02)
    cbar1.set_label(r"$G_{\mathrm{init}}$", fontsize=14)
    axs[0].set_xlabel(r"Spike Strength ($\gamma$)", fontsize=14)
    axs[0].set_ylabel(r"Signal-to-Noise Ratio ($\rho^2/\sigma^2$)", fontsize=14)
    axs[0].set_title(r"$G_{\mathrm{init}}$ Heatmap", fontsize=16)

    pcm2 = axs[1].pcolormesh(gammas, snrs, Gfeats, cmap="viridis", shading="gouraud")
    cbar2 = fig.colorbar(pcm2, ax=axs[1], pad=0.02)
    cbar2.set_label(r"$G_{\mathrm{feat}}$", fontsize=14)
    axs[1].set_xlabel(r"Spike Strength ($\gamma$)", fontsize=14)
    axs[1].set_ylabel(r"Signal-to-Noise Ratio ($\rho^2/\sigma^2$)", fontsize=14)
    axs[1].set_title(r"$G_{\mathrm{feat}}$ Heatmap", fontsize=16)
    # use same colorbar scale for both G_init and G_feat heatmaps
    vmin = min(float(np.nanmin(Ginits)), float(np.nanmin(Gfeats)))
    vmax = max(float(np.nanmax(Ginits)), float(np.nanmax(Gfeats)))
    pcm1.set_clim(vmin=vmin, vmax=vmax)
    pcm2.set_clim(vmin=vmin, vmax=vmax)

    # make a separate plot with symmetric colormap for the difference G_feat - G_init
    Delta = Gfeats - Ginits
    M = float(np.nanmax(np.abs(Delta)))
    norm = mpl.colors.Normalize(vmin=-M, vmax=M)
    pcm3 = axs[2].pcolormesh(gammas, snrs, Delta, cmap="RdBu_r", norm=norm, shading="gouraud")
    cbar3 = fig.colorbar(pcm3, ax=axs[2], pad=0.02)
    cbar3.set_label(r"$\Delta = G_{\mathrm{feat}} - G_{\mathrm{init}}$", fontsize=14)
    axs[2].set_xlabel(r"Spike Strength ($\gamma$)", fontsize=14)
    axs[2].set_ylabel(r"Signal-to-Noise Ratio ($\rho^2/\sigma^2$)", fontsize=14)
    axs[2].set_title(r"$\Delta$ Heatmap", fontsize=16)

    fig.tight_layout()
    save_stem = "G_init_feat_heatmaps_psi=0.2_sigma=1_kl=10"
    _savefig(fig, save_stem)


# ---- joint (lambda, k_l) optimum phase diagrams -----------------------------
def _joint_axes(ax, gammas, snrs):
    ax.set_xlim(gammas.min(), gammas.max())
    ax.set_ylim(snrs.min(), snrs.max())
    ax.set_xlabel(r"Spike Strength ($\gamma$)", fontsize=16)
    ax.set_ylabel(r"Signal-to-Noise Ratio ($\rho^2/\sigma^2$)", fontsize=16)
    ax.grid(False)


def plot_joint_G_opt(pkl_path, cmap="viridis", save_stem=None):
    """Heatmap 1: minimal G jointly over lambda >= 0 and 0 <= k_l <= k_max."""
    set_paper_style()
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    gammas, snrs, G = d["gammas"], d["snrs"], d["G_opt"]

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    pcm = ax.pcolormesh(gammas, snrs, G, cmap=cmap, shading="gouraud")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Optimal Generalization Error", fontsize=15)
    _joint_axes(ax, gammas, snrs)
    fig.tight_layout()
    if save_stem is None:
        save_stem = f"joint_G_opt_psi={d['psi']:g}_sigma={d['sigma']:g}_kmax={d['k_max']:g}"
    _savefig(fig, save_stem)
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


def plot_joint_delta(pkl_path, cmap="Blues_r", save_stem=None):
    """
    Heatmap 2: G_opt - G_baseline. This is <= 0 everywhere by construction
    (k_l = 0 is inside the feasible set), so a sequential map over the actual
    [min, 0] range is used -- white = no gain from feature learning.
    """
    set_paper_style()
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    gammas, snrs, Delta = d["gammas"], d["snrs"], d["Delta"]

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    pcm = ax.pcolormesh(gammas, snrs, Delta, cmap=cmap,
                        vmin=float(np.nanmin(Delta)), vmax=0.0, shading="gouraud")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Generalization Error Difference", fontsize=15)
    _joint_axes(ax, gammas, snrs)
    fig.tight_layout()
    if save_stem is None:
        save_stem = f"joint_delta_psi={d['psi']:g}_sigma={d['sigma']:g}_kmax={d['k_max']:g}"
    _savefig(fig, save_stem)
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


def plot_joint_k_opt(pkl_path, cmap="magma", display_cap=10.0,
                     mark_saturated=True, save_stem=None):
    """
    Heatmap 3: the optimal k_l, shown on [0, display_cap] with the top bin
    labelled '>= display_cap'. When the data were generated with k_max = inf,
    k_opt is genuinely infinite wherever the optimum sits at the ceiling; those
    cells are clipped to display_cap for plotting.

    The dashed contour is the boundary of that region: outside it a finite
    optimal k_l exists; inside it G is still decreasing in c at the ceiling, so
    more feature learning always helps and the optimum runs to k_l -> infinity.
    """
    set_paper_style()
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    gammas, snrs, K = d["gammas"], d["snrs"], d["k_opt"]
    k_max, Sat = float(d["k_max"]), d["saturated"]
    cap = float(min(display_cap, k_max))

    K_disp = np.where(np.isfinite(K), K, cap)     # inf -> top of the display range
    K_disp = np.minimum(K_disp, cap)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    pcm = ax.pcolormesh(gammas, snrs, K_disp, cmap=cmap, vmin=0.0, vmax=cap,
                        shading="gouraud")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, extend="max")
    cbar.set_label(r"Optimal $k_\ell$", fontsize=15)
    ticks = list(np.linspace(0, cap, 6))
    cbar.set_ticks(ticks)
    labels = [f"{t:g}" for t in ticks]
    labels[-1] = rf"$\geq {cap:g}$"
    cbar.set_ticklabels(labels)

    if mark_saturated and Sat.any() and not Sat.all():
        # single boundary contour (white halo + dashed black) -- reads far better
        # than hatching a region that covers most of the panel
        ax.contour(gammas, snrs, Sat.astype(float), levels=[0.5],
                   colors="w", linewidths=2.2, zorder=4)
        ax.contour(gammas, snrs, Sat.astype(float), levels=[0.5],
                   colors="k", linewidths=1.2, linestyles="--", zorder=5)

    _joint_axes(ax, gammas, snrs)
    fig.tight_layout()
    if save_stem is None:
        save_stem = f"joint_k_opt_psi={d['psi']:g}_sigma={d['sigma']:g}_kmax={d['k_max']:g}"
    _savefig(fig, save_stem)
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


def plot_all_joint_phase(directory=JOINT_PHASE_DIR):
    """Render all three joint-optimum heatmaps for every dataset in `directory`."""
    for path in sorted(glob.glob(os.path.join(directory, "joint_phase_*.pkl"))):
        plot_joint_G_opt(path)
        plot_joint_delta(path)
        plot_joint_k_opt(path)


def plot_joint_cap_consistency(exact_pkl, capped_pkl,
                               out_dir=os.path.join(HERE, "consistency_checks"),
                               save_stem="joint_cap_consistency"):
    """
    Diagnostic (not a paper figure): where does the k_l cap actually matter?
    Heatmap of G_opt(exact, k_l unbounded) - G_opt(capped), which is <= 0 since
    the exact problem optimizes over a strictly larger feasible set.
    """
    set_paper_style()
    with open(exact_pkl, "rb") as fh:
        ex = pickle.load(fh)
    with open(capped_pkl, "rb") as fh:
        cp = pickle.load(fh)
    diff = ex["G_opt"] - cp["G_opt"]
    gammas, snrs = ex["gammas"], ex["snrs"]

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    pcm = ax.pcolormesh(gammas, snrs, diff, cmap="Blues_r",
                        vmin=float(np.nanmin(diff)), vmax=0.0, shading="gouraud")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(rf"$G^{{\rm opt}}_{{k_\ell\leq\infty}} - "
                   rf"G^{{\rm opt}}_{{k_\ell\leq {cp['k_max']:g}}}$", fontsize=14)
    _joint_axes(ax, gammas, snrs)
    ax.set_title(f"effect of the $k_\\ell$ cap "
                 f"(max {np.nanmin(diff):.4f}, mean {np.nanmean(diff):.5f})",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{save_stem}.{ext}"))
    plt.close(fig)
    print(f"[saved] {os.path.join(out_dir, save_stem)}.pdf / .png")
    return os.path.join(out_dir, f"{save_stem}.pdf")


# ---- io --------------------------------------------------------------------
def _savefig(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"))
    print(f"[saved] {os.path.join(OUT_DIR, stem)}.pdf / .png")


if __name__ == "__main__":
    plot_single_run_scaled("new_spiked_sweep/spiked_gamma=10_rho=0_D=3000_n=1500_sigma=1_kl=10_ntrials=100.pkl", xscale="symlog")
    plot_single_run_scaled("new_spiked_sweep/spiked_gamma=0_rho=0_D=3000_n=1500_sigma=1_kl=10_ntrials=100.pkl", xscale="symlog")
    print(1/0)

    # plot_c_lambda_heatmap('c_lambda_data/c_lambda_heatmap_gamma=0_rho=0_D=1000_n=500_sigma=0.5.pkl', cmap="cool_r", vmin=1.0, vmax=None, save_stem=None)
    # plot_c_lambda_heatmap('c_lambda_data/c_lambda_heatmap_gamma=10_rho=0.6_D=1000_n=500_sigma=0.5.pkl', cmap="cool_r", vmin=1.0, vmax=None, save_stem=None)
    plot_c_lambda_lines('c_lambda_data/c_lambda_lines_gamma=0_rho=0_D=1000_n=500_sigma=0.5_ntrials=100.pkl', mark_every_lambda=0.5, upper_lambda=None,
                        ylim=None, save_stem=None)
    plot_c_lambda_lines('c_lambda_data/c_lambda_lines_gamma=10_rho=0.6_D=1000_n=500_sigma=0.5_ntrials=100.pkl', mark_every_lambda=0.5, upper_lambda=None,
                            ylim=None, save_stem=None)
    print(1./0)

    # plot_single_run('new_spiked_sweep/spiked_gamma=0.25_rho=1_D=5000_n=1000_sigma=1_kl=10_ntrials=100.pkl', marker_lambda_spacing=3, upper_lambda=None,
    #                     save_stem=None)
    # plot_single_run('new_spiked_sweep/spiked_gamma=15_rho=0.1_D=5000_n=1000_sigma=1_kl=10_ntrials=100.pkl', marker_lambda_spacing=0.25, upper_lambda=None,
    #                     save_stem=None)
    # plot_single_run('new_spiked_sweep/spiked_gamma=10_rho=0.5_D=5000_n=1000_sigma=1_kl=10_ntrials=100.pkl', marker_lambda_spacing=0.5, upper_lambda=None,
    #                 save_stem=None)
    # plot_single_run('spiked_sweep/spiked_gamma=0_rho=0_D=1000_n=500_sigma=0.5_kl=10_ntrials=100.pkl', marker_lambda_spacing=0.1, upper_lambda=None,
    #                 save_stem=None)
    # plot_single_run('new_spiked_sweep/spiked_gamma=20_rho=0_D=2000_n=1000_sigma=0.5_kl=10_ntrials=100.pkl', marker_lambda_spacing=0.1, upper_lambda=None,
    #                 save_stem=None)
    # save_single_run_legend()

    # plot_f1_grid()
    # plot_isotropic()
    # plot_all_snr_phase()
    plot_snr_phase_diagram(pkl_path=os.path.join(SNR_PHASE_DIR, "snr_phase_psi=0.2_sigma=1_kl=10_gmax=20_snrmax=1_gammastep=0.1_snrstep=0.001.pkl"), 
                           # gamma_range=(19., 20.), snr_range=(0., 0.02), 
                           save=True, have_legend=False)
    print(1/0)

    base = 'new_spiked_sweep/spiked_{}_D=5000_n=1000_sigma=1_kl=10_ntrials=100.pkl'
    panels = [('gamma=0.25_rho=1',  None),
            ('gamma=10_rho=0.5',  1.5),
            ('gamma=15_rho=0.1',  None)]

    for tag, ym in panels:
        plot_single_run_scaled(
            base.format(tag),
            xscale='symlog',
            ymax=ym,
            show_scale_in_title=False,
            out_dir='paper_figures',
            save_stem=f'G_{tag}_D=5000_n=1000_sigma=1_symlog',
        )