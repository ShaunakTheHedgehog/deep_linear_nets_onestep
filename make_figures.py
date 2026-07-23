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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(HERE, "new_spiked_sweep")                  # clean parallel run
SPIKED_DIR = os.path.join(HERE, "spiked_covariance_experiments")  # legacy runs
CLAMBDA_DIR = os.path.join(HERE, "c_lambda_data")                 # c_lambda study data
BV_DIR = os.path.join(HERE, "bias_variance_data")                 # bias/variance study data
SNR_PHASE_DIR = os.path.join(HERE, "snr_phase_data")             # SNR phase-diagram data
OUT_DIR = os.path.join(HERE, "paper_figures")

# ---- colour / style semantics (consistent across every figure) -------------
INIT_COLOR = "#6E6E6E"   # grey  : baseline  f_init
FEAT_COLOR = "#0072B2"   # blue  : feature-learning  f_feat  (colourblind-safe)

# per-k_l colours for the c_lambda line plots (colourblind-safe)
KL_COLORS = {0.0: "#6E6E6E", 1.0: "#0072B2", 10.0: "#D55E00"}


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


def _draw_lambda_star_tick(ax, lam, color):
    """Small upward caret at the bottom spine marking a curve's minimiser."""
    ax.plot([lam], [0.0], marker="^", markersize=5, color=color,
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)


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
            _draw_lambda_star_tick(ax, _lambda_star(lam_th, d["init_gen_error_theory"])[0], INIT_COLOR)
            _draw_lambda_star_tick(ax, _lambda_star(lam_th, d["feat_gen_error_theory"])[0], FEAT_COLOR)

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
    k_l = d.get("k_l", d.get("beta_coeff"))
    gamma, rho = d["spike_strength"], d["rho"]
    psi = n / D

    lam_th = d.get("lambdas_theory", d["lambdas"])
    lam = d["lambdas"]
    if upper_lambda is None:
        upper_lambda = float(lam.max())

    # convert the requested lambda-unit spacing into an index stride
    lam_step = float(np.median(np.diff(lam))) if len(lam) > 1 else marker_lambda_spacing
    me = max(1, round(marker_lambda_spacing / lam_step))

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

    _draw_lambda_star_tick(ax, _lambda_star(lam_th, d["init_gen_error_theory"])[0], INIT_COLOR)
    _draw_lambda_star_tick(ax, _lambda_star(lam_th, d["feat_gen_error_theory"])[0], FEAT_COLOR)

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

    ax.set_xlabel(r"ridge $\lambda$")
    ax.set_ylabel(r"$G(\hat w)$")
    ax.set_title(rf"$\gamma={gamma:g}$, $\rho={rho:g}$")
    ax.legend(frameon=False, loc="best")

    ntrials, seed = d.get("ntrials", "?"), d.get("seed", "?")
    caption = (
        rf"$\psi={psi:g}$, $D={D}$, $n={n}$, $\sigma={sigma:g}$, $k_\ell={k_l:g}$; "
        rf"{ntrials} dataset draws; seed$={seed}$. Markers every $\Delta\lambda\approx{marker_lambda_spacing:g}$."
    )
    fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=8.0, color="0.25")

    fig.tight_layout()
    if save_stem is None:
        save_stem = f"G_gamma={gamma:g}_rho={rho:g}_D={D}_n={n}_sigma={sigma:g}"
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
        ax.plot(lam, d["c_theory"][a], color=color, zorder=3)
        ax.errorbar(lam, d["c_emp_mean"][a], yerr=d["c_emp_sem"][a], linestyle="none",
                    marker=marker_cycle[a % len(marker_cycle)], markersize=4.5,
                    markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.0,
                    ecolor=color, elinewidth=0.9, capsize=2.0,
                    markevery=me, errorevery=me, zorder=4)

    ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)  # c_lambda = 1 floor
    ax.set_xlim(0, upper_lambda)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(r"ridge $\lambda$")
    ax.set_ylabel(r"$c_\lambda$")
    title = rf"$\gamma={gamma:g}$, $\rho={rho:g}$"
    ax.set_title(title)

    # legend: one colour entry per k_l, plus theory/sim style key
    handles = [Line2D([0], [0], color=_kl_color(float(k_l), a),
                      marker=marker_cycle[a % len(marker_cycle)], markerfacecolor="none",
                      label=rf"$k_\ell={k_l:g}$")
               for a, k_l in enumerate(k_ls)]
    handles += [
        Line2D([0], [0], color="0.2", label="theory"),
        Line2D([0], [0], color="0.2", marker="o", markerfacecolor="none",
               linestyle="none", label=r"sim. ($\pm$s.e.m.)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=1)

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
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"$c_\lambda$")

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
def plot_snr_phase_diagram(pkl_path, cmap="RdBu_r", show_upper=True, save_stem=None):
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
    psi, sigma, k_l = d["psi"], d["sigma"], d["k_l"]
    snr_max = float(snrs.max())

    # symmetric scale: white at 0, equal extent on both sides (no distortion)
    M = float(np.nanmax(np.abs(Delta)))
    norm = mpl.colors.Normalize(vmin=-M, vmax=M)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    pcm = ax.pcolormesh(gammas, snrs, Delta, cmap=cmap, norm=norm, shading="gouraud")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, extend="neither")
    cbar.set_label(r"$\Delta^\star = \inf_\lambda \mathcal{G}_{\mathrm{feat}}"
                   r" - \inf_\lambda \mathcal{G}_{\mathrm{init}}$")

    # --- analytic SNR bound curves ---
    gc = np.linspace(max(gammas.min(), 1e-4), gammas.max(), 2000)
    upper = (1 + gc * psi) ** 3 / (gc * (1 + gc) * (1 - psi) ** 3)
    lower = (1 + gc * psi**2) / (gc * (1 - psi) ** 2)      # L_glob (corrected)
    lower_plot = np.where(lower <= snr_max, lower, np.nan)

    # find all gamma, SNR pairs where the difference is approximately zero and mark each of them with a small black dot
    # This is to highlight the boundary where feature learning neither helps nor hurts
    ax.contour(gammas, snrs, Delta, levels=[0], colors="dimgray", linewidths=1.5, linestyles="dotted", zorder=6)

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
                    edgecolor="0.15", hatch="////", linewidth=0.0, zorder=4)

    if show_upper:
        ax.plot(gc, np.where(upper <= snr_max, upper, np.nan),
                color="k", lw=2.0, ls="-", zorder=5)
    ax.plot(gc, lower_plot, color="k", lw=2.0, ls="--", zorder=5)

    ax.set_xlim(gammas.min(), gammas.max())
    ax.set_ylim(0, snr_max)
    ax.set_xlabel(r"spike strength $\gamma$")
    ax.set_ylabel(r"$\rho^2/\sigma^2$")
    ax.grid(False)

    handles = []
    if show_upper:
        handles.append(Line2D([0], [0], color="k", lw=2.0, ls="-",
                       label=r"$U(\gamma)=\dfrac{(1+\gamma\psi)^3}{\gamma(1+\gamma)(1-\psi)^3}$"))
    handles.append(Line2D([0], [0], color="k", lw=2.0, ls="--",
                   label=r"$L(\gamma)=\dfrac{1+\gamma\psi^2}{\gamma(1-\psi)^2}$"))
    handles.append(Patch(facecolor="none", edgecolor="0.15", hatch="////", label="SNR window"))
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92,
              edgecolor="0.7", fontsize=10)

    caption = (rf"$\psi={psi:g}$, $\sigma={sigma:g}$, $k_\ell={k_l:g}$. "
               rf"Blue: feature learning lowers optimal error; red: it raises it.")
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8.5, color="0.25")

    fig.tight_layout()
    if save_stem is None:
        save_stem = f"snr_phase_psi={psi:g}_sigma={sigma:g}_kl={k_l:g}"
    _savefig(fig, save_stem)
    plt.close(fig)
    return os.path.join(OUT_DIR, f"{save_stem}.pdf")


def plot_all_snr_phase(directory=SNR_PHASE_DIR):
    """Render every SNR phase-diagram dataset found in `directory`."""
    for path in sorted(glob.glob(os.path.join(directory, "snr_phase_*.pkl"))):
        plot_snr_phase_diagram(path)


# ---- io --------------------------------------------------------------------
def _savefig(fig, stem):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"))
    print(f"[saved] {os.path.join(OUT_DIR, stem)}.pdf / .png")


if __name__ == "__main__":
    # plot_f1_grid()
    # plot_isotropic()
    plot_all_snr_phase()
