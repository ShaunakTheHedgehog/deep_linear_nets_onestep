"""
finite_width_experiments.py — finite-width validation of the one-step feature
kernel limit (Proposition 4.1 / "Prop. feature_kernel_limits").

This is a standalone NumPy driver. It simulates ONE step of gradient descent in
the deep linear network of the paper (Eqs. 2.1--2.3, with the GD update 2.7 and
the backprop recursion 2.8) at finite width N, and measures how the layer-ell
feature kernel changes:

    Phi^ell_{t=1} - Phi^ell_{t=0}  ->  (s_ell * eta^2 / n^2) * K_x y y^T K_x   (N -> infinity)

where K_x = X^T X / D and the *depth coefficient* is

    s_ell = 1^2 + 2^2 + ... + ell^2 = ell (ell + 1) (2 ell + 1) / 6.

(The coefficient s_ell replaces the ell^2 that would arise if the layer-ell
increment were 2 ell - 1; the extra (ell - 1)^2 per layer comes from the
dependence of the once-updated hidden state on W^{ell-1} through the
backpropagated gradient.)

Key simplification used here: in a *linear* network the backprop gradient
g^ell = (1/sqrt(N)) W^ell^T g^{ell+1} is input-independent, so the one-step
weight update Delta W^ell is rank one. We therefore never materialize a dense
N x N update; the whole simulation is cheap and its only real cost is storing
the initial weight matrices.

Run `python finite_width_experiments.py` to (re)generate the depth-coefficient
figure in paper_figures/. Depends only on numpy + matplotlib.
"""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "paper_figures")


def s_ell(L):
    """Theoretical depth coefficients s_ell = ell(ell+1)(2ell+1)/6 for ell=1..L."""
    ell = np.arange(1, L + 1)
    return ell * (ell + 1) * (2 * ell + 1) / 6.0


# ---------------------------------------------------------------------------
# fixed dataset (Prop. 4.1 holds for ANY fixed dataset, so we fix one instance)
# ---------------------------------------------------------------------------
def make_fixed_dataset(D, n, sigma=0.3, seed=12345):
    """Isotropic Gaussian inputs, noisy linear targets; returns (X [D x n], y [n])."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((D, n))
    w_star = rng.standard_normal(D)
    w_star *= np.sqrt(D) / np.linalg.norm(w_star)          # ||w_star|| = sqrt(D)
    y = (X.T @ w_star) / np.sqrt(D) + sigma * rng.standard_normal(n)
    return X, y


# ---------------------------------------------------------------------------
# one gradient step in the deep linear network (Eqs. 2.1-2.3, 2.7, 2.8)
# ---------------------------------------------------------------------------
def simulate_one_step_delta_kernels(X, y, L, eta, N, seed):
    """
    Perform one GD step at width N and return the list of feature-kernel updates
    [Phi^1_{t=1} - Phi^1_{t=0}, ..., Phi^L_{t=1} - Phi^L_{t=0}], each n x n.

    The t=0 and t=1 kernels are formed from the SAME weight draw, so the O(1/sqrt(N))
    baseline fluctuation cancels in the difference.
    """
    rng = np.random.default_rng(seed)
    D, n = X.shape
    N_in = [D] + [N] * (L - 1)                              # fan-in of W^ell (ell=0..L-1)

    # --- initialization: W^ell_{ij} ~ N(0,1),  w^L ~ N(0,1) ---
    W = [rng.standard_normal((N, D))] + [rng.standard_normal((N, N)) for _ in range(L - 1)]
    wL = rng.standard_normal(N)

    # --- forward pass at t=0:  h^{ell+1} = W^ell h^ell / sqrt(N_ell) ---
    H0 = [X]                                                # H0[ell] is the layer-ell hidden state (columns = samples)
    for ell in range(L):
        H0.append(W[ell] @ H0[ell] / np.sqrt(N_in[ell]))
    z = (wL @ H0[L]) / N
    Delta = y.reshape(-1) - z.reshape(-1)                                        # output error signal Delta_mu

    # --- backprop gradients at t=0 (input-independent):  g^ell = W^ell^T g^{ell+1} / sqrt(N) ---
    g = [None] * (L + 1)
    g[L] = wL
    for ell in range(L - 1, 0, -1):
        g[ell] = W[ell].T @ g[ell + 1] / np.sqrt(N)        # g^1..g^{L-1}; g^L = wL

    # --- one GD step + forward pass at t=1, applying Delta W^ell = c_ell g^{ell+1} u_ell^T as a rank-1 op ---
    #     u_ell = sum_mu Delta_mu h^ell_mu(0);   Delta W^ell = (eta / (n sqrt(N_ell))) g^{ell+1} u_ell^T
    dPhi = []
    Ht1 = X                                                # H^ell_{t=1}
    for ell in range(L):
        u = H0[ell] @ Delta                                # N_ell-vector
        scale = eta / (n * np.sqrt(N_in[ell]))
        Ht1 = (W[ell] @ Ht1 + scale * np.outer(g[ell + 1], Ht1.T @ u)) / np.sqrt(N_in[ell])
        Phi0 = H0[ell + 1].T @ H0[ell + 1] / N             # Phi^{ell+1}_{t=0}
        Phi1 = Ht1.T @ Ht1 / N                             # Phi^{ell+1}_{t=1}
        dPhi.append(Phi1 - Phi0)
    return dPhi


def depth_coefficient(dPhi_ell, M, eta, n):
    """Least-squares scalar coefficient of (eta^2/n^2) M in dPhi_ell (M = K_x y y^T K_x)."""
    return float((np.sum(dPhi_ell * M) / np.sum(M * M)) * (n ** 2 / eta ** 2))

def cosine_similarity(dPhi_ell, M):
    return float( np.sum(dPhi_ell * M) / (np.sum(M * M)**0.5 * np.sum(dPhi_ell * dPhi_ell)**0.5) )


def measure_depth_coefficients(X, y, L, eta, Ns, ntrials=40, base_seed=0):
    """
    For each width N, return an (ntrials, L) array of empirical depth coefficients
    d_ell (one column per layer ell=1..L). The dataset (X, y) is fixed; only the
    network weights are resampled across trials.
    """
    D, n = X.shape
    K_x = (X.T @ X) / D
    Ky = K_x @ y
    M = np.outer(Ky, Ky)                                   # K_x y y^T K_x
    coeffs_by_N = {}
    cos_sims_by_N = {}
    for N in Ns:
        coeffs = np.zeros((ntrials, L))
        cosine_sims = np.zeros((ntrials, L))
        for t in range(ntrials):
            dPhi = simulate_one_step_delta_kernels(X, y, L, eta, N, seed=base_seed + 1 + t)
            for ell in range(L):
                coeffs[t, ell] = depth_coefficient(dPhi[ell], M, eta, n)
                cosine_sims[t, ell] = cosine_similarity(dPhi[ell], M)
        coeffs_by_N[N] = coeffs
        cos_sims_by_N[N] = cosine_sims
        print(f"[finite width] N={N}: mean d_ell = {np.round(coeffs.mean(0), 3)} "
              f"(theory s_ell = {s_ell(L).astype(int)})", flush=True)
        print(f'mean cosine similarity to expected rank-one update: {np.round(cosine_sims.mean(0), 3)}')
    return coeffs_by_N, cos_sims_by_N


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def _set_style():
    mpl.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 12,
        "axes.titlesize": 14, "axes.labelsize": 14, "legend.fontsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def make_depth_coefficient_figure(eta=1.0, D=30, n=10, L=5, Ns=(100, 500, 1000, 5000),
                                  ntrials=50, sigma=0.3, dataset_seed=12345,
                                  base_seed=0, depth_coeffs=None,
                                  save_stem="finite_width_depth_coefficient",
                                  out_dir=OUT_DIR, save=True):
    """Depth coefficient d_ell vs. layer index: empirics per N (markers + s.e.m.) vs. theory.

    Pass `depth_coeffs` (dict N -> (ntrials, L) array from measure_depth_coefficients) to
    reuse an existing measurement; otherwise the simulation is run here.
    """
    _set_style()
    if depth_coeffs is None:
        X, y = make_fixed_dataset(D, n, sigma=sigma, seed=dataset_seed)
        depth_coeffs, _ = measure_depth_coefficients(X, y, L, eta, Ns, ntrials=ntrials, base_seed=base_seed)
    Ns = list(depth_coeffs.keys())
    L = depth_coeffs[Ns[0]].shape[1]

    ell = np.arange(1, L + 1)
    theory = s_ell(L)
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(Ns)))
    markers = ["o", "s", "D", "^", "v"]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for i, N in enumerate(Ns):
        m = depth_coeffs[N].mean(0)
        sem = depth_coeffs[N].std(0, ddof=1) / np.sqrt(depth_coeffs[N].shape[0])
        ax.errorbar(ell, m, yerr=sem, marker=markers[i % len(markers)], markersize=6.5,
                    lw=2.0, capsize=3.0, color=colors[i], markerfacecolor=colors[i],
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3,
                    label=rf"$N={N}$")
    ax.plot(ell, theory, color="black", lw=2.5, ls="--", markersize=11,
            markerfacecolor="black", zorder=4,
            label=r"theory: $\ell(\ell+1)(2\ell+1)/6$")

    ax.set_xlabel(r"Layer index $\ell$", fontsize=18)
    ax.set_ylabel(r"Depth coefficient $d_\ell$", fontsize=18)
    ax.set_title(r"$\Delta \Phi^{\ell}$ has a depth coefficient of $\ell(\ell+1)(2\ell+1)/6$", fontsize=22, pad=15)
    ax.set_xticks(ell)
    ax.grid(True, alpha=0.25, lw=0.6)
    # ax.legend(frameon=False, loc="upper left")
    cap = (rf"$\Delta\Phi^\ell = \Phi^\ell_{{t=1}} - \Phi^\ell_{{t=0}}$ projected onto "
           rf"$\frac{{\eta^2}}{{n^2}} K_x y y^\top K_x$; $\eta={eta:g}$, $D={D}$, $n={n}$, "
           rf"$\sigma={sigma:g}$; {ntrials} weight draws (fixed dataset); $\pm$1 s.e.m.")
    fig.text(0.5, -0.03, cap, ha="center", va="top", fontsize=8.0, color="0.25")
    ax.tick_params(axis='both', labelsize=16)

    fig.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"{save_stem}.{ext}"))
        print(f"[saved] {os.path.join(out_dir, save_stem)}.pdf / .png", flush=True)
    plt.close(fig)
    return depth_coeffs


def make_cosine_similarity_figure(eta=1.0, D=30, n=10, L=5, Ns=(100, 500, 1000, 5000),
                                  ntrials=50, sigma=0.3, dataset_seed=12345,
                                  base_seed=0, cos_sims=None,
                                  save_stem="finite_width_cosine_similarity",
                                  out_dir=OUT_DIR, save=True):
    """
    Cosine similarity of the update dPhi^ell to the rank-one K_x y y^T K_x vs. layer
    index: empirics per N (markers + s.e.m.), with a dashed line at cosine = 1 (the
    theory's perfect structural match). y-axis auto-zoomed near 1.

    Pass `cos_sims` (dict N -> (ntrials, L) array from measure_depth_coefficients) to
    reuse an existing measurement; otherwise the simulation is run here.
    """
    _set_style()
    if cos_sims is None:
        X, y = make_fixed_dataset(D, n, sigma=sigma, seed=dataset_seed)
        _, cos_sims = measure_depth_coefficients(X, y, L, eta, Ns, ntrials=ntrials, base_seed=base_seed)
    Ns = list(cos_sims.keys())
    L = cos_sims[Ns[0]].shape[1]
    nt = cos_sims[Ns[0]].shape[0]

    ell = np.arange(1, L + 1)
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(Ns)))
    markers = ["o", "s", "D", "^", "v"]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    lowers = []
    for i, N in enumerate(Ns):
        m = cos_sims[N].mean(0)
        sem = cos_sims[N].std(0, ddof=1) / np.sqrt(cos_sims[N].shape[0])
        lowers.append(float((m - sem).min()))
        ax.errorbar(ell, m, yerr=sem, marker=markers[i % len(markers)], markersize=6.5,
                    lw=2.0, capsize=3.0, color=colors[i], markerfacecolor=colors[i],
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3,
                    label=rf"$N={N}$")
    ax.axhline(1.0, color="black", ls="--", lw=1.8, zorder=2,
               label=r"theoretical prediction")

    # data-driven zoom near 1 (keep the cos = 1 line near the top with a little headroom)
    span = max(1.0 - min(lowers), 5e-3)
    ax.set_ylim(1.0 - 1.15 * span, 1.0 + 0.15 * span)

    ax.set_xlabel(r"Layer index $\ell$", fontsize=18)
    ax.set_ylabel(r"Cosine similarity", fontsize=18)
    ax.set_title(r"$\Delta \Phi^{\ell}$ aligns with $K_x y y^\top K_x$", fontsize=22, pad=15)
    ax.set_xticks(ell)
    ax.grid(True, alpha=0.25, lw=0.6)
    # ax.legend(frameon=False, loc="lower right")
    cap = (rf"Cosine similarity between $\Delta\Phi^\ell = \Phi^\ell_{{t=1}} - \Phi^\ell_{{t=0}}$ and "
           rf"$K_x y y^\top K_x$ (Frobenius); $\eta={eta:g}$, $D={D}$, $n={n}$, "
           rf"$\sigma={sigma:g}$; {nt} weight draws (fixed dataset); $\pm$1 s.e.m.")
    fig.text(0.5, -0.03, cap, ha="center", va="top", fontsize=8.0, color="0.25")
    ax.tick_params(axis='both', labelsize=16)

    fig.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"{save_stem}.{ext}"))
        print(f"[saved] {os.path.join(out_dir, save_stem)}.pdf / .png", flush=True)
    plt.close(fig)
    return cos_sims


def make_shared_legend(Ns=(100, 500, 1000, 5000), theory_label="theory", ncol=None,
                       save_stem="finite_width_legend", out_dir=OUT_DIR, save=True):
    """
    Standalone legend shared by both finite-width figures: a dashed black line for
    theory plus one colored line+marker per width N, matching the styling used in
    make_depth_coefficient_figure / make_cosine_similarity_figure. Saved cropped
    tight to just the legend box. Pass the SAME Ns used to draw the figures.
    """
    from matplotlib.lines import Line2D
    _set_style()
    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(Ns)))
    markers = ["o", "s", "D", "^", "v"]
    handles = [Line2D([0], [0], color="black", lw=2.0, ls="--", label=theory_label)]
    for i, N in enumerate(Ns):
        handles.append(Line2D([0], [0], color=colors[i], lw=2.0,
                              marker=markers[i % len(markers)], markersize=6.5,
                              markerfacecolor=colors[i], markeredgecolor="white",
                              markeredgewidth=0.7, label=rf"$N={N}$"))
    if ncol is None:
        ncol = len(handles)                                 # single horizontal row by default

    fig, ax = plt.subplots(figsize=(2.2, 2.5))
    ax.axis("off")
    leg = ax.legend(handles=handles, loc="center", frameon=False, ncol=ncol,
                    handlelength=2.2, columnspacing=1.6)
    if save:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"{save_stem}.{ext}"),
                        bbox_inches="tight", bbox_extra_artists=(leg,))
        print(f"[saved] {os.path.join(out_dir, save_stem)}.pdf / .png", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    eta, D, n, L = 0.5, 30, 10, 10
    Ns = (100, 500, 1000, 5000)
    ntrials, sigma, base_seed = 50, 0.3, 0

    # Measure once (both quantities share the same draws), then draw both figures.
    X, y = make_fixed_dataset(D, n, sigma=sigma, seed=12345)
    depth_coeffs, cos_sims = measure_depth_coefficients(X, y, L, eta, Ns, ntrials=ntrials, base_seed=base_seed)
    make_depth_coefficient_figure(eta=eta, D=D, n=n, L=L, Ns=Ns, ntrials=ntrials, sigma=sigma,
                                  base_seed=base_seed, depth_coeffs=depth_coeffs)
    make_cosine_similarity_figure(eta=eta, D=D, n=n, L=L, Ns=Ns, ntrials=ntrials, sigma=sigma,
                                  base_seed=base_seed, cos_sims=cos_sims)
    make_shared_legend(Ns=Ns, ncol=1)                               # shared legend matching both figures
