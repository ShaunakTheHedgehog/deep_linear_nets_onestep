import os
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

NUMERICAL_TOLERANCE = 1e-6


def mp_stieltjes_transform(z, c):
    """
    Stieltjes transform m(z) of the Marchenko–Pastur law
    in the regime c = n/D < 1.
    Uses the mathematicians' sign convention:
        m(z) = ∫ 1/(λ - z) dμ_MP(λ)
    """
    z = np.asarray(z, dtype=complex) 
    if z == 0:
        # z += NUMERICAL_TOLERANCE
        return 1./(1 - c)

    a = (1 - np.sqrt(c))**2
    b = (1 + np.sqrt(c))**2

    Delta_a = np.sqrt((z - a))  # correct branch automatically
    Delta_b = np.sqrt((z - b))

    m = (1 - c - z + Delta_a * Delta_b) / (2 * c * z)
    return m


def mp_stieltjes_derivative(z, c):
    """
    Computes m'(z) for the MP law in the c<1 regime.
    """
    z = np.asarray(z, dtype=complex) 
    if z == 0:
        # z+= NUMERICAL_TOLERANCE
        return 1./((1 - c)**3)

    a = (1 - np.sqrt(c))**2
    b = (1 + np.sqrt(c))**2

    Delta_a = np.sqrt((z - a))  # correct branch automatically
    Delta_b = np.sqrt((z - b))

    Delta_prime = Delta_a / (2 * Delta_b) + Delta_b / (2 * Delta_a)

    m = (1 - c - z + Delta_a * Delta_b) / (2 * c * z)

    m_prime = ((-1 + Delta_prime) / (2 * c * z)) - (m / z)
    return m_prime


def compute_linear_model_bias_and_variance(n, D, ridge_lambda, noise_std):
    """
    Computes the bias, variance, and generalization error of the baseline linear model
    in the proportional asymptotics regime using the MP Stieltjes transform.
    """
    psi = n / D
    if psi >= 1:
        raise ValueError("This function only supports the overparameterized regime (psi < 1).")

    mp_stieltjes = mp_stieltjes_transform(-ridge_lambda, psi)
    mp_steltjes_prime = mp_stieltjes_derivative(-ridge_lambda, psi)

    alpha_1 = psi * (1 - ridge_lambda * mp_stieltjes)
    alpha_2 = psi * (mp_stieltjes - ridge_lambda * mp_steltjes_prime)

    noise_1 = psi * mp_stieltjes * noise_std**2
    noise_2 = psi * mp_steltjes_prime * noise_std**2

    bias = (1 - alpha_1)**2 
    variance = alpha_1 - alpha_1**2 - ridge_lambda * alpha_2 + noise_1 - ridge_lambda * noise_2 
    gen_error = bias + variance

    bias = bias.real
    variance = variance.real
    gen_error = gen_error.real

    return bias, variance, gen_error


# def compute_feature_learning_model_bias_and_variance(n, D, beta_coeff, ridge_lambda, noise_std):
#     """
#     Computes the bias, variance, and generalization error of the feature learning model
#     in the proportional asymptotics regime using the MP Stieltjes transform.
#     """
#     psi = n / D
#     k = beta_coeff
#     if psi >= 1:
#         raise ValueError("This function only supports the overparameterized regime (psi < 1).")

#     mp_stieltjes = mp_stieltjes_transform(-ridge_lambda, psi)
#     mp_steltjes_prime = mp_stieltjes_derivative(-ridge_lambda, psi)

#     alpha_1 = psi * (1 - ridge_lambda * mp_stieltjes)
#     alpha_2 = psi * (mp_stieltjes - ridge_lambda * mp_steltjes_prime)

#     noise_1 = psi * mp_stieltjes * noise_std**2
#     noise_2 = psi * mp_steltjes_prime * noise_std**2

#     y_norm_term = 1 + noise_std**2
#     y_trace_term = 1 - ridge_lambda * mp_stieltjes + (noise_std**2)*mp_stieltjes

#     num = (ridge_lambda * k / psi) * (y_norm_term - ridge_lambda * y_trace_term)
#     denom = 1 + k * (1 + ((1 + noise_std**2) / psi) - (ridge_lambda * y_norm_term / psi) + (ridge_lambda**2 / psi) * y_trace_term ) 
#     c_lambda = 1. + (num / denom)

#     bias = (1 - c_lambda * alpha_1)**2
#     variance = (c_lambda**2) * (alpha_1 - alpha_1**2 - ridge_lambda * alpha_2 + noise_1 - ridge_lambda * noise_2)
#     gen_error = bias + variance

#     bias = bias.real
#     variance = variance.real
#     gen_error = gen_error.real

#     return bias, variance, gen_error


def compute_spiked_covariance_intermediate_quantities(n, D, k_l, ridge_lambda, spike_strength, rho, noise_std):
    psi = n / D
    gamma = spike_strength 
    if psi >= 1:
        raise ValueError("This function only supports the overparameterized regime (psi < 1).")

    mp_stieltjes = mp_stieltjes_transform(-ridge_lambda, psi)
    mp_steltjes_prime = mp_stieltjes_derivative(-ridge_lambda, psi)

    tau_1 = psi * (1 - ridge_lambda * mp_stieltjes)
    tau_2 = psi * (mp_stieltjes - ridge_lambda * mp_steltjes_prime)

    a = tau_1
    b = gamma * tau_1 * (1 - tau_1) / (1 + gamma * tau_1)  #1 - tau_1 - (1. / (1./(1 - tau_1) + gamma * psi * mp_stieltjes))

    # T1 = gamma * psi * mp_stieltjes + (tau_1 / (1. - tau_1))
    # T2 = gamma * psi * mp_steltjes_prime + (tau_2 / ((1 - tau_1)**2))

    a_tilde = tau_2
    b_tilde = tau_2 * ( (gamma + 1)/((1 + gamma * tau_1)**2) - 1 )  #(T2 / (1 + T1)**2) - tau_2 

    # noise_1 = psi * mp_stieltjes * noise_std**2
    # noise_2 = psi * mp_steltjes_prime * noise_std**2

    y_norm_term = 1 + noise_std**2 + gamma * rho**2
    y_resolvent_term = ((a + b * rho**2) / psi) + (noise_std**2)*mp_stieltjes
    y_Kx_y_term = 1. + (1 + noise_std**2)/psi + gamma * (rho**2) * (2 + gamma + 1./psi)

    num_term = (ridge_lambda / psi) * (y_norm_term - ridge_lambda * y_resolvent_term)
    num = k_l * num_term
    denom = 1. + k_l * (y_Kx_y_term - (ridge_lambda / psi) * y_norm_term + (ridge_lambda**2 / psi) * y_resolvent_term)
    c_lambda = 1. + (num / denom)

    intermediate_quantities = {'psi': psi, 'mp_stieltjes': mp_stieltjes, 'mp_stieltjes_prime': mp_steltjes_prime,
                               'tau_1': tau_1, 'tau_2': tau_2, 'a': a, 'b': b,  
                               'a_tilde': a_tilde, 'b_tilde': b_tilde, 
                               'c_num': num, 'c_num_term': num_term, 'c_denom': denom, 'c_lambda': c_lambda}
    
    return intermediate_quantities


def compute_spiked_covariance_model_bias_and_variance(n, D, k_l, ridge_lambda, spike_strength, rho, noise_std):
    '''
    Computes the bias, variance, and generalization error of the spiked covariance model.
    
    Arguments:
        n: number of samples
        D: data dimension
        k_l: feature learning update strength
        ridge_lambda: ridge regularization strength
        spike_strength: strength of the spike in the covariance
        rho: alignment between true regression vector and the spike direction
        noise_std: standard deviation of the noise in the outputs
    '''
    psi = n / D
    gamma = spike_strength 
    if psi >= 1:
        raise ValueError("This function only supports the overparameterized regime (psi < 1).")

    vars = compute_spiked_covariance_intermediate_quantities(n, D, k_l, ridge_lambda, spike_strength, rho, noise_std)
    a = vars['a']
    b = vars['b']
    a_tilde = vars['a_tilde']
    b_tilde = vars['b_tilde']
    c_lambda = vars['c_lambda']

    bias_term1 = (1 - c_lambda * a)**2
    bias_term2 = (rho**2) * ( (b * c_lambda)**2 - 2 * c_lambda * b * (1 - a * c_lambda) )
    bias_term3 = gamma * (rho**2) * (1 - c_lambda * (a + b))**2
    bias = bias_term1 + bias_term2 + bias_term3

    var1 = a + b * rho**2
    var2 = a_tilde + b_tilde * rho**2
    init_var = (var1 - ridge_lambda * var2 + (a_tilde * noise_std**2) - (a**2) - (rho**2) * (2*a*b + b**2))
    variance = (c_lambda**2) * init_var

    gen_error = bias + variance 

    bias = bias.real
    variance = variance.real
    gen_error = gen_error.real

    return bias, variance, gen_error


def compute_spiked_covariance_dG_dc(n, D, k_l, ridge_lambda, spike_strength, rho, noise_std):
    gamma = spike_strength
    vars = compute_spiked_covariance_intermediate_quantities(n, D, k_l, ridge_lambda, spike_strength, rho, noise_std)
    a = vars['a']
    b = vars['b']
    a_tilde = vars['a_tilde']
    b_tilde = vars['b_tilde']
    # noise_1 = vars['noise_1']
    # noise_2 = vars['noise_2']
    c_lambda = vars['c_lambda']
    c_num_term = vars['c_num_term']
    c_denom = vars['c_denom']

    dB_dc = -2 * a * (1 - c_lambda * a) + (rho**2) * b * (b * c_lambda - 2 * (1 - c_lambda * a)) + (rho**2) * b * c_lambda * (b + 2 * a) - 2 * (a + b) * gamma * (rho**2) * (1 - c_lambda * (a + b))
    
    var1 = a + b * rho**2
    var2 = a_tilde + b_tilde * rho**2
    init_var = (var1 - ridge_lambda * var2 + (a_tilde * noise_std**2) - (a**2) - (rho**2) * (2*a*b + b**2))
    dV_dc = 2 * c_lambda * init_var
    dG_dc = dB_dc + dV_dc 

    dc_dk = c_num_term / (c_denom**2)      # will be greater than 0, as c_lambda increases as k increases

    dG_dk = dG_dc * dc_dk 

    return dG_dk, dG_dc, dc_dk 

def calculate_dG_finite_differencing(Gs, spacing):
    return np.diff(Gs) / spacing


def dG_dlambda_dk_at_zero(psi, gamma, rho, noise_std):
    """
    Computes the derivative of the generalization error with respect to k_l and lambda
    at k_l = 0 and lambda = 0 for the spiked covariance model.
    """
    coeff = -2. * (1 + noise_std**2 + gamma * rho**2)
    rho_coeff = (1 - psi) * gamma * (1 + gamma) / (1 + gamma * psi)**2
    sigma_coeff = -1. / (1 - psi) 
    main_term = rho_coeff * rho**2 + sigma_coeff * noise_std**2
    dG_dlambda_dk = coeff * main_term
    return dG_dlambda_dk


def visualize_mixed_partial_at_zero(psi, gammas, rhos, noise_stds, ylim=None, save=False):
    '''
    Visualizes the mixed partial derivative of the generalization error with respect to k_l and lambda
    at k_l = 0 and lambda = 0 for the spiked covariance model.
    Arguments:
        psi: ratio of n/D
        gamma: list of spike strengths
        rhos: array of alignment values, or a single float
        noise_stds: array of noise standard deviations, or a single float
    '''
    # add description
    
    # assert that either rhos and noise_stds are 1d arrays of the same length OR one of them is a float scalar
    assert (isinstance(rhos, (float, int)) or isinstance(noise_stds, (float, int)) or (len(rhos) == len(noise_stds)))

    # assert that noise_stds has no zero values 
    assert np.all(noise_stds > 0), "noise_stds must be greater than 0"

    SNRs = rhos**2 / noise_stds**2

    # sample colors from a nice smooth colormap for gammas
    colors = plt.cm.viridis(np.linspace(0, 1, len(gammas)))

    # make a beautiful, paper-ready ML theory style plot of SNRs against dG_dlambda_dks
    # make the plot grid style white, with grid lines in light gray, and the axes spines in black
    # make sure to label the point where each curve crosses the x-axis
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_facecolor('white')
    ax.grid(True, color='lightgray', linestyle='--')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    for i, gamma in enumerate(gammas):
        dG_dlambda_dks = dG_dlambda_dk_at_zero(psi, gamma, rhos, noise_stds)
        ax.plot(SNRs, dG_dlambda_dks, linestyle='-', color=colors[i], lw=2.5, label=f'$\\gamma$={gamma:g}')
        # find point where dG_dlambda_dks crosses the x-axis and mark it with a dot
        if np.any(dG_dlambda_dks < 0):
            crossing_index = np.where(dG_dlambda_dks < 0)[0][0]
            ax.scatter(SNRs[crossing_index], dG_dlambda_dks[crossing_index], color=colors[i], s=50, zorder=5)
    crit_SNR = psi**2 / (1 - psi)**2
    ax.axvline(crit_SNR, color='black', linestyle='dotted', lw=1.5)
    ax.axhline(0, color='gray', linestyle='--', lw=1.5)

    # shade the negative y-axis region in light blue, and the positive y-axis region in light red
    ax.fill_between(SNRs, dG_dlambda_dks.min()-0.5, 0, color='lightblue', alpha=0.2)
    ax.fill_between(SNRs, 0, dG_dlambda_dks.max()+0.5, color='lightcoral', alpha=0.2)

    ax.set_xlabel('Signal-to-Noise Ratio $\\left( \\rho^2/\\sigma^2\\right)$', fontsize=16)
    ax.set_ylabel('$\\left. \\frac{\\partial^2 G}{\\partial k_\\ell \\partial \\lambda} \\right|_{k_\\ell=0, \\lambda=0}$', fontsize=22)
    ax.set_title(f'Local Feature Learning Advantage \nEmerges above a Critical SNR', fontsize=17)
    # increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=16)
    # make legend have 2 rows
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=14, title='Spike Strength $\\gamma$', title_fontsize=14, ncol=2)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlim(SNRs.min(), SNRs.max())
    if save:
        plt.savefig(f"paper_figures/feature_learning_advantage_psi={psi:g}.pdf", bbox_inches='tight')
    else:
        plt.show()


def depth_lambda_dG_dk_heatmap(n, D, spike_strength, rho, noise_std):
    q = D / n
    lambdas = np.arange(0, 1.01, 0.01)
    k_ls = np.arange(0, 1.01, 0.01)
    dG_dks = np.zeros((len(lambdas), len(k_ls)), dtype=complex)

    k_sub = k_ls[0::20]
    lambdas_sub = lambdas[0::20]

    for i in range(len(lambdas)):
        lmbda = lambdas[i]
        dG_dk_row, _, _ = compute_spiked_covariance_dG_dc(n, D, k_ls, lmbda, spike_strength, rho, noise_std)
        dG_dks[i] = dG_dk_row
    
    dG_dks = np.real(dG_dks)
    fig, ax = plt.subplots()
    img = ax.imshow(dG_dks, cmap='viridis', origin='upper') # 'upper' is typical for matrices
    
    k_indices = np.arange(0, len(k_ls), 20)
    lambda_indices = np.arange(0, len(lambdas), 20)
    ax.set_xticks(k_indices)
    ax.set_yticks(lambda_indices)

    # 4. Set the custom labels
    ax.set_xticklabels(np.round(k_sub, 2))
    ax.set_yticklabels(np.round(lambdas_sub, 2))

    # Optional: Add a color bar and ensure layout is nice
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    plt.xlabel("$k_\\ell$")
    plt.ylabel("$\\lambda$")
    plt.title("$dG/dk_\\ell$")
    plt.show()

    print(f'Max derivative: {dG_dks.max()}')
    print(f'Min derivative: {dG_dks.min()}')


def explore_depth_effect_on_gen_error(n, D, ridge_lambda, spike_strength, rho, noise_std):
    # L = 50
    # layers = np.arange()
    # c_ls = np.arange()

    ks = np.arange(0, 10.01, 0.01) #10.

    _, _, feat_gen_errors = compute_spiked_covariance_model_bias_and_variance(n, D, ks, ridge_lambda, spike_strength, rho, noise_std)

    plt.figure()
    plt.plot(ks, feat_gen_errors, lw=2.4)
    plt.xlabel('Feature Learning Update Strength ($k_\\ell$)')
    plt.ylabel('Generalization Error')
    plt.show()

    return

def generate_gen_error_advantage_curve():
    return


# ---------------------------------------------------------------------------
# SNR phase diagram: Delta* = min_lambda G_feat - min_lambda G_init over (gamma, rho^2/sigma^2)
# ---------------------------------------------------------------------------
def _gen_error_at_lambda(ridge_lambda, psi, gamma, rho, sigma, k_l):
    """Asymptotic generalization error at a single ridge lambda (only psi matters)."""
    D_ref = 1000.0
    n_ref = psi * D_ref
    _, _, G = compute_spiked_covariance_model_bias_and_variance(
        n_ref, D_ref, k_l, ridge_lambda, gamma, rho, sigma)
    return float(G)

def _gen_error_at_lambda_with_u(u, psi, gamma, rho, sigma, k_l):
    ridge_lambda = _convert_u_to_lambda(u)
    return _gen_error_at_lambda(ridge_lambda, psi, gamma, rho, sigma, k_l)


def _convert_lambda_to_u(ridge_lambda):
    """Convert ridge lambda to u = lambda / (1 + lambda) in [0, 1)."""
    return ridge_lambda / (1. + ridge_lambda)

def _convert_u_to_lambda(u):
    """Convert u = lambda / (1 + lambda) in [0, 1) to ridge lambda."""
    return u / (1. - u)


def min_gen_error_over_lambda(psi, gamma, rho, sigma, k_l):
                              # lambda_max=15.0, n_coarse=151, max_extend=5):
    """
    Robustly minimize the asymptotic generalization error over lambda >= 0.

    Strategy: (1) coarse grid over [0, lambda_max] to globally bracket the min
    (guards against multimodality); (2) if the coarse argmin sits on the upper
    boundary, double lambda_max and retry (never miss a min beyond the range);
    (3) refine within the winning bracket via a bounded scalar optimizer.

    Returns (G_min, lambda_star).
    """
    # lam_hi = float(lambda_max)
    # grid = Gs = idx = None
    # for _ in range(max_extend):
    #     grid = np.linspace(0.0, lam_hi, n_coarse)
    #     Gs = np.array([_gen_error_at_lambda(l, psi, gamma, rho, sigma, k_l) for l in grid])
    #     idx = int(np.argmin(Gs))
    #     if idx < len(grid) - 1:
    #         break
    #     lam_hi *= 2.0   # min at the top boundary -> extend and retry

    # lo = grid[max(idx - 1, 0)]
    # hi = grid[min(idx + 1, len(grid) - 1)]
    # G_best, lam_best = float(Gs[idx]), float(grid[idx])
    # if hi > lo:
    #     res = minimize_scalar(
    #         _gen_error_at_lambda, bounds=(lo, hi), method="bounded",
    #         args=(psi, gamma, rho, sigma, k_l), options={"xatol": 1e-7})
    #     if float(res.fun) <= G_best:
    #         G_best, lam_best = float(res.fun), float(res.x)
    u_bounds = (0., 1.)
    res = minimize_scalar(
            _gen_error_at_lambda_with_u, bounds=u_bounds, method="bounded",
            args=(psi, gamma, rho, sigma, k_l), options={"xatol": 1e-7})
    G_best, u_best = float(res.fun), float(res.x)
    lam_best = _convert_u_to_lambda(u_best)
    return G_best, lam_best


def _verify_min_finder_varying_rho(psi, sigma, k_l, cells, lambda_max, n_dense=5000, rng_seed=0):
    """
    Sanity check: on a random subset of (gamma, rho) cells, recompute the min
    over lambda with an ultra-dense grid and compare to min_gen_error_over_lambda.
    Returns the worst absolute discrepancy found.
    """
    rng = np.random.default_rng(rng_seed)
    picks = cells[rng.choice(len(cells), size=min(len(cells), 150), replace=False)]
    worst = 0.0
    for gamma, rho in picks:
        for kl in (0.0, k_l):
            G_fast, _ = min_gen_error_over_lambda(psi, gamma, rho, sigma, kl) #, lambda_max)
            dense = np.linspace(0.0, lambda_max, n_dense)
            G_dense = min(_gen_error_at_lambda(l, psi, gamma, rho, sigma, kl) for l in dense)
            worst = max(worst, abs(G_fast - G_dense))
    return worst

def _verify_min_finder_varying_sigma(psi, rho, k_l, cells, lambda_max, n_dense=10_000, rng_seed=0):
    """
    Sanity check: on a random subset of (gamma, sigma) cells, recompute the min
    over lambda with an ultra-dense grid and compare to min_gen_error_over_lambda.
    Returns the worst absolute discrepancy found.
    """
    rng = np.random.default_rng(rng_seed)
    picks = cells[rng.choice(len(cells), size=min(len(cells), 150), replace=False)]
    worst = 0.0
    for gamma, sigma in picks:
        for kl in (0.0, k_l):
            G_fast, _ = min_gen_error_over_lambda(psi, gamma, rho, sigma, kl) #, lambda_max)
            dense = np.linspace(0.0, lambda_max * (1. + sigma**2), n_dense * int(1 + sigma**2))
            G_dense = min(_gen_error_at_lambda(l, psi, gamma, rho, sigma, kl) for l in dense)
            worst = max(worst, abs(G_fast - G_dense))
    return worst


def compute_snr_phase_diagram(psi=0.5, sigma=0.2, k_l=10.0,
                              gamma_max=20.0, gamma_step=0.1,
                              snr_max=25.0, snr_step=0.1,
                              lambda_max=20.0, out_dir="snr_phase_data",
                              save=True, verify=True, rho=None, vary_sigma=False):
    """
    Build the phase diagram of Delta* = min_lambda G_feat - min_lambda G_init.

    x-axis: gamma in [0, gamma_max]. y-axis: rho^2/sigma^2 in [0, snr_max],
    sampled UNIFORMLY (rho = sigma * sqrt(snr), clipped to [0, 1]). For each cell,
    G_feat uses k_l and G_init uses k_l = 0, each minimized over lambda >= 0.
    """
    gammas = np.round(np.arange(0.0, gamma_max + 1e-9, gamma_step), 6)
    snrs = np.round(np.arange(0.0, snr_max + 1e-9, snr_step), 6)
    fixed_tag = f'sigma={sigma:g}' if not vary_sigma else f'rho={rho:g}'
    if vary_sigma:
        assert sigma is None, "sigma must be None if vary_sigma is True"
        assert rho is not None, "rho must be provided if vary_sigma is True"
        snrs = np.round(np.arange(snr_step, snr_max + 1e-9, snr_step), 6)
        snrs = np.concatenate((np.array([snr_step/10.]), snrs))
    else:
        assert rho is None, "rho must be None if vary_sigma is False"
        assert sigma is not None, "sigma must be provided if vary_sigma is False"

    Delta = np.full((len(snrs), len(gammas)), np.nan)
    Gfeat = np.full_like(Delta, np.nan)
    Ginit = np.full_like(Delta, np.nan)

    print(f"[snr phase] psi={psi} {fixed_tag} k_l={k_l} | "
          f"{len(gammas)} gammas x {len(snrs)} snr rows", flush=True)
    for i, snr in enumerate(snrs):
        curr_rho, curr_sigma = rho, sigma
        if vary_sigma:
            curr_sigma = np.sqrt(curr_rho**2 / snr)
        else:
            curr_rho = curr_sigma * np.sqrt(snr)
            assert curr_rho <= 1.0 
        # rho = min(curr_sigma * np.sqrt(snr), 1.0)   # rho^2/sigma^2 = snr

        for j, g in enumerate(gammas):
            Gf, _ = min_gen_error_over_lambda(psi, g, curr_rho, curr_sigma, k_l)
            Gi, _ = min_gen_error_over_lambda(psi, g, curr_rho, curr_sigma, 0.0)
            Gfeat[i, j] = Gf
            Ginit[i, j] = Gi
            Delta[i, j] = Gf - Gi
        if i % 25 == 0:
            print(f"  row {i+1}/{len(snrs)} (snr={snr:g})", flush=True)

    worst = None
    if verify and not vary_sigma:
        cells = np.array([(g, min(sigma * np.sqrt(s), 1.0)) for s in snrs for g in gammas])
        worst = _verify_min_finder_varying_rho(psi, sigma, k_l, cells, lambda_max)
        print(f"[snr phase] verification: worst |fast-dense| min discrepancy = {worst:.2e}",
              flush=True)
    if verify and vary_sigma:
        cells = np.array([(g, np.sqrt(rho**2 / s)) for s in snrs for g in gammas])
        worst = _verify_min_finder_varying_sigma(psi, curr_rho, k_l, cells, lambda_max)
        print(f"[snr phase] verification: worst |fast-dense| min discrepancy = {worst:.2e}",
              flush=True)

    results = dict(
        kind="snr_phase_diagram", psi=psi, sigma=sigma, k_l=k_l,
        gammas=gammas, snrs=snrs, Delta=Delta, Gfeat=Gfeat, Ginit=Ginit,
        lambda_max=lambda_max, verify_worst=worst, vary_sigma=vary_sigma, rho=rho,
    )
    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = (f"snr_phase_psi={psi:g}_{fixed_tag}_kl={k_l:g}"
                 f"_gmax={gamma_max:g}_snrmax={snr_max:g}.pkl")
        path = os.path.join(out_dir, fname)
        with open(path, "wb") as f:
            pkl.dump(results, f)
        print(f"[snr phase] saved -> {path}", flush=True)
    return results


if __name__ == "__main__":
    compute_snr_phase_diagram(psi=0.5, sigma=0.2, k_l=10.0,
                              gamma_max=20.0, gamma_step=0.1,
                              snr_max=25.0, snr_step=0.1,
                              lambda_max=20.0, out_dir="snr_phase_data",
                              save=True, verify=True)
    compute_snr_phase_diagram(psi=0.5, sigma=0.5, k_l=10.0,
                              gamma_max=10.0, gamma_step=0.1,
                              snr_max=4.0, snr_step=0.01,
                              lambda_max=20.0, out_dir="snr_phase_data",
                              save=True, verify=True)
    compute_snr_phase_diagram(psi=0.5, sigma=None, k_l=10.0,
                              gamma_max=20.0, gamma_step=0.1,
                              snr_max=25.0, snr_step=0.1,
                              lambda_max=20.0, out_dir="snr_phase_data",
                              save=True, verify=True, vary_sigma=True, rho=0.5)
    print(1./0)
    gammas = [0., 1., 2., 4., 8., 16., 32., 64.]
    # visualize_mixed_partial_at_zero(psi=0.1, gammas=gammas, rhos=np.arange(0, 1.001, 0.001), noise_stds=0.4, ylim=None, save=False)

    # print(1./0)
    n = 500
    D = 1000 
    spike_strength = 10.0
    rho = 1.
    noise_std = 0.5
    # ridge_lambda = 0.1

    # k_ls = np.arange(0, 1.01, 0.01)
    # spacing = 0.01
    # dG_dk, dG_dc, dc_dk = compute_spiked_covariance_dG_dc(n, D, k_ls, ridge_lambda, spike_strength, rho, noise_std)

    # _, _, Gs = compute_spiked_covariance_model_bias_and_variance(n, D, k_ls, ridge_lambda, spike_strength, rho, noise_std)
    # dGs_manual = calculate_dG_finite_differencing(Gs, spacing)

    # plt.figure()
    # plt.plot(k_ls, dG_dk)
    # plt.plot(k_ls[:-1], dGs_manual, linestyle='dashed')
    # plt.show()

    # explore_depth_effect_on_gen_error(n, D, ridge_lambda, spike_strength, rho, noise_std)

    # depth_lambda_dG_dk_heatmap(n, D, spike_strength, rho, noise_std)
    # # print(1./0)

    k_l = 10.
    lambdas = np.arange(0., 10000, 100.)

    init_biases = np.zeros_like(lambdas)
    init_variances = np.zeros_like(lambdas)
    init_gen_errors = np.zeros_like(lambdas)

    feat_biases = np.zeros_like(lambdas)
    feat_variances = np.zeros_like(lambdas)
    feat_gen_errors = np.zeros_like(lambdas)


    for i in range(len(lambdas)):
        ridge_lambda = lambdas[i]
        # init_bias, init_variance, init_gen_error = compute_linear_model_bias_and_variance(n, D, ridge_lambda, noise_std)
        init_bias, init_variance, init_gen_error = compute_spiked_covariance_model_bias_and_variance(n, D, 0., ridge_lambda, spike_strength, rho, noise_std)

        init_biases[i] = init_bias
        init_variances[i] = init_variance 
        init_gen_errors[i] = init_gen_error 

        # feat_bias, feat_variance, feat_gen_error = compute_feature_learning_model_bias_and_variance(n, D, beta_coeff, ridge_lambda, noise_std)
        feat_bias, feat_variance, feat_gen_error = compute_spiked_covariance_model_bias_and_variance(n, D, k_l, ridge_lambda, spike_strength, rho, noise_std)

        feat_biases[i] = feat_bias
        feat_variances[i] = feat_variance 
        feat_gen_errors[i] = feat_gen_error 

    plt.figure()
    # plt.plot(lambdas, init_biases, color='gray', lw=2., linestyle='dashed', label='Bias')
    # plt.plot(lambdas, init_variances, color='gray', lw=2., linestyle='dotted', label='Variance')
    plt.plot(lambdas, init_gen_errors, color='gray', lw=2.5, label='Generalization Error')

    # plt.plot(lambdas, feat_biases, color='royalblue', lw=2., linestyle='dashed', label='Bias')
    # plt.plot(lambdas, feat_variances, color='royalblue', lw=2., linestyle='dotted', label='Variance')
    plt.plot(lambdas, feat_gen_errors, color='royalblue', lw=2.5, label='Generalization Error')

    # plot blue and green stars at minimum of generalization error for init and feat learn, respectively
    plt.scatter(lambdas[np.argmin(init_gen_errors)], np.min(init_gen_errors), color='gray', marker='o', s=50, label='Init Min Gen Error')
    plt.scatter(lambdas[np.argmin(feat_gen_errors)], np.min(feat_gen_errors), color='royalblue', marker='o', s=50, label='Feat Learn Min Gen Error')

    # plt.ylim(0, 0.65)
    plt.xlabel('Ridge Regularization Strength')
    # plt.legend()
    plt.show()
    # plt.savefig(f'bias_variance_plots/n={n}_D={D}_psi={n/D:.2f}_rho={rho}_gamma={spike_strength}_noise={noise_std}.pdf', bbox_inches='tight')

    print(f'Init min gen error: {np.min(init_gen_errors)} at lambda={lambdas[np.argmin(init_gen_errors)]}')
    print(f'Feat learn min gen error: {np.min(feat_gen_errors)} at lambda={lambdas[np.argmin(feat_gen_errors)]}')