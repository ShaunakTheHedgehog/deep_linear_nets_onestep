import numpy as np
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random
import math
from matplotlib.colors import TwoSlopeNorm

from stieltjes_asymptotics import *


def compute_w_init(X, y, ridge_lambda=0.):
    '''
    Arguments: 
    X               :   D x n matrix of training inputs
    y               :   n-dim vector of targets
    ridge_lambda    :   ridge regularization strength
    '''
    D, n = X.shape
    K_x = (1./D) * (X.T @ X)
    A = K_x + ridge_lambda * np.eye(n)
    w_init = (1./D**0.5) * X @ np.linalg.pinv(A) @ y

    return w_init

def f_init(X_test, X, y, ridge_lambda=0.):
    '''
    Arguments: 
    X_test   :   D x n_test matrix of test inputs
    X        :   D x n matrix of training inputs
    y        :   n-dim vector of targets
    '''
    D, n = X.shape
    K_x = (1./D) * (X.T @ X)
    A = K_x + ridge_lambda * np.eye(n)
    w_init = (1./D**0.5) * X @ np.linalg.pinv(A) @ y
    f_inits = (1./D**0.5) * X_test.T @ w_init 

    return w_init, f_inits 


def compute_w_feat(X, y, k_l, ridge_lambda=0.):
    '''
    Arguments: 
    X               :   D x n matrix of training inputs
    y               :   n-dim vector of targets
    k_l             :   feature learning update strength
    ridge_lambda    :   ridge regularization strength
    '''
    D, n = X.shape
    K_x = (1./D) * (X.T @ X)
    A = K_x + ridge_lambda * np.eye(n)
    A_inv = np.linalg.pinv(A)
    w_init = (1./D**0.5) * X @ A_inv @ y
    # w_init = compute_w_init(X, y, ridge_lambda)
    beta = k_l * D / n**2

    num = ridge_lambda * beta * np.dot(y, K_x @ A_inv @ y)
    xy = K_x @ y
    denom = 1 + beta * np.dot(xy, A_inv @ xy)

    w_feat = (1. + num / denom) * w_init

    return w_feat


def compute_gen_error(w_star, Sigma, X, y, k_l=0., ridge_lambda=0.):
    D = len(w_star)
    w_est = compute_w_feat(X, y, k_l=k_l, ridge_lambda=ridge_lambda)
    w_error = w_star - w_est 
    G = np.dot(w_error, Sigma @ w_error) / D 

    return G 


def get_isotropic_small_lambda_coeffs(D, n, beta=0., noise_std=0.):
    q = 1. * D / n 
    psi = 1./q 
    beta_coeff = beta * (n**2 / D)
    r = 1. / ( 1 + ((1 + noise_std**2) / psi) + (1. / beta_coeff) )

    G_init_order0 = ( 1 - psi + (noise_std**2) * ( psi / (1 - psi) ) )
    G_init_order1_coeff = -2 * (noise_std**2) * (psi / (1 - psi)**3) 
    G_init_order2_coeff = ( (psi / (1 - psi)**3) + ( 3 * (noise_std**2) * psi * (1 + psi) / ((1 - psi)**5) ) ) 

    term1_order1_coeff = 2 * (noise_std**2) * (1 + noise_std**2) * r / (1 - psi)
    term1_order2_coeff = -2 * (r / (1 - psi)) * ( ((1 + noise_std**2) * (1 + 2*(noise_std**2) / ((1 - psi)**3))) - ( (noise_std**2) * (r * q * ((1 + noise_std**2)**2) - 1 - ((noise_std**2) * psi / (1 - psi)) ) ) ) 

    term2_order2_coeff = (r**2) * q * ((1 + noise_std**2)**2) * ( 1 + (noise_std**2) / (1 - psi) ) 

    G_init_coeffs = (G_init_order0, G_init_order1_coeff, G_init_order2_coeff)
    G_feat_coeffs = (G_init_order0, G_init_order1_coeff+term1_order1_coeff, G_init_order2_coeff+term1_order2_coeff+term2_order2_coeff)

    return G_init_coeffs, G_feat_coeffs 


def get_quadratic_opt_lambda(D, n, beta=0., noise_std=0.):
    G_init_coeffs, G_feat_coeffs = get_isotropic_small_lambda_coeffs(D, n, beta, noise_std)

    a0, a1, a2 = G_init_coeffs
    init_opt_lambda = -a1 / (2. * a2)
    init_opt_G = a0 - (a1**2 / (4 * a2**2))

    b0, b1, b2 = G_feat_coeffs
    feat_opt_lambda = -b1 / (2. * b2)
    feat_opt_G = b0 - (b1**2 / (4 * b2**2))

    return init_opt_lambda, init_opt_G, feat_opt_lambda, feat_opt_G



def compute_isotropic_small_lambda_gen_error(D, n, beta=0., ridge_lambda=0., noise_std=0.):
    G_init_coeffs, G_feat_coeffs = get_isotropic_small_lambda_coeffs(D, n, beta, noise_std)

    a0, a1, a2 = G_init_coeffs
    b0, b1, b2 = G_feat_coeffs
    
    G_init = a0 + a1 * ridge_lambda + a2 * ridge_lambda**2
    
    G_feat = b0 + b1 * ridge_lambda + b2 * ridge_lambda**2 

    return G_init, G_feat 


# create a power-law decaying w_star vector, normalized so that w_star^T \Sigma w_star = D
def generate_power_law_w_star(D, beta, Sigma):
    ns = np.arange(1, D+1)
    w_star = ns**(-beta)
    Sigma_norm_squared = np.dot(w_star, Sigma @ w_star)
    w_star *= (np.sqrt(D) / np.sqrt(Sigma_norm_squared))
    # norm = np.linalg.norm(w_star)
    # w_star *= (np.sqrt(D) / norm)
    return w_star 

# create a power-law decaying (diagonal) covariance matrix Sigma, normalized so that Tr(Sigma) = D
def generate_power_law_covariance(D, alpha):
    ns = np.arange(1, D+1)
    diag = ns**(-alpha)
    sum = np.sum(diag)
    diag *= (D / sum)       # ensure that the trace is D
    return np.diag(diag)


def generate_spiked_covariance(D, v, gamma):
    '''
    Generate a spiked covariance matrix of the form Sigma = I + gamma * v v^T
    where v is a unit vector (||v||^2 = 1)
    '''
    assert D == len(v)
    assert np.isclose(np.linalg.norm(v), 1.0), "v must be a unit vector"
    Sigma = np.eye(D) + gamma * np.outer(v, v)
    return Sigma

def generate_random_alignment_vector(v, rho):
    '''
    Generate a random vector w_star of length sqrt(D) that has alignment such that (w_star / sqrt(D)) has alignment rho with the unit vector v.
    That is, (w_star^T v) / (||w_star|| * ||v||) = rho
    '''
    assert np.isclose(np.linalg.norm(v), 1.0), "v must be a unit vector"
    D = len(v)

    # Generate a random vector
    w_random = np.random.randn(D)
    w_random = w_random - np.dot(w_random, v) * v  # make it orthogonal to v
    w_random = w_random / np.linalg.norm(w_random)  # normalize

    # Combine to get desired alignment
    w_star = rho * np.sqrt(D) * v + np.sqrt(1 - rho**2) * np.sqrt(D) * w_random
    return w_star


def generate_data(D, n, Sigma, noise_std=0., w_star=None, rng=None):
    rng = np.random.default_rng(rng)
    assert D == Sigma.shape[0]

    # Factorization (Cholesky is fastest/stable if Sigma is SPD)
    L = np.linalg.cholesky(Sigma)

    # Standard normal samples
    Z = rng.standard_normal(size=(D, n))

    # Apply covariance structure
    X = L @ Z

    # initialize some direction vector of length sqrt(D)
    if w_star is None:
        w_star = np.random.randn(D)
        w_star /= np.linalg.norm(w_star)
        w_star *= np.sqrt(D)

    epsilon = np.random.normal(size=n) * noise_std

    # generate targets
    y = (1./np.sqrt(D)) * (X.T @ w_star) + epsilon

    return X, y, w_star 

def calclulate_alpha_beta_gen_error_heatmap(D, n, alpha_exps, beta_exps, ridge_lambda, noise_std, learn_step_coeff=10., plot=True):
    psi = 1.*n/D 
    q = 1.*D/n 

    beta = learn_step_coeff * D / n**2

    gen_error_diffs = np.zeros((len(alpha_exps), len(beta_exps)))
    relative_gen_error_diffs = np.zeros((len(alpha_exps), len(beta_exps)))

    for i in range(len(alpha_exps)):
        alpha_exp = alpha_exps[i]
        for j in range(len(beta_exps)):
            beta_exp = beta_exps[j]

            Sigma = generate_power_law_covariance(D, alpha_exp)
            w_star = generate_power_law_w_star(D, beta_exp, Sigma)

            X, y, w_star = generate_data(D, n, Sigma, noise_std=noise_std, w_star=w_star)

            init_gen_error = compute_gen_error(w_star, Sigma, X, y, beta=0., ridge_lambda=ridge_lambda)
            feat_gen_error = compute_gen_error(w_star, Sigma, X, y, beta=beta, ridge_lambda=ridge_lambda)

            diff = feat_gen_error - init_gen_error
            gen_error_diffs[i, j] = diff 
            relative_gen_error_diffs[i, j] = diff / init_gen_error
    
    if plot:
        # Define normalization centered at 0
        norm1 = TwoSlopeNorm(vmin=np.min(gen_error_diffs), vcenter=0, vmax=np.max(gen_error_diffs))
        plt.figure(figsize=(8, 9))
        plt.pcolormesh(alpha_exps, beta_exps, gen_error_diffs.T, shading='auto',
                    cmap='bwr', norm=norm1)  # Note the transpose!
        plt.xlabel('$\\alpha$')
        plt.ylabel('$\\beta$')
        plt.title(f'Generalization error difference, with \n$D/n={q}, \\lambda={ridge_lambda}, \\sigma={noise_std}$')
        plt.colorbar(label='$G_f - G_i$')
        plt.show()


        norm2 = TwoSlopeNorm(vmin=np.min(relative_gen_error_diffs), vcenter=0, vmax=np.max(relative_gen_error_diffs))
        plt.figure(figsize=(8, 9))
        plt.pcolormesh(alpha_exps, beta_exps, relative_gen_error_diffs.T, shading='auto',
                    cmap='bwr', norm=norm2)  # Note the transpose!
        plt.xlabel('$\\alpha$')
        plt.ylabel('$\\beta$')
        plt.title(f'Fractional generalization error difference, with \n$D/n={q}, \\lambda={ridge_lambda}, \\sigma={noise_std}$')
        plt.colorbar(label='$\\frac{G_f - G_i}{G_i}$')
        plt.show()

    return gen_error_diffs, relative_gen_error_diffs


def spiked_covariance_model_exploration():
    D = 1000
    n = 500
    q = 1.*D/n
    psi = 1./q
    noise_std = 0.5
    rho = 0.4
    spike_strength = 5.0
    k_l = 1.
    lambdas = np.arange(0, 1.001, 0.02)

    ntrials = 100

    # first, generate a fixed unit vector v for the spike direction
    v = np.random.randn(D)
    v /= np.linalg.norm(v)

    # next, generate the spiked covariance for the Gaussian input distribution
    Sigma = generate_spiked_covariance(D, v, spike_strength)

    # now, generate a fixed w_star vector with alignment rho with v
    w_star = generate_random_alignment_vector(v, rho)

    mark_every = 0.1
    # small_lambdas = np.arange(0, 0.15001, 0.02)

    init_gen_errors = np.zeros_like(lambdas)
    feat_gen_errors = np.zeros_like(init_gen_errors)

    init_gen_error_theory = np.zeros_like(lambdas)
    feat_gen_error_theory = np.zeros_like(init_gen_error_theory)

    # now, compute trial-averaged generalization error over multiple realizations of the data-generating process
    print('Starting numerics...')

    for trial in range(ntrials):
        X, y, _ = generate_data(D, n, Sigma, noise_std=noise_std, w_star=w_star, rng=trial)

        for j in range(len(lambdas)):
            ridge_lambda = lambdas[j]
            init_gen_errors[j] += compute_gen_error(w_star, Sigma, X, y, k_l=0., ridge_lambda=ridge_lambda) / ntrials
            feat_gen_errors[j] += compute_gen_error(w_star, Sigma, X, y, k_l=k_l, ridge_lambda=ridge_lambda) / ntrials

        print(f'Completed trial {trial+1}/{ntrials} numerics.')
    
    # next, compute theoretical predictions for bias, variance, and generalization error
    for j in range(len(lambdas)):
        ridge_lambda = lambdas[j]
        init_bias, init_variance, init_gen_error_theory[j] = compute_spiked_covariance_model_bias_and_variance(n, D, 0., ridge_lambda, spike_strength, rho, noise_std)
        feat_bias, feat_variance, feat_gen_error_theory[j] = compute_spiked_covariance_model_bias_and_variance(n, D, k_l, ridge_lambda, spike_strength, rho, noise_std)
       
    print('Completed theory computations.')

    # finally, save results in a dictionary
    results_dict = {
        'D': D,
        'n': n,
        'rho': rho,
        'spike_strength': spike_strength,
        'noise_std': noise_std,
        'k_l': k_l,
        'lambdas': lambdas,
        'init_gen_errors': init_gen_errors,
        'feat_gen_errors': feat_gen_errors,
        'init_bias_theory': init_bias,
        'init_variance_theory': init_variance,
        'init_gen_error_theory': init_gen_error_theory,
        'feat_bias_theory': feat_bias,
        'feat_variance_theory': feat_variance,
        'feat_gen_error_theory': feat_gen_error_theory
    }
    with open(f'spiked_covariance_experiments/asymptotics_fixed_wstar_D_n_ratio={q}_spike_strength={spike_strength}_rho={rho}_noise_std={noise_std}_ntrials={ntrials}.pkl', 'wb') as f:
        pkl.dump(results_dict, f)

    plt.figure(figsize=(10, 8))
    plt.xlabel('Ridge Parameter (lambda)')
    plt.ylabel('Generalization Error')
    plt.plot(lambdas, init_gen_errors, linestyle='-', marker='o', markevery=mark_every,
            color='blue', label='Baseline empirics')
    plt.plot(lambdas, feat_gen_errors, linestyle='-', marker='s', markevery=mark_every,
            color='orange', label='Feature learning empirics')
    plt.plot(lambdas, init_gen_error_theory, linestyle='--', color='blue', label='Baseline theory')
    plt.plot(lambdas, feat_gen_error_theory, linestyle='--', color='orange', label='Feature learning theory')
    plt.legend()
    plt.savefig(f'spiked_covariance_experiments/asymptotic_gen_error_fixed_wstar_D_n_ratio={q}_spike_strength={spike_strength}_rho={rho}_noise_std={noise_std}_ntrials={ntrials}.png')



def numerical_exploration():
    D = 3_000
    n = 1_000
    q = 1.*D/n
    psi = 1./q
    noise_std = 0.3
    k_l = 10.
    lambdas = np.arange(0, 1.001, 0.02)

    ntrials = 20

    Sigma = np.eye(D)
    w_star = np.random.randn(D)
    w_star /= np.linalg.norm(w_star)
    w_star *= np.sqrt(D)

    mark_every = 0.1
    # small_lambdas = np.arange(0, 0.15001, 0.02)

    init_gen_errors = np.zeros_like(lambdas)
    feat_gen_errors = np.zeros_like(init_gen_errors)

    init_gen_error_theory = np.zeros_like(lambdas)
    feat_gen_error_theory = np.zeros_like(init_gen_error_theory)

    print('Starting numerics...')

    # first, compute trial-averaged generalization errors
    for trial in range(ntrials):
        # Sigma = np.eye(D)
        # w_star = None
        # Sigma = generate_power_law_covariance(D, alpha_exp)
        # w_star = generate_power_law_w_star(D, beta_exp, Sigma)

        X, y, _ = generate_data(D, n, Sigma, noise_std=noise_std, w_star=w_star, rng=trial)

        for j in range(len(lambdas)):
            ridge_lambda = lambdas[j]
            init_gen_errors[j] += compute_gen_error(w_star, Sigma, X, y, k_l=0., ridge_lambda=ridge_lambda) / ntrials
            feat_gen_errors[j] += compute_gen_error(w_star, Sigma, X, y, k_l=k_l, ridge_lambda=ridge_lambda) / ntrials

        print(f'Completed trial {trial+1}/{ntrials} numerics.')
    
    # next, compute theoretical predictions
    for j in range(len(lambdas)):
        ridge_lambda = lambdas[j]
        _, _, init_gen_error_theory[j] = compute_linear_model_bias_and_variance(n, D, ridge_lambda, noise_std)
        _, _, feat_gen_error_theory[j] = compute_feature_learning_model_bias_and_variance(n, D, k_l, ridge_lambda, noise_std)
       
    print('Completed theory computations.')

    results_dict = {
        'D': D,
        'n': n,
        'noise_std': noise_std,
        'k_l': k_l,
        'lambdas': lambdas,
        'init_gen_errors': init_gen_errors,
        'feat_gen_errors': feat_gen_errors,
        'init_gen_error_theory': init_gen_error_theory,
        'feat_gen_error_theory': feat_gen_error_theory
    }
    with open(f'isotropic_stieltjes_asymptotics_fixed_wstar_D_n_ratio={q}_noise_std={noise_std}_ntrials={ntrials}.pkl', 'wb') as f:
        pkl.dump(results_dict, f)

    # plt.figure(figsize=(10, 8))
    # plt.xlabel('Ridge Parameter (lambda)')
    # plt.ylabel('Generalization Error')
    # plt.plot(lambdas, init_gen_errors, linestyle='-', marker='o', markevery=mark_every,
    #         color='blue', label='Baseline empirics')
    # plt.plot(lambdas, feat_gen_errors, linestyle='-', marker='s', markevery=mark_every,
    #         color='orange', label='Feature learning empirics')
    # plt.plot(lambdas, init_gen_error_theory, linestyle='--', color='blue', label='Baseline theory')
    # plt.plot(lambdas, feat_gen_error_theory, linestyle='--', color='orange', label='Feature learning theory')
    # plt.legend()
    # plt.savefig(f'isotropic_stieltjes_asymptotics_fixed_wstar_D_n_ratio={q}_noise_std={noise_std}_ntrials={ntrials}.png')


def numerical_sweep_exploration():
    Ds = np.array([9_000])
    n = 3_000
    noise_std = 0.3
    phis = n / Ds
    qs = Ds / n 

    # alpha_exp = 0.5
    # beta_exp = 3.0
    beta_coeff = 10.
    betas = beta_coeff * Ds / n**2
    lambdas = np.arange(0, 1.001, 0.02)

    mark_every = 0.1
    # small_lambdas = np.arange(0, 0.15001, 0.02)

    init_gen_errors = np.zeros((len(Ds), len(lambdas)))
    feat_gen_errors = np.zeros_like(init_gen_errors)

    init_gen_error_theory = np.zeros((len(Ds), len(lambdas)))
    feat_gen_error_theory = np.zeros_like(init_gen_error_theory)

    print('Starting numerics...')

    for i in range(len(Ds)):
        D = Ds[i]
        beta = betas[i]

        Sigma = np.eye(D)
        w_star = None
        # Sigma = generate_power_law_covariance(D, alpha_exp)
        # w_star = generate_power_law_w_star(D, beta_exp, Sigma)

        X, y, w_star = generate_data(D, n, Sigma, noise_std=noise_std, w_star=w_star)

        for j in range(len(lambdas)):
            ridge_lambda = lambdas[j]
            init_gen_errors[i, j] = compute_gen_error(w_star, Sigma, X, y, beta=0., ridge_lambda=ridge_lambda)
            feat_gen_errors[i, j] = compute_gen_error(w_star, Sigma, X, y, beta=beta, ridge_lambda=ridge_lambda)

            _, _, init_gen_error_theory[i, j] = compute_linear_model_bias_and_variance(n, D, ridge_lambda, noise_std)
            _, _, feat_gen_error_theory[i, j] = compute_feature_learning_model_bias_and_variance(n, D, beta_coeff, ridge_lambda, noise_std)
            # if j < len(small_lambdas):
            #     init_gen_error_theory[i, j], feat_gen_error_theory[i, j] = compute_isotropic_small_lambda_gen_error(D, n, beta=beta, ridge_lambda=ridge_lambda, noise_std=noise_std)
        
        print(f'Completed D={D} numerics.')

    results_dict = {
        'Ds': Ds,
        'n': n,
        'noise_std': noise_std,
        'betas': betas,
        'lambdas': lambdas,
        'init_gen_errors': init_gen_errors,
        'feat_gen_errors': feat_gen_errors,
        'init_gen_error_theory': init_gen_error_theory,
        'feat_gen_error_theory': feat_gen_error_theory
    }
    with open('krr_isotropic_numerical_exploration_results.pkl', 'wb') as f:
        pkl.dump(results_dict, f)

    cmap = plt.cm.cool
    norm = matplotlib.colors.Normalize(vmin=qs.min(), vmax=qs.max())
    cmap = plt.cm.cool

    plt.figure(figsize=(10, 8))
    plt.xlabel('Ridge Parameter (lambda)')
    plt.ylabel('Generalization Error')

    for i in range(len(qs)):
        q = qs[i]

        color = cmap(norm(q))
    
        # Baseline empirics
        plt.plot(lambdas, init_gen_errors[i], linestyle='-', marker='o', markevery=mark_every,
                color=color, label=f'Baseline empirics' if i == 0 else None)
        
        # Model empirics
        plt.plot(lambdas, feat_gen_errors[i], linestyle='-', marker='s', markevery=mark_every,
                color=color, label=f'Feature learning empirics' if i == 0 else None)
        


        # Baseline theory
        plt.plot(lambdas, init_gen_error_theory[i], linestyle='--', linewidth=2,
                color='green', label=f'Baseline theory' if i == 0 else None)
        
        # Model theory
        plt.plot(lambdas, feat_gen_error_theory[i], linestyle=':', linewidth=2,
                color='green', label=f'Feature learning theory' if i == 0 else None)
    
    # print(init_gen_error_theory)

    # print(feat_gen_error_theory)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # needed for colorbar
    cbar = plt.colorbar(sm)
    cbar.set_label("D/n")

    # Legend outside the plot
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),   # center it, push down
        ncol=2,                        # spread entries horizontally
        fontsize=9,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave extra space at bottom

    plt.show()


def plot_isotropic_results():
    q = 3.0
    noise_std = 0.3
    ntrials1 = 30
    ntrials2 = 20
    tot_ntrials = ntrials1 + ntrials2
    mark_every = 0.1
    lambdas = np.arange(0, 1.01, 0.02)

    with open(f'isotropic_stieltjes_asymptotics_fixed_wstar_D_n_ratio={q}_noise_std={noise_std}_ntrials={ntrials1}.pkl', 'rb') as file1:
        dict1 = pkl.load(file1)

    with open(f'isotropic_stieltjes_asymptotics_fixed_wstar_D_n_ratio={q}_noise_std={noise_std}_ntrials={ntrials2}.pkl', 'rb') as file2:
        dict2 = pkl.load(file2)

    init_gen_error_theory = dict1['init_gen_error_theory']
    feat_gen_error_theory = dict1['feat_gen_error_theory']

    init_gen_errors1 = dict1['init_gen_errors']
    init_gen_errors2 = dict2['init_gen_errors']
    init_gen_errors = (ntrials1 * init_gen_errors1 + ntrials2 * init_gen_errors2) / tot_ntrials

    feat_gen_errors1 = dict1['feat_gen_errors']
    feat_gen_errors2 = dict2['feat_gen_errors']
    feat_gen_errors = (ntrials1 * feat_gen_errors1 + ntrials2 * feat_gen_errors2) / tot_ntrials

    plt.figure(figsize=(10, 8))
    plt.xlabel('Ridge Parameter (lambda)')
    plt.ylabel('Generalization Error')
    plt.plot(lambdas, init_gen_errors, linestyle='-', marker='o', markevery=mark_every,
            color='blue', label='Baseline empirics')
    plt.plot(lambdas, feat_gen_errors, linestyle='-', marker='s', markevery=mark_every,
            color='orange', label='Feature learning empirics')
    plt.plot(lambdas, init_gen_error_theory, linestyle='--', color='blue', label='Baseline theory')
    plt.plot(lambdas, feat_gen_error_theory, linestyle='--', color='orange', label='Feature learning theory')
    plt.legend()
    plt.savefig(f'isotropic_stieltjes_asymptotics_fixed_wstar_D_n_ratio={q}_noise_std={noise_std}_ntrials={tot_ntrials}.png')



if __name__=="__main__":
    spiked_covariance_model_exploration()
    # numerical_exploration()
    



    # D = 3_000
    # n = 500
    # q = 1.*D/n
    # alpha_exps = np.arange(0, 5.1, 0.1)[:20]
    # beta_exps = np.arange(0, 5.1, 0.1)
    # noise_std = 0.3
    # ridge_lambda = 1.0

    # # gen_error_diffs, relative_gen_error_diffs = calclulate_alpha_beta_gen_error_heatmap(D, n, alpha_exps, beta_exps, ridge_lambda, noise_std, learn_step_coeff=10., plot=False)
    
    # gen_error_diffs = np.load(f'gen_error_diffs_sigma{noise_std}_lambda{ridge_lambda}.npy')
    # gen_error_diffs = gen_error_diffs[:20]
    # relative_gen_error_diffs = np.load(f'relative_gen_error_diffs_sigma{noise_std}_lambda{ridge_lambda}.npy')

    # plt.figure(figsize=(8, 10))
    # vmax = np.max(np.abs(gen_error_diffs))
    # norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # plt.pcolormesh(alpha_exps, beta_exps, gen_error_diffs.T, shading='auto', cmap='bwr', norm=norm)
    # cbar = plt.colorbar(label=r"$G_f - G_i$")

    # plt.xlabel('$\\alpha$')
    # plt.ylabel('$\\beta$')
    # plt.title(f'Generalization error difference, with \n$D/n={q}, \\lambda={ridge_lambda}, \\sigma={noise_std}$')
    # plt.show()



