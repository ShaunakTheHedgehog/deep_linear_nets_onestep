import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from stieltjes_asymptotics import *


def compute_A_inv(X, ridge_lambda):
    '''
    Arguments: 
    X               :   D x n matrix of training inputs
    ridge_lambda    :   ridge regularization strength
    '''
    D, n = X.shape
    K_x = (1./D) * (X.T @ X)
    A = K_x + ridge_lambda * np.eye(n)
    A_inv = np.linalg.pinv(A)

    return A_inv


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


def compute_w_feat(X, y, k_l, ridge_lambda=0., A_inv=None):
    '''
    Arguments: 
    X               :   D x n matrix of training inputs
    y               :   n-dim vector of targets
    k_l             :   feature learning update strength
    ridge_lambda    :   ridge regularization strength
    A_inv           :   precomputed A_inv = (K_x + ridge_lambda I_n)^(-1), where K_x = (1/D) X^T X, if available
    '''
    D, n = X.shape
    K_x = (1./D) * (X.T @ X)
    A = K_x + ridge_lambda * np.eye(n)
    A_inv = A_inv if A_inv is not None else np.linalg.pinv(A)
    w_init = (1./D**0.5) * X @ A_inv @ y
    # w_init = compute_w_init(X, y, ridge_lambda)
    beta = k_l * D / n**2

    num = ridge_lambda * beta * np.dot(y, K_x @ A_inv @ y)
    xy = K_x @ y
    denom = 1 + beta * np.dot(xy, A_inv @ xy)

    w_feat = (1. + num / denom) * w_init

    return w_feat



def compute_gen_error(w_star, Sigma, X, y, k_l=0., ridge_lambda=0., A_inv=None):
    '''
    Arguments:
    w_star          :   D-dim vector of true weights
    Sigma           :   D x D covariance matrix of the input distribution
    X               :   D x n matrix of training inputs
    y               :   n-dim vector of targets
    k_l             :   feature learning update strength
    ridge_lambda    :   ridge regularization strength
    A_inv           :   precomputed A_inv = (K_x + ridge_lambda I_n)^(-1), where K_x = (1/D) X^T X, if available
    '''

    D = len(w_star)
    w_est = compute_w_feat(X, y, k_l=k_l, ridge_lambda=ridge_lambda, A_inv=A_inv)
    w_error = w_star - w_est 
    G = np.dot(w_error, Sigma @ w_error) / D 

    return G 


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
    with open(f'asymptotics_fixed_wstar_D_n_ratio={q}_spike_strength={spike_strength}_rho={rho}_noise_std={noise_std}_ntrials={ntrials}.pkl', 'wb') as f:
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
    plt.savefig(f'asymptotic_gen_error_fixed_wstar_D_n_ratio={q}_spike_strength={spike_strength}_rho={rho}_noise_std={noise_std}_ntrials={ntrials}.png')


if __name__=="__main__":
    spiked_covariance_model_exploration()



