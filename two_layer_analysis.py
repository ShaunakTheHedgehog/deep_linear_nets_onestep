import jax
import jax.numpy as jnp
import jax.random as jr
from jax.random import multivariate_normal
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax import debug
from jax.tree_util import tree_map
import optax
import optax.tree_utils as otu
from functools import partial
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import functools
import random
import math

def two_layer_linear_net(params, x, gamma0):
    W0 = params['W0']   # N x D
    w1 = params['w1']   # N

    D = W0.shape[1]
    N = W0.shape[0]

    h = 1./jnp.sqrt(D) * W0 @ x                 # N
    f_pre = 1./jnp.sqrt(N) * jnp.dot(w1, h)     # 1
    f = 1./(gamma0 * jnp.sqrt(N)) * f_pre       # 1

    return f, h 

def initialize_weights(N, D, key):
    skeys = jr.split(key, 3)
    params = {'W0' : jr.normal(skeys[1], (N, D)), 'w1' : jr.normal(skeys[2], (N,))}
    return params

batched_two_layer_linear_net = vmap(two_layer_linear_net, in_axes=(None, 0, None))

def compute_grads(params, x, y, gamma0, eta0, learning_rule, rho, key):
    assert learning_rule in ['GF', 'p-FA', 'DFA', 'Hebb', 'GLN']
    if learning_rule != 'p-FA':
            assert rho is None

    W0 = params['W0']   # N x D
    w1 = params['w1']   # N

    D = W0.shape[1]
    N = W0.shape[0]

    @jit
    def _compute_pseudograd(delta, w1, h):
        g_tilde = jnp.zeros(N)
        if learning_rule == 'GF':
            g_tilde = w1 
        elif learning_rule == 'p-FA':
            w_tilde = jr.normal(key, (N, ))
            g_tilde = rho * w1 + jnp.sqrt(1 - rho**2) * w_tilde 
        elif learning_rule == 'DFA':
            g_tilde = jr.normal(key, (N, ))
        elif learning_rule == 'Hebb':
            g_tilde = delta * h
        else:
            print('GLNs not yet supported!')
            print(1./0)
        
        return g_tilde

    f, h = two_layer_linear_net(params, x, gamma0)
    delta = y - f
    g_tilde = _compute_pseudograd(delta, w1, h)

    w1_grad = eta0 * gamma0 * (delta * h)
    W0_grad = eta0 * gamma0/jnp.sqrt(D) * (delta * jnp.outer(g_tilde, x))
    return w1_grad, W0_grad

batched_compute_grads = vmap(compute_grads, in_axes=(None, 0, 0, None, None, None, None, None))

def compute_fullbatch_grads(params, xs, ys, gamma0, eta0, learning_rule, rho, key):
    w1_grads, W0_grads = batched_compute_grads(params, xs, ys, gamma0, eta0, learning_rule, rho, key)
    w1_grad = jnp.mean(w1_grads, axis=0)
    W0_grad = jnp.mean(W0_grads, axis=0)
    return w1_grad, W0_grad 

def run_onestep_update(params, xs, ys, gamma0, eta0, learning_rule, rho, key):
    w1_grad, W0_grad = compute_fullbatch_grads(params, xs, ys, gamma0, eta0, learning_rule, rho, key)
    params['w1'] = params['w1'] + w1_grad
    params['W0'] = params['W0'] + W0_grad
    return params 

def single_index_target(beta, xs):
    D = xs.shape[1]
    return 1./D * (xs @ beta)

def compute_empirical_feature_kernel(params, xs):
    W0 = params['W0']
    hs = 1./jnp.sqrt(D) * xs @ W0.T     # n x N 
    N = hs.shape[1]

    Phi = 1./N * hs @ hs.T
    return Phi 

def compute_small_time_theoretical_feature_kernels(xs, ts, beta, Sigma, gamma0, learning_rule, init_feature_kernel=None):
    assert learning_rule in ['GF', 'p-FA', 'DFA', 'Hebb']
    D = xs.shape[1]
    Phi_init = (1./D) * xs @ xs.T 
    if init_feature_kernel is not None:
        Phi_init = init_feature_kernel

    Phis = None 
    if learning_rule != 'Hebb':
        X_tilde = xs @ Sigma @ beta 
        d2_Phi = (gamma0**2 * 1./D**4) * jnp.outer(X_tilde, X_tilde)
        coeff = 2.
        if learning_rule == 'GF':
            coeff = 4. 
        
        delta_Phi = 0.5 * coeff * d2_Phi[..., None] * ts**2       # n x n x T
        Phis = Phi_init[..., None] + delta_Phi           # n x n x T
    else:
        d_Phi = (2. * gamma0 / D**4) * ((xs @ Sigma @ xs.T) * (jnp.dot(Sigma @ beta, beta)) + 2 * jnp.outer(xs @ Sigma @ beta, xs @ Sigma @ beta))

        b_S_b = jnp.dot(Sigma @ beta, beta) 
        b_S2_b = jnp.dot(Sigma**2 @ beta, beta)

        first_d2_term = (4. * gamma0**2 / D**7) * (b_S_b**2 * (xs @ Sigma**2 @ xs.T) + 2 * b_S_b * jnp.outer(xs @ Sigma**2 @ beta, xs @ Sigma @ beta) + 2 * b_S_b * jnp.outer(xs @ Sigma @ beta, xs @ Sigma**2 @ beta) + 4 * b_S2_b * jnp.outer(xs @ Sigma @ beta, xs @ Sigma @ beta) )
        second_d2_term = (-4. * gamma0 / D**5) * (b_S2_b * (xs @ Sigma @ xs.T) + jnp.outer(xs @ Sigma**2 @ beta, xs @ Sigma @ beta) + jnp.outer(xs @ Sigma @ beta, xs @ Sigma**2 @ beta))
        d2_Phi = first_d2_term + second_d2_term

        delta_Phi = d_Phi[..., None] * ts + 0.5 * d2_Phi[..., None] * ts**2           # n x n x T
        Phis = Phi_init[..., None] + delta_Phi           # n x n x T

    return Phis

def compute_small_time_theoretical_pseudograd_kernels(xs, ts, beta, Sigma, gamma0, rho, learning_rule, init_pseudograd_kernel=None):
    assert learning_rule in ['GF', 'p-FA', 'DFA', 'Hebb']
    D = xs.shape[1]
    n = xs.shape[0]
    G_tilde_init = jnp.ones((n, n))
    if learning_rule == 'p-FA':
        G_tilde_init *= rho 
    if learning_rule in ['DFA', 'Hebb']:
        G_tilde_init *= 0.

    if init_pseudograd_kernel is not None:
        G_tilde_init = init_pseudograd_kernel

    G_tildes = None 
    if learning_rule != 'Hebb':
        X_tilde = xs @ Sigma @ beta 
        d2_Phi = (gamma0**2 * 1./D**4) * jnp.outer(X_tilde, X_tilde)
        coeff = 2.
        if learning_rule == 'GF':
            coeff = 4. 
        
        delta_Phi = 0.5 * coeff * d2_Phi[..., None] * ts**2       # n x n x T
        Phis = Phi_init[..., None] + delta_Phi           # n x n x T
    else:
        d_Phi = (2. * gamma0 / D**4) * ((xs @ Sigma @ xs.T) * (jnp.dot(Sigma @ beta, beta)) + 2 * jnp.outer(xs @ Sigma @ beta, xs @ Sigma @ beta))

        b_S_b = jnp.dot(Sigma @ beta, beta) 
        b_S2_b = jnp.dot(Sigma**2 @ beta, beta)

        first_d2_term = (4. * gamma0**2 / D**7) * (b_S_b**2 * (xs @ Sigma**2 @ xs.T) + 2 * b_S_b * jnp.outer(xs @ Sigma**2 @ beta, xs @ Sigma @ beta) + 2 * b_S_b * jnp.outer(xs @ Sigma @ beta, xs @ Sigma**2 @ beta) + 4 * b_S2_b * jnp.outer(xs @ Sigma @ beta, xs @ Sigma @ beta) )
        second_d2_term = (-4. * gamma0 / D**5) * (b_S2_b * (xs @ Sigma @ xs.T) + jnp.outer(xs @ Sigma**2 @ beta, xs @ Sigma @ beta) + jnp.outer(xs @ Sigma @ beta, xs @ Sigma**2 @ beta))
        d2_Phi = first_d2_term + second_d2_term

        delta_Phi = d_Phi[..., None] * ts + 0.5 * d2_Phi[..., None] * ts**2           # n x n x T
        Phis = Phi_init[..., None] + delta_Phi           # n x n x T

    return Phis

def compute_eNTK_over_time(Phis, G_tildes, K_x):
    eNTK = Phis + G_tildes * K_x[..., None]
    return eNTK

def get_average_kernel_behavior(Phis):
    on_diag_avg = jnp.trace(Phis, axis1=0, axis2=1) / n  # shape (T,)

    total_sum = jnp.sum(Phis, axis=(0, 1))  # shape (T,)
    num_off_diag = n**2 - n
    off_diag_sum = total_sum - jnp.trace(Phis, axis1=0, axis2=1)
    off_diag_avg = off_diag_sum / num_off_diag  # shape (T,)

    return on_diag_avg, off_diag_avg


if __name__=="__main__":
    # Set up PRNGKey
    key = jr.PRNGKey(203)
    skeys = jr.split(key, 5)

    D = 5
    n = 500
    N = 10_000
    gamma0 = 10.0
    eta0 = 5e-4

    learning_rule = 'GF'    # gradient flow
    rho = None 

    T = 1e-1
    num_steps = int(T // eta0)
    ts = jnp.linspace(0, T, num_steps + 1, endpoint=True)   # (num_steps + 1)

    beta = jr.normal(skeys[1], (D,))

    # Define mean vector and covariance matrix
    mu = jnp.zeros(D)               # shape (D,)
    Sigma = jnp.eye(D)  # shape (D, D)

    # Sample n iid multivariate normals
    xs = jr.multivariate_normal(skeys[2], mean=mu, cov=Sigma, shape=(n,))   # n x D
    ys = single_index_target(beta, xs)

    init_params = initialize_weights(N, D, skeys[3])

    # simulate gradient flow up to time T << 1
    sskey = skeys[4]
    params = init_params
    init_feature_kernel = compute_empirical_feature_kernel(init_params, xs)
    empirical_Phis = jnp.expand_dims(init_feature_kernel, -1)
    for i in range(num_steps):
        _, sskey = jr.split(sskey, 2)
        params = run_onestep_update(params, xs, ys, gamma0, eta0, learning_rule, rho, sskey)

        curr_Phis = compute_empirical_feature_kernel(params, xs)
        empirical_Phis = jnp.concatenate([empirical_Phis, jnp.expand_dims(curr_Phis, -1)], axis=-1)
    
    theoretical_Phis = compute_small_time_theoretical_feature_kernels(xs, ts, beta, Sigma, gamma0, learning_rule, init_feature_kernel=init_feature_kernel)

    # pick some on-diagonal element and see how it evolves
    # then, do same for some off-diagonal element

    avg_ondiag_empirical, avg_offdiag_empirical = get_average_kernel_behavior(empirical_Phis)
    avg_ondiag_theoretical, avg_offdiag_theoretical = get_average_kernel_behavior(theoretical_Phis)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax[0].set_title('On-diagonal kernel elements')
    ax[0].plot(ts, avg_ondiag_empirical, color='green', lw=3)
    ax[0].plot(ts, avg_ondiag_theoretical, color='black', linestyle='dashed', lw=2)
    # ax[0].set_ylim(0, 1.5)

    ax[1].set_title('Off-diagonal kernel elements')
    ax[1].plot(ts, avg_offdiag_empirical, color='green', lw=3)
    ax[1].plot(ts, avg_offdiag_theoretical, color='black', linestyle='dashed', lw=2)
    # ax[1].set_ylim(-3, 3)

    plt.show()


    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax[0].set_title('On-diagonal kernel elements')
    ax[0].plot(ts, empirical_Phis[10, 10], color='green', lw=3)
    ax[0].plot(ts, theoretical_Phis[10, 10], color='black', linestyle='dashed', lw=2)
    # ax[0].set_ylim(1.5, 2.5)

    ax[1].set_title('Off-diagonal kernel elements')
    ax[1].plot(ts, empirical_Phis[2, 5], color='green', lw=3)
    ax[1].plot(ts, theoretical_Phis[2, 5], color='black', linestyle='dashed', lw=2)
    # ax[1].set_ylim(-2, 2)

    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax[0].set_title('On-diagonal kernel elements')
    ax[0].plot(ts, empirical_Phis[52, 52], color='green', lw=3)
    ax[0].plot(ts, theoretical_Phis[52, 52], color='black', linestyle='dashed', lw=2)
    # ax[0].set_ylim(1.5, 2.5)

    ax[1].set_title('Off-diagonal kernel elements')
    ax[1].plot(ts, empirical_Phis[23, 98], color='green', lw=3)
    ax[1].plot(ts, theoretical_Phis[23, 98], color='black', linestyle='dashed', lw=2)
    # ax[1].set_ylim(-2, 2)

    plt.show()



