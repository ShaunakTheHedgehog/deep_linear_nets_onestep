import jax
import jax.numpy as jnp
import jax.random as jr
from jax.random import multivariate_normal
from jax.scipy.linalg import sqrtm
from jax import grad, vmap, jit, value_and_grad
from jax import lax
from jax import debug
from jax.tree_util import tree_map
from jax.lax import fori_loop
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

from kernel_ridge_regression import generate_spiked_covariance, generate_random_alignment_vector, generate_data
from kernel_ridge_regression import compute_gen_error
from stieltjes_asymptotics import compute_spiked_covariance_model_bias_and_variance


def get_nonlinearity(nln):
    """
    nln = nonlinearity to use

    Returns the JAX function corresponding to the specified nonlinearity.
    """
    if nln == "relu":
        return jax.nn.relu
    elif nln == "tanh":
        return jnp.tanh
    elif nln == "erf":
        return jax.scipy.special.erf
    elif nln == "linear":
        return lambda x: x
    else:
        raise ValueError(f"Unknown nonlinearity {nln}")


def init_params(input_dim, width, depth, key):
    """
    Arguments:

    input_dim = input dimension (D)
    width = neural network hidden size (N)
    depth = number of hidden layers (L)

    Returns a PyTree of NN weights.
    """
    keys = jr.split(key, 4)
    
    params = {
        "W0": jr.normal(keys[1], (width, input_dim)),
        "W_hidden": jr.normal(keys[2], (depth-1, width, width)),  # pre-stacked intermediate weight matrices
        "w_L": jr.normal(keys[3], (width,))
    }

    return params


def hidden_state_features(params, x, nln, layer):
    """
    Arguments:

    params  :   weights of NN
    x       :   input, with shape (D, )
    nln     :   nonlinearity used
    layer   :   which layer to compute features for

    Returns phi(h^layer) across a batch of inputs.
    """
    phi = get_nonlinearity(nln)
    N, D = params['W0'].shape
    
    h = (params['W0'] @ x) / jnp.sqrt(D)

    # if layer == 1:
    #     return phi(h)
    
    # def forward_step(h, W_l):
    #     return (W_l @ phi(h)) / jnp.sqrt(N), None
    
    # h, _ = lax.scan(forward_step, h, params['W_hidden'][:layer-1])
    def forward_step(h, W_l):
        return (W_l @ phi(h)) / jnp.sqrt(N), None
    h, _ = lax.scan(forward_step, h, params['W_hidden'][:layer-1])

    return phi(h)   # shape (n, N)

batched_hidden_state_features = jax.jit(
    vmap(hidden_state_features, in_axes=(None, 0, None, None)), 
    static_argnames=("nln", "layer")
    )


@partial(jax.jit, static_argnames=("nln", "layer"))
def get_feature_kernel(params, X, nln, layer):
    """
    Arguments:

    params  :   weights of NN
    X       :   batch of training inputs, with shape (n, D)
    nln     :   nonlinearity used
    layer   :   layer of the feature kernel to be computed

    Computes the feature kernel at the specified layer of the network.
    """
    features = batched_hidden_state_features(params, X, nln, layer) # shape: (n, N)
    return (features @ features.T / features.shape[1])


def forward(params, x, nln):
    """
    Arguments:

    params  :   weights of NN
    x       :   input, with shape (D, )
    nln     :   nonlinearity used

    Returns batch of outputs of mean-field parameterized neural network.
    """
    phi = get_nonlinearity(nln)
    N, D = params['W0'].shape
    
    h = (params['W0'] @ x) / jnp.sqrt(D)
    
    def forward_step(h, W_l):
        return (W_l @ phi(h)) / jnp.sqrt(N), None
    
    h, _ = lax.scan(forward_step, h, params['W_hidden'])
    
    return jnp.dot(params['w_L'], phi(h)) / N

batched_forward = vmap(forward, in_axes=(None, 0, None))


@partial(jax.jit, static_argnames=("nln",))
def train_step(params, X, y, nln, eta0):
    """
    Arguments:

    params  :   weights of NN
    X       :   batched inputs, with shape (n, D)
    y       :   batched outputs, with shape (n,)
    nln     :   nonlinearity used
    eta0    :   base learning rate

    Computes one step of batch GD and updates network parameters.
    """
    def mse_loss(p):
        f = batched_forward(p, X, nln)
        return 0.5 * jnp.mean((f - y) ** 2)
    
    N = params["W0"].shape[0]
    eta = eta0 * N

    loss, grads = value_and_grad(mse_loss)(params)

    params = jax.tree_util.tree_map(
        lambda p, g: p - eta * g,
        params, grads
    )

    return params, loss


def train_network(params, X, y, nln, eta0, num_steps, print_loss_every=1):
    """
    Trains a feedforward deep neural network by GD for a specified number of steps.
    Arguments:

    params      :   weights of NN
    X           :   batched inputs, with shape (n, D)
    y           :   batched outputs, with shape (n,)
    nln         :   nonlinearity used
    eta0        :   base learning rate
    num_steps   :   number of GD steps done during training

    Returns network parameters after training for the specified number of GD steps.
    """
    for step in range(num_steps):
        params, loss = train_step(params, X, y, nln, eta0)

        if step % print_loss_every == 0:
            print(f"step {step}, loss {loss:.4f}")

    return params


def estimate_KRR_gen_error(params, X, y, nln, layer, w_star, v, num_test_samples, ridge_lambda, spike_strength, key, num_trials=1):
    N, D = params['W0'].shape
    n = len(y)
    m = num_test_samples
    gamma = spike_strength

    # for spiked covariance Sigma = I_D + \gamma v v^T
    Sigma_sqrt = jnp.eye(D) + (jnp.sqrt(1 + gamma) - 1) * jnp.outer(v, v)

    h_X = batched_hidden_state_features(params, X, nln, layer)  # n x N
    A = (1. / N) * h_X @ h_X.T + ridge_lambda * jnp.eye(n)
    A_inv_y = jnp.linalg.solve(A, y)    # size n

    @jit
    def compute_single_trial(key):
        X_tilde = jr.normal(key, (m, D)) @ Sigma_sqrt
        h_X_tilde = batched_hidden_state_features(params, X_tilde, nln, layer)
        K_overlap = (1./N) * h_X_tilde @ h_X.T
        y_true = (1. / jnp.sqrt(D)) * X_tilde @ w_star
        f_hat = K_overlap @ A_inv_y
        
        # Return both predictions and true values for MSE computation
        return f_hat, y_true

    # Generate independent random keys for each trial
    keys = jr.split(key, num_trials)

    # Vectorize over trials
    f_hat_all, y_true_all = jax.vmap(compute_single_trial)(keys)

    error = jnp.mean((f_hat_all - y_true_all)**2)
    return error 


if __name__=="__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    key = jr.PRNGKey(17)
    skeys = jr.split(key, 2)
    noise_std = 0.3
    rho = 0.5
    spike_strength = 10.0
    n = 50
    D = 100
    N = 1_000
    L = 3
    num_test_samples = 200

    # first, generate a fixed unit vector v for the spike direction
    v = np.random.randn(D)
    v /= np.linalg.norm(v)

    # next, generate the spiked covariance for the Gaussian input distribution
    Sigma = generate_spiked_covariance(D, v, spike_strength)

    # now, generate a fixed w_star vector with alignment rho with v
    w_star = generate_random_alignment_vector(v, rho)
    X, y, _ = generate_data(D, n, Sigma, noise_std=noise_std, w_star=w_star)
    
    X_data = jnp.asarray(X).T
    y = jnp.asarray(y)
    print(X_data.shape)
    print(y.shape)

    params_0 = init_params(D, N, L, skeys[0])
    eta0_base = 1e-1
    eta0 = eta0_base * jnp.sqrt(D)
    num_steps = 2
    nln = 'erf'

    params_1 = train_network(params_0, X_data, y, nln, eta0, num_steps=1, print_loss_every=1)
    params_2 = train_network(params_0, X_data, y, nln, eta0, num_steps=2, print_loss_every=1)
    params_3 = train_network(params_0, X_data, y, nln, eta0, num_steps=3, print_loss_every=1)

    l = L 
    k_l = (eta0_base**2) * l**2
    beta_l = k_l * D / (n**2)

    # empirical kernel & gen error
    # Phi_l = get_feature_kernel(params, X_data, nln, l)

    lambdas = np.arange(0, 1.001, 0.01)
    gen_errors_0 = np.zeros_like(lambdas)
    gen_errors_1 = np.zeros_like(lambdas)
    gen_errors_2 = np.zeros_like(lambdas)
    gen_errors_3 = np.zeros_like(lambdas)
    baseline_theory = np.zeros_like(lambdas)
    one_step_theory = np.zeros_like(lambdas)

    key = skeys[1]
    for i in range(len(lambdas)):
        _, key = jr.split(key)
        ridge_lambda = lambdas[i]
        gen_errors_0[i] = estimate_KRR_gen_error(params_0, X_data, y, nln, l, w_star, v, num_test_samples, ridge_lambda, spike_strength, key, num_trials=50)
        gen_errors_1[i] = estimate_KRR_gen_error(params_1, X_data, y, nln, l, w_star, v, num_test_samples, ridge_lambda, spike_strength, key, num_trials=50)
        gen_errors_2[i] = estimate_KRR_gen_error(params_2, X_data, y, nln, l, w_star, v, num_test_samples, ridge_lambda, spike_strength, key, num_trials=50)
        gen_errors_3[i] = estimate_KRR_gen_error(params_3, X_data, y, nln, l, w_star, v, num_test_samples, ridge_lambda, spike_strength, key, num_trials=50)
        # _, _, baseline_theory[i] = compute_spiked_covariance_model_bias_and_variance(n, D, 0., ridge_lambda, spike_strength, rho, noise_std)
        # _, _, one_step_theory[i] = compute_spiked_covariance_model_bias_and_variance(n, D, k_l, ridge_lambda, spike_strength, rho, noise_std)

    
    plt.figure()
    plt.plot(lambdas, gen_errors_0, label='0 GD steps')
    plt.plot(lambdas, gen_errors_1, label='1 GD step')
    plt.plot(lambdas, gen_errors_2, label='2 GD steps')
    plt.plot(lambdas, gen_errors_3, label='3 GD steps')
    # plt.plot(lambdas, one_step_theory, linestyle='dashed', label='1 step theory')
    # plt.plot(lambdas, baseline_theory, linestyle='dashed', label='Baseline theory')
    plt.xlabel('$\\lambda$')
    plt.title('Generalization Error using Learned Features')
    plt.legend()
    plt.show()
        