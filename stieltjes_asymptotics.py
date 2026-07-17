import numpy as np
import matplotlib.pyplot as plt 

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
    


if __name__ == "__main__":
    n = 10_000
    D = 20_000 
    spike_strength = 3.0
    rho = 0.1
    noise_std = 0.
    ridge_lambda = 0.1

    k_ls = np.arange(0, 1.01, 0.01)
    spacing = 0.01
    dG_dk, dG_dc, dc_dk = compute_spiked_covariance_dG_dc(n, D, k_ls, ridge_lambda, spike_strength, rho, noise_std)

    _, _, Gs = compute_spiked_covariance_model_bias_and_variance(n, D, k_ls, ridge_lambda, spike_strength, rho, noise_std)
    dGs_manual = calculate_dG_finite_differencing(Gs, spacing)

    plt.figure()
    plt.plot(k_ls, dG_dk)
    plt.plot(k_ls[:-1], dGs_manual, linestyle='dashed')
    plt.show()

    explore_depth_effect_on_gen_error(n, D, ridge_lambda, spike_strength, rho, noise_std)

    depth_lambda_dG_dk_heatmap(n, D, spike_strength, rho, noise_std)
    # print(1./0)

    beta_coeff = 10.
    lambdas = np.arange(0., 1.001, 0.01)

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
        feat_bias, feat_variance, feat_gen_error = compute_spiked_covariance_model_bias_and_variance(n, D, beta_coeff, ridge_lambda, spike_strength, rho, noise_std)

        feat_biases[i] = feat_bias
        feat_variances[i] = feat_variance 
        feat_gen_errors[i] = feat_gen_error 

    plt.figure()
    plt.plot(lambdas, init_biases, color='blue', lw=2., linestyle='dashed', label='Bias')
    plt.plot(lambdas, init_variances, color='blue', lw=2., linestyle='dotted', label='Variance')
    plt.plot(lambdas, init_gen_errors, color='blue', lw=2.5, label='Generalization Error')

    plt.plot(lambdas, feat_biases, color='green', lw=2., linestyle='dashed', label='Bias')
    plt.plot(lambdas, feat_variances, color='green', lw=2., linestyle='dotted', label='Variance')
    plt.plot(lambdas, feat_gen_errors, color='green', lw=2.5, label='Generalization Error')

    # plt.ylim(0, 0.65)
    plt.xlabel('Ridge Regularization Strength')
    # plt.legend()
    plt.show()
    # plt.savefig(f'spiked_covariance_bias_variance_example_q={q:.2f}_noisestd={noise_std}.png')

    print(f'Init min gen error: {np.min(init_gen_errors)} at lambda={lambdas[np.argmin(init_gen_errors)]}')
    print(f'Feat learn min gen error: {np.min(feat_gen_errors)} at lambda={lambdas[np.argmin(feat_gen_errors)]}')