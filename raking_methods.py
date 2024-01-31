"""
This module implements the raking methods in the k-dimensional case
"""

import numpy as np

from scipy.linalg import solve

def get_margin_matrix_vector(v_i, v_j, mu_i, mu_j):
    """
    In 2D, transform the I + J conditions on the margins
    into a matrix formulation
    """
    assert isinstance(v_i, np.ndarray), \
        'Coefficients for the margin in the first dimension should be a Numpy array.'
    assert isinstance(v_j, np.ndarray), \
        'Coefficients for the margin in the second dimension should be a Numpy array.'
    assert isinstance(mu_i, np.ndarray), \
        'Values of the margin in the first dimension should be a Numpy array.'
    assert isinstance(mu_j, np.ndarray), \
        'Values of the margin in the second dimension should be a Numpy array.'
    assert len(v_i) == len(mu_j), \
        'Coefficients in the first dimension and margin values in the second dimension should have the same size.'
    assert len(v_j) == len(mu_i), \
        'Coefficients in the second dimension and margin values in the first dimension should have the same size.'
    assert abs(np.sum(mu_i) - np.sum(mu_j)) < 1e-10, \
        'Sum of margins over rows and columns should be equal.'

    I = len(mu_i)
    J = len(mu_j)
    A = np.zeros((I + J, I * J))
    y = np.zeros(I + J)
    # Partial sums in the first dimension
    for i in range(0, I):
        for j in range(0, J):
            A[i, j * I + i] = A[i, j * I + i] + v_i[j]
            A[I + j, j * I + i] = A[I + j, j * I + i] + v_j[i]
    for i in range(0, I):
        y[i] = mu_i[i]
    for j in range(0, J):
        y[I + j] = mu_j[j]
    return (A, y)

def raking_chi2_distance(x, q, A, y):
    """
    Raking using the chi2 distance (mu - x)^2 / 2x.
    Input:
      x: 1D Numpy array, observed values x_n = x_i,j if n = j I + i
      q: 1D Numpy array, weights for the observations
      A: 2D Numpy array, linear constraints
      y: 1D Numpy array, partial sums
    Output:
      mu: 1D Numpy array, raked values
    """
    assert isinstance(x, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q, np.ndarray), \
        'Weights should be a Numpy array.'
    assert len(x) == len(q), \
        'Observations and weights arrays should have the same size.'
    assert isinstance(A, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert isinstance(y, np.ndarray), \
        'Partial sums should be a Numpy array.'
    assert np.shape(A)[0] == len(y), \
        'The number of linear constraints should be equal to the number of partial sums.'
    assert np.shape(A)[1] == len(x), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    y_hat = np.matmul(A, x)
    Phi = np.matmul(A, np.transpose(A * x * q))
    # Compute Moore-Penrose pseudo inverse to solve the system
    svd = np.linalg.svd(Phi)
    U = svd.U
    V = np.transpose(svd.Vh)
    S = np.diag(svd.S)
    Sinv = 1.0 / svd.S
    Sinv[np.abs(svd.S) < 1.0e-10] = 0.0
    Sinv = np.diag(Sinv)
    Phi_plus = np.matmul(np.matmul(V, Sinv), np.transpose(U))
    lambda_k = np.matmul(Phi_plus, y_hat - y)
    mu = x * (1 - q * np.matmul(np.transpose(A), lambda_k))
    return mu

def raking_general_distance(alpha, x, q, A, y, max_iter=500):
    """
    Raking using the general distance 1/alpha (x/alpha+1 (mu/x)^alpha+1 - mu + cte)
    Input:
      alpha: Scalar to define the distance
      x: 1D Numpy array, observed values x_n = x_i,j if n = j I + i
      q: 1D Numpy array, weights for the observations
      A: 2D Numpy array, linear constraints
      y: 1D Numpy array, partial sums
      max_iter: Integer, number of iterations for Newton's root finding method
    Output:
      mu: 1D Numpy array, raked values
    """
    assert isinstance(x, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(q, np.ndarray), \
        'Weights should be a Numpy array.'
    assert len(x) == len(q), \
        'Observations and weights arrays should have the same size.'
    assert isinstance(A, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert isinstance(y, np.ndarray), \
        'Partial sums should be a Numpy array.'
    assert np.shape(A)[0] == len(y), \
        'The number of linear constraints should be equal to the number of partial sums.'
    assert np.shape(A)[1] == len(x), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    if alpha == 1:
        mu = raking_chi2_distance(x, q, A, y)
    else:
        y_hat = np.matmul(A, x)
        lambda_k = np.zeros(np.shape(A)[0])
        epsilon = 1
        iter_eps = 0
        while (epsilon > 1.0e-5) & (iter_eps < max_iter):
            if alpha == 0:
                Phi = np.matmul(A, x * (1 - np.exp(- q * \
                    np.matmul(np.transpose(A), lambda_k))))
                D = np.diag(x * q * np.exp(- q * \
                    np.matmul(np.transpose(A), lambda_k)))
            else:
                Phi = np.matmul(A, x * (1 - np.power(1 - alpha * q * \
                    np.matmul(np.transpose(A), lambda_k), 1.0 / alpha)))
                D = np.diag(x * q * np.power(1 - alpha * q * \
                    np.matmul(np.transpose(A), lambda_k), 1.0 / alpha - 1))
            J = np.matmul(np.matmul(A, D), np.transpose(A))
            # Compute Moore-Penrose pseudo inverse to solve the system
            svd = np.linalg.svd(J)
            U = svd.U
            V = np.transpose(svd.Vh)
            S = np.diag(svd.S)
            Sinv = 1.0 / svd.S
            Sinv[np.abs(svd.S) < 1.0e-10] = 0.0
            Sinv = np.diag(Sinv)
            J_plus = np.matmul(np.matmul(V, Sinv), np.transpose(U))
            # Make sure that the new value of lambda is still valid
            # This could be a problem only if alpha > 0.5 or alpha < -1
            if (alpha > 0.5) or (alpha < -1.0):
                gamma = 1.0
                iter_gam = 0
                while (np.any(1 - alpha * q * np.matmul(np.transpose(A), \
                    lambda_k - np.matmul(J_plus, Phi - y_hat + y)) <= 0.0)) & \
                      (iter_gam < max_iter):
                    gamma = gamma / 2.0
                    iter_gam = iter_gam + 1
            else:
                gamma = 1.0
            epsilon = np.sum(np.abs(gamma * np.matmul(J_plus, Phi - y_hat + y)))
            lambda_k = lambda_k - gamma * np.matmul(J_plus, Phi - y_hat + y)
            iter_eps = iter_eps + 1
        if alpha == 0:
            mu = x * np.exp(- q * np.matmul(np.transpose(A), lambda_k))
        else:
            mu = x * np.power(1 - alpha * q * \
                np.matmul(np.transpose(A), lambda_k), 1.0 / alpha)
    return mu

def logit_raking(x, l, h, q, A, y, max_iter=500):
    """
    Logit raking ensuring that l < mu < h
    Input:
      x: 1D Numpy array, observed values x_n = x_i,j if n = j I + i
      l: 1D Numpy array, lower bound for the observations
      h: 1D Numpy array, upper bound for the observations
      q: 1D Numpy array, weights for the observations
      A: 2D Numpy array, linear constraints
      y: 1D Numpy array, partial sums
      max_iter: Integer, number of iterations for Newton's root finding method
    Output:
      mu: 1D Numpy array, raked values
    """
    assert isinstance(x, np.ndarray), \
        'Observations should be a Numpy array.'
    assert isinstance(l, np.ndarray), \
        'Lower bounds should be a Numpy array.'
    assert isinstance(h, np.ndarray), \
        'Upper bounds should be a Numpy array.'
    assert isinstance(q, np.ndarray), \
        'Weights should be a Numpy array.'
    assert len(x) == len(l), \
        'Observations and lower bounds arrays should have the same size.'
    assert len(x) == len(h), \
        'Observations and upper bounds arrays should have the same size.'
    assert len(x) == len(q), \
        'Observations and weights arrays should have the same size.'
    assert isinstance(A, np.ndarray), \
        'Linear constraints should be a Numpy array.'
    assert isinstance(y, np.ndarray), \
        'Partial sums should be a Numpy array.'
    assert np.shape(A)[0] == len(y), \
        'The number of linear constraints should be equal to the number of partial sums.'
    assert np.shape(A)[1] == len(x), \
        'The number of coefficients for the linear constraints should be equal to the number of observations.'

    lambda_k = np.zeros(np.shape(A)[0])
    epsilon = 1
    iter_eps = 0
    while (epsilon > 1.0e-5) & (iter_eps < max_iter):
        Phi = np.matmul(A, (l * (h - x) + h * (x - l) * \
            np.exp(- q * np.matmul(np.transpose(A), lambda_k))) / \
            ((h - x) + (x - l) * \
             np.exp(- q * np.matmul(np.transpose(A), lambda_k))))
        D = np.diag(- q * ((x - l) * (h - x) * (h - l)) / \
            np.square((h - x) + (x - l) * \
            np.exp(- q * np.matmul(np.transpose(A), lambda_k))))    
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        # Compute Moore-Penrose pseudo inverse to solve the system
        svd = np.linalg.svd(J)
        U = svd.U
        V = np.transpose(svd.Vh)
        S = np.diag(svd.S)
        Sinv = 1.0 / svd.S
        Sinv[np.abs(svd.S) < 1.0e-10] = 0.0
        Sinv = np.diag(Sinv)
        J_plus = np.matmul(np.matmul(V, Sinv), np.transpose(U))
        epsilon = np.sum(np.abs(np.matmul(J_plus, Phi - y)))
        lambda_k = lambda_k - np.matmul(J_plus, Phi - y)
        iter_eps = iter_eps + 1
    mu = (l * (h - x) + h * (x - l) * \
        np.exp(- q * np.matmul(np.transpose(A), lambda_k))) / \
        ((h - x) + (x - l) * \
        np.exp(- q * np.matmul(np.transpose(A), lambda_k)))
    return mu

