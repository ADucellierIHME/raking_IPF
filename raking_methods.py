"""
This module implements the raking methods in the k-dimensional case
"""

import numpy as np
import pandas as pd

from math import floor

def raking_vectorized_IPF(df, agg_vars, constant_vars=[], max_iter=500, return_num_iter=False):
    """
    Raking using the l2 distance (mu - x)^2 / 2.
    Input:
      df: Pandas dataframe, containing the columns:
          - value = Values to be raked
          - agg_var = Variables over which we want to rake.
          - constant_vars = Several other variables (not to be raked).
          - all_'agg_var'_value = Partial sums
      agg_vars: List of strings, variables over which we do the raking
      constant_vars: List of strings, several other variables (not to be raked).
      max_iter: Integer, number of iterations for IPF algorithm.
      return_num_iter: Boolean, return the number of iterations.
    Output:
      df: Pandas dataframe with another additional column value_raked
    """
    assert 'value' in df.columns, \
        'The dataframe should contain a column with the values to be raked.'
    assert len(agg_vars)== 2, \
        'Currently, raking can only be done over two variables.'
    for agg_var in agg_vars:
        assert agg_var in df.columns, \
            'The dataframe should contain a column ' + agg_var + '.'
        assert 'all_' + agg_var + '_value' in df.columns, \
            'The dataframe should contain a column with the margins for' + agg_var + '.' 
    if len(constant_vars) > 0:
        for var in constant_vars:
            assert var in df.columns, \
                'The dataframe should contain a column ' + var + '.'

    df['entropic_distance'] = df['value']
    df.sort_values(by=constant_vars + agg_vars, inplace=True)
    epsilon = 1.0
    iter_eps = 0
    while (epsilon > 1.0e-6) & (iter_eps < max_iter):
        df['sum_value_raked'] = df.groupby(constant_vars + [agg_vars[1]])['entropic_distance'].transform('sum')
        df['entropic_distance'] = df['entropic_distance'] * df['all_' + agg_vars[0] + '_value'] / df['sum_value_raked']
        sum0 = np.mean(np.abs(df['all_' + agg_vars[0] + '_value'].to_numpy() - df['sum_value_raked'].to_numpy()))
        df['sum_value_raked'] = df.groupby(constant_vars + [agg_vars[0]])['entropic_distance'].transform('sum')
        df['entropic_distance'] = df['entropic_distance'] * df['all_' + agg_vars[1] + '_value'] / df['sum_value_raked']
        sum1 = np.mean(np.abs(df['all_' + agg_vars[1] + '_value'].to_numpy() - df['sum_value_raked'].to_numpy()))
        epsilon = sum0 + sum1
        iter_eps = iter_eps + 1
    df.drop(columns=['sum_value_raked'], inplace=True)
    if return_num_iter:
        return [df, iter_eps]
    else:
        return df

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
#    assert abs(np.sum(mu_i) - np.sum(mu_j)) < 1e-6, \
#        'Sum of margins over rows and columns should be equal.'
    print(abs(np.sum(mu_i) - np.sum(mu_j)))

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

def get_full_margin_matrix_vector(mu_i, I, J, K):
    """
    In 3D, transform the I conditions on the margins for the causes
    into a matrix formulation
    """
    assert len(mu_i) == I, \
        'We should have ' + str(I) + ' margins for the causes.'

    A = np.zeros((I, I * J * K))
    y = np.zeros(I)
    for i in range(0, I):
        for j in range(0, J):
            for k in range(0, K):
                A[i, k * I * J + j * I + i] = 1
        y[i] = mu_i[i]
    return (A, y)

def get_constrained_matrix_vector(mu_i, I, J, K):
    """
    In 3D, transform the I conditions on the margins for the causes
    and the null conditions in other directions into a matrix formulation
    """
    assert len(mu_i) == I + 1, \
        'We should have ' + str(I + 1) + ' margins for the causes.'

    A = np.zeros((1 + I + 2 * K + 2 * J * K, (I + 1) * (J + 1) * K))
    y = np.zeros(1 + I + 2 * K + 2 * J * K)
    # Constraint sum_k=0,...,K-1 mu_0,0,k = mu_0
    for k in range(0, K):
        A[0, k * (I + 1) * (J + 1)] = \
        A[0, k * (I + 1) * (J + 1)] + 1
    y[0] = mu_i[0]
    # Constraint sum_k=0,...,K-1 mu_i,0,k = mu_i for i=1,...,I
    for i in range(1, I + 1):
        for k in range(0, K):
            A[i, k * (I + 1) * (J + 1) + i] = \
            A[i, k * (I + 1) * (J + 1) + i] + 1
        y[i] = mu_i[i]
    # Constraint sum_i=1,...,I mu_i,0,k - mu_0,0,k = 0 for k=0,...,K-1
    for k in range(0, K):
        for i in range(1, I + 1):
            A[1 + I + k, k * (I + 1) * (J + 1) + i] = \
            A[I + 1 + k, k * (I + 1) * (J + 1) + i] + 1
        A[1 + I + k, k * (I + 1) * (J + 1)] = \
        A[I + 1 + k, k * (I + 1) * (J + 1)] - 1
        y[I + 1 + k] = 0
    # Constraint sum_j=1,...,J mu_0,j,k - mu_0,0,k = 0 for k=0,...,K-1
    for k in range(0, K):
        for j in range(1, J + 1):
            A[1 + I + K + k, k * (I + 1) * (J + 1) + j * (I + 1)] = \
            A[1 + I + K + k, k * (I + 1) * (J + 1) + j * (I + 1)] + 1
        A[1 + I + K + k, k * (I + 1) * (J + 1)] = \
        A[1 + I + K + k, k * (I + 1) * (J + 1)] - 1
        y[1 + I + K + k] = 0
    # Constraint sum_i=1,...,I mu_i,j,k - mu_0,j,k = 0 for j=1,...,J and k=0,...,K-1
    for k in range(0, K):
        for j in range(1, J + 1):
            for i in range(1, I + 1):
                A[1 + I + 2 * K + k * J + j - 1, k * (I + 1) * (J + 1) + j * (I + 1) + i] = \
                A[1 + I + 2 * K + k * J + j - 1, k * (I + 1) * (J + 1) + j * (I + 1) + i] + 1
            A[1 + I + 2 * K + k * J + j - 1, k * (I + 1) * (J + 1) + j * (I + 1)] = \
            A[1 + I + 2 * K + k * J + j - 1, k * (I + 1) * (J + 1) + j * (I + 1)] - 1
            y[1 + I + 2 * K + k * J + j - 1] = 0
    # Constraint sum_j=1,...,J mu_i,j,k - mu_i,0,k = 0 for i=1,...,I and k=0,...,K-1
    for k in range(0, K):
        for i in range(1, I + 1):
            for j in range(1, J + 1):
                A[1 + I + 2 * K + J * K + k * I + i - 1, k * (I + 1) * (J + 1) + j * (I + 1) + i] = \
                A[1 + I + 2 * K + J * K + k * I + i - 1, k * (I + 1) * (J + 1) + j * (I + 1) + i] + 1
            A[1 + I + 2 * K + J * K + k * I + i - 1, k * (I + 1) * (J + 1) + i] = \
            A[1 + I + 2 * K + J * K + k * I + i - 1, k * (I + 1) * (J + 1) + i] - 1
            y[1 + I + 2 * K + J * K + k * I + i - 1] = 0
    return (A, y)
    
def raking_chi2_distance(x, q, A, y, direct=True):
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

    if direct:
        y_hat = np.matmul(A, x)
        Phi = np.matmul(A, np.transpose(A * x * q))
        # Compute Moore-Penrose pseudo inverse to solve the system
        U, S, Vh = np.linalg.svd(Phi, full_matrices=True)
        V = np.transpose(Vh)
        # Invert diagonal matrix while dealing with 0 and near 0 values
        S[np.abs(S) <= 1.0e-10] = 1.0e-10
        Sinv = 1.0 / S
        Sinv[np.abs(S) <= 1.0e-10] = 0.0
        Sinv = np.diag(Sinv)
        Phi_plus = np.matmul(np.matmul(V, Sinv), np.transpose(U))
        lambda_k = np.matmul(Phi_plus, y_hat - y)
        mu = x * (1 - q * np.matmul(np.transpose(A), lambda_k))
    else:
        Phi = np.concatenate((np.concatenate((np.diag(1.0 / (q * x)), \
            np.transpose(A)), axis=1), \
            np.concatenate((A, np.zeros((len(y), len(y)))), axis=1)), axis=0)
        b = np.concatenate((1.0 / q, y), axis=0)
        # Compute Moore-Penrose pseudo inverse to solve the system
        U, S, Vh = np.linalg.svd(Phi, full_matrices=True)
        V = np.transpose(Vh)
        # Invert diagonal matrix while dealing with 0 and near 0 values
        S[np.abs(S) <= 1.0e-10] = 1.0e-10
        Sinv = 1.0 / S
        Sinv[np.abs(S) <= 1.0e-10] = 0.0
        Sinv = np.diag(Sinv)
        Phi_plus = np.matmul(np.matmul(V, Sinv), np.transpose(U))
        result = np.matmul(Phi_plus, b)
        mu = result[0:len(x)]
        lambda_k = result[len(x):(len(x) + len(y))]
    return (mu, lambda_k)

def raking_entropic_distance(x, q, A, y, max_iter=500, return_num_iter=False):
    """
    Raking using the entropic distance mu log(mu/x) + x - mu.
    Input:
      x: 1D Numpy array, observed values x_n = x_i,j if n = j I + i
      q: 1D Numpy array, weights for the observations
      A: 2D Numpy array, linear constraints
      y: 1D Numpy array, partial sums
      max_iter: Integer, number of iterations for Newton's root finding method
      return_num_iter: boolean, whether we return the number of iterations
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
    lambda_k = np.zeros(np.shape(A)[0])
    mu = np.copy(x)
    epsilon = 1.0e6
    iter_eps = 0
    while (epsilon > 1.0e-6) & (iter_eps < max_iter):
        Phi = np.matmul(A, x * (1 - np.exp(- q * \
            np.matmul(np.transpose(A), lambda_k))))
        D = np.diag(x * q * np.exp(- q * \
            np.matmul(np.transpose(A), lambda_k)))
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        # Compute Moore-Penrose pseudo inverse to solve the system
        U, S, Vh = np.linalg.svd(J, full_matrices=True)
        V = np.transpose(Vh)
        # Invert diagonal matrix while dealing with 0 and near 0 values
        S[np.abs(S) <= 1.0e-10] = 1.0e-10
        Sinv = 1.0 / S
        Sinv[np.abs(S) <= 1.0e-10] = 0.0
        Sinv = np.diag(Sinv)
        J_plus = np.matmul(np.matmul(V, Sinv), np.transpose(U))
        # Make sure that epsilon is decreasing
        gamma = 1.0
        iter_gam = 0
        lambda_k = lambda_k - gamma * np.matmul(J_plus, Phi - y_hat + y)
        mu = x * np.exp(- q * np.matmul(np.transpose(A), lambda_k))
        if iter_eps > 0:
            while np.mean(np.abs(y - np.matmul(A, mu))) > epsilon:
                gamma = gamma / 2.0
                iter_gam = iter_gam + 1
                lambda_k = lambda_k - gamma * np.matmul(J_plus, Phi - y_hat + y)
                mu = x * np.exp(- q * np.matmul(np.transpose(A), lambda_k))
        epsilon = np.mean(np.abs(y - np.matmul(A, mu)))
        iter_eps = iter_eps + 1
    if return_num_iter:
        return [mu, iter_eps]
    else:
        return mu

def raking_general_distance(alpha, x, q, A, y, max_iter=500, return_num_iter=False):
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
        while (epsilon > 1.0e-6) & (iter_eps < max_iter):
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
            U, S, Vh = np.linalg.svd(J, full_matrices=True)
            V = np.transpose(Vh)
            # Invert diagonal matrix while dealing with 0 and near 0 values
            S[np.abs(S) <= 1.0e-10] = 1.0e-10
            Sinv = 1.0 / S
            Sinv[np.abs(S) <= 1.0e-10] = 0.0
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
            lambda_k = lambda_k - gamma * np.matmul(J_plus, Phi - y_hat + y)
            if alpha == 0:
                mu = x * np.exp(- q * np.matmul(np.transpose(A), lambda_k))
            else:
                mu = x * np.power(1 - alpha * q * \
                    np.matmul(np.transpose(A), lambda_k), 1.0 / alpha)
            epsilon = np.mean(np.abs(y - np.matmul(A, mu)))
            iter_eps = iter_eps + 1
    if return_num_iter:
        return [mu, iter_eps]
    else:
        return mu

def raking_l2_distance(x, q, A, y):
    """
    Raking using the l2 distance (mu - x)^2 / 2.
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
    Phi = np.matmul(A, np.transpose(q * A))
    # Compute Moore-Penrose pseudo inverse to solve the system
    U, S, Vh = np.linalg.svd(Phi, full_matrices=True)
    V = np.transpose(Vh)
    # Invert diagonal matrix while dealing with 0 and near 0 values
    S[np.abs(S) <= 1.0e-10] = 1.0e-10
    Sinv = 1.0 / S
    Sinv[np.abs(S) <= 1.0e-10] = 0.0
    Sinv = np.diag(Sinv)
    Phi_plus = np.matmul(np.matmul(V, Sinv), np.transpose(U))
    lambda_k = np.matmul(Phi_plus, y_hat - y)
    mu = x - np.matmul(np.transpose(q * A), lambda_k)
    return mu

def raking_vectorized_l2_distance(df, agg_vars, constant_vars=[]):
    """
    Raking using the l2 distance (mu - x)^2 / 2.
    Input:
      df: Pandas dataframe, containing the columns:
          - value = Values to be raked
          - agg_var = Variables over which we want to rake.
          - constant_vars = Several other variables (not to be raked).
          - all_'agg_var'_value = Partial sums
      agg_vars: List of strings, variables over which we do the raking
      constant_vars: List of strings, several other variables (not to be raked).
    Output:
      df: Pandas dataframe with another additional column value_raked
    """
    assert 'value' in df.columns, \
        'The dataframe should contain a column with the values to be raked.'
    assert len(agg_vars)== 2, \
        'Currently, raking can only be done over two variables.'
    for agg_var in agg_vars:
        assert agg_var in df.columns, \
            'The dataframe should contain a column ' + agg_var + '.'
        assert 'all_' + agg_var + '_value' in df.columns, \
            'The dataframe should contain a column with the margins for' + agg_var + '.' 
    if len(constant_vars) > 0:
        for var in constant_vars:
            assert var in df.columns, \
                'The dataframe should contain a column ' + var + '.'

    # Get values to be raked
    df.sort_values(by=constant_vars + agg_vars, inplace=True)
    x = df.value.to_numpy()

    # Get margins
    name0 = 'all_' + agg_vars[0] + '_value'
    name1 = 'all_' + agg_vars[1] + '_value'
    mu_i = df.groupby(constant_vars + [agg_vars[1]]).agg({name0: np.mean}).reset_index()
    mu_j = df.groupby(constant_vars + [agg_vars[0]]).agg({name1: np.mean}).reset_index()
    y = pd.concat([mu_i.drop(columns=[agg_vars[1]]).rename(columns={name0: 'value'}), \
                   mu_j.drop(columns=[agg_vars[0]]).rename(columns={name1: 'value'})])
    y.sort_values(by=constant_vars, inplace=True)
    y = y.value.to_numpy()

    # Get linear constraints
    I = len(df[agg_vars[1]].unique())
    J = len(df[agg_vars[0]].unique())
    A = np.zeros((I + J, I * J))
    for i in range(0, I):
        for j in range(0, J):
            A[i, j * I + i] = 1
            A[I + j, j * I + i] = 1

    # Compute Moore-Penrose pseudo inverse to solve the system
    Phi = np.matmul(A, np.transpose(A))
    U, S, Vh = np.linalg.svd(Phi, full_matrices=True)
    V = np.transpose(Vh)
    # Invert diagonal matrix while dealing with 0 and near 0 values
    S[np.abs(S) <= 1.0e-10] = 1.0e-10
    Sinv = 1.0 / S
    Sinv[np.abs(S) <= 1.0e-10] = 0.0
    Sinv = np.diag(Sinv)
    Phi_plus = np.matmul(np.matmul(V, Sinv), np.transpose(U))
    Phi_1 = np.eye(I * J) - np.matmul(np.matmul(np.transpose(A), Phi_plus), A)
    Phi_2 = np.matmul(np.transpose(A), Phi_plus)

    # Aggregate and compute raked values
    N1 = int(floor(len(x) / (I * J)))
    N2 = int(floor(len(y) / (I + J)))
    assert N1 == N2, \
        'Inconsistency between number of values to be raked and number of marginal totals.'
    Phi_1_big = np.zeros((N1 * I * J, N1 * I * J))
    Phi_2_big = np.zeros((N1 * I * J, N1 * (I + J)))
    for n in range(0, N1):
        Phi_1_big[n * I * J : (n + 1) * I * J, n * I * J : (n + 1) * I * J] = Phi_1
        Phi_2_big[n * I * J : (n + 1) * I * J, n * (I + J) : (n + 1) * (I + J)] = Phi_2
    mu = np.matmul(Phi_1_big, x) + np.matmul(Phi_2_big, y)
    df['l2_distance'] = mu
    return df

def raking_logit(x, l, h, q, A, y, max_iter=500, return_num_iter=False):
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
    while (epsilon > 1.0e-6) & (iter_eps < max_iter):
        Phi = np.matmul(A, (l * (h - x) + h * (x - l) * \
            np.exp(- q * np.matmul(np.transpose(A), lambda_k))) / \
            ((h - x) + (x - l) * \
             np.exp(- q * np.matmul(np.transpose(A), lambda_k))))
        D = np.diag(- q * ((x - l) * (h - x) * (h - l)) / \
            np.square((h - x) + (x - l) * \
            np.exp(- q * np.matmul(np.transpose(A), lambda_k))))    
        J = np.matmul(np.matmul(A, D), np.transpose(A))
        # Compute Moore-Penrose pseudo inverse to solve the system
        U, S, Vh = np.linalg.svd(J, full_matrices=True)
        V = np.transpose(Vh)
        # Invert diagonal matrix while dealing with 0 and near 0 values
        S[np.abs(S) <= 1.0e-10] = 1.0e-10
        Sinv = 1.0 / S
        Sinv[np.abs(S) <= 1.0e-10] = 0.0
        Sinv = np.diag(Sinv)
        J_plus = np.matmul(np.matmul(V, Sinv), np.transpose(U))
        lambda_k = lambda_k - np.matmul(J_plus, Phi - y)
        mu = (l * (h - x) + h * (x - l) * \
            np.exp(- q * np.matmul(np.transpose(A), lambda_k))) / \
            ((h - x) + (x - l) * \
            np.exp(- q * np.matmul(np.transpose(A), lambda_k)))
        epsilon = np.mean(np.abs(y - np.matmul(A, mu)))
        iter_eps = iter_eps + 1
    if return_num_iter:
        return [mu, iter_eps]
    else:
        return mu

