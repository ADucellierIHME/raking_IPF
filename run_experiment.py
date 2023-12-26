import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def IPF(x, mu_i, mu_j):
    """
    Function to do iterative proportional fitting (IPF) in 2D.
    Input:
      x: 2D Numpy array, values to be raked
      mu_i: 1D Numpy array, partial sums over the columns
      mu_j: 1D Numpy array, partial sums over the rows
    Output:
      mu: 2D Numpy array, raked values
    """
    assert isinstance(x, np.ndarray), \
        'Values to be raked should be a Numpy array'
    assert isinstance(mu_i, np.ndarray), \
        'Partial sums over columns should be a Numpy array'
    assert isinstance(mu_j, np.ndarray), \
        'Partial sums over rows should be a Numpy array'
    assert len(np.shape(x)) == 2, \
        'We need to rake a two-dimensional matrix'
    assert np.shape(x)[0] == len(mu_i), \
        'Partial sums over columns should be equal to the number of rows'
    assert np.shape(x)[1] == len(mu_j), \
        'Partial sums over rows should be equal to the number of columns'
    assert abs(np.sum(mu_i) - np.sum(mu_j)) < 1e-10, \
        'Sum of partial sums over rows and columns should be equal'

    alpha = np.ones((np.shape(x)[0], np.shape(x)[1]))
    epsilon = 1
    diffs = [0.5 * np.mean(np.abs((mu_j - np.sum(alpha * x, 0)) / mu_j)) + \
             0.5 * np.mean(np.abs((mu_i - np.sum(alpha * x, 1)) / mu_i))]
    while epsilon > 1e-10:
        alpha_old = np.copy(alpha)
        alpha = alpha * mu_j / np.sum(alpha * x, 0)
        alpha = np.transpose(np.transpose(alpha) * mu_i / np.sum(alpha * x, 1))
        epsilon = np.mean(np.abs((alpha_old - alpha) / alpha_old))
        diffs.append( \
            0.5 * np.mean(np.abs((mu_j - np.sum(alpha * x, 0)) / mu_j)) + \
            0.5 * np.mean(np.abs((mu_i - np.sum(alpha * x, 1)) / mu_i)))
    mu = alpha * x
    return (mu, diffs)

def IPF_torch(x, mu_i, mu_j):
    """
    Function to do iterative proportional fitting (IPF) in 2D
    using Torch tensors instead of Numpy.
    Input:
      x: 2D Torch tensor, values to be raked
      mu_i: 1D Torch tensor, partial sums over the columns
      mu_j: 1D Torch tensor, partial sums over the rows
    Output:
      mu: 2D Torch tensor, raked values
    """
    assert torch.is_tensor(x), \
        'Values to be raked should be a Torch tensor'
    assert torch.is_tensor(mu_i), \
        'Partial sums over columns should be a Torch tensor'
    assert torch.is_tensor(mu_j), \
        'Partial sums over rows should be a Torch tensor'
    assert len(x.shape) == 2, \
        'We need to rake a two-dimensional tensor'
    assert x.shape[0] == mu_i.shape[0], \
        'Partial sums over columns should be equal to the number of rows'
    assert x.shape[1] == mu_j.shape[0], \
        'Partial sums over rows should be equal to the number of columns'
    assert torch.abs(torch.sum(mu_i) - torch.sum(mu_j)) < 1e-5, \
        'Sum of partial sums over rows and columns should be equal'

    x = torch.log(x)
    beta = torch.zeros(x.shape[0], x.shape[1])
    epsilon = 1
    diffs = (0.5 * torch.mean(torch.abs((torch.log(mu_j) -
                torch.logsumexp(x + beta, 0)) / torch.log(mu_j))) + \
             0.5 * torch.mean(torch.abs((torch.log(mu_i) -
                torch.logsumexp(x + beta, 1)) / torch.log(mu_i)))).reshape(1)
    while epsilon > 1e-10:
        beta_old = torch.clone(beta)
        beta = beta + torch.log(mu_j) - torch.logsumexp(x + beta, 0)
        beta = torch.transpose(torch.transpose(beta, 0, 1) +
            torch.log(mu_i) - torch.logsumexp(x + beta, 1), 0, 1)
        epsilon = torch.mean(torch.abs(
            (torch.exp(beta_old) - torch.exp(beta)) / torch.exp(beta_old)))
        diffs = torch.cat((diffs,
            (0.5 * torch.mean(torch.abs((torch.log(mu_j) -
                torch.logsumexp(x + beta, 0)) / torch.log(mu_j))) + \
             0.5 * torch.mean(torch.abs((torch.log(mu_i) -
                torch.logsumexp(x + beta, 1)) / torch.log(mu_i)))).reshape(1)),
            dim=0)
    mu = torch.exp(x + beta)
    return (mu, diffs)

def create_experiment():
    """
    Function to test the IPF algorithm.
    Returns initial values and raked values.
    """
    # Set seed for reproducibility
    np.random.seed(0)

    # Choose values for x, mu_i, mu_j
    n = 50
    m = 40
    x = np.random.uniform(0.004, 0.006, m * n).reshape((n, m))
    mu_i = np.random.uniform(0.15, 0.25, n)
    mu_i = mu_i / np.sum(mu_i)
    mu_j = np.random.uniform(0.2, 0.3, m)
    mu_j = mu_j / np.sum(mu_j)

    # Rake matrix using IPF algorithm
    (mu, diffs) = IPF(x, mu_i, mu_j)

    # Print results to show that it works
    # Print sums over i
    for i in range(0, n):
        print('Row ', i, ' - True value = ', mu_i[i], \
                         ' - Computed value = ', np.sum(mu, 1)[i])
    # Print sums over j
    for j in range(0, m):
        print('Column ', j, ' - True value = ', mu_j[j], \
                            ' - Computed value = ', np.sum(mu, 0)[j])
    # Print overall sum
    print('Total = ', np.sum(mu_i), ' and ', np.sum(mu_j), \
          ' - Computed value = ', np.sum(mu))

    # Show how fast it converges
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(np.arange(1, len(diffs) + 1), diffs, 'bo')
    plt.plot(np.arange(1, len(diffs) + 1), diffs, 'b-')
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('MAPE', fontsize=20)
    ax.set_xticks(np.arange(1, len(diffs) + 1).tolist())
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Convergence of IPF algorithm', fontsize=20)
    plt.tight_layout()
    plt.savefig('IPF_convergence.png')

    # Return values for analysis
    return (x, mu_i, mu_j, mu)

def create_experiment_torch():
    """
    Function to test the IPF algorithm with Torch tensors.
    Returns initial values and raked values.
    """
    # Set seed for reproducibility
    np.random.seed(0)

    # Choose values for x, mu_i, mu_j
    n = 50
    m = 40
    x = np.random.uniform(0.004, 0.006, m * n).reshape((n, m))
    mu_i = np.random.uniform(0.15, 0.25, n)
    mu_i = mu_i / np.sum(mu_i)
    mu_j = np.random.uniform(0.2, 0.3, m)
    mu_j = mu_j / np.sum(mu_j)

    # Convert to Torch tensors
    x = torch.from_numpy(x)
    mu_i = torch.from_numpy(mu_i)
    mu_j = torch.from_numpy(mu_j)

    # Rake matrix using IPF algorithm
    (mu, diffs) = IPF_torch(x, mu_i, mu_j)

    # Print results to show that it works
    # Print sums over i
    for i in range(0, n):
        print('Row ', i, ' - True value = ', mu_i[i], \
                         ' - Computed value = ', torch.sum(mu, 1)[i])
    # Print sums over j
    for j in range(0, m):
        print('Column ', j, ' - True value = ', mu_j[j], \
                            ' - Computed value = ', torch.sum(mu, 0)[j])
    # Print overall sum
    print('Total = ', torch.sum(mu_i), ' and ', torch.sum(mu_j), \
          ' - Computed value = ', torch.sum(mu))

    # Show how fast it converges
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(np.arange(1, len(diffs) + 1), diffs, 'bo')
    plt.plot(np.arange(1, len(diffs) + 1), diffs, 'b-')
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('MAPE', fontsize=20)
    ax.set_xticks(np.arange(1, len(diffs) + 1).tolist())
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Convergence of IPF algorithm', fontsize=20)
    plt.tight_layout()
    plt.savefig('IPF_torch_convergence.png')

    # Return values for analysis
    return (x, mu_i, mu_j, mu)

def transform_to_pandas(x, mu_i, mu_j, mu):
    """
    """
    cause = np.repeat(np.array(['cause1', 'cause2', 'cause3', 'cause4', 'cause5']).reshape((5, 1)), 4, axis=1)
    cause_value = np.repeat(mu_i.reshape((5, 1)), 4, axis=1)
    county = np.repeat(np.array(['county1', 'county2', 'county3', 'county4']).reshape((1, 4)), 5, axis=0)
    county_value = np.repeat(mu_j.reshape((1, 4)), 5, axis=0)
    df_x = pd.DataFrame({'value': x.reshape((-1, 1)).reshape(-1),
                         'cause': cause.reshape((-1, 1)).reshape(-1),
                         'cause_value': cause_value.reshape((-1, 1)).reshape(-1),
                         'county': county.reshape((-1, 1)).reshape(-1),
                         'county_value': county_value.reshape((-1, 1)).reshape(-1)})
    df_mu = pd.DataFrame({'value': mu.reshape((-1, 1)).reshape(-1),
                          'cause': cause.reshape((-1, 1)).reshape(-1),
                          'cause_value': cause_value.reshape((-1, 1)).reshape(-1),
                          'county': county.reshape((-1, 1)).reshape(-1),
                          'county_value': county_value.reshape((-1, 1)).reshape(-1)})
    return (df_x, df_mu)

if __name__ == "__main__":

    (x, mu_i, mu_j, mu) = create_experiment()
    (x, mu_i, mu_j, mu) = create_experiment_torch()
#    (df_x, df_mu) = transform_to_pandas(x, mu_i, mu_j, mu)
#    df_x.to_csv('initial_values.csv')
#    df_mu.to_csv('raked_values.csv')

