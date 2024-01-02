import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from timeit import default_timer as timer

def IPF(x, mu_i, mu_j):
    """
    Function to do iterative proportional fitting (IPF)
    with two categorical variables.
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
        'Partial sums over columns should have length equal to the number of rows'
    assert np.shape(x)[1] == len(mu_j), \
        'Partial sums over rows should have length equal to the number of columns'
    assert abs(np.sum(mu_i) - np.sum(mu_j)) < 1e-10, \
        'Sum of partial sums over rows and columns should be equal'

    # Initialization
    alpha = np.ones((np.shape(x)[0], np.shape(x)[1]))
    epsilon = 1
    diffs = [0.5 * np.mean(np.abs((mu_j - np.sum(alpha * x, 0)) / mu_j)) + \
             0.5 * np.mean(np.abs((mu_i - np.sum(alpha * x, 1)) / mu_i))]
    # Loop until convergence
    while epsilon > 1e-10:
        alpha_old = np.copy(alpha)
        alpha = alpha * mu_j / np.sum(alpha * x, 0)
        alpha = np.transpose(np.transpose(alpha) * mu_i / np.sum(alpha * x, 1))
        epsilon = np.mean(np.abs((alpha_old - alpha) / alpha_old))
        diffs.append( \
            0.5 * np.mean(np.abs((mu_j - np.sum(alpha * x, 0)) / mu_j)) + \
            0.5 * np.mean(np.abs((mu_i - np.sum(alpha * x, 1)) / mu_i)))
    # Return raked values
    mu = alpha * x
    return (mu, diffs)

def IPF_torch(x, mu_i, mu_j):
    """
    Function to do iterative proportional fitting (IPF)
    with two categorical variables using Torch tensors instead of Numpy.
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
        'Partial sums over columns should have length equal to the number of rows'
    assert x.shape[1] == mu_j.shape[0], \
        'Partial sums over rows should have length equal to the number of columns'
    assert torch.abs(torch.sum(mu_i) - torch.sum(mu_j)) < 1e-5, \
        'Sum of partial sums over rows and columns should be equal'

    # We use log values instead of actual values for better convergence
    x = torch.log(x)
    # Initialization
    beta = torch.zeros(x.shape[0], x.shape[1])
    epsilon = 1
    diffs = (0.5 * torch.mean(torch.abs((torch.log(mu_j) -
                torch.logsumexp(x + beta, 0)) / torch.log(mu_j))) + \
             0.5 * torch.mean(torch.abs((torch.log(mu_i) -
                torch.logsumexp(x + beta, 1)) / torch.log(mu_i)))).reshape(1)
    # Loop until convergence
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
    # Go back to experiment and return raked values
    mu = torch.exp(x + beta)
    return (mu, diffs)

def create_experiment(n, m, plot=True):
    """
    Function to test the IPF algorithm.
    Input:
      n: Integer, number of categories for first variable
      m: Integer, number of categories for second variable
    Output:
      x: 2D Numpy array, values to be raked
      mu_i: 1D Numpy array, partial sums over the columns
      mu_j: 1D Numpy array, partial sums over the rows
      mu: 2D Numpy array, raked values
      num_iter: Integer, number of iterations until convergence
      time: scalar, computation time
    """

    # Set seed for reproducibility
    np.random.seed(n * m)

    # Choose values for x, mu_i, mu_j
    x = np.random.uniform(0.01 / (n * m), 1.99 / (n * m), m * n).reshape((n, m))
    mu_i = np.random.uniform(0.75 / n, 1.25 / n, n)
    mu_i = mu_i / np.sum(mu_i)
    mu_j = np.random.uniform(0.75 / m, 1.25 / m, m)
    mu_j = mu_j / np.sum(mu_j)

    # Rake matrix using IPF algorithm
    time_start = timer()
    (mu, diffs) = IPF(x, mu_i, mu_j)
    time_end = timer()

    # Show how fast it converges
    if plot:
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
    num_iter = len(diffs)
    return (x, mu_i, mu_j, mu, num_iter, time_end - time_start)

def create_experiment_torch(n, m, plot=True):
    """
    Function to test the IPF algorithm with Torch tensors.
    Input:
      n: Integer, number of categories for first variable
      m: Integer, number of categories for second variable
    Output:
      x: 2D Torch tensor, values to be raked
      mu_i: 1D Torch tensor, partial sums over the columns
      mu_j: 1D Torch tensor, partial sums over the rows
      mu: 2D Torch tensor, raked values
      num_iter: Integer, number of iterations until convergence
      time: scalar, computation time
    """

    # Set seed for reproducibility
    np.random.seed(n * m)

    # Choose values for x, mu_i, mu_j
    x = np.random.uniform(0.01 / (n * m), 1.99 / (n * m), m * n).reshape((n, m))
    mu_i = np.random.uniform(0.75 / n, 1.25 / n, n)
    mu_i = mu_i / np.sum(mu_i)
    mu_j = np.random.uniform(0.75 / m, 1.25 / m, m)
    mu_j = mu_j / np.sum(mu_j)

    # Convert to Torch tensors
    x = torch.from_numpy(x)
    mu_i = torch.from_numpy(mu_i)
    mu_j = torch.from_numpy(mu_j)

    # Rake matrix using IPF algorithm
    time_start = timer()
    (mu, diffs) = IPF_torch(x, mu_i, mu_j)
    time_end = timer()

    # Show how fast it converges
    if plot:
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
    num_iter = len(diffs)
    return (x, mu_i, mu_j, mu, num_iter, time_end - time_start)

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

    # Run experiment once to show the convergence
    (x, mu_i, mu_j, mu, num, time) = create_experiment(50, 40, True)
    (x, mu_i, mu_j, mu, num_torch, time_torch) = create_experiment_torch(50, 40, True)

    # Run experiments with different matrix sizes
    # and compare number of iterations and computation time
    ns = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ms = np.array([4, 8, 16, 40, 80, 160, 400, 800, 1600, 4000, 8000, 16000])
    xticks = [1, 4, 7, 10]
    xticklabels = ['20', '2e3', '2e5', '2e7']
    num_iters = np.zeros(12)
    num_iters_torch = np.zeros(12)
    times = np.zeros(12)
    times_torch = np.zeros(12)
    for count, (n, m) in enumerate(zip(ns, ms)):
        (x, mu_i, mu_j, mu, num, time) = create_experiment(n, m, False)
        num_iters[count] = num
        times[count] = time        
        print(count, 'Numpy', num_iters[count], times[count])
        (x, mu_i, mu_j, mu, num, time) = create_experiment_torch(n, m, False)
        num_iters_torch[count] = num
        times_torch[count] = time
        print(count, 'Torch', num_iters_torch[count], times_torch[count])

    # Plot results of experiment
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Speed of convergence of IPF algorithm', fontsize=20)
    # Iterations
    ax1.plot(np.arange(1, len(num_iters) + 1), num_iters, 'bo')
    ax1.plot(np.arange(1, len(num_iters) + 1), num_iters, 'b-', label='Numpy')
    ax1.plot(np.arange(1, len(num_iters_torch) + 1), num_iters_torch, 'ro')
    ax1.plot(np.arange(1, len(num_iters_torch) + 1), num_iters_torch, 'r-', label='Torch')
    ax1.set_xlabel('Matrix size', fontsize=20)
    ax1.set_ylabel('Number of iterations', fontsize=20)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, fontsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.legend(loc='upper left', fontsize=20)
    # Computation time
    ax2.plot(np.arange(1, len(times) + 1), np.log(times), 'bo')
    ax2.plot(np.arange(1, len(times) + 1), np.log(times), 'b-', label='Numpy')
    ax2.plot(np.arange(1, len(times_torch) + 1), np.log(times_torch), 'ro')
    ax2.plot(np.arange(1, len(times_torch) + 1), np.log(times_torch), 'r-', label='Torch')
    ax2.set_xlabel('Matrix size', fontsize=20)
    ax2.set_ylabel('Log Computation time', fontsize=20)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels, fontsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    ax2.legend(loc='upper left', fontsize=20)
    # End figure
    plt.tight_layout()
    plt.savefig('IPF_convergence_speed.png')

#    (df_x, df_mu) = transform_to_pandas(x, mu_i, mu_j, mu)
#    df_x.to_csv('initial_values.csv')
#    df_mu.to_csv('raked_values.csv')

