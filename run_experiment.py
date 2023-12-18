import matplotlib.pyplot as plt
import numpy as np

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
    assert np.sum(mu_i) == np.sum(mu_j), \
        'Sum of partial sums over rows and columns should be equal'

    alpha = np.ones((np.shape(x)[0], np.shape(x)[1]))
    epsilon = 1
    diffs = [0.5 * np.mean(np.abs((mu_j - np.sum(alpha * x, 0)) / mu_j)) + \
             0.5 * np.mean(np.abs((mu_i - np.sum(alpha * x, 1)) / mu_i))]
    while epsilon > 0.0001:
        alpha_old = np.copy(alpha)
        alpha = alpha * mu_j / np.sum(alpha * x, 0)
        alpha = np.transpose(np.transpose(alpha) * mu_i / np.sum(alpha * x, 1))
        epsilon = np.mean(np.abs((alpha_old - alpha) / alpha_old))
        diffs.append( \
            0.5 * np.mean(np.abs((mu_j - np.sum(alpha * x, 0)) / mu_j)) + \
            0.5 * np.mean(np.abs((mu_i - np.sum(alpha * x, 1)) / mu_i)))
    mu = alpha * x
    return (mu, diffs)

if __name__ == "__main__":

    # Set seed for reproducibility
    np.random.seed(0)

    # Choose values for x, mu_i, mu_j
    n = 5
    m = 4
    x = np.random.uniform(0.04, 0.06, m * n).reshape((n, m))
    mu_i = np.random.uniform(0.15, 0.25, n)
    mu_i = mu_i / np.sum(mu_i)
    mu_j = np.random.uniform(0.2, 0.3, m)
    mu_j = mu_j / np.sum(mu_j)

    # rake matrix using IPF algorithm
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

