import numpy as np

def IPF(mu, N, fi, fj):
    """
    Function to do iterative proportional fitting in 2D.
    Input:
      mu: 2D Numpy array, values to be raked
      N: scalar, total over rows and columns
      fi: 1D Numpy array, partial sums over the columns
      fj: 1D Numpy array, partial sums over the rows
    Output:
      r: 2D Numpy array, raked values
    """
    assert isinstance(mu, np.ndarray), \
        'Values to be raked should be a Numpy array'
    assert isinstance(fi, np.ndarray), \
        'Partial sums over columns should be a Numpy array'
    assert isinstance(fj, np.ndarray), \
        'Partial sums over rows should be a Numpy array'
    assert len(np.shape(mu)) == 2, \
        'We need to rake a two-dimensional matrix'
    assert np.shape(mu)[0] == len(fi), \
        'Partial sums over columns should be equal to the number of rows'
    assert np.shape(mu)[1] == len(fj), \
        'Partial sums over rows should be equal to the number of columns'

    r = np.ones((np.shape(mu)[0], np.shape(mu)[1]))
    epsilon = 1
    while epsilon > 0.001:
        r_old = np.copy(r)
        r = r * fj * N / np.sum(r * mu, 0)
        r = np.transpose(np.transpose(r) * fi * N / np.sum(r * mu, 1))
        epsilon = np.mean(np.abs((r_old - r) / r_old))
    return r

if __name__ == "__main__":

    # Set seed for reproducibility
    np.random.seed(0)

    # Choose values for mu, N, fi, fj
    n = 5
    m = 4
    mu = np.random.uniform(0.04, 0.06, m * n).reshape((n, m))
    fi = np.random.uniform(0.15, 0.25, n)
    fi = fi / np.sum(fi)
    fj = np.random.uniform(0.2, 0.3, m)
    fj = fj / np.sum(fj)
    N = 1

    # rake matrix using IPF algorithm
    r = IPF(mu, N, fi, fj)

    # Print results to show that it works
    # Print sums over i
    for i in range(0, n):
        print('Row ', i, ' - True value = ', fi[i], ' - Computed value = ', np.sum(r * mu, 1)[i])
    # Print sums over j
    for j in range(0, m):
        print('Column ', j, ' - True value = ', fj[j], ' - Computed value = ', np.sum(r * mu, 0)[j])
    # Print overall sum
    print('Total = ', N, ' - Computed value = ', np.sum(r * mu))
