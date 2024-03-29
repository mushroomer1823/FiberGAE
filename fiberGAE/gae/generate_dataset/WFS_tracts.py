import numpy as np


def WFS_tracts(tract, para, k):
    """
    Computes the WFS (Weighted Finite Sums) of a curve.

    Args:
    tract: 3 x n_vertex coordinates of 3D curve
    para: arc-length parameterization
    k: degree

    Returns:
    wfs: weighted finite sums
    beta: beta coefficients
    """

    tract = tract.T

    # print(tract)
    # print(para)

    n_vertex = len(para)
    para_even = np.hstack((-para[::-1][1:], para))
    # print(para_even)
    tract_even = np.hstack((tract[:, ::-1][:, 1:], tract))
    # print(tract_even)

    Y = np.zeros((2 * n_vertex - 1, k + 1))
    para_even = np.tile(para_even, (k + 1, 1)).T
    pi_factors = np.tile(np.arange(k + 1), (2 * n_vertex - 1, 1)) * np.pi
    Y = np.cos(para_even * pi_factors) * np.sqrt(2)

    beta = np.linalg.pinv(Y.T @ Y) @ Y.T @ tract_even.T

    hat = Y @ beta

    wfs = hat[n_vertex:(n_vertex * 2 - 1), :].T

    return wfs, beta