import numpy as np


def parameterize_arclength(tract):
    """
    Computes the arc length and performs the unit length parameterization of a curve.

    Args:
    tract: 3 x n_vertex or 2 x n_vertex, the curve data

    Returns:
    arc_length: total arc length
    para: array of values between 0 and 1 that maps the track to a unit interval
    """

    tract = tract.T
    n_vertex = tract.shape[1]

    p0 = tract[:, :-1]
    p1 = tract[:, 1:]
    disp = p1 - p0

    L2 = np.sqrt(np.sum(disp ** 2, axis=0))

    arc_length = np.sum(L2)

    cum_len = np.cumsum(L2) / arc_length
    para = np.zeros(n_vertex)
    para[1:] = cum_len

    return arc_length, para


def parameterize_arclength2(tract):
    """
    Computes the arc length and performs the unit length parameterization of a curve.

    Args:
    tract: 3 x n_vertex or 2 x n_vertex, the curve data

    Returns:
    arc_length: total arc length
    para: array of values between 0 and 1 that maps the track to a unit interval
    """

    tract = tract.T
    n_vertex = tract.shape[1]

    p0 = tract[:, :-1]
    p1 = tract[:, 1:]
    disp = p1 - p0

    L2 = np.sqrt(np.sum(disp ** 2, axis=0))

    arc_length = np.sum(L2)
    return arc_length
