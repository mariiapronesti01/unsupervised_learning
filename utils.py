import numpy as np



def compute_angle(p1, p2, p3):
    """
    Compute the angle formed by the two line segments (p1 to p2) and (p2 to p3).

    Args:
    - p1, p2, p3: Points in Euclidean space.

    Returns:
    - angle: Angle in radians.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    if dot_product > 0:
        cos_angle = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
    else: 
        cos_angle = 0
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return angle


def find_colinear_neighbor(data, i, neighbor_idx, neighbors_of_i):
    """
    Find the most colinear neighbor of the neighbor point with respect to the given data point.

    Args:
    - data: A numpy array of shape (n, d) where n is the number of data points and d is the dimensionality.
    - i: Index of the data point.
    - neighbor_idx: Index of the neighbor point.
    - neighbors_of_i: List of indices corresponding to the k-nearest neighbors of data point i.

    Returns:
    - colinear_neighbor_idx: Index of the most colinear neighbor of the neighbor point.
    """
    min_angle_diff_to_pi = np.inf
    colinear_neighbor_idx = None
    for neighbor_of_i_idx in neighbors_of_i:
        if neighbor_of_i_idx != neighbor_idx:
            angle = compute_angle(data[i], data[neighbor_idx], data[neighbor_of_i_idx])
            angle_diff_to_pi = np.abs(np.pi - angle)
            if angle_diff_to_pi < min_angle_diff_to_pi:
                min_angle_diff_to_pi = angle_diff_to_pi
                colinear_neighbor_idx = neighbor_of_i_idx

    return colinear_neighbor_idx


def compute_distances(X, k, indices, n):

    """""
    Input: X: data, k: number of nearest neighbours
    Output: distances of a point from its k nearest neighbours, 
            average distance over all points

    """""
    dist = []
    for i in range(n):
        for j in range(1, k+1):
        # compute euclidean distance between the point and its k nearest neighbours
            dist.append(np.linalg.norm(X[i] - X[indices[i,j]]))

    mean_dist = np.mean(dist)/k
    return mean_dist



