import numpy as np



def compute_angle(p1, vertex, p3):
    """
    Compute the angle formed by the two line segments (p1 to vertex) and (vertex to p3).

    Args:
    - p1, p3: Points in Euclidean space.
    - vertex: is the vertex of the angle.

    Returns:
    - angle: Angle in radians.
    """
    v1 = p1 - vertex
    v2 = p3 - vertex
    dot_product = np.dot(v1, v2)
    
    cos_angle = (dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle = np.arccos(cos_angle)
    
    return angle


def create_col_angle_matrix(X, indices):
    """
    Create a matrix of most colinear angles between each point and its k nearest neighbours.

    Args:
    - X: Data matrix.
    - indices: Indices of the k nearest neighbours.

    Returns:
    - theta: Matrix of most colinear angles.
    """

    theta = np.zeros((indices.shape[0], indices.shape[1]))

    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):
                for k in range(1, indices.shape[1]):
                    temp = compute_angle(X[indices[i,0]], X[indices[i,j]], X[indices[j, k]])
                    if temp > theta[i,j]:
                        theta[i,j] = temp       
    return theta
    


def compute_distances(X, k, indices, n):

    """""
    Args: 
    - X: Data matrix,
    - k: number of nearest neighbours, 
    - indices: indices of nearest neighbours,
    - n: number of points
 
    Retunrs: 
    - dist: array of distances of a point from its k nearest neighbours, 
    - mean_dist: average distance over all points

    """""
    dist =  np.zeros((indices.shape[0], indices.shape[1]))
    for i in range(n):
        for j in range(1, k):
        # compute euclidean distance between the point and its k nearest neighbours
            dist[i,j] = np.linalg.norm(X[indices[i,0]] - X[indices[i,j]])
    
    mean_dist = np.mean(dist)     # maybe we should divide by k

    return dist, mean_dist



def compute_relationships(X, k, indices, n):
    
    """""
    Compute a series of relationships between each point in X and its k nearest neighbours

    Input:  X: data, 
            k: number of nearest neighbours, 
            indices: indices of nearest neighbours, 
            n: number of points
            
    Output: angle: array of the angle between the point i and its k nearest neighbours,
            dist: array of distances between point i and its k nearest neighbours,
            average_dist: global average distance between all the neighbors of all points

    """""
    angle = create_col_angle_matrix(X, indices)
    dist, average_dist = compute_distances(X, k, indices, n)
    return angle, dist, average_dist



def compute_error(X, k, indices, n, i,original_dist, original_angle, original_average_dist):
    """
    Compute the error between the original and the new relationships for a given point i.

    Args:
    - i: Index of the point.
    - original_dist: Original distances.
    - original_angle: Original angles.
    - original_average_dist: Original average distance.

    Returns:
    - err: Error between the original and the new relationships.
    """

    new_angle, new_dist, _ = compute_relationships(X, k, indices, n)

    err = 0.

    for j in range(original_dist.shape[1]):
        err += ((original_dist[i,j] - new_dist[i,j]) ** 2) / original_average_dist
        err += np.max([0,((original_angle[i,j] - new_angle[i,j]) ** 2) / np.pi])


    return err


def adjust_point(X, k, indices, n, p,i, eta, original_dist, original_angle, original_average_dist, D_preserved):
   
    """
    Adjust the point p in order to minimize the error between the original and the new relationships.

    Args:
    - p: Point to adjust.
    - i: Index of the point.
    - eta: Learning rate.

    Returns:
    - s: Number of steps to reach the minimum error.
    """

    s = -1
    improved = True
    while improved:
        s += 1
        improved = False
        error0 = compute_error(X, k, indices, n, i,original_dist, original_angle, original_average_dist)
        p[:D_preserved] += eta
        error = compute_error(X, k, indices, n, i,original_dist, original_angle, original_average_dist)
        if error > error0:
            p[:D_preserved] -= 2 * eta
            error = compute_error(X, k, indices, n, i,original_dist, original_angle, original_average_dist)
            if error > error0:
                p[:D_preserved] += eta
            else:
                improved = True
        else:
            improved = True
    return s



