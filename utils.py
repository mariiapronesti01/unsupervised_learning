import numpy as np
from sklearn.metrics import pairwise_distances
import os
import matplotlib.pyplot as plt



def compute_original_relationships(data, k=10):
        """""
        Compute a series of relationships between each point in data and its k nearest neighbours

        Input:  
        - data: orginal data, 
        - k: number of nearest neighbours, 
                
        Output: 
        - neighbors: array of the indexes of the k nearest neighbours,
        - dist: array of distances between each point in data and its k nearest neighbours,
        - colinear: array of the indexes of the most colinear points,
        - theta: array of the angles, 
        - mean_dist: global average distance between all the neighbors of all points

        """""

        distances = pairwise_distances(data, metric='euclidean')
        
        neighbors = np.argpartition(distances, range(1, k+1))[:, 1:k+1]
        dist = np.take_along_axis(distances, neighbors, axis=1)

        mean_dist = np.mean(dist)


        theta = np.zeros((data.shape[0], k)) #angles with colinear points
        colinear = np.zeros((data.shape[0], k), dtype=int) #indexes of colinear points

        for i in range(data.shape[0]):
            for j in range(k):
                n = int(neighbors[i, j])
                v1 = data[i,:] - data[n,:]

                angles = np.zeros(data.shape[0])
                for z in range(k):
                    m = int(neighbors[n, z])
                    if m != i:
                        v2 = data[m,:] - data[n,:]
                        angles[m] = np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

                
                colinear[i,j] = np.argmin(np.abs(angles - np.pi))
                theta[i,j] = angles[colinear[i,j]]

        return neighbors, dist, colinear, theta, mean_dist



def get_avg_dist(data, neighbors):
        """
        Compute the average distance each point in data and all its neighbors.

        Input:  
        - data: orginal data, 
        - neighbors: array of indices of nearest neighbours, 
                
        Output:
        - avg_dist: average distance between each point in data and all its neighbors

        """
        
        avg_dist = 0
        k = neighbors.shape[1]
        for i in range(data.shape[0]):
            for j in range(k):
                avg_dist += np.linalg.norm(data[i,:] - data[neighbors[i,j],:])/k

        return (avg_dist / data.shape[0])



def PCA_rotation(data):
        """
        Perform PCA on the data and rotate it accordingly.

        Input:  
        - data: orginal data,

        Output: 
        - rotated data, 
        - idx: indexes of the principal components
        """

        data = data - data.mean(axis=0)
        cov = np.dot(data.T, data) / data.shape[0]
        eig_values, eig_vectors = np.linalg.eig(cov)
        idx = eig_values.argsort()[::-1]
        eig_vectors = eig_vectors[:, idx]

        return np.dot(data, eig_vectors), idx


def compute_error(data, neighbors, colinear, dist0, angles0, curr_idx, adj_data, avg_dist0):
        """
        Compute the error between the original and the new relationships for a given point i.

        Input:
        - data
        - neighbors: Array of indices of nearest neighbours.
        - colinear: Array of indices of colinear points.
        - dist0: Original distances.
        - angles0: Original angles.
        - curr_idx: Index of the point.
        - adj_data: List of adjusted point.
        - avg_dist0: Original average distance.

        Returns:
        - error: Error between the original and the new relationships for point curr_idx.
        """
        c = 10
        error = 0

        for i,j in enumerate(neighbors[curr_idx, :]):
            # weight of the error
            w = 1
            # change weight if point j has already been adjusted
            if j in adj_data:
                w = c

            # distance between current point and point j
            new_dist = np.linalg.norm(data[curr_idx, :] - data[j, :])

            # angle between current point and colinear point with current one, through point j
            v1 = data[curr_idx, :] - data[j, :] 
            v2 = data[colinear[curr_idx,i], :] - data[j, :] 

            if (np.linalg.norm(v1) * np.linalg.norm(v2)) == 0:
                new_angle = 0
            else:
                new_angle = np.arccos(np.clip(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1,1))

            # obtain final error
            error_dist = ((dist0[curr_idx, i] - new_dist)/(2*avg_dist0))**2
            error_angle  = ((angles0[curr_idx, i] - new_angle)/np.pi)**2

            error += w *( error_dist + error_angle )

        return error


def adjust_points(data, dpres, curr_idx, eta, adj_data, neighbors, colinear, dist0, angles0, avg_dist0):
        """
        Adjust the curr_idx point in order to minimize the error between the original and the new relationships.

        Input:
        - data
        - dpres: Dimensions to preserve.
        - curr_idx: Index of the point.
        - eta: Learning rate.
        - adj_data: List of adjusted points.
        - neighbors: Array of indices of nearest neighbours.
        - colinear: Array of indices of colinear points.
        - dist0: Original distances.
        - angles0: Original angles.
        - avg_dist0: Original average distance.

        Returns:
        - s: Number of steps to reach the minimum error.
        - data: Adjusted data.
        """
        s = 0
        improved = True

        eta = 0.3*eta

        while improved:
                s += 1
                improved = False

                error = compute_error(data, neighbors, colinear, dist0, angles0, curr_idx, adj_data, avg_dist0)

                for d in dpres:
                    data[curr_idx, d] += eta

                    new_error = compute_error(data, neighbors, colinear, dist0, angles0, curr_idx, adj_data, avg_dist0)
                    if new_error > error:
                        data[curr_idx, d] -= 2*eta

                        new_error = compute_error(data, neighbors, colinear, dist0, angles0, curr_idx, adj_data, avg_dist0)
                        if  new_error > error:
                            data[curr_idx, d] += eta
                        else:
                            improved = True
                    else:
                        improved = True

        return s, data


def plot(data, epoch, plot_dim, color):
    """
    Plot the data in the specified directory.

    Input:
    - data: Data to plot.
    - epoch: Current epoch.
    - plot_dim: Dimension of the plot - either 2d or 3d
    - color: Color of the points.
    """

    # Create the directory if it does not exist
    directory = f's_data_{data.shape[0]}points_{plot_dim}d'
    if not os.path.exists(directory):
        os.makedirs(directory)

    if plot_dim == 2:
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.scatter(data[:,0], data[:,1], c=color, cmap=plt.cm.viridis)
    else:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(data[:,0], data[:,1],data[:,2], c=color, cmap=plt.cm.viridis)

    # Save the plot in the specified directory
    file_path = os.path.join(directory, f'epoch_{epoch}_plot.png')
    plt.savefig(file_path)
    plt.close()

