import numpy as np
from collections import deque
import copy
import os

from utils import compute_original_relationships, get_avg_dist, PCA_rotation, compute_error, adjust_points, plot

def Manifold_Sculpting(data, k=12, n_components=2, n_iter=800, sigma=0.9, th=0.0001, align=True, verbose=True, color=None, dim=2):
    '''
    Manifold Sculpting algorithm for dimensionality reduction.

    Input:  
    data: np.array
        The data to be reduced
    k: int, default=12
        Number of nearest neighbors to consider
    n_components: int, default=2
        Number of dimensions in the output space
    n_iter: int, default=800
        Maximum number of iterations
    sigma: float, default=0.9
        Scaling factor for the discarded dimensions
    th: float, default=0.0001
        Threshold for the change in the data to stop the algorithm
    align: bool, default=True
        Whether to align the data along the principal components
    verbose: bool, default=True
        Whether to print information about the algorithm
    color: np.array, default=None
        The colors of the points to plot

    Output:
    embedding: np.array
        The data in the new space
    '''

    # Step 1 & 2: find the k nearest neighbors for each point and compute original relationships between each point and its neighbors
    neighbors, dist0, colinear, angles0, avg_dist0 = compute_original_relationships(data, k)
    eta = copy.deepcopy(avg_dist0)

    # Step 3: Optionally align data according PCA rotation
    if align:
        x_pca, idx = PCA_rotation(data)
    else:
        x_pca = data.copy()

    dpres = idx[:n_components]
    dscal = idx[n_components:]

    if verbose:
        print("Preserved dimensions: ", dpres)
        print("Discarded dimensions: ", dscal)

    prev_data = copy.deepcopy(x_pca)

    # Step 4: Iteratively transform the data

    for iter in range(n_iter):
        # 4a: scale the data along the discarded dimensions
        for j in range(x_pca.shape[0]):
            x_pca[j, dscal] *= sigma
        
        # 4b: the values in dpres are scaled up to keep
        # the average neighbor distance equal to avg_dist
        while get_avg_dist(x_pca, neighbors) < avg_dist0:
            for j in range(x_pca.shape[0]):
                x_pca[j, dpres] /= sigma

        # create queue of points and add a random point to the queue
        start_point = np.random.randint(0, x_pca.shape[0])
        q = deque([start_point])

        # keep list of adjusted points
        adj_data = set()

        step = 0

        # while queue is not empty
        while q:
            # pick point from queue
            curr_idx = q.popleft()

            if curr_idx not in adj_data:  # if current point has not been adjusted yet
                s, x_pca = adjust_points(x_pca, dpres, curr_idx, eta, adj_data, neighbors, colinear, dist0, angles0, avg_dist0)
                step += s

                # add current point to adjusted points
                adj_data.add(curr_idx)

                # add neighbors to queue
                for n in neighbors[curr_idx, :]:
                    q.append(int(n))

        # stop criterion
        change = np.sum(np.abs(x_pca - prev_data))
        prev_data = x_pca.copy()

        if change < th:
            if verbose:
                print(f"Converged after {iter} iterations")
            break

        if (iter % 10 == 0 and iter != 0 and verbose):
            print(f"Iteration: {iter}, change: {change}")
            plot(x_pca, iter, dim, color)

            # save checkpoint
            os.makedirs(f'checkpoints_{x_pca.shape[0]}', exist_ok=True)
            checkpoint_path = f'checkpoints_{x_pca.shape[0]}/embedding_iter_{iter}.npy'
            np.save(checkpoint_path, x_pca)

    # Step 5: Project data by discarding the unwanted dimensions
    embedding = x_pca[:, dpres]

    return embedding



