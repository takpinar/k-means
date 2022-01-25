"""
    This is a file you will have to fill in.
    It contains helper functions required by K-means method via iterative improvement
"""
import numpy as np
from random import sample
import sys


def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids
    Hint: use random.sample (it is already imported for you!)
    :param k: number of cluster centroids
    :param inputs: a 2D Numpy array, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    k_cluster_centroids = np.array(sample(inputs.tolist(), k))
    return k_cluster_centroids


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance
    :param inputs: inputs of data, a 2D Numpy array
    :param centroids: a Numpy array of k current centroids
    :return: a Numpy array of centroid indices, one for each row of the inputs
    """
    centroid_indices = np.zeros(inputs.shape[0])

    for xi, x in enumerate(inputs):

        smallest_distance = sys.maxsize

        for centroid_index, centroid in enumerate(centroids):

            distance_to_centroid = np.linalg.norm(x - centroid)

            if distance_to_centroid < smallest_distance:

                smallest_distance = distance_to_centroid
                centroid_indices[xi] = centroid_index

    return centroid_indices


def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster
    :param inputs: inputs of data, a 2D Numpy array
    :param indices: a Numpy array of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """

    k_cluster_centroids = np.zeros((k, inputs.shape[1]))

    for cluster in range(k):

        cluster_indices = np.where(indices == cluster, True, False)
        cluster_examples = inputs[cluster_indices]
        cluster_centroid = np.sum(cluster_examples, axis=0) / cluster_examples.shape[0]
        k_cluster_centroids[cluster] = cluster_centroid

    return k_cluster_centroids


def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    :param inputs: inputs of data, a 2D Numpy array
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: the tolerance we determine convergence with when compared to the ratio as stated on handout
    :return: a Numpy array of k cluster centroids, one per row
    """
    # Initialize Centroids Randomly
    centroids = init_centroids(k, inputs)

    iterations = 0
    shift = sys.maxsize

    while (iterations < max_iter) or (tol < shift):

        print("\n iterations: ", iterations)
        # Assign examples to new clusters
        centroid_indices = assign_step(inputs, centroids)
        # Get new list of centroids form clusters
        new_centroids = update_step(inputs, centroid_indices, k)

        # Find largest shift for any centroid
        diffs = np.linalg.norm(new_centroids - centroids, axis=1)
        old_norms = np.linalg.norm(centroids, axis=1)
        shift = np.max(diffs / old_norms)

        # Update Centroids
        centroids = new_centroids
        iterations += 1

    return centroids
