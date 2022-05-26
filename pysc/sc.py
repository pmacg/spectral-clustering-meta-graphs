"""
Implementations of the spectral clustering algorithm with fewer eigenvectors.
"""
import scipy as sp
import scipy.sparse.linalg
from sklearn.cluster import KMeans
import sgtl
import sgtl.clustering
from typing import List, Type
from .objfunc import ClusteringObjectiveFunction
from .sclogging import logger


def sc_precomputed_eigenvectors(eigvecs, num_clusters, num_eigenvectors):
    """
    Given an array of eigenvectors, run the k-means step of spectral clustering using the given number of eigenvectors.

    :param eigvecs: The precomputed eigenvectors
    :param num_clusters: The number of clusters to find
    :param num_eigenvectors: The number of eigenvectors to use for clustering
    :return: the found clusters
    """
    # Perform k-means on the eigenvectors to find the clusters
    labels = KMeans(n_clusters=num_clusters).fit_predict(eigvecs[:, :num_eigenvectors])

    # Split the clusters.
    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    return clusters
