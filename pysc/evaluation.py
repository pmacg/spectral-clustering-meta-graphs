"""
Methods for evaluating the output of a clustering algorithm.
"""
from sklearn.metrics import rand_score, adjusted_rand_score, adjusted_mutual_info_score


def clusters_to_labels(clusters, num_data_points=None):
    """
    Given a list of clusters (a list of lists), convert it to a list of labels.

    :param clusters: A list of lists giving the members of each cluster
    :param num_data_points: The total number of data points in the data set
    :return: A single list containing the label for each datapoint.
    """
    if num_data_points is None:
        num_data_points = sum([len(cluster) for cluster in clusters])

    labels = [0] * num_data_points

    for i, cluster in enumerate(clusters):
        for elem in cluster:
            labels[elem] = i

    return labels


def adjusted_rand_index(gt_labels, found_clusters) -> float:
    """
    Compute the rand index of a given clustering, with respect to the ground truth clustering.

    :param gt_labels: A list of true cluster labels for each data point
    :param found_clusters: The candidate cluster labels for each data point
    :return: the rand index of the candidate clusters
    """
    if isinstance(found_clusters[0], list):
        # Need to convert the list of clusters to a list of labels
        found_labels = clusters_to_labels(found_clusters, num_data_points=len(gt_labels))
    else:
        found_labels = found_clusters

    return adjusted_rand_score(gt_labels, found_labels)


def rand_index(gt_labels, found_clusters) -> float:
    """
    Compute the rand index of a given clustering, with respect to the ground truth clustering.

    :param gt_labels: A list of true cluster labels for each data point
    :param found_clusters: The candidate cluster labels for each data point
    :return: the rand index of the candidate clusters
    """
    if isinstance(found_clusters[0], list):
        # Need to convert the list of clusters to a list of labels
        found_labels = clusters_to_labels(found_clusters, num_data_points=len(gt_labels))
    else:
        found_labels = found_clusters

    return rand_score(gt_labels, found_labels)


def mutual_information(gt_labels, found_clusters) -> float:
    """
    Compute the adjusted mutual information of a given clustering, with respect to the ground truth clustering.

    :param gt_labels: A list of true cluster labels for each data point
    :param found_clusters: The candidate cluster labels for each data point
    :return: the adjusted mutual information of the candidate clusters
    """
    if isinstance(found_clusters[0], list):
        # Need to convert the list of clusters to a list of labels
        found_labels = clusters_to_labels(found_clusters, num_data_points=len(gt_labels))
    else:
        found_labels = found_clusters

    return adjusted_mutual_info_score(gt_labels, found_labels)

