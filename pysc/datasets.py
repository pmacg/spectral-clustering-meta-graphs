"""
This module provides functions for interacting with different datasets.
"""
import os.path
from typing import Optional, List, Dict
import keras.datasets.mnist
import numpy
import sgtl
import sgtl.graph
import sgtl.random
import networkx
from networkx import grid_graph
from sklearn.datasets import fetch_20newsgroups_vectorized
import skimage.transform
import skimage.measure
import skimage.filters
import pickle
import h5py
import scipy as sp
import scipy.sparse
import math
from matplotlib import image
from .sclogging import logger


class Dataset(object):
    """
    This base class represents a dataset, for clustering. A dataset might consist of some combination of:
      - raw high-demensional numerical data
      - naturally graph structured data
      - a ground truth clustering
    """

    def __init__(self, data_file=None, gt_clusters_file=None, graph_file=None, num_data_points=None, graph_type="knn10"):
        """
        Intiialise the dataset, optionally specifying a data file.
        """
        # Raw data can be a dense or sparse numpy or scipy vector
        self.raw_data = None
        self.gt_clusters: Optional[List[List[int]]] = None
        self.gt_labels: Optional[List[int]] = None
        self.graph: Optional[sgtl.Graph] = None
        self.num_data_points = num_data_points

        # We only load the data if there is no graph file supplied
        if graph_file is None:
            self.load_data(data_file)

        self.load_gt_clusters(gt_clusters_file)
        self.load_graph(graph_file, graph_type=graph_type)

    @staticmethod
    def set_default_files(data_file: Optional[str],
                          gt_clusters_file: Optional[str],
                          graph_file: Optional[str],
                          kwargs: Dict[str, Optional[str]]):
        """Set the default values of the data file and gt_clusters file in the keyword arguments."""
        if 'data_file' not in kwargs:
            kwargs['data_file'] = data_file
        if 'gt_clusters_file' not in kwargs:
            kwargs['gt_clusters_file'] = gt_clusters_file
        if 'graph_file' not in kwargs:
            kwargs['graph_file'] = graph_file
        return kwargs

    def load_data(self, data_file):
        """
        Load the raw data from the given data file.
        :param data_file:
        :return:
        """
        # TODO:: Implement this method correctly
        if data_file is not None:
            self.raw_data = None

    def load_gt_clusters(self, gt_clusters_file):
        """
        Load the ground truth clusters from the specified file.
        :param gt_clusters_file:
        :return:
        """
        # TODO:: Implement this method correctly
        if gt_clusters_file is not None:
            self.gt_clusters = None

    def load_graph(self, graph_file=None, graph_type="knn10"):
        """
        Load the graph from the specified file, or create a graph for the dataset if the file is not specified.

        The 'graph_type' parameter can be used to specify how the graph should be constructed if it is not otherwise
        specified. Valid formats are:
            'knn10' - the k nearest neighbour graph, with 10 neighbours. Can replace 10 as needed.

        :param graph_file: (optional) the file containing the edgelist of the graph to load.
        :param graph_type: (optional) if there is no edgelist, the type of graph to be constructed.
        """
        if graph_file is not None:
            logger.info(f"Loading edgelist graph for the {self.__class__.__name__} from {graph_file}...")
            self.graph = sgtl.graph.from_edgelist(graph_file, num_vertices=self.num_data_points)
        elif self.raw_data is not None:
            # Construct the graph using the method specified
            if graph_type[:3] == "knn":
                logger.info(f"Constructing the KNN graph for the {self.__class__.__name__}...")

                # We will construct the k-nearest neighbour graph
                k = int(graph_type[3:])
                self.graph = sgtl.graph.knn_graph(self.raw_data, k)
            elif graph_type[:3] == "rbf":
                logger.info(f"Constructing the RBF graph for {self}...")
                self.graph = sgtl.graph.rbf_graph(self.raw_data, variance=20)
        else:
            logger.debug(f"Skipping constructing graph for the {self.__class__.__name__}.")

    def construct_and_save_graph(self, graph_filename: str, graph_type="knn10"):
        """
        Construct a graph representing this dataset (using knn10 by default), and save it to the specified file.

        :param graph_filename: the name of the file to save the graph to
        :param graph_type: which type of graph to construct from the data
        """
        # Construct the graph
        self.load_graph(graph_file=None, graph_type=graph_type)

        # Save the graph
        logger.info(f"Saving the graph to {graph_filename}.")
        sgtl.graph.to_edgelist(self.graph, graph_filename)

    def ground_truth(self) -> Optional[List[List[int]]]:
        """
        Return the ground truth clusters or None if they are unknown.
        :return:
        """
        return self.gt_clusters

    def __repr__(self):
        return self.__str__()


class MnistDataset(Dataset):

    def __init__(self, *args, k=10, downsample=None, **kwargs):
        """
        Load the mnist dataset using the KNN graph.

        :param k: (Optional) the value of k in the KNN graph to use
        :param downsample: Set to an integer to downsample each image to an n * n image after normalisation.
        """
        self.downsample = downsample
        self.k = k
        self.graph = None

        # Generate the edgelist file if it doesn't already exist.
        if self.downsample is not None:
            edgelist_filename = f"data/mnist/mnist_{downsample}_knn{k}.edgelist"
        else:
            edgelist_filename = f"data/mnist/mnist_knn{k}.edgelist"
        if not os.path.exists(edgelist_filename):
            self.load_data(None)
            self.construct_and_save_graph(edgelist_filename, graph_type=f"knn{k}")

        kwargs = self.set_default_files(None, None, edgelist_filename, kwargs)
        super(MnistDataset, self).__init__(*args, **kwargs)

    def load_data(self, data_file):
        """
        For the MNIST dataset, we will load the data using the keras api.
        :param data_file:
        :return:
        """
        # We only load the data if the graph hasn't been loaded yet.
        if self.graph is None:
            logger.info("Loading MNIST raw data from Keras...")

            # Load the basic data
            self.raw_data = []
            (train_x, _), (test_x, _) = keras.datasets.mnist.load_data()

            # Downsample and normalise the data
            if self.downsample is not None:
                self.raw_data = numpy.stack(
                    [skimage.transform.resize(im, (self.downsample, self.downsample)) for im in train_x], 0)
                self.raw_data = numpy.reshape(self.raw_data, (len(train_x), -1))
            else:
                # Normalise each number to be between 0 and 1 by dividing through by 255.
                self.raw_data = numpy.reshape(train_x, (len(train_x), -1))
                self.raw_data = self.raw_data / 255

            # Set the total number of data points.
            self.num_data_points = len(train_x)

            # TODO: add the test set as well

    def load_gt_clusters(self, gt_clusters_file):
        """
        Load the ground truth clusters using the keras api.
        :param gt_clusters_file:
        :return:
        """
        logger.info("Loading MNIST GT clusters from Keras...")

        self.gt_clusters = [[] for _ in range(10)]
        self.gt_labels = []

        (_, train_y), (_, test_y) = keras.datasets.mnist.load_data()
        for i, y in enumerate(train_y):
            self.gt_clusters[y].append(i)
            self.gt_labels.append(y)

        # TODO: add the test set as well

    def __str__(self):
        return f"mnist({self.k}, {self.downsample})"

    def __repr__(self):
        self.__str__()


class UspsDataset(Dataset):

    def __init__(self, *args, k=10, downsample=None, **kwargs):
        """
        Load the usps dataset using the KNN graph.

        :param k: (Optional) the value of k in the KNN graph to use
        :param downsample: Set to an integer to downsample each image to an n * n image after normalisation.
        """
        self.downsample = downsample
        self.k = k
        self.graph = None

        # Generate the edgelist file if it doesn't already exist.
        if self.downsample is not None:
            edgelist_filename = f"data/usps/usps_{downsample}_knn{k}.edgelist"
        else:
            edgelist_filename = f"data/usps/usps_knn{k}.edgelist"
        if not os.path.exists(edgelist_filename):
            self.load_data(None)
            self.construct_and_save_graph(edgelist_filename, graph_type=f"knn{k}")

        kwargs = self.set_default_files(None, None, edgelist_filename, kwargs)
        super(UspsDataset, self).__init__(*args, **kwargs)

    def load_data(self, data_file):
        """
        For the USPS dataset, we will load the data from the h5 file.
        :param data_file:
        :return:
        """
        data_file = "data/usps/usps.h5"

        # We only load the data if the graph hasn't been loaded yet.
        if self.graph is None:
            logger.info("Loading USPS raw data...")

            # Load the basic datas
            with h5py.File(data_file, 'r') as hf:
                train = hf.get('train')
                X_tr = train.get('data')[:]
            self.raw_data = X_tr
            self.num_data_points = self.raw_data.shape[0]

            # Downsample and normalise the data
            if self.downsample is not None:
                self.raw_data = numpy.stack(
                    [skimage.transform.resize(im, (self.downsample, self.downsample)) for im in self.raw_data], 0)
                self.raw_data = numpy.reshape(self.raw_data, (self.num_data_points, -1))
            else:
                # Normalise each number to be between 0 and 1 by dividing through by 255.
                self.raw_data = numpy.reshape(self.raw_data, (self.num_data_points, -1))
                self.raw_data = self.raw_data / 255

    def load_gt_clusters(self, gt_clusters_file):
        """
        Load the ground truth clusters.
        :param gt_clusters_file:
        :return:
        """
        logger.info("Loading USPS GT clusters...")

        # Load the basic data
        data_file = "data/usps/usps.h5"
        with h5py.File(data_file, 'r') as hf:
            train = hf.get('train')
            self.gt_labels = train.get('target')[:]

        self.gt_clusters = [[] for _ in range(10)]

        for i, y in enumerate(self.gt_labels):
            self.gt_clusters[y].append(i)

    def __str__(self):
        return f"usps({self.k}, {self.downsample})"

    def __repr__(self):
        self.__str__()


class SbmCycleDataset(Dataset):

    def __init__(self, *args, k=6, n=50, p=0.3, q=0.05, **kwargs):
        self.k, self.n, self.p, self.q = k, n, p, q
        super(SbmCycleDataset, self).__init__(self, *args, num_data_points=(n * k), **kwargs)

    def load_data(self, data_file):
        # The SBM dataset has no data
        pass

    def load_graph(self, graph_file=None, graph_type="knn10"):
        # Generate the graph from the sbm
        logger.info(f"Generating {self} graph from sbm...")
        prob_mat = self.p * sp.sparse.eye(self.k) + self.q * sgtl.graph.cycle_graph(self.k).adjacency_matrix()
        self.graph = sgtl.random.sbm_equal_clusters(self.n * self.k, self.k, prob_mat.toarray())

    def load_gt_clusters(self, gt_clusters_file):
        logger.info(f"Loading GT clusters for {self}...")

        # We can just generate the ground truth clusters as needed
        self.gt_clusters = [list(range(i * self.n, (i * self.n) + self.n)) for i in range(self.k)]
        self.gt_labels = []
        for cluster in range(self.k):
            for _ in range(self.n):
                self.gt_labels.append(cluster)

    def __str__(self):
        return f"sbmCycle({self.k}, {self.n}, {self.p}, {self.q})"

    def __repr__(self):
        return self.__str__()


class SBMGridDataset(Dataset):

    def __init__(self, *args, d=3, n=50, p=0.3, q=0.05, **kwargs):
        self.d, self.n, self.p, self.q = d, n, p, q
        super(SBMGridDataset, self).__init__(self, *args, num_data_points=(n * (d * d)), **kwargs)

    def load_data(self, data_file):
        # The SBM dataset has no data
        pass

    def load_graph(self, graph_file=None, graph_type="knn10"):
        # Generate the graph from the sbm
        logger.info(f"Generating {self} graph from sbm...")
        prob_mat = self.p * sp.sparse.eye(self.d * self.d) + self.q * \
            networkx.to_numpy_matrix(grid_graph((self.d, self.d)))
        self.graph = sgtl.random.sbm_equal_clusters(self.n * self.d * self.d, self.d * self.d, prob_mat.tolist())

    def load_gt_clusters(self, gt_clusters_file):
        logger.info(f"Loading GT clusters for {self}...")

        # We can just generate the ground truth clusters as needed
        self.gt_clusters = [list(range(i * self.n, (i * self.n) + self.n)) for i in range(self.d * self.d)]
        self.gt_labels = []
        for cluster in range(self.d * self.d):
            for _ in range(self.n):
                self.gt_labels.append(cluster)

    def __str__(self):
        return f"sbmGrid({self.d}, {self.n}, {self.p}, {self.q})"

    def __repr__(self):
        return self.__str__()


class BSDSDataset(Dataset):

    def __init__(self, img_idx, *args, downsample_factor=None,
                 data_directory="data/bsds/BSR/bench/data/images/", blur_variance=1, **kwargs):
        """Construct a dataset from a single image in the BSDS dataset.

        We will construct a graph from the image based on the gaussian radial basis function.

        :param img_idx: The number of the image in the dataset.
        :param downsample_factor: The factor by which do downsample the image. If none is supplied, then the factor is
                                  computed to make the total number of vertices in the dataset graph roughly equal to
                                  20,000.
        :param blur_variance: The variance of the gaussian blur applied to the downsampled image
        :param data_directory: The base directory containing the dataset images.
        """
        self.img_idx = img_idx
        self.image_filename = os.path.join(data_directory, f"{img_idx}.jpg")
        self.original_image_dimensions = []
        self.downsampled_image_dimensions = []
        self.downsample_factor = downsample_factor
        self.blur_variance = blur_variance
        super(BSDSDataset, self).__init__(*args, graph_type="rbf", **kwargs)

    def load_graph(self, *args, **kwargs):
        super(BSDSDataset, self).load_graph(*args, **kwargs)

        # Add a grid to the graph, with weight 0.01.
        logger.info(f"Adding grid graph to image graph for {self}...")
        grid_graph_adj_mat = sp.sparse.lil_matrix((self.num_data_points, self.num_data_points))
        for x in range(self.downsampled_image_dimensions[0]):
            for y in range(self.downsampled_image_dimensions[1]):
                this_data_point = x * self.downsampled_image_dimensions[1] + y

                # Add the four orthogonal edges
                if x > 0:
                    that_data_point = (x - 1) * self.downsampled_image_dimensions[1] + y
                    grid_graph_adj_mat[this_data_point, that_data_point] = 0.1
                    grid_graph_adj_mat[that_data_point, this_data_point] = 0.1
                if y > 0:
                    that_data_point = x * self.downsampled_image_dimensions[1] + y - 1
                    grid_graph_adj_mat[this_data_point, that_data_point] = 0.1
                    grid_graph_adj_mat[that_data_point, this_data_point] = 0.1
        grid_graph = sgtl.graph.Graph(grid_graph_adj_mat)
        self.graph += grid_graph

    def load_data(self, data_file):
        """
        Load the dataset from the image. Each pixel in the image is a data point. Each data point will have 5
        dimensions, namely the normalised 'rgb' values and the (x, y) coordinates of the pixel in the image.

        To reformat the data to be a manageable size, we downsample by a factor of 3.

        :param data_file:
        :return:
        """
        img = image.imread(self.image_filename)
        self.original_image_dimensions = (img.shape[0], img.shape[1])

        # Compute the downsample factor if needed
        if self.downsample_factor is None:
            current_num_vertices = self.original_image_dimensions[0] * self.original_image_dimensions[1]

            if current_num_vertices > 20000:
                self.downsample_factor = int(math.sqrt(current_num_vertices / 20000))
            else:
                self.downsample_factor = 1

        # Do the downsampling here
        img_l1 = skimage.measure.block_reduce(img[:, :, 0], self.downsample_factor, func=numpy.mean)
        img_l2 = skimage.measure.block_reduce(img[:, :, 1], self.downsample_factor, func=numpy.mean)
        img_l3 = skimage.measure.block_reduce(img[:, :, 2], self.downsample_factor, func=numpy.mean)
        img = numpy.stack([img_l1, img_l2, img_l3], axis=2)
        self.downsampled_image_dimensions = (img.shape[0], img.shape[1])

        # Blur the image slightly
        img = skimage.filters.gaussian(img, sigma=self.blur_variance)

        # Extract the data points from the image
        self.num_data_points = img.shape[0] * img.shape[1]
        self.raw_data = []
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                self.raw_data.append([img[x, y, 0],
                                      img[x, y, 1],
                                      img[x, y, 2],
                                      x,
                                      y])
        self.raw_data = numpy.array(self.raw_data)

    def save_downsampled_image(self, filename):
        """
        Save the downsampled image to the given file.

        :param filename:
        """
        # Load and downsample the image
        img = image.imread(self.image_filename)
        img_l1 = skimage.measure.block_reduce(img[:, :, 0], self.downsample_factor, func=numpy.mean)
        img_l2 = skimage.measure.block_reduce(img[:, :, 1], self.downsample_factor, func=numpy.mean)
        img_l3 = skimage.measure.block_reduce(img[:, :, 2], self.downsample_factor, func=numpy.mean)
        img = numpy.stack([img_l1, img_l2, img_l3], axis=2)

        # Blur the image slightly
        img = skimage.filters.gaussian(img, sigma=self.blur_variance)

        # Save the image to the given file
        image.imsave(filename, img / 255)

    def __str__(self):
        return f"bsds({self.img_idx})"
