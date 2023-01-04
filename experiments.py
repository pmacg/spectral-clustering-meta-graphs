"""
Run experiments with the new spectral clustering approach.
"""
import multiprocessing
import time
import argparse
import scipy as sp
import scipy.sparse
import scipy.io
import numpy
from multiprocessing import Process, Queue
import os
import os.path
import math
import sgtl
import sgtl.random
import sgtl.clustering
import pysc.objfunc
import pysc.sc
import pysc.datasets
import pysc.evaluation
import pysc.objfunc
from pysc.sclogging import logger


def basic_experiment_sub_process(dataset, k, num_eigenvalues: int, q):
    logger.info(f"Starting clustering: {dataset} with {num_eigenvalues} eigenvalues.")
    start_time = time.time()
    found_clusters = sgtl.clustering.spectral_clustering(dataset.graph, num_clusters=k,
                                                         num_eigenvectors=num_eigenvalues)
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Finished clustering: {dataset} with {num_eigenvalues} eigenvalues.")
    this_rand_score = pysc.evaluation.adjusted_rand_index(dataset.gt_labels, found_clusters)
    this_mutual_info = pysc.evaluation.mutual_information(dataset.gt_labels, found_clusters)
    this_conductance = pysc.objfunc.KWayExpansion.apply(dataset.graph, found_clusters)
    q.put((num_eigenvalues, this_rand_score, this_mutual_info, this_conductance, total_time))


def basic_experiment(dataset, k):
    """
    Run a basic experiment with a given dataset, in which we do spectral clustering with a variety of numbers of
    eigenvectors, and compare the resulting clustering.

    :param dataset: A `pysc.datasets.Dataset` object.
    :param k: The number of clusters.
    """
    logger.info(f"Running basic experiment with {dataset.__class__.__name__}.")

    # Start all of the sub-processes to do the clustering with different numbers of eigenvalues
    rand_scores = {}
    mutual_info = {}
    conductances = {}
    times = {}
    q = Queue()
    processes = []
    for i in range(2, k + 1):
        p = Process(target=basic_experiment_sub_process, args=(dataset, k, i, q))
        p.start()
        processes.append(p)

    logger.info(f"All sub-processes started for {dataset}.")

    for p in processes:
        p.join()

    logger.info(f"All sub-processes finished for {dataset}.")

    # Get all of the data from the subprocesses
    while not q.empty():
        num_vectors, this_rand_sc, this_mut_info, this_conductance, this_time = q.get()
        rand_scores[num_vectors] = this_rand_sc
        mutual_info[num_vectors] = this_mut_info
        conductances[num_vectors] = this_conductance
        times[num_vectors] = this_time

    return rand_scores, mutual_info, conductances, times


############################################
# Experiments on MNIST and USPS
############################################
def mnist_experiment_instance(d, nn, q):
    this_rand_scores, this_mut_info, this_conductances, this_times = basic_experiment(
        pysc.datasets.MnistDataset(k=nn, downsample=d), 10)
    q.put((d, nn, this_rand_scores, this_mut_info, this_conductances, this_times))

def run_mnist_experiment():
    """
    Run experiments on the MNIST dataset.
    """

    # We will construct the 3-NN graph for the MNIST dataset.
    k = 3

    # Kick off the experiment in a sub-process.
    q = Queue()
    p = Process(target=mnist_experiment_instance, args=(None, k, q))
    p.start()
    p.join()

    # Write out the results
    with open("results/mnist/results.csv", 'w') as fout:
        fout.write("k, d, eigenvectors, rand\n")

        while not q.empty():
            downsample, k, rand_scores, _, _, _ = q.get()
            for i in range(2, 11):
                fout.write(f"{k}, {downsample}, {i}, {rand_scores[i]}\n")

def usps_experiment_instance(d, nn, q):
    this_rand_scores, this_mut_info, this_conductances, this_times = basic_experiment(
        pysc.datasets.UspsDataset(k=nn, downsample=d), 10)
    q.put((d, nn, this_rand_scores, this_mut_info, this_conductances, this_times))

def run_usps_experiment():
    """
    Run experiments on the USPS dataset.
    """

    # We will construct the 3-NN graph
    k = 3

    # Kick off the experiment ina sub-process
    q = Queue()
    p = Process(target=usps_experiment_instance, args=(None, k, q))
    p.start()
    p.join()

    # Write out the results
    with open("results/usps/results.csv", 'w') as fout:
        fout.write("k, d, eigenvectors, rand\n")

        while not q.empty():
            downsample, k, rand_scores, _ ,_, _ = q.get()
            for i in range(2, 11):
                fout.write(f"{k}, {downsample}, {i}, {rand_scores[i]}\n")


############################################
# Experiments on synthetic data
############################################
class SBMJobRunner(Process):

    def __init__(self, k, n, prob_p, prob_q, queue, num_runs=1, use_grid=False, **kwargs):
        super(SBMJobRunner, self).__init__(**kwargs)
        self.k, self.n, self.prob_p, self.prob_q = k, n, prob_p, prob_q
        self.queue = queue
        self.num_runs = num_runs
        self.use_grid = use_grid
        self.d = self.k
        if use_grid:
            self.k = self.d * self.d

    def run(self) -> None:
        # We will run the whole experiment self.num_runs times, and take the average results.
        total_conductances = [0] * self.k
        total_rand_scores = [0] * self.k

        for run_no in range(self.num_runs):
            if self.use_grid:
                dataset = pysc.datasets.SBMGridDataset(d=self.d, n=self.n, p=self.prob_p, q=self.prob_q)
            else:
                dataset = pysc.datasets.SbmCycleDataset(k=self.k, n=self.n, p=self.prob_p, q=self.prob_q)
            logger.info(f"Starting experiment: {dataset}, run number {run_no}")

            # Pre-compute the eigenvectors of the graph
            laplacian_matrix = dataset.graph.normalised_laplacian_matrix()
            _, eigvecs = scipy.sparse.linalg.eigsh(laplacian_matrix, self.k+1, which='SM')

            for num_eigenvalues in range(1, self.k + 1):
                found_clusters = pysc.sc.sc_precomputed_eigenvectors(eigvecs, self.k, num_eigenvalues)
                logger.info(f"Finished clustering: {dataset}, run number {run_no} with {num_eigenvalues} eigenvectors.")
                total_conductances[num_eigenvalues - 1] += pysc.objfunc.KWayExpansion.apply(dataset.graph, found_clusters)
                total_rand_scores[num_eigenvalues - 1] += pysc.evaluation.rand_index(dataset.gt_labels, found_clusters)

            logger.info(f"Finished experiment: {dataset}, run number {run_no}")

        # Get the average, and submit to the queue
        for num_eigenvalues in range(1, self.k + 1):
            self.queue.put((self.k, self.n, self.prob_p, self.prob_q, num_eigenvalues,
                            total_conductances[num_eigenvalues - 1] / self.num_runs,
                            total_rand_scores[num_eigenvalues - 1] / self.num_runs))

        # let the calling code know that we're done
        self.queue.put(None)


def run_sbm_experiment(n, k, prob_p, use_grid=False):
    logger.info(f"Running experiment with SBM data.")

    # For each set of SBM parameters, run 10 times.
    num_runs = 10

    # Start all of the sub-processes to do the clustering with different numbers of eigenvalues
    processes = []
    if use_grid:
        results_filename = "results/sbm/grid_results.csv"
    else:
        results_filename = "results/sbm/cycle_results.csv"
    with open(results_filename, 'w') as fout:
        fout.write("k, n, p, q, poverq, eigenvectors, conductance, rand\n")
        fout.flush()
        for prob_q in numpy.linspace(prob_p / 10, prob_p, num=10):
            q = Queue()
            p = SBMJobRunner(k, n, prob_p, prob_q, q, num_runs=num_runs, use_grid=use_grid)
            p.start()
            processes.append(p)

            # Keep at most 20 sub-processes
            if len(processes) >= 20:
                for p in processes:
                    # Save all of the data to the output file
                    while True:
                        process_result = p.queue.get()

                        if process_result is None:
                            break
                        else:
                            k, n, prob_p, prob_q, num_vectors, this_conductance, this_rand_score = process_result
                            fout.write(f"{k}, {n}, {prob_p}, {prob_q}, {prob_p/prob_q}, {num_vectors}, {this_conductance}, {this_rand_score}\n")
                            fout.flush()
                    p.join()
                processes = []

        logger.info(f"All sub-processes started for sbm experiments.")

        for p in processes:
            # Save all of the data to the output file
            while True:
                process_result = p.queue.get()

                if process_result is None:
                    break
                else:
                    k, n, prob_p, prob_q, num_vectors, this_conductance, this_rand_score = process_result
                    fout.write(f"{k}, {n}, {prob_p}, {prob_q}, {prob_p/prob_q}, {num_vectors}, {this_conductance}, {this_rand_score}\n")
                    fout.flush()
            p.join()

    logger.info(f"All sub-processes finished for sbm experiments.")


############################################
# Experiments on BSDS
############################################
def segment_bsds_image(bsds_dataset, num_segments, num_eigenvectors_l):
    """
    Given a loaded bsds dataset, find a segmentation into the given number of segments.

    :param bsds_dataset: the already loaded bsds_dataset
    :param num_segments: the number of segments to find
    :param num_eigenvectors_l: a list with different numbers of eigenvectors to use to find the segmentation
    :return: a list of segmentations
    """
    all_segmentations = []

    # First, compute all of the eigenvectors up front
    laplacian_matrix = bsds_dataset.graph.normalised_laplacian_matrix()
    _, eigvecs = scipy.sparse.linalg.eigsh(laplacian_matrix, max(num_eigenvectors_l), which='SM')

    for num_eigenvectors in num_eigenvectors_l:
        logger.info(f"Segmenting {bsds_dataset} into {num_segments} segments with {num_eigenvectors} eigenvectors.")
        found_clusters = pysc.sc.sc_precomputed_eigenvectors(eigvecs, num_segments, num_eigenvectors)
        all_segmentations.append(found_clusters)

    return all_segmentations


def save_bsds_segmentations(bsds_dataset, segmentations, eigenvectors_l, filename, upscale=True):
    """
    Save the given segmentation of the given dataset to the given filename.
    Save in matlab file format so the analysis can be done with matlab tools.

    :param bsds_dataset: the bsds image in question, as a dataset object
    :param segmentations: a list of segmentations
    :param eigenvectors_l: a list of the number of eigenvectors used for each segmentation
    :param filename: the filename to save the segmentation to
    :param upscale: whether to scale the segmentation back up to the original size
    :return:
    """
    seg_cell = numpy.empty((len(segmentations, )), dtype=object)
    eigs_cell = numpy.empty((len(segmentations, )), dtype=object)

    for seg_i, segmentation in enumerate(segmentations):
        # Construct the labels rather than the list of clusters.
        pixel_labels = [0] * bsds_dataset.num_data_points
        for i, segment in enumerate(segmentation):
            for pixel in segment:
                pixel_labels[pixel] = i

        # Construct the labelled image with the downsampled dimensions
        labelled_image = numpy.array(pixel_labels, dtype="int32")
        labelled_image = numpy.reshape(labelled_image, bsds_dataset.downsampled_image_dimensions) + 1

        # Scale up the segmentation by taking the appropriate tensor product
        labelled_image_upsample =\
            numpy.kron(labelled_image, numpy.ones((bsds_dataset.downsample_factor, bsds_dataset.downsample_factor)))
        labelled_image_upsample = labelled_image_upsample[:bsds_dataset.original_image_dimensions[0],
                                                          :bsds_dataset.original_image_dimensions[1]]

        seg_cell[seg_i] = labelled_image_upsample if upscale else labelled_image
        eigs_cell[seg_i] = eigenvectors_l[seg_i]

    # Save the labelled image to the given file
    data_to_save = {'segs': seg_cell, 'eigs': eigs_cell}
    sp.io.savemat(filename, data_to_save)


def get_bsd_num_cluster(gt_filename) -> int:
    """
    Given the path to the filename containing the ground truth clustering for an element of the bsds dataset, get the
    number of clusters that we should try to find.

    :param gt_filename:
    :return:
    """
    # Take the minimum number of clusters from the ground truth segmentations.
    gt_data = sp.io.loadmat(gt_filename)
    num_ground_truth_segmentations = gt_data["groundTruth"].shape[1]

    nums_segments = []
    for i in range(num_ground_truth_segmentations):
        this_segmentation = gt_data["groundTruth"][0, i][0][0][0]
        this_num_segments = numpy.max(this_segmentation)
        nums_segments.append(this_num_segments)

    # Return the median number of segments, and at least 2.
    return max(2, int(numpy.median(nums_segments)))


def run_bsds_experiment(image_id=None):
    """
    Run experiments on the BSDS dataset.
    :image_files: a list of the BSDS image files to experiment with
    :return:
    """
    if image_id is None:
        # If no image file is provided, then run the experiment on all image files in the test data.
        ground_truth_directory = "data/bsds/BSR/BSDS500/data/groundTruth/test/"
        images_directory = "data/bsds/BSR/BSDS500/data/images/test/"
        output_directory = "results/bsds/segs/"
        image_files = os.listdir(images_directory)
    else:
        # If an image filename is provided, then work out whether it is in the test or training data
        image_filename = image_id + '.jpg'
        images_directory = "data/bsds/BSR/BSDS500/data/images/test/"
        if image_filename in os.listdir(images_directory):
            ground_truth_directory = "data/bsds/BSR/BSDS500/data/groundTruth/test/"
            output_directory = "results/bsds/segs/"
            image_files = [image_filename]
        else:
            images_directory = "data/bsds/BSR/BSDS500/data/images/train/"
            ground_truth_directory = "data/bsds/BSR/BSDS500/data/groundTruth/train/"
            output_directory = "results/bsds/segs/"
            image_files = [image_filename]

            if image_filename not in os.listdir(images_directory):
                # If the target file is not in the training directory, then it's a lost cause.
                raise Exception("BSDS image ID not found.")
        
    for i, file in enumerate(image_files):
        id = file.split(".")[0]

        # Ignore any images we've already tried.
        if os.path.exists(output_directory + id + ".mat"):
            logger.debug(f"Skipping image {file} - output already exists.")
            continue

        logger.info(f"Running BSDS experiment with image {file}. (Image {i+1}/{len(image_files)})")

        # Get the number of clusters to look for in this image.
        k = get_bsd_num_cluster(os.path.join(ground_truth_directory, f"{id}.mat"))

        # Create the list of numbers of eigenvectors to use for the clustering - get 10 data points for each image.
        num_eigenvectors_l = list(range(2, k, int(math.ceil(k/10))))
        if len(num_eigenvectors_l) == 0 or num_eigenvectors_l[-1] != k:
            num_eigenvectors_l.append(k)

        dataset = pysc.datasets.BSDSDataset(id, blur_variance=0, data_directory=images_directory)
        segmentations = segment_bsds_image(dataset, k, num_eigenvectors_l)

        # Save the downscaled image
        output_filename = f"results/bsds/downsamples/{dataset.img_idx}.jpg"
        dataset.save_downsampled_image(output_filename)

        # Save the upscaled segmentations
        for i, num_eigenvectors in enumerate(num_eigenvectors_l):
            output_filename = f"results/bsds/segs/{dataset.img_idx}.mat"
            save_bsds_segmentations(dataset, segmentations, num_eigenvectors_l, output_filename)

        # Save the downscaled segmentation
        for i, num_eigenvectors in enumerate(num_eigenvectors_l):
            output_filename = f"results/bsds/downsampled_segs/{dataset.img_idx}.mat"
            save_bsds_segmentations(dataset, segmentations, num_eigenvectors_l, output_filename, upscale=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the experiments.')
    parser.add_argument('experiment', type=str, choices=['cycle', 'grid', 'mnist', 'usps', 'bsds'],
                        help="which experiment to perform")
    parser.add_argument('bsds_image', type=str, nargs='?', help="(optional) the BSDS ID of a single BSDS image file to segment")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment == 'cycle':
        run_sbm_experiment(1000, 10, 0.01)
    elif args.experiment == 'grid':
        run_sbm_experiment(1000, 4, 0.01, use_grid=True)
    elif args.experiment == 'mnist':
        run_mnist_experiment()
    elif args.experiment == 'usps':
        run_usps_experiment()
    elif args.experiment == 'bsds':
        if args.bsds_image is None:
            logger.warning("\nThe BSDS experiment is very resource-intensive. We recommend running on a compute server.")
            logger.info("Waiting 10 seconds before starting the experiment...")
            time.sleep(10)
            run_bsds_experiment()
        else:
            run_bsds_experiment(image_id=args.bsds_image)


if __name__ == "__main__":
    main()
