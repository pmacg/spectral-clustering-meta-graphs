# Beyond Spectral Clustering

This directory contains the code to reproduce the results in the paper "A Tighter Analysis of Spectral Clustering, and Beyond", published in 
ICML 2022.

## Preparing your environment
Our code is primarily written in Python 3. There is also a matlab
script for analysing the results of the BSDS experiment.

We recommend running the python code inside a virtual environment.

To install the dependencies of the project, run

```bash
pip install -r requirements.txt
```

If you would like to run the experiments on the BSDS dataset, you should untar the data file
in the `data/bsds` directory.

## Running the experiments
To run one of the experiments described in the paper, run

```bash
python experiment.py {experiment_name}
```

where ```{experiment_name}``` is one of `cycle`, `grid`, `mnist`, `usps`, or `bsds`.

The MNIST and USPS experiments will run easily on a laptop or desktop. The `cycle` and `grid` experiments will also run
on a personal computer but could take a few minutes since they must run multiple trials for each number of eigenvectors.

**Please note that the BSDS experiment is quite resource-intensive, and we recommend running on a compute server.**

## Output
The output from the experiments will be in the `results` directory, under the appropriate experiment name.
The BSDS results can be analysed using the matlab script `analyseBsdsResults.m` which will callthe
BSDS benchmarking code to evaluate the image segmentation output.
