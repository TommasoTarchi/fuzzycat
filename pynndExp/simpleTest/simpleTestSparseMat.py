"""
This script can be used to generate an initial clusters dataset, using Gaussian
noise, then apply pynndescent.

Args:
    - Number of samples, i.e. realizations of the data points (default: 100).
"""

from pathlib import Path
import shutil
import os
import sys
import numpy as np
from scipy.sparse import csr_matrix
import sklearn.datasets as data
from fuzzycat import FuzzyData
from pynndescent import NNDescent

if __name__ == '__main__':
    # Clean clusters data directory
    clustersDirPath = Path.cwd() / "Clusters"
    if clustersDirPath.is_dir():
        for item in clustersDirPath.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # Set number of points and points realizations
    nPoints = 400000
    nSamples = 100
    if len(sys.argv) > 1:
        nSamples = int(sys.argv[1])

    # Generate mean values
    np.random.seed(0)

    print("Generating clusters...")

    background = np.random.uniform(-10, 10, (int(nPoints / 2), 2))

    clustersWrite, _ = data.make_blobs(
        n_samples=int(nPoints / 2),
        centers=20,                 # 20 distinct clusters
        cluster_std=0.4,            # Spread of each cluster
        # Keeps clusters inside the background noise boundaries
        center_box=(-9, 9),
        random_state=0,
    )

    P = np.vstack([background, clustersWrite])

    # Generate dataset
    covP = 0.05**2
    FuzzyData.clusteringsFromRandomSamples(P, covP, nSamples)

    print("Fetching and formatting clusters data...")

    # Read clusters data from directory and build input for pynndescent
    # (use scipy sparse matrices)
    files = sorted(os.listdir(clustersDirPath))
    clustersRead = [np.load(os.path.join(clustersDirPath, f)) for f in files]

    nClusters = len(clustersRead)

    rows, cols = [], []
    for clusterId, pointIds in enumerate(clustersRead):
        rows.extend([clusterId] * len(pointIds))
        cols.extend(pointIds)

    belongData = np.ones(len(rows), dtype=np.bool_)
    belongMatrix = csr_matrix((belongData, (rows, cols)),
                              shape=(nClusters, nPoints))

    print("Running pynndescent...")

    # Compute approximated Jaccard indices and nearest neighbors using
    # pynndescent
    indices = NNDescent(
        belongMatrix,
        metric="jaccard",
        random_state=42,
    )

    neighborIndices, neighborDistances = indices.neighbor_graph
    neighborSimilarities = 1.0 - neighborDistances

    print(neighborIndices[0])
    print(neighborIndices[1])
    print(neighborIndices[2])
    print()
    print(neighborDistances[0])
    print(neighborDistances[1])
    print(neighborDistances[2])
