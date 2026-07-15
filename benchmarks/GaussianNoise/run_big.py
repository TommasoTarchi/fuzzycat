"""
This script can be used to generate an initial clusters dataset, using Gaussian
noise, then apply FuzzyCat and visualize the resulting soft clustering.

Args:
    - Number of samples, i.e. realizations of the data points (default: 100).
"""

from pathlib import Path
import shutil
import sys
import numpy as np
import sklearn.datasets as data
from fuzzycat import FuzzyData
from fuzzycat import FuzzyCat
from fuzzycat import FuzzyPlots

if __name__ == '__main__':
    # Clean clusters data directory
    clustersDirPath = Path.cwd() / "Clusters"
    if clustersDirPath.is_dir():
        for item in clustersDirPath.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # Set number of points realizations
    nSamples = 100
    if len(sys.argv) > 1:
        nSamples = int(sys.argv[1])

    # Generate mean values
    np.random.seed(0)

    background = np.random.uniform(-10, 10, (200000, 2))

    clusters, _ = data.make_blobs(
        n_samples=200000,
        centers=20,             # 20 distinct clusters
        cluster_std=0.4,        # Spread of each cluster
        center_box=(-9, 9),     # Keeps clusters inside the background noise boundaries
        random_state=0
    )

    P = np.vstack([background, clusters])

    # Generate dataset
    covP = 0.05**2
    FuzzyData.clusteringsFromRandomSamples(P, covP, nSamples)

    # Run FuzzyCat
    nPoints = P.shape[0]

    fc = FuzzyCat(nSamples, nPoints, verbose=2)
    fc.run()

    # Plot fuzzy clusters
    FuzzyPlots.plotFuzzyLabelsOnX(fc, P, membersOnly=True)
