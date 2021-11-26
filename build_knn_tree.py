import h5py
import json
import os
from sklearn.neighbors import NearestNeighbors
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train kNN classifier from any data. The tree is meant for fast search of nearest samples.')
    parser.add_argument('path_features', type=str, help='path to h5 files with features')
    parser.add_argument('--feature_name', type=str, help='name of features in input h5 file.', default="avgpool")
    parser.add_argument('--k_neighbors', "-k", type=int, help='number of closest features to keep.', default=10)
    parser.add_argument('output', type=str, help='path to output')
    args = parser.parse_args()

    f = h5py.File(args.path_features, "r")
    training_filenames = f["filenames"][:]
    training_data = f[args.feature_name]

    nn = NearestNeighbors(n_neighbors=args.k_neighbors, n_jobs=4)
    nn.fit(training_data)

    output = {"filenames": training_filenames, "tree": nn}

    # Its important to use binary mode
    nnPickle = open(args.output, 'wb')
    # source, destination
    pickle.dump(output, nnPickle, protocol=pickle.HIGHEST_PROTOCOL)
    nnPickle.close()
