import pickle
import h5py
import cv2
import os
from PIL import Image
import numpy as np


def read_tree(filename):
    tree = pickle.load(open(filename, "rb"))

    return tree


def find_similar_features(data, tree, k=2, key="avgpool"):
    data_raw = data[key]
    nearest = tree.kneighbors(data_raw, k)

    return nearest


def draw_nearest(image_path, input_filename, input_feature, tree, filenames, k=2):
    nearest = tree.kneighbors(input_feature.reshape(1, -1), k)
    im = Image.open(os.path.join(image_path, input_filename))
    im = np.array(im)

    show = False

    for i, (distance, index) in enumerate(zip(*nearest[0], *nearest[1])):
        if i == 0:
            continue

        if distance > 5.0:
            continue

        show = True
        cv2.namedWindow(str(distance))
        cv2.moveWindow(str(distance), i*350, 100)
        im_sample = Image.open(os.path.join(image_path, filenames[index]))
        im_sample = np.array(im_sample)
        cv2.imshow(str(distance), im_sample)

    if show:
        cv2.imshow("query", im)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    tree_filename = "watermarks_resnet50_knn.p"
    data_filename = "watermarks_index.h5"

    tree_meta = read_tree(tree_filename)

    tree = tree_meta["tree"]
    filenames = tree_meta["filenames"]

    data = h5py.File(data_filename, "r")

    # for i in range(len(filenames)):
    #     draw_nearest("watermarks", filenames[i], data["avgpool"][i], tree, filenames, k=2)

    nearest = find_similar_features(data, tree)
    nearest_mat = np.hstack((nearest[0][:, 1].reshape(-1, 1), nearest[1][:, 1].reshape(-1, 1)))
    np.save("nearest_samples.npy", nearest_mat)

