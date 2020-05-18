import argparse
import h5py
import os
import numpy as np
from PIL import Image
from networks import DeepFeatureExtractor
import time
import sys


def extract_and_add_to_h5(images, filenames, h5_file):
    features = dfe.get_features(images)

    if "filenames" not in h5_file:
        index = 0
        dt = h5py.string_dtype()
        h5_file.create_dataset("filenames", shape=(len(images),), maxshape=(None,), dtype=dt)
    else:
        index = h5_file["filenames"].shape[0]
        h5_file["filenames"].resize((h5_file["filenames"].shape[0] + len(images),))

    h5_file["filenames"][index:] = filenames

    for layer, feat in features.items():
        if layer not in h5_file:
            index = 0
            h5_file.create_dataset(layer, shape=feat.data.shape, maxshape=(None, feat.data.shape[1]),
                                   dtype=np.float32)
        else:
            index = h5_file[layer].shape[0]
            h5_file[layer].resize((h5_file[layer].shape[0] + feat.data.shape[0], feat.data.shape[1]))

        h5_file[layer][index:] = feat.data

    h5_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute deep features on images from a path and store as h5.')
    parser.add_argument('path', type=str, help='path to images')
    parser.add_argument('model', type=str, help='what model to use (vgg16, resnet50, ...)')
    parser.add_argument('--batch_size', type=int, help='how many images to process at once (default 1)', default=1)
    parser.add_argument('--use_cuda', type=bool, help='whether to use CUDA', default=False)
    parser.add_argument('--data_range', type=str, help='range of data to process (12:123)')
    parser.add_argument('output', type=str, help='full-path to output')
    args = parser.parse_args()

    dfe = DeepFeatureExtractor(args.model, gpu=args.use_cuda)

    image_filenames = os.listdir(args.path)
    accepted_extentions = ("jpg", "jp2", "jpeg", "png", "bmp", "tiff")

    valid_image_filenames = []
    for im in image_filenames:
        if im.lower().endswith(accepted_extentions):
            valid_image_filenames.append(im)

    h5_file = h5py.File(args.output, "w")

    if args.data_range is not None:
        data_range = args.data_range.split(":")
        if len(data_range) != 2:
            raise ValueError("Incorrect data range!")

        if data_range[1] == "":
            data_range[1] = "{}".format(len(valid_image_filenames))

        data_range = [int(d) for d in data_range]
    else:
        data_range = [0, len(valid_image_filenames)]

    eta = None
    images = []
    filenames = []
    index = 0
    if eta is None:
        start_time = time.time()

    total_start = time.time()

    for filename in valid_image_filenames[data_range[0]:data_range[1]]:
        try:
            im = Image.open(os.path.join(args.path, filename))
        except IOError:
            continue

        images.append(im)
        filenames.append(filename)

        if len(images) == args.batch_size:
            index += args.batch_size
            extract_and_add_to_h5(images, filenames, h5_file)
            images = []
            filenames = []
            print("Processing image {}/{}".format(index, len(valid_image_filenames[data_range[0]:data_range[1]])))

            if eta is None:
                end_time = time.time()

                batch_time = end_time - start_time
                sample_time = batch_time / args.batch_size
                eta = len(valid_image_filenames[data_range[0]:data_range[1]]) * sample_time
                print("Avg. time per sample = {} s".format(sample_time))
                print("ETA: {} s".format(eta))

            sys.stdout.flush()

    if len(images) != 0:
        extract_and_add_to_h5(images, filenames, h5_file)

    print("Total time: {} s".format(time.time() - total_start))

    h5_file.close()
