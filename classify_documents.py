import pickle
import argparse
import json
import h5py
import os
import numpy as np
from shutil import copyfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify the deep features using kNN.')
    parser.add_argument('path_knn', type=str, help='path to pickle with kNN model')
    parser.add_argument('path_id_list', type=str, help='path to pickle with list of IDs')
    parser.add_argument('path_data', type=str, help='path to data with deep features that will be classified')
    parser.add_argument('layer', type=str, help='deep features level')
    parser.add_argument('--annotations', type=str, help='optional path to json annotations to compute the precision')
    parser.add_argument('--save_debug', type=str, help='whether to save debug images', default=False)
    parser.add_argument('--path_images', type=str, help='path to images')
    parser.add_argument('output', type=str, help='path to output (only dir)')
    args = parser.parse_args()

    id_list_pickle = open(args.path_id_list, 'rb')
    id_list = pickle.load(id_list_pickle)
    id_list_pickle.close()

    knn_pickle = open(args.path_knn, 'rb')
    knn = pickle.load(knn_pickle)
    knn_pickle.close()

    f = h5py.File(args.path_data, 'r')

    data = f[args.layer][:]

    results = knn.predict(data).tolist()

    os.makedirs(args.output, exist_ok=True)
    json_output = open(os.path.join(args.output, "results.json"), "wt", encoding="utf8")

    if args.annotations is not None:
        confusion_matrix = np.zeros((len(id_list), len(id_list)))
    else:
        json_ann = None
        confusion_matrix = None

    acc = 0
    total_items = 0

    results_dict = {}

    for idx, res in enumerate(results):
        if args.annotations is not None:
            json_ann = json.load(
                open(os.path.join(args.annotations, "{}.json".format(f["filenames"][idx])), "rt", encoding="utf8"))
            try:
                target = id_list.index(json_ann["page_type"])
            except ValueError:
                print("Unknown target: {}".format(json_ann["page_type"]))
                continue

            try:
                if json_ann["manually_verified"] == "yes":
                    continue
                if json_ann["correct_label"] != "yes":
                    continue
            except KeyError:
                print("Annotation {} is not in version 0.1.0".format(f["filenames"][idx]))
                pass

            results_dict[f["filenames"][idx]] = {}
            results_dict[f["filenames"][idx]]["label"] = json_ann["page_type"]
            results_dict[f["filenames"][idx]]["predict"] = id_list[res]

            if args.save_debug:
                if res != target:
                    os.makedirs(os.path.join(args.output, "debug", json_ann["page_type"], id_list[res]), exist_ok=True)
                    copyfile(os.path.join(args.path_images, f["filenames"][idx]),
                             os.path.join(args.output, "debug", json_ann["page_type"], id_list[res], f["filenames"][idx]))

            if res == target:
                acc += 1

            total_items += 1

            confusion_matrix[target, res] += 1

    acc /= total_items
    print("Accuracy: {}".format(acc))

    if args.annotations is not None:
        np.savetxt(os.path.join(args.output, "confusion_matrix.txt"), confusion_matrix, "%4d")
        json_output.write(json.dumps(results_dict, indent=3, sort_keys=True))

    json_output.close()
