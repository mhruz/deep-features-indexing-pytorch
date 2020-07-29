import os
import json
import argparse
import h5py

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Index the deep features using euclidean distance. The input is many H5 files'
                    ' and associated JSON annotations.')
    parser.add_argument('path_h5', type=str, help='path to h5 files with deep features')
    parser.add_argument('path_json', type=str, help='path to json files with annotations')
    parser.add_argument('output', type=str, help='path to output')
    args = parser.parse_args()

    h5_files = os.listdir(args.path_h5)
    h5_files = [x for x in h5_files if x.endswith(".h5")]

    os.makedirs(os.path.split(args.output)[0], exist_ok=True)
    out_h5 = h5py.File(args.output, "w")

    json_files = os.listdir(args.path_json)
    json_files = [x for x in json_files if x.endswith(".json")]

    for h5 in h5_files:
        f = h5py.File(os.path.join(args.path_h5, h5), "r")
        for key in f.keys():
            if key not in out_h5:
                index = 0
                out_h5.create_dataset(key, shape=f[key].shape, dtype=f[key].dtype, maxshape=(None, *f[key].shape[1:]))
            else:
                index = out_h5[key].shape[0]
                out_h5[key].resize((index + f[key].shape[0], *f[key].shape[1:]))

            out_h5[key][index:] = f[key]

        f.close()

    out_h5.create_dataset("labels", shape=(len(out_h5["filenames"]),), dtype=int)

    labels_to_index = {"unknown": 0}
    labels = []
    for file in out_h5["filenames"]:
        if "{}.json".format(file) not in json_files:
            print("Invalid filename {} in H5. There is no associated annotation!".format(file))
            labels.append(0)
            continue

        json_data = json.load(open(os.path.join(args.path_json, "{}.json".format(file)), encoding="utf8"))
        label = json_data["page_type"]

        if label not in labels_to_index:
            labels_to_index[label] = len(labels_to_index)

        labels.append(labels_to_index[label])

    out_h5["labels"][:] = labels

    dt = h5py.string_dtype()
    out_h5.create_dataset("index_to_labels", shape=(len(labels_to_index),), dtype=dt)
    out_h5["index_to_labels"][:] = list(labels_to_index.keys())

    out_h5.close()
