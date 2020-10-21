import h5py
import json
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train kNN classifier from manually verified data.')
    parser.add_argument('path_features', type=str, help='path to h5 files with features')
    parser.add_argument('list_of_files', type=str, help='TXT file with a list of manually verified data')
    parser.add_argument('--ignore_validity', type=bool,
                        help='whether to ignore validity of file (correct_label, manually verified)', default=False)
    parser.add_argument('labels', type=str, help='path to jsons with manual annotations')
    parser.add_argument('output', type=str, help='path to output')
    args = parser.parse_args()

    f = open(args.list_of_files, "rt")
    list_of_files = f.readlines()
    list_of_files = [x.strip() for x in list_of_files]
    training_data = []
    labels = []

    ids = []

    list_of_h5 = os.listdir(args.path_features)

    files_to_h5 = {}
    for h in list_of_h5:
        fh5 = h5py.File(os.path.join(args.path_features, h), "r")

        for i, ff in enumerate(fh5["filenames"]):
            files_to_h5[ff] = (i, fh5)

    for fil in list_of_files:
        ann_file = fil + ".json"
        ann_data = json.load(open(os.path.join(args.labels, ann_file), "rt"))

        if not args.ignore_validity:
            if ann_data["manually_verified"] == "no":
                print("Data {} was not manually verified! Skipping...".format(fil))
                continue
            if ann_data["correct_label"] != "yes":
                print("Data {} is not correct! Skipping...".format(fil))
                continue

        if ann_data["page_type"] not in ids:
            target = len(ids)
            ids.append(ann_data["page_type"])
        else:
            target = ids.index(ann_data["page_type"])

        labels.append(target)
        training_data.append(files_to_h5[fil][1]["avgpool"][files_to_h5[fil][0]])

    print("Data processing ready. Valid files: {}".format(len(training_data)))

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(training_data, labels)

    # Its important to use binary mode
    knnPickle = open(os.path.join(args.output, 'resnet50_pytorch_index.p'), 'wb')
    # source, destination
    pickle.dump(knn, knnPickle, protocol=pickle.HIGHEST_PROTOCOL)
    knnPickle.close()

    id_list_pickle = open(os.path.join(args.output, 'id_list_resnet50_pytorch.p'), 'wb')
    pickle.dump(ids, id_list_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    id_list_pickle.close()

    labels_pickle = open(os.path.join(args.output, 'labels_resnet50_pytorch.p'), 'wb')
    pickle.dump(labels, labels_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    labels_pickle.close()