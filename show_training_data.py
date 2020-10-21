from shutil import copyfile
import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copies the data from a list to a given location.')
    parser.add_argument('path_images', type=str, help='path to images for training')
    parser.add_argument('list_of_files', type=str, help='TXT file with a list of manually verified data')
    parser.add_argument('labels', type=str, help='path to jsons with manual annotations')
    parser.add_argument('output', type=str, help='path to output')
    args = parser.parse_args()

    f = open(args.list_of_files, "rt")
    list_of_files = f.readlines()
    list_of_files = [x.strip() for x in list_of_files]
    training_data = []
    labels = []

    ids = []

    for fil in list_of_files:
        ann_file = fil + ".json"
        ann_data = json.load(open(os.path.join(args.labels, ann_file), "rt"))
        if ann_data["manually_verified"] == "no":
            print("Data {} was not manually verified! Skipping...".format(fil))
            continue
        if ann_data["correct_label"] != "yes":
            print("Data {} is not correct! Skipping...".format(fil))
            continue

        if ann_data["multi_type"] != "no":
            print("Data {} is multi_type! Skipping...".format(fil))
            continue

        target_path = os.path.join(args.output, ann_data["page_type"], fil)
        target_dir = os.path.join(args.output, ann_data["page_type"])
        source_path = os.path.join(args.path_images, fil)

        os.makedirs(target_dir, exist_ok=True)
        copyfile(source_path, target_path)