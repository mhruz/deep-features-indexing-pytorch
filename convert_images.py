import argparse
import os

from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a directory structure of images into desired format and save')
    parser.add_argument('path_source', type=str, help='path to root of the input directory structure')
    parser.add_argument('out_type', type=str, help='string representing the extention of target images (eg. jpg)')
    parser.add_argument('path_output', type=str, help='path to output root of the structure')
    args = parser.parse_args()

    accepted_extentions = ("jpg", "jp2", "jpeg", "png", "bmp", "tiff", "tif")

    for root, dirs, files in os.walk(args.path_source):
        for file in files:
            if not file.endswith(accepted_extentions):
                continue

            im = Image.open(os.path.join(root, file))
            im = im.resize((im.size[0]//2, im.size[1]//2))
            cp = os.path.commonpath([args.path_source, root])
            target_path = root.replace(cp, args.path_output)
            os.makedirs(target_path, exist_ok=True)
            out_filename = os.path.basename(file)
            out_filename = os.path.splitext(out_filename)[0]
            out_filename += ".{}".format(args.out_type)

            im.save(os.path.join(target_path, out_filename), quality=50)
