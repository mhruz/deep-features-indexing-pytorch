import argparse
import os

from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a directory structure of images into desired format and save')
    parser.add_argument('path_source', type=str, help='path to root of the input directory structure')
    parser.add_argument('out_type', type=str, help='string representing the extention of target images (eg. jpg)')
    parser.add_argument('path_output', type=str, help='path to output root of the structure')
    parser.add_argument('--resize', type=float, help='relative size of output, default=1.0', default=1.0)
    parser.add_argument('--rewrite', type=bool, help='whether to rewrite output or keep existing, default=False',
                        default=False)
    args = parser.parse_args()

    accepted_extentions = ("jpg", "jp2", "jpeg", "png", "bmp", "tiff", "tif")

    for root, dirs, files in os.walk(args.path_source):
        for file in files:
            if not file.endswith(accepted_extentions):
                continue

            im = Image.open(os.path.join(root, file))
            if args.resize != 1.0:
                im = im.resize((int(round(im.size[0] * args.resize)), int(round(im.size[1] * args.resize))))

            cp = os.path.commonpath([args.path_source, root])
            target_path = root.replace(cp, args.path_output)

            if os.path.isfile(target_path) and args.rewrite is not True:
                continue
                
            os.makedirs(target_path, exist_ok=True)
            out_filename = os.path.basename(file)
            out_filename = os.path.splitext(out_filename)[0]
            out_filename += ".{}".format(args.out_type)

            im.save(os.path.join(target_path, out_filename), quality=50)
