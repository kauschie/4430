import imageio
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF from images in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing images.")
    parser.add_argument("output", type=str, help="Output GIF file name.")
    args = parser.parse_args()

    with imageio.get_writer(args.output, mode='I', duration=0.02) as writer:
        for filename in sorted(os.listdir(args.directory)):
            if filename.endswith(".png") and filename.startswith("frame_"):
                image = imageio.imread(os.path.join(args.directory, filename))
                writer.append_data(image)