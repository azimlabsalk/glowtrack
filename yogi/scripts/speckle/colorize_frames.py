import argparse
import glob
from pprint import pprint

from yogi.matching import colorize_frames
from matplotlib import pyplot as plt

def main(input_frames, template_frame, colorized_template, output_dir, image_bg, propagation_window, radius):
    colorize_frames(input_frames, template_frame, colorized_template, output_dir, image_bg=image_bg, propagation_window=propagation_window, radius=radius)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Uses SIFT matching to transfer color from a template frame to other frames.")

    # input arguments
    parser.add_argument("-i", dest="input_frames", help="directory or text file containing frames to colorize")
    parser.add_argument("-m", dest="image_bg", action='store_true', default=False, help="use original image as background, for visualization")
    parser.add_argument("-p", dest="propagation_window", default=0, type=int, help="number of previous frames to match against (in addition to template)")
    parser.add_argument("-t", dest="template_frame", help="template frame")
    parser.add_argument("-c", dest="colorized_template", help="colorized template")
    parser.add_argument("-o", dest="output_dir", help="output dir")
    parser.add_argument("-r", dest="radius", default=10, type=int, help="color keypoint radius")
    
    args = parser.parse_args()
    args_dict = vars(args)
    pprint(args_dict)
    
    main(**args_dict)
