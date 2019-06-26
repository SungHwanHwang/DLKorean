import argparse
import glob
import io
import os
import random

import numpy
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, '../label/2350-common-hangul.txt')
DEFAULT_PNG_FILE = os.path.join(SCRIPT_PATH, '../image-data/hangul-images/')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../image-data')
DISTORTION_COUNT = 3

def generate_hangul_csv(label_file, output_dir):
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w', encoding='utf-8')

    total_count = 0
    prev_count = 0

    folder_list = os.listdir(args.png_file)

    for i in range(len(folder_list)):
        file_list = os.listdir(args.png_file+folder_list[i])

        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))
        for p in range(len(file_list)):
            total_count += 1
            file_path = args.png_file+folder_list[i] + '/' + file_list[p]
            labels_csv.write(u'{},{}\n'.format(file_path, labels[i]))
            print(args.png_file + folder_list[i] + '/' + file_list[p] + '---' + labels[i])
    print('Finished generating {} images.'.format(total_count))
    labels_csv.close()


def elastic_distort(image, alpha, sigma):
    random_state = numpy.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and label CSV file.')
    parser.add_argument('--png-file', type=str, dest='png_file',
                        default=DEFAULT_PNG_FILE,
                        help='png image directory')
    args = parser.parse_args()

    generate_hangul_csv(args.label_file, args.output_dir)
