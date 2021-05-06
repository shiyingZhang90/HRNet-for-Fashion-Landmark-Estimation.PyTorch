#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   evaluate.py.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Evaluation Scripts
@License :   This source code is licensed under the license found in the 
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image, ImageChops, ImageOps
import cv2

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import network
from datasets import SCHPDataset, transform_logits


dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label':['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--restore-weight", type=str, default='', help="restore pretrained model parameters.")
    parser.add_argument("--input", type=str, default='', help="path of input image folder.")
    parser.add_argument("--output", type=str, default='', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_palette_mask(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)

    for j in range(4, 11):
        lab = j
        palette[j * 3 + 0] = 255
        palette[j * 3 + 1] = 255
        palette[j * 3 + 2] = 255


 #   palette[7 * 3 + 0] = 0
 #   palette[7 * 3 + 1] = 0
 #   palette[7 * 3 + 2] = 0

    return palette

def main():
    args = get_arguments()
    SRC_DIR = args.input
    DST_DIR=args.output
    os.makedirs(DST_DIR, exist_ok=True)
    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']


    src_files = sorted([ f for f in os.listdir(SRC_DIR) if f[0] not in '._' and f.endswith('png')])
    for src_file in src_files:

        input_image_name = os.path.join(SRC_DIR, src_file)
        original_mask = Image.open(input_image_name)
        palette = get_palette(num_classes)

        parsing_result_path = os.path.join(DST_DIR, src_file)
        original_mask.putpalette(palette)
        original_mask.save(parsing_result_path)



    return


if __name__ == '__main__':
    main()
