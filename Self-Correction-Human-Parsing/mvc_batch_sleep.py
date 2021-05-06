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

    for j in range(4, 8):
        lab = j
        palette[j * 3 + 0] = 255
        palette[j * 3 + 1] = 255
        palette[j * 3 + 2] = 255


  #  palette[7 * 3 + 0] = 255
  #  palette[7 * 3 + 1] = 255
  #  palette[7 * 3 + 2] = 255

  #  palette[5 * 3 + 0] = 255
  #  palette[5 * 3 + 1] = 255
  #  palette[5 * 3 + 2] = 255
    return palette

def main():
    args = get_arguments()

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']

    model = network(num_classes=num_classes, pretrained=None).cuda()
    model = nn.DataParallel(model)
    state_dict = torch.load(args.restore_weight)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    subdirs = [x[1] for x in os.walk(args.input)]
    for subdir in subdirs[0]:
        DIR = args.input + '/' + subdir
        dataset = SCHPDataset(root=DIR, input_size=input_size, transform=transform)
        dataloader = DataLoader(dataset)

        PARSING_DIR=args.input + '/test_label'
        os.makedirs(PARSING_DIR, exist_ok=True)
        CLOTHING_DIR=args.input + '/test_color'
        os.makedirs(CLOTHING_DIR, exist_ok=True)
        CLOTHMASK_DIR=args.input + '/test_edge'
        os.makedirs(CLOTHMASK_DIR, exist_ok=True)


        palette = get_palette(num_classes)
        mask_palette = get_palette_mask(num_classes)

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):

                image, meta = batch
                img_name = meta['name'][0]
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].numpy()[0]
                h = meta['height'].numpy()[0]

                output = model(image.cuda())
                upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
                upsample_output = upsample(output)
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(1, 2, 0) #CHW -> HWC

                logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
                parsing_result = np.argmax(logits_result, axis=2)
                

                parsing_result_path = os.path.join(PARSING_DIR,img_name[:-4]+'.png')
                cloth_mask_path = os.path.join(CLOTHMASK_DIR,img_name[:-4]+'.jpg')
                cloth_path = os.path.join(CLOTHING_DIR,img_name[:-4]+'.png')
                output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                #original mask all class
                original_mask = output_img
                original_mask.save(parsing_result_path)
                # original_mask.putpalette(palette)
                # original_mask.save(parsing_result_path)
                if not '-p-' in img_name : continue
                #new mask with only upper cloth
                new_mask = output_img
                new_mask.putpalette(mask_palette)
                RGB_mask = new_mask.convert('RGB')
                L_mask = new_mask.convert('L')
                original_image_path = os.path.join(DIR, img_name)
                print('original_image_path is ', original_image_path)
                original_image = Image.open(original_image_path)
                #original_save_path = os.path.join(OUTDIR ,img_name[:-4]+'.png')
                #original_image.save(original_save_path)
                masked_image = ImageChops.multiply(original_image, RGB_mask)
                bg_image = Image.new("RGBA", masked_image.size,(255,255,255,255))
                white_masked_image = bg_image.paste(masked_image.convert('RGBA'),(0,0),L_mask) 
                #reverse_mask = ImageOps.invert(RGB_mask)
                #white = Image.new("RGB", original_image.size, "white")
                #background = ImageChops.multiply(white, reverse_mask)
                #masked_image.paste(white, (0,0), reverse_mask)         
                L_mask.save(cloth_mask_path)
                bg_image.save(cloth_path)



                if args.logits:
                    logits_result_path = os.path.join(args.output, img_name[:-4] + '.npy')
                    np.save(logits_result_path, logits_result)
    return


if __name__ == '__main__':
    main()
