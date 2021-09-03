#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import torch
import numpy as np
from skimage import transform


class Rescale(object):

    def __init__(self, output_size):
        super(Rescale, self).__init__()
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[0], image.shape[1]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):

    def __init__(self, output_size):
        super(RandomCrop, self).__init__()
        if isinstance(output_size, int):
            self.output_size = output_size, output_size
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[0], image.shape[1]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)

        image = image[top: top + new_h, left: left + new_w]
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):

    def __call__(self, sample):
        image, landmarkers = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarkers)}


















