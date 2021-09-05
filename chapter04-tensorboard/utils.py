#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/9/3

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def get_labels():
    return ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


def show_image(image, one_channel=False):
    if one_channel:
        image = image.mean(dim=0)
    image = image / 2 + 0.5

    if one_channel:
        image = image.cpu()
        plt.imshow(image.numpy(), cmap='Greys')
    else:
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.show()


def image_to_probs(model, images):
    output = model(images)
    _, pred_tensor = torch.max(output, 1)
    pred_tensor = pred_tensor.cpu()
    preds = np.squeeze(pred_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(model, images, labels):
    preds, probs = image_to_probs(model, images)
    fig = plt.figure(figsize=(12, 28))
    classes = get_labels()
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        show_image(images[idx], one_channel=True)
        ax.set_title('{0}, {1:.1f}%\n(label:{2}'.format(classes[preds[idx]],
                                                        probs[idx] * 100.0,
                                                        classes[labels[idx]]),
                     color=('green' if preds[idx] == labels[idx].item() else 'red'))
    return fig


def add_pr_curve_tensorboard(writer, class_index, test_probs, test_preds, global_step=0):

    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]
    classes = get_labels()
    writer.add_pr_curve(classes[class_index], tensorboard_preds,
                        tensorboard_probs, global_step=global_step)