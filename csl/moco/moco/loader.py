# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torchvision.transforms as transforms

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, overlap):
        self.base_transform = base_transform
        self.overlap = overlap
        assert self.overlap in ['O', 'NO'], 'Set overlap either to "O" or "NO"'

    def __call__(self, x):
        if self.overlap == 'O':
            crop = transforms.RandomCrop(size = (600,400),
                padding=None,
                pad_if_needed=True,
                fill = (255, 255, 255),
                padding_mode='constant')
            x = crop(x)
    	q = self.base_transform(x)
    	k = self.base_transform(x)
    	return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
