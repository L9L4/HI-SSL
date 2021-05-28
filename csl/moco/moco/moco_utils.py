import os, glob, random, math, cv2
import numpy as np
import pickle as pkl
import torchvision.transforms.functional as F
from tqdm import tqdm

def compute_mean_and_std(root, CHANNEL_NUM = 3, amount = 0.1, selection = False):

    types = ('*.png', '*.jpg')
    training_images = []
    for files in types:
        training_images.extend(glob.glob(root + '/*/' + files)) 

    if selection:
        training_images = random.sample(training_images, math.ceil(len(training_images)*amount))

    pixel_num = 0
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for i in tqdm(training_images):
        im = cv2.imread(i)
        im = im/255.0

        pixel_num += (im.size/CHANNEL_NUM)
        channel_sum += np.sum(im, axis = (0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum/pixel_num
    bgr_std = np.sqrt(channel_sum_squared/pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    stats = [rgb_mean, rgb_std]
    with open(root + os.sep + 'rgb_stats.pkl', 'wb') as f:
        pkl.dump(stats, f) 

    return rgb_mean, rgb_std

def load_rgb_mean_std(root):
    try:
        stats = []
        
        with open(root + os.sep + 'rgb_stats.pkl', 'rb') as f:
            stats = pkl.load(f)
        
        mean_ = stats[0]
        std_ = stats[1]
    except:
        mean_, std_ = compute_mean_and_std(root = root, amount = 0.3, selection = False)

    return mean_, std_

class Invert(object):

    def __call__(self, x):
    	x = F.invert(x)
    	return x

    def __str__(self):
        str_transforms = f"Invert RGB channels"
        return str_transforms