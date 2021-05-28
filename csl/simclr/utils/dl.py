# from https://github.com/lightly-ai/lightly/blob/develop/lightly/data/collate.py

import torch, glob, cv2, math, random, os
import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from typing import List

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

import lightly.data as data

imagenet_normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

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

def load_rgb_mean_std(root, amount, selection):
    try:
        stats = []
        
        with open(root + os.sep + 'rgb_stats.pkl', 'rb') as f:
            stats = pkl.load(f)
        
        mean_ = stats[0]
        std_ = stats[1]
    except:
        mean_, std_ = compute_mean_and_std(root = root, amount = amount, selection = selection)

    return mean_, std_

def test_transforms(img_crop_size, norm_dict):
	
	TT = T.Compose([
		T.RandomCrop(size = img_crop_size, padding = None, 
			pad_if_needed = True, fill = (255, 255, 255), 
			padding_mode = 'constant'),
		T.ToTensor(),
		T.Normalize(mean = norm_dict['mean'],
			std = norm_dict['std'])])
	return TT

class Invert(object):

    def __call__(self, x):
    	x = F.invert(x)
    	return x

    def __str__(self):
        str_transforms = f"Invert RGB channels"
        return str_transforms

class BaseCollateFunction_MS(nn.Module):
	def __init__(self, transform: torchvision.transforms.Compose):

		super(BaseCollateFunction_MS, self).__init__()
		self.transform = transform
		self.crop = T.RandomCrop(size = (600,400), 
			padding=None, 
			pad_if_needed=True, 
			fill = (255, 255, 255), 
			padding_mode='constant')

	def forward(self, batch: List[tuple]):

		batch_size = len(batch)

		batch = [(self.crop(batch[i % batch_size][0]), batch[i % batch_size][1], batch[i % batch_size][2]) for i in range(batch_size)]
		
		transforms = [self.transform(batch[i % batch_size][0]).unsqueeze_(0) for i in range(2 * batch_size)]

		labels = torch.LongTensor([item[1] for item in batch])

		fnames = [item[2] for item in batch]


		transforms = (
			torch.cat(transforms[:batch_size], 0),
			torch.cat(transforms[batch_size:], 0)
			)

		return transforms, labels, fnames

class ImageCollateFunction_MS(BaseCollateFunction_MS):
	def __init__(self,
		img_crop_size: int = 380, 		
		cjitter: dict = {'brightness': [0.4, 1.3], 'contrast': 0.6, 'saturation': 0.6,'hue': 0.4}, 
		cjitter_p: float = 1., 
		randaffine: dict = {'degrees': [-10,10], 'translate': [0.2, 0.2], 'scale': [1.3, 1.4], 'shear': 1}, 
		randpersp: dict = {'distortion_scale': 0.1, 'p': 0.2}, 
		gray_p: float = 0.2, 
		gaussian_blur: dict = {'kernel_size': 3, 'sigma': [0.1, 0.5]},
		rand_eras: dict = {'p': 0.5, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3], 'value': 0}, 
		invert_p: float = 0.05,
		normalize: dict = imagenet_normalize):

		if isinstance(img_crop_size, tuple):
			img_crop_size_ = max(img_crop_size)
		else:
			img_crop_size_ = img_crop_size

		transform = [
			T.RandomCrop(size = img_crop_size_, padding = None, pad_if_needed = True, fill = (255, 255, 255), padding_mode = 'constant'),
			T.RandomApply([T.ColorJitter(**cjitter)], p=cjitter_p),
			T.RandomAffine(**randaffine),
			T.RandomPerspective(**randpersp),
			T.GaussianBlur(**gaussian_blur),
			T.RandomGrayscale(gray_p),
			T.ToTensor(),
			T.RandomErasing(**rand_eras),
			T.RandomApply([Invert()], p=invert_p),
			]


		if normalize:
			transform += [
				T.Normalize(
					mean=normalize['mean'],
					std=normalize['std'])
				]
           
		transform = T.Compose(transform)

		super(ImageCollateFunction_MS, self).__init__(transform)

class MSCollateFunction(ImageCollateFunction_MS):
	def __init__(self,
		img_crop_size: int = 380, 		
		cjitter: dict = {'brightness': [0.4, 1.3], 'contrast': 0.6, 'saturation': 0.6,'hue': 0.4}, 
		cjitter_p: float = 1., 
		randaffine: dict = {'degrees': [-10,10], 'translate': [0.2, 0.2], 'scale': [1.3, 1.4], 'shear': 1}, 
		randpersp: dict = {'distortion_scale': 0.1, 'p': 0.2}, 
		gray_p: float = 0.2, 
		gaussian_blur: dict = {'kernel_size': 3, 'sigma': [0.1, 0.5]},
		rand_eras: dict = {'p': 0.5, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3], 'value': 0}, 
		invert_p: float = 0.05,
		normalize: dict = imagenet_normalize):

		super(MSCollateFunction, self).__init__(
			img_crop_size = img_crop_size,
			cjitter = cjitter,
			cjitter_p = cjitter_p,
			randaffine = randaffine,
			randpersp = randpersp,
			gray_p = gray_p,
			gaussian_blur = gaussian_blur,
			rand_eras = rand_eras,
			invert_p = invert_p,
			normalize = normalize
			)

class Lightly_DataLoader(Dataset):
	def __init__(self, dir_, train_dir, transforms_params, batch_size, num_workers, train = True, 
		shuffle = True, amount = 0.3, selection = False):
		
		self.dir_ = dir_
		self.train_dir = train_dir 		
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.train = train
		self.shuffle = shuffle
		mean_, std_ = load_rgb_mean_std(self.train_dir, amount, selection)
		self.norm_dict = {'mean': mean_, 'std': std_}
		self.transforms_params = transforms_params
		self.transforms_params['normalize'] = self.norm_dict

	def generate_dl(self):
		
		if self.train:
			dataset = data.LightlyDataset(input_dir = self.dir_)
			collate_fn = MSCollateFunction(**self.transforms_params)
			dataloader = DataLoader(dataset,
				batch_size = self.batch_size,
				shuffle = self.shuffle,
				collate_fn = collate_fn,
				num_workers = self.num_workers)
		
		else:
			TT = test_transforms(self.transforms_params['img_crop_size'],
				self.norm_dict)
			dataset = data.LightlyDataset(input_dir = self.dir_,
				transform = TT)
			dataloader = DataLoader(dataset,
				batch_size = self.batch_size,
				shuffle = False,
				drop_last = False,
				num_workers = self.num_workers)
		
		return dataloader