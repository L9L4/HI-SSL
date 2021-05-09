import os, torch, cv2, random, math, glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as T
import torchvision.transforms.functional as F
import pickle as pkl

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

class Invert(object):

    def __call__(self, x):
    	x = F.invert(x)
    	return x

    def __str__(self):
        str_transforms = f"Invert RGB channels"
        return str_transforms

class Standard_DataLoader():
	
	def __init__(self, directory, transforms_params, batch_size, weighted_sampling = True, train = True, mean = [0, 0, 0], std = [1, 1, 1],
		shuffle = True, amount = 0.3, selection = False):

		self.directory = directory
		self.transforms_params = transforms_params
		self.batch_size = batch_size
		self.weighted_sampling = weighted_sampling
		self.train = train
		self.shuffle = shuffle
		self.amount = amount
		self.selection = selection
		if (mean == [0, 0, 0] and std == [1, 1, 1]):
			self.mean, self.std = compute_mean_and_std(self.directory, 3, self.amount, self.selection)
		else:
			self.mean = mean
			self.std = std

	def make_weights_for_balanced_classes(self, images, nclasses): # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
		count = [0] * nclasses                                                      
		for item in images:
			count[item[1]] += 1
		weight_per_class = [0.] * nclasses
		N = float(sum(count))
		for i in range(nclasses):
			weight_per_class[i] = N/float(count[i])
		weight = [0] * len(images)
		for idx, val in enumerate(images):
			weight[idx] = weight_per_class[val[1]]
		return weight    

	def compose_transform(self, 
		img_crop_size = 380, 
		cjitter = {'brightness': [0.4, 1.3], 'contrast': 0.6, 'saturation': 0.6,'hue': 0.4}, 
		cjitter_p = 1, 
		randaffine = {'degrees': [-10,10], 'translate': [0.2, 0.2], 'scale': [1.3, 1.4], 'shear': 1}, 
		randpersp = {'distortion_scale': 0.1, 'p': 0.2}, 
		gray_p = 0.2, 
		gaussian_blur= {'kernel_size': 3, 'sigma': [0.1, 0.5]},
		rand_eras = {'p': 0.5, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3], 'value': 0}, 
		invert_p = 0.05):
		

		randaffine['interpolation'] = randpersp['interpolation'] = T.InterpolationMode.BILINEAR
		randaffine['fill'] = randpersp['fill'] = [255, 255, 255]

		train_transforms = T.Compose([
			T.RandomCrop(size = img_crop_size, padding = None, pad_if_needed = True, fill = (255, 255, 255), padding_mode = 'constant'),
			T.RandomApply([T.ColorJitter(**cjitter)], p=cjitter_p),
			T.RandomAffine(**randaffine),
			T.RandomPerspective(**randpersp),
			T.GaussianBlur(**gaussian_blur), 
			T.RandomGrayscale(gray_p),
			T.ToTensor(),
			T.RandomErasing(**rand_eras),
			T.RandomApply([Invert()], p=invert_p),
			T.Normalize(self.mean, self.std)
			])	

		val_transforms = T.Compose([
			T.RandomCrop(size = img_crop_size, padding = None, pad_if_needed = True, fill = (255, 255, 255), padding_mode = 'constant'),
			T.ToTensor(),
			T.Normalize(self.mean, self.std)
			])
		
		if self.train == True:
			return train_transforms			
		else:
			return val_transforms

	def generate_dataset(self):
		return datasets.ImageFolder(root = self.directory, 
			transform = self.compose_transform(**self.transforms_params))

	def load_data(self):
		dataset = self.generate_dataset()
		if self.train:
			if self.weighted_sampling:
				weights = self.make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
				weights = torch.DoubleTensor(weights)
				sampler = WeightedRandomSampler(weights, len(weights))
				loader = DataLoader(dataset, batch_size = self.batch_size, sampler = sampler)
			else:
				loader = DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle)
		else:
			loader = DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle)
		return dataset, loader 

class Dataset_Generator_SN(Dataset):
	
	def __init__(self, directory, transforms_params, train, mean, std, amount, selection):
		super(Dataset_Generator_SN, self).__init__()
		
		self.directory = directory
		self.transforms_params = transforms_params
		self.train = train
		
		self.classes = os.listdir(self.directory)
		self.num_classes = len(self.classes)

		self.img_files = [glob.glob(os.path.join(self.directory,'*/*.png'))][0] + [glob.glob(os.path.join(self.directory,'*/*.jpg'))][0]
		self.num_elements = len(self.img_files)

		self.amount = amount
		self.selection = selection		
		if (mean == [0, 0, 0] and std == [1, 1, 1]):
			self.mean, self.std = compute_mean_and_std(self.directory, 3, self.amount, self.selection)
		else:
			self.mean = mean
			self.std = std
		
		self.dict_classes = {}
		for class_ in self.classes:
			self.dict_classes[class_] = [glob.glob(os.path.join(self.directory, class_ + '/*.png'))][0] + [glob.glob(os.path.join(self.directory, class_ + '/*.jpg'))][0]

	def compose_transform(self, 
		img_crop_size = 380, 
		cjitter = {'brightness': [0.4, 1.3], 'contrast': 0.6, 'saturation': 0.6,'hue': 0.4}, 
		cjitter_p = 1, 
		randaffine = {'degrees': [-10,10], 'translate': [0.2, 0.2], 'scale': [1.3, 1.4], 'shear': 1}, 
		randpersp = {'distortion_scale': 0.1, 'p': 0.2}, 
		gray_p = 0.2, 
		gaussian_blur= {'kernel_size': 3, 'sigma': [0.1, 0.5]},
		rand_eras = {'p': 0.5, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3], 'value': 0}, 
		invert_p = 0.05):
		

		randaffine['interpolation'] = randpersp['interpolation'] = T.InterpolationMode.BILINEAR
		randaffine['fill'] = randpersp['fill'] = [255, 255, 255]

		train_transforms = T.Compose([
			T.RandomCrop(size = img_crop_size, padding = None, pad_if_needed = True, fill = (255, 255, 255), padding_mode = 'constant'),
			T.RandomApply([T.ColorJitter(**cjitter)], p=cjitter_p),
			T.RandomAffine(**randaffine),
			T.RandomPerspective(**randpersp),
			T.GaussianBlur(**gaussian_blur), 
			T.RandomGrayscale(gray_p),
			T.ToTensor(),
			T.RandomErasing(**rand_eras),
			T.RandomApply([Invert()], p=invert_p),
			T.Normalize(self.mean, self.std)
			])	

		val_transforms = T.Compose([
			T.RandomCrop(size = img_crop_size, padding = None, pad_if_needed = True, fill = (255, 255, 255), padding_mode = 'constant'),
			T.ToTensor(),
			T.Normalize(self.mean, self.std)
			])
		
		if self.train == True:
			return train_transforms			
		else:
			return val_transforms

	def get_pn(self, index):
		class_anchor = self.img_files[index].split(os.path.sep)[-2]
		
		list_positives = self.dict_classes[class_anchor]
		list_negatives = []
		for class_ in list(self.dict_classes.keys()):
			if class_ != class_anchor:
				list_negatives += self.dict_classes[class_]
			else:
				continue

		positive = random.choice(list_positives)
		negative = random.choice(list_negatives)

		return positive, negative

	def __getitem__(self, index):
		positive, negative = self.get_pn(index)

		anchor = Image.open(self.img_files[index])				
		positive = Image.open(positive)
		negative = Image.open(negative)

		data_transform = self.compose_transform(**self.transforms_params)

		return data_transform(anchor), data_transform(positive), data_transform(negative)
    
	def __len__(self):
		return self.num_elements

class Data_Loader_SN():
	
	def __init__(self, directory, transforms_params, batch_size, train = True, mean = [0, 0, 0], std = [1, 1, 1], amount = 0.3, selection = False, shuffle = True):
		self.dataset = Dataset_Generator_SN(directory, transforms_params, train, mean, std, amount, selection)
		self.batch_size = batch_size
		self.shuffle = shuffle

	def load_data(self):		
		return DataLoader(self.dataset, batch_size = self.batch_size, shuffle = self.shuffle)