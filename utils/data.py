import os, torch, cv2, random, math, glob, numbers
from copy import deepcopy
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

def compute_min_page_size(root):

	try:
		sizes = []
		with open(root + os.sep + 'crop_size.pkl', 'rb') as f:
			sizes = pkl.load(f)
		return sizes[0], sizes[1]
	
	except:
		min_width = 10**6
		min_height = 10**6
		
		classes = [class_ for class_ in os.listdir(root) if os.path.isdir(os.path.join(root, class_))]
		
		for class_ in classes:
			sample_name = os.listdir(os.path.join(root, class_))[random.randint(0, len(os.listdir(os.path.join(root, class_))) - 1)]
			sample = cv2.imread(os.path.join(os.path.join(root, class_), sample_name))
			h, w, c = sample.shape
			
			min_width = min(min_width, w)
			min_height = min(min_height, h)
			
		sizes = [min_height, min_width]
		
		with open(root + os.sep + 'crop_size.pkl', 'wb') as f:
			pkl.dump(sizes, f)
			
		return min_height, min_width

def n_random_crops(img, x, y, h, w):

	crops = []
	for i in range(len(x)):
		new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
		crops.append(new_crop)
	return tuple(crops)

class Invert(object):

    def __call__(self, x):
        x = F.invert(x)
        return x

    def __str__(self):
        str_transforms = f"Invert RGB channels"
        return str_transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        randn_tensor = torch.randn(1, tensor.size()[1], tensor.size()[2])
        randn_tensor = randn_tensor.repeat(3,1,1)
        return tensor + randn_tensor * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# https://gist.github.com/amirhfarzaneh/66251288d07c67f6cfd23efc3c1143ad
class NRandomCrop(object):

	def __init__(self, size, n=1, padding=0, pad_if_needed=False):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding
		self.pad_if_needed = pad_if_needed
		self.n = n

	@staticmethod
	def get_params(img, output_size, n):
		w, h = img.size
		th, tw = output_size
		if w == tw and h == th:
			return 0, 0, h, w

		i_list = [random.randint(0, h - th) for i in range(n)]
		j_list = [random.randint(0, w - tw) for i in range(n)]
		return i_list, j_list, th, tw

	def __call__(self, img):
		
		if self.padding > 0:
			img = F.pad(img, self.padding)

		if self.pad_if_needed and img.size[0] < self.size[1]:
			img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))

		if self.pad_if_needed and img.size[1] < self.size[0]:
			img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

		i, j, h, w = self.get_params(img, self.size, self.n)

		return n_random_crops(img, i, j, h, w)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class Standard_DataLoader():
	
	def __init__(self, directory, transforms_params, batch_size, weighted_sampling = True, phase = 'train', mean = [0, 0, 0], std = [1, 1, 1],
		shuffle = True, amount = 0.3, selection = False):

		self.directory = directory
		self.transforms_params = transforms_params
		self.batch_size = batch_size
		self.weighted_sampling = weighted_sampling
		self.phase = phase
		self.shuffle = shuffle
		self.amount = amount
		self.selection = selection
		if (mean == [0, 0, 0] and std == [1, 1, 1]):
			self.mean, self.std = compute_mean_and_std(self.directory, 3, self.amount, self.selection)
		else:
			self.mean = mean
			self.std = std

		self.min_height, self.min_width = compute_min_page_size(self.directory)

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
		rc_p = 1.0, 
		cjitter = {'brightness': [0.4, 1.3], 'contrast': 0.6, 'saturation': 0.6,'hue': 0.4}, 
		cjitter_p = 1, 
		randaffine = {'degrees': [-10,10], 'translate': [0.2, 0.2], 'scale': [1.3, 1.4], 'shear': 1}, 
		randpersp = {'distortion_scale': 0.1, 'p': 0.2}, 
		gray_p = 0.2, 
		gaussian_blur = {'kernel_size': 3, 'sigma': [0.1, 0.5]},
		rand_eras = {'p': 0.5, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3], 'value': 0}, 
		invert_p = 0.05,
		gaussian_noise = {'mean': 0., 'std': 0.004},
		gn_p = 0.0,
		n_test_crops = 10):

		assert ((1-rc_p)**2 + rc_p**2) == 1, "rc_p should be equal to 0.0 or 1.0"

		randaffine['interpolation'] = randpersp['interpolation'] = T.InterpolationMode.BILINEAR
		randaffine['fill'] = randpersp['fill'] = [255, 255, 255]

		train_transforms = T.Compose([
			T.RandomApply([T.RandomCrop(size = img_crop_size, padding = None, pad_if_needed = True, fill = (255, 255, 255), padding_mode = 'constant')], p = rc_p),
			T.RandomApply([T.CenterCrop(size = (self.min_height, self.min_width))], p = 1 - rc_p),
			T.RandomApply([T.ColorJitter(**cjitter)], p=cjitter_p),
			T.RandomAffine(**randaffine),
			T.RandomPerspective(**randpersp),
			T.GaussianBlur(**gaussian_blur), 
			T.RandomGrayscale(gray_p),
			T.ToTensor(),
			T.RandomErasing(**rand_eras),
			T.RandomApply([Invert()], p=invert_p),
			T.Normalize(self.mean, self.std),
			T.RandomApply([AddGaussianNoise(**gaussian_noise)], p=gn_p)
			])	

		val_transforms = T.Compose([
			T.RandomApply([T.RandomCrop(size = img_crop_size, padding = None, pad_if_needed = True, fill = (255, 255, 255), padding_mode = 'constant')], p = rc_p),
			T.RandomApply([T.CenterCrop(size = (self.min_height, self.min_width))], p = 1 - rc_p),
			T.ToTensor(),
			T.Normalize(self.mean, self.std)
			])

		if rc_p == 1:
			test_transforms = T.Compose([
				NRandomCrop(size = img_crop_size, n = n_test_crops, pad_if_needed = True),
				T.Lambda(lambda crops: torch.stack([T.Normalize(self.mean, self.std)(T.ToTensor()(crop)) for crop in crops]))
				])
		else:
			test_transforms = T.Compose([
				T.CenterCrop(size = (self.min_height, self.min_width)),
				T.ToTensor(),
				T.Normalize(self.mean, self.std)
				])
		
		if self.phase == 'train':
			return train_transforms			
		elif self.phase == 'val':
			return val_transforms
		elif self.phase == 'test':
			return test_transforms

	def generate_dataset(self):
		return datasets.ImageFolder(root = self.directory, 
			transform = self.compose_transform(**self.transforms_params))

	def load_data(self):
		dataset = self.generate_dataset()
		if self.phase == 'train':
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
	
	def __init__(self, directory, transforms_params, phase = 'train', mean = [0, 0, 0], std = [1, 1, 1], amount = 0.3, selection = False):
		super(Dataset_Generator_SN, self).__init__()
		
		self.directory = directory
		self.transforms_params = transforms_params
		self.phase = phase
		
		self.classes = [class_ for class_ in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory, class_))]
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
		self.weight_per_class = {}
		for class_ in self.classes:
			self.dict_classes[class_] = [glob.glob(os.path.join(self.directory, class_ + '/*.png'))][0] + [glob.glob(os.path.join(self.directory, class_ + '/*.jpg'))][0]
			self.weight_per_class[class_] = float(self.num_elements)/float(len(self.dict_classes[class_]))

		self.weight = [0]*len(self.img_files)
		for idx, val in enumerate(self.img_files):
			self.weight[idx] = self.weight_per_class[self.img_files[idx].split(os.path.sep)[-2]]

	def compose_transform(self, 
		img_crop_size = 380, 
		cjitter = {'brightness': [0.4, 1.3], 'contrast': 0.6, 'saturation': 0.6,'hue': 0.4}, 
		cjitter_p = 1, 
		randaffine = {'degrees': [-10,10], 'translate': [0.2, 0.2], 'scale': [1.3, 1.4], 'shear': 1}, 
		randpersp = {'distortion_scale': 0.1, 'p': 0.2}, 
		gray_p = 0.2, 
		gaussian_blur= {'kernel_size': 3, 'sigma': [0.1, 0.5]},
		rand_eras = {'p': 0.5, 'scale': [0.02, 0.33], 'ratio': [0.3, 3.3], 'value': 0}, 
		invert_p = 0.05,
		gaussian_noise = {'mean': 0., 'std': 0.004},
		gn_p = 0.0,
		n_test_crops = 10):
		

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
			T.Normalize(self.mean, self.std),
			T.RandomApply([AddGaussianNoise(**gaussian_noise)], p=gn_p)
			])	

		val_transforms = T.Compose([
			T.RandomCrop(size = img_crop_size, padding = None, pad_if_needed = True, fill = (255, 255, 255), padding_mode = 'constant'),
			T.ToTensor(),
			T.Normalize(self.mean, self.std)
			])
		
		if self.phase == 'train':
			return train_transforms
		elif self.phase == 'val':
			return val_transforms

	def get_pn(self, index):
		class_anchor = self.img_files[index].split(os.path.sep)[-2]
		
		list_positives = self.dict_classes[class_anchor]
		
		dict_negatives = deepcopy(self.dict_classes)
		dict_negatives.pop(class_anchor)
		
		list_negatives = sum(dict_negatives.values(), [])

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
	
	def __init__(self, directory, transforms_params, batch_size, weighted_sampling = True, phase = 'train', mean = [0, 0, 0], std = [1, 1, 1], amount = 0.3, selection = False, shuffle = True):
		self.dataset = Dataset_Generator_SN(directory, transforms_params, phase, mean, std, amount, selection)
		self.batch_size = batch_size
		self.weighted_sampling = weighted_sampling
		self.phase = phase
		self.shuffle = shuffle

	def load_data(self):
		if self.phase == 'train':
			if self.weighted_sampling:
				weights = self.dataset.weight
				weights = torch.DoubleTensor(weights)
				sampler = WeightedRandomSampler(weights, len(weights))
				loader = DataLoader(self.dataset, batch_size = self.batch_size, sampler = sampler)
			else:
				loader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = self.shuffle)
		else:
			loader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = self.shuffle)
		return loader