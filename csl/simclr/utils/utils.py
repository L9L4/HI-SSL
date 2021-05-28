import lightly.models as models
import torchvision
import torch.nn as nn

from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import matplotlib.pyplot as plt

def SimCLR_builder(arch, num_ftrs, out_dim):
	base_encoder = torchvision.models.__dict__[arch](num_classes = num_ftrs)
	model = models.SimCLR(base_encoder, num_ftrs=num_ftrs, out_dim = out_dim)
	return model

def cp_call(monitor, dirpath, save_top_k, mode):
	return ModelCheckpoint(
		monitor = monitor,
		dirpath = dirpath,
		filename = f'checkpoints-{{epoch:03d}}-{{{monitor}:.3f}}',
		save_top_k = save_top_k, 
		mode = mode
		)

def create_logger(save_dir, name):
	return TensorBoardLogger(save_dir = save_dir,
		name = name)

def plot_loss(log_path, test_ID):
	tfevents_path = glob.glob(log_path + '/*/*.tfevents.*')[0]

	event_acc = EventAccumulator(tfevents_path)
	event_acc = event_acc.Reload()

	losses = [event_acc.Scalars('loss')[i][2] for i in range(len(event_acc.Scalars('loss')))]

	plt.plot(losses)
	plt.title('Model loss')
	plt.ylabel('Loss [-]')
	plt.xlabel('Epoch [-]')
	plt.legend(['Training'], loc='best')
	plt.savefig(log_path + f'/{test_ID}_losses.png')
	plt.close()	