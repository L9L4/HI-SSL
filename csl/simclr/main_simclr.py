# Global Batch Normalization --> 1 GPU
# LARS optimizer: replaced with SGD (the batch size is not so big here)
# Augmentations: fixed

import torch, yaml, pathlib, os

import pytorch_lightning as pl
import lightly.loss as loss
import lightly.embedding as embedding
from utils.dl import Lightly_DataLoader
from utils.utils import *
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from argparse import ArgumentParser 

def get_args():
  parser = ArgumentParser(description = 'Hyperparameters', add_help = True)
  parser.add_argument('-dir', '--root_directory',
  	type = str,
  	help = 'Root directory.',
  	dest = 'PATH',
  	default = './')
  parser.add_argument('-td', '--train_im_directory',
  	type = str,
  	help = 'Training image directory.',
  	dest = 'T_IM_DIR',
  	default = './Sample_Training/')
  parser.add_argument('-vd', '--v_im_directory',
  	type = str,
  	help = 'Validation image directory.',
  	dest = 'V_IM_DIR',
  	default = './Sample_Validation/')  
  parser.add_argument('-c', '--config',
  	type=str,
  	required=True,
  	default="",
  	help='Config file with parameters of the experiment.')
  parser.add_argument('-g', '--gpus',
  	type=int,
  	required=True,
  	default=1,
  	dest = 'GPUS',
  	help='Number of gpus.')
  args = parser.parse_args()

  full_config_path = pathlib.Path(args.PATH) / 'config' / (args.config + '.yaml')
  print(f'Loading experiment {full_config_path}')
  with open(full_config_path, 'r') as f:
  	args.exp_config = yaml.load(f, Loader=yaml.SafeLoader)
  return args

if __name__ == '__main__':
	
	args = get_args()
	PATH = args.PATH
	TEST_ID = args.exp_config['setup']['test_id']	
	SEED = args.exp_config['setup']['seed']
	TRANSFORMS = args.exp_config['transforms']
	BATCH_SIZE = args.exp_config['hyperparameters']['batch_size']
	NW = args.exp_config['setup']['num_workers']
	MODEL_PARAMS = args.exp_config['model']	
	TEMP = args.exp_config['hyperparameters']['temperature']
	OPTIM_PARAMS = args.exp_config['hyperparameters']['optimizer']
	SCHED_PARAMS = args.exp_config['hyperparameters']['scheduler']
	EPOCHS = args.exp_config['setup']['num_epochs']
	CP_PATH = os.path.join(PATH,TEST_ID)
	LOG_PATH = os.path.join(CP_PATH,TEST_ID)

	if not os.path.exists(CP_PATH):
		os.makedirs(CP_PATH)

	pl.seed_everything(SEED)

	TD = Lightly_DataLoader(args.T_IM_DIR, args.T_IM_DIR, TRANSFORMS, BATCH_SIZE, NW)
	train_dataloader = TD.generate_dl()
	VD = Lightly_DataLoader(args.V_IM_DIR, args.T_IM_DIR, TRANSFORMS, BATCH_SIZE, NW, False)
	val_dataloader = VD.generate_dl()

	model = SimCLR_builder(**MODEL_PARAMS)

	criterion = loss.NTXentLoss(temperature=TEMP)

	optimizer = torch.optim.SGD(model.parameters(), **OPTIM_PARAMS)

	scheduler = LinearWarmupCosineAnnealingLR(optimizer, **SCHED_PARAMS)

	encoder = embedding.SelfSupervisedEmbedding(model,
		criterion,
		optimizer,
		train_dataloader,
		scheduler
		)

	gpus = args.GPUS if torch.cuda.is_available() else 0

	checkpoint_callback = cp_call('loss',
		CP_PATH,
		1,
		'min')

	logger = create_logger(CP_PATH, 
		TEST_ID)

	encoder.train_embedding(gpus=gpus,
		distributed_backend='ddp',
		progress_bar_refresh_rate=1,
		max_epochs=EPOCHS,
		default_root_dir = CP_PATH,
		checkpoint_callback = checkpoint_callback,
		logger = logger,
		log_every_n_steps = len(train_dataloader),
		resume_from_checkpoint=None
		)

	plot_loss(LOG_PATH, TEST_ID)