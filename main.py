from utils.data import *
from utils.model import *
from utils.train import *
from torch.utils import *
from utils.torch_utils import *
import torch, yaml, pathlib
from argparse import ArgumentParser 
import pytorch_lightning as pl

def get_args():
  parser = ArgumentParser(description = 'Hyperparameters', add_help = True)
  parser.add_argument('-dir', '--root_directory',
  	type = str,
  	help = 'Root directory',
  	dest = 'PATH',
  	default = './')
  parser.add_argument('-td', '--train_im_directory',
  	type = str,
  	help = 'Training image directory',
  	dest = 'T_IM_DIR',
  	default = './Training/')
  parser.add_argument('-vd', '--v_im_directory',
  	type = str,
  	help = 'Validation image directory',
  	dest = 'V_IM_DIR',
  	default = './Validation/')
  parser.add_argument('-c', '--config',
  	type=str,
  	required=True,
  	default="",
  	help='Config file with parameters of the experiment.')

  args = parser.parse_args()

  full_config_path = pathlib.Path(args.PATH) / 'config' / (args.config + '.yaml')
  print(f'Loading experiment {full_config_path}')
  with open(full_config_path, 'r') as f:
    args.exp_config = yaml.load(f, Loader=yaml.SafeLoader)
  return args

if __name__ == '__main__':
	
	args = get_args()
	test_ID = args.exp_config['general']['test_id']
	test_type = args.exp_config['general']['test_type']
	NUM_CLASSES = args.exp_config['model']['num_classes']
	EMB_WIDTH = args.exp_config['model']['emb_width']
	ARCH = args.exp_config['model']['feature_extractor_arch']
	PRETRAINING = args.exp_config['model']['pretraining']
	MODE = args.exp_config['model']['mode']
	CP_PATH = args.exp_config['model']['cp_path']
	ALPHA = args.exp_config['model']['alpha']
	ALPHA_VALUE = args.exp_config['model']['alpha_value']
	EMB_TYPE = args.exp_config['model']['emb_type']
	SAMPLES = args.exp_config['model']['samples']
	EPOCHS = args.exp_config['optim']['num_epochs']
	OPTIM = args.exp_config['optim']
	img_size = args.exp_config['data']['transforms']['img_crop_size']
	TRANSFORMS = args.exp_config['data']['transforms']
	BATCH_SIZE = args.exp_config['data']['batch_size']
	WS = args.exp_config['data']['weighted_sampling']
	SEED = args.exp_config['general']['seed']

	assert test_type in ['MCC', 'TL', 'SN'], 'Set test type either to "MCC", "TL" or "SN"'

	history_path, model_path = make_dirs(args.PATH, test_ID)

	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.empty_cache()
	torch.backends.cudnn.benchmark = True	

	mean_, std_ = load_rgb_mean_std(args.T_IM_DIR)

	if test_type in ['MCC', 'TL']:
		t_ds = Standard_DataLoader(args.T_IM_DIR, TRANSFORMS, BATCH_SIZE, WS, 'train', mean_, std_, True)
		v_ds = Standard_DataLoader(args.V_IM_DIR, TRANSFORMS, BATCH_SIZE, False, 'val', mean_, std_, True)
		tds, t_dl = t_ds.load_data()
		vds, v_dl = v_ds.load_data()
		pl.seed_everything(SEED)
		if test_type == 'MCC':
			assert NUM_CLASSES == len(t_dl.dataset.classes), f'Wrong number of classes: ({len(t_dl.dataset.classes)} classes in dataset).\n'
			model = Model_MCC(pretrained = PRETRAINING, mode = MODE, emb_width = EMB_WIDTH, arch = ARCH, cp_path = CP_PATH, num_classes = NUM_CLASSES)
			model = model.to(DEVICE)
			save_model(model_path, test_ID, test_type, model)
			trainer = Trainer_MCC(model, t_dl, v_dl, DEVICE, OPTIM, model_path, history_path, test_ID, EPOCHS)
			trainer()
		else:
			model = Model_TL(pretrained = PRETRAINING, mode = MODE, emb_width = EMB_WIDTH, arch = ARCH, cp_path = CP_PATH, alpha = ALPHA, alpha_value = ALPHA_VALUE, emb_type = EMB_TYPE, samples = SAMPLES)
			model = model.to(DEVICE)
			save_model(model_path, test_ID, test_type, model)
			trainer = Trainer_TL(model, tds, vds, t_dl, v_dl, DEVICE, OPTIM, model_path, history_path, test_ID, EPOCHS)
			trainer()
	else:
		t_ds = Data_Loader_SN(args.T_IM_DIR, TRANSFORMS, BATCH_SIZE, WS, 'train', mean_, std_)
		v_ds = Data_Loader_SN(args.V_IM_DIR, TRANSFORMS, BATCH_SIZE, False, 'val', mean_, std_)
		t_dl = t_ds.load_data()
		v_dl = v_ds.load_data()
		pl.seed_everything(SEED)
		model = Model_SN(pretrained = PRETRAINING, mode = MODE, emb_width = EMB_WIDTH, arch = ARCH, cp_path = CP_PATH)
		model = model.to(DEVICE)
		save_model(model_path, test_ID, test_type, model)
		trainer = Trainer_SN(model, t_dl, v_dl, DEVICE, OPTIM, model_path, history_path, test_ID, EPOCHS)
		trainer()