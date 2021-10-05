import torch, yaml, pathlib
from argparse import ArgumentParser
from utils.model import *
from utils.torch_utils import *
from utils.test import *

def get_args():
  parser = ArgumentParser(description = 'Parameters', add_help = True)
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
  parser.add_argument('-ted', '--test_im_directory',
  	type = str,
  	help = 'Test image directory',
  	dest = 'TE_IM_DIR',
  	default = './Test/')	  
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
	model_pars = args.exp_config['model']	
	BATCH_SIZE = args.exp_config['data']['batch_size']	
	TRANSFORMS = args.exp_config['data']['transforms']
	test_params = args.exp_config['test']

	assert test_type in ['MLC', 'TL', 'SN'], 'Set test type either to "MLC", "TL" or "SN"'

	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	try:
		torch.cuda.empty_cache()
		torch.backends.cudnn.benchmark = True
	except:
		pass

	model_train, model_val = load_models(args.PATH, test_ID, test_type, model_pars, DEVICE)

	print_losses(args.PATH, test_ID, test_type)
	if test_type in ['TL','MLC']:
		print_accs(args.PATH, test_ID, test_type)

	dir_ = {'train': args.T_IM_DIR, 'val': args.V_IM_DIR, 'test': args.TE_IM_DIR}
	mod_ = {'train': model_train, 'val': model_val, 'test': model_val}

	if test_type in ['TL','SN']:
		
		for ph in ['train', 'val']:
			f_a = feature_analysis(args.PATH, args.T_IM_DIR, dir_[ph], mod_[ph], BATCH_SIZE, TRANSFORMS, DEVICE, test_ID, test_type, test_params, phase = ph)
			f_a()

	else:
		
		for ph in ['train', 'val', 'test']:
			mlca = mlc_accuracy(args.PATH, args.T_IM_DIR, dir_[ph], mod_[ph], TRANSFORMS, DEVICE, test_ID, test_type, phase = ph)
			mlca()		