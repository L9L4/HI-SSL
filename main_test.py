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
  	default = './Sample_Training/')
  parser.add_argument('-vd', '--v_im_directory',
  	type = str,
  	help = 'Validation image directory',
  	dest = 'V_IM_DIR',
  	default = './Sample_Validation/')
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

	assert test_type in ['TL', 'SN'], 'Set test type either to "TL" or "SN"'

	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	try:
		torch.cuda.empty_cache()
		torch.backends.cudnn.benchmark = True
	except:
		pass

	model_train, model_val = load_models(args.PATH, test_ID, test_type, model_pars, DEVICE)

	print_losses(args.PATH, test_ID, test_type)
	print_accs(args.PATH, test_ID, test_type)

	f_a_train = feature_analysis(args.PATH, args.T_IM_DIR, args.T_IM_DIR, model_train, BATCH_SIZE, TRANSFORMS, DEVICE, test_ID, test_type, test_params, phase = 'train')
	f_a_train()

	f_a_val = feature_analysis(args.PATH, args.T_IM_DIR, args.V_IM_DIR, model_val, BATCH_SIZE, TRANSFORMS, DEVICE, test_ID, test_type, test_params, phase = 'val')
	f_a_val()		