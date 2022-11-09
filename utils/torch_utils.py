import torch, os
import torch.nn as nn
from copy import deepcopy
import numpy as np
from utils.data import compute_mean_and_std
from utils.model import *
import pickle as pkl

def make_dirs(path, id_):
    model_path = path + os.sep + 'model/'
    history_path = path + os.sep + 'data/'
    new_directories = [model_path, model_path + '/checkpoints_' + id_, history_path]

    for new_directory in new_directories:
        try:
            os.mkdir(new_directory)
        except:
            pass

    return history_path, model_path    

def save_model(model_path, id_, type_, model):
    
    with open(model_path + os.sep + 'Test_' + id_ + '_' + type_ + '.txt','w+') as f:
        for child in model.children():
            f.write(f'{child}\n')

def load_models(root, test_ID, test_type, model_pars, device):
    
    model_path = root + os.sep + 'model' + os.sep + 'checkpoints_' + test_ID
    train_cp = os.path.join(model_path, 'Test_' + test_ID + '_' + test_type + '_train_best_model.pth')
    val_cp = os.path.join(model_path, 'Test_' + test_ID + '_' + test_type + '_val_best_model.pth')

    if test_type == 'TL':
        model = Model_TL(pretrained = model_pars['pretraining'], emb_width = model_pars['emb_width'], arch = model_pars['feature_extractor_arch'], cp_path = model_pars['cp_path'],  alpha = model_pars['alpha'], alpha_value = model_pars['alpha_value'], emb_type = model_pars['emb_type'], samples = model_pars['samples'])
    elif test_type == 'SN':
        model = Model_SN(pretrained = model_pars['pretraining'], emb_width = model_pars['emb_width'], arch = model_pars['feature_extractor_arch'], cp_path = model_pars['cp_path'])
    elif test_type == 'MCC':
        model = Model_MCC(pretrained = model_pars['pretraining'], emb_width = model_pars['emb_width'], arch = model_pars['feature_extractor_arch'], cp_path = model_pars['cp_path'], num_classes = model_pars['num_classes'])

    model_train = model.to(device)
    model_val = deepcopy(model_train)

    if torch.cuda.is_available():
        model_train.load_state_dict(torch.load(train_cp)['model_state_dict'])
        model_val.load_state_dict(torch.load(val_cp)['model_state_dict'])
    else:
        model_train.load_state_dict(torch.load(train_cp, map_location=torch.device('cpu'))['model_state_dict'])
        model_val.load_state_dict(torch.load(val_cp, map_location=torch.device('cpu'))['model_state_dict'])

    if test_type in ['TL','MCC']:
        return model_train, model_val
    elif test_type == 'SN':
        return nn.Sequential(model_train.enc, model_train.fc_layers), nn.Sequential(model_val.enc, model_val.fc_layers)

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

def average_precision(label_query, labels_list, num_elements):
    ap = 0
    tp = 0
    for j in range(len(labels_list)):
        if labels_list[j] == label_query:
            tp += 1
            ap += tp/(j+1)
        else:
            continue
    return ap/num_elements
