import torch
t = torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def load_model(pretrained, mode, cp_path, arch):
	
	if pretrained == 'obow':
		cp = torch.load(cp_path)['network']
		n_classes = len(cp[list(cp.keys())[-1]])
		encoder = models.__dict__[arch](num_classes = n_classes)
		encoder.load_state_dict(cp)
		if mode == 'freezed':
			for param in encoder.parameters():
				param.requires_grad = False

	elif pretrained == 'moco':
		cp = torch.load(cp_path)['state_dict']
		arch = torch.load(cp_path)['arch']
		moco_filtered = {}
		for key in list(cp.keys()):
			if 'module.encoder_q' in key:
				moco_filtered[key[17:]] = cp[key]

		n_classes = len(moco_filtered[list(moco_filtered.keys())[-1]])
		encoder = models.__dict__[arch](num_classes = n_classes)

		if 'fc.0.weight' in list(moco_filtered.keys()):
			dim_mlp = encoder.fc.weight.shape[1]
			encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), encoder.fc)

		encoder.load_state_dict(moco_filtered)

		if mode == 'freezed':
			for param in encoder.parameters():
				param.requires_grad = False

	elif pretrained == 'simclr':
		cp = torch.load(cp_path)['state_dict']
		n_classes = len(cp['model.backbone.fc.weight'])
		encoder = models.__dict__[arch](num_classes = n_classes)
		simclr_filtered = {}
		for key in list(cp.keys()):
			if 'model.backbone' in key:
				simclr_filtered[key[15:]] = cp[key]

		encoder.load_state_dict(simclr_filtered)

		if mode == 'freezed':
			for param in encoder.parameters():
				param.requires_grad = False

	elif pretrained == None:
		encoder = models.__dict__[arch]()
		n_classes = list(encoder.children())[-1].out_features
		
	return encoder, n_classes

def add_fc_layer(type_fc_layer, in_f, out_f):
    if type_fc_layer == 'last':
        fc_block = nn.Sequential(*[nn.Linear(in_features = in_f, out_features = out_f)])
    elif type_fc_layer in ['first', 'hidden']:
        fc_block = nn.Sequential(*[nn.Linear(in_features = in_f, out_features = out_f),
                    nn.BatchNorm1d(out_f), nn.ReLU(), nn.Dropout(p = 0.3)])
    else:
        raise Exception('Wrong fully connected layer type')
    
    return fc_block        

class Model_TL(nn.Module):
    def __init__(self, pretrained = None, mode = None, arch = 'resnet50', cp_path = './', emb_width = 512, num_fc_layers = 1):
        super().__init__()
        self.pretrained = pretrained
        self.mode = mode
        self.arch = arch
        self.cp_path = cp_path
        self.enc, self.num_words = load_model(self.pretrained, self.mode, self.cp_path, self.arch)
        self.emb_width = emb_width
        self.num_fc_layers = num_fc_layers
        self.fc_layers = nn.Sequential()
        if self.num_fc_layers == 1:
            new_layer = add_fc_layer('last', self.num_words,  self.emb_width)
            self.fc_layers.add_module('fc0', new_layer)
        else:
            for layer in range(self.num_fc_layers):
                if layer == 0:
                    new_layer = add_fc_layer('first', self.num_words, self.emb_width*(2**(self.num_fc_layers - layer - 1)))
                    self.fc_layers.add_module('fc' + str(layer), new_layer)
                elif layer == (self.num_fc_layers - 1):
                    new_layer = add_fc_layer('last', self.emb_width*(2**(self.num_fc_layers - layer)), self.emb_width)
                    self.fc_layers.add_module('fc' + str(layer), new_layer)
                else:
                    new_layer = add_fc_layer('hidden', self.emb_width*(2**(self.num_fc_layers - layer)),
                                                   self.emb_width*(2**(self.num_fc_layers - layer - 1)))
                    self.fc_layers.add_module('fc' + str(layer), new_layer)
                                       
    def forward(self, x):
        x = self.enc(x)
        x = self.fc_layers(x)
        return F.normalize(x, p = 2, dim = 1)
        # return x

class Model_SN(nn.Module):
    def __init__(self, pretrained = None, mode = None, arch = 'resnet50', cp_path = './', emb_width = 512, num_fc_layers = 1):
        super().__init__()
        self.pretrained = pretrained
        self.mode = mode
        self.arch = arch
        self.cp_path = cp_path
        self.enc, self.num_words = load_model(self.pretrained, self.mode, self.cp_path, self.arch)
        self.emb_width = emb_width
        self.num_fc_layers = num_fc_layers
        self.fc_layers = nn.Sequential()
        if self.num_fc_layers == 1:
            new_layer = add_fc_layer('last', self.num_words,  self.emb_width)
            self.fc_layers.add_module('fc0', new_layer)
        else:
            for layer in range(self.num_fc_layers):
                if layer == 0:
                    new_layer = add_fc_layer('first', self.num_words, self.emb_width*(2**(self.num_fc_layers - layer - 1)))
                    self.fc_layers.add_module('fc' + str(layer), new_layer)
                elif layer == (self.num_fc_layers - 1):
                    new_layer = add_fc_layer('last', self.emb_width*(2**(self.num_fc_layers - layer)), self.emb_width)
                    self.fc_layers.add_module('fc' + str(layer), new_layer)
                else:
                    new_layer = add_fc_layer('hidden', self.emb_width*(2**(self.num_fc_layers - layer)),
                                                   self.emb_width*(2**(self.num_fc_layers - layer - 1)))
                    self.fc_layers.add_module('fc' + str(layer), new_layer)
                                       
        self.fc = nn.Linear(in_features = self.emb_width, out_features = 1)

    def forward(self, x1, x2):
        x1 = self.fc_layers(self.enc(x1))
        x2 = self.fc_layers(self.enc(x2))
        return torch.sigmoid(self.fc(torch.abs(x1 - x2)))

class Model_MLC(nn.Module):
    def __init__(self, pretrained = None, mode = None, arch = 'resnet50', cp_path = './', emb_width = 512, num_fc_layers = 1, num_classes = 8):
        super().__init__()
        self.pretrained = pretrained
        self.mode = mode
        self.arch = arch
        self.cp_path = cp_path
        self.enc, self.num_words = load_model(self.pretrained, self.mode, self.cp_path, self.arch)
        self.emb_width = emb_width
        self.num_fc_layers = num_fc_layers
        self.fc_layers = nn.Sequential()
        self.num_classes = num_classes
        if self.num_fc_layers == 1:
            new_layer = add_fc_layer('hidden', self.num_words,  self.emb_width)
            class_layer = add_fc_layer('last', self.emb_width, self.num_classes)
            self.fc_layers.add_module('fc0', new_layer)
            self.fc_layers.add_module('fc1', class_layer)
        else:
            for layer in range(self.num_fc_layers):
                if layer == 0:
                    new_layer = add_fc_layer('first', self.num_words, self.emb_width*(2**(self.num_fc_layers - layer - 1)))
                    self.fc_layers.add_module('fc' + str(layer), new_layer)
                else:
                    new_layer = add_fc_layer('hidden', self.emb_width*(2**(self.num_fc_layers - layer)),
                                                   self.emb_width*(2**(self.num_fc_layers - layer - 1)))
                    self.fc_layers.add_module('fc' + str(layer), new_layer)

            class_layer = add_fc_layer('last', self.emb_width, self.num_classes)
            self.fc_layers.add_module('fc' + str(self.num_fc_layers), class_layer)

    def forward(self, x):
        x = self.enc(x)
        x = self.fc_layers(x)        
        return F.softmax(x,1)
