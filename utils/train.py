import os
import torch
t = torch
import pickle as pkl
from tqdm import tqdm
import torch.optim as optim
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

# Intra-Class Variance Minimization Loss
class IaCVM(object):
	def __init__(self, first_loss, magnitude):
		self.first_loss = first_loss
		self.magnitude = magnitude

	def IaCVM_func(self, samples, labels):
		weight = torch.zeros(labels.max()+1, samples.shape[0]).to(samples.device)
		weight[labels, torch.arange(samples.shape[0]).unsqueeze(0)] = 1
		embeddings_per_class = torch.mul(weight.unsqueeze(2),samples.unsqueeze(0))
		var = torch.stack([torch.sum(torch.var(embeddings_per_class[i][embeddings_per_class[i].nonzero(as_tuple=True)].view(-1, embeddings_per_class[i].shape[1]), 
			dim = 0)) for i in range(embeddings_per_class.shape[0])])

		label_freq = torch.bincount(labels)
		label_freq[label_freq == 1] = 0
		label_freq = label_freq.type(torch.FloatTensor).to(samples.device)
        
		mean_variance = torch.matmul(label_freq, torch.nan_to_num(var, nan=0.0))/torch.sum(label_freq)
		return mean_variance

	def __call__(self, samples, labels):
		FL = self.first_loss(samples, labels)
		SL = self.IaCVM_func(samples, labels)

		RATIO = 100*SL/(FL + SL)

		print(f'\nRatio between Intra-Class Variance Minimization Loss and overall loss: {round(RATIO.item(), 1)} %\n')
		return FL + SL*self.magnitude

# Batch Hard Triplet Loss (online_triplet_loss)
class BHTL(object): 

	def __init__(self, squared, margin):
		self.squared = squared
		self.margin = margin

	def __call__(self, output, target):
		from online_triplet_loss.losses import batch_hard_triplet_loss
		loss = batch_hard_triplet_loss(target, output, squared = self.squared, margin = self.margin)
		return loss

class save_results():
    def __init__(self, history_path, checkpoint_path, test_name):
        self.history_path = history_path
        self.checkpoint_path = checkpoint_path
        self.test_name = test_name

    def save_pkl(self, name, list_):
        with open(self.history_path + os.sep + self.test_name + name +'.pkl', 'wb') as f:
            pkl.dump(list_, f) 

    def save_checkpoints(self, ep_loss, min_loss, model, optimizer, name):
        if ep_loss <= min_loss:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': ep_loss}, self.checkpoint_path + os.sep + self.test_name + name + '_best_model.pth')
            return ep_loss
        else:
            return min_loss

def set_optimizer(optim_params, model):
    if optim_params['optim_type'] == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
            lr = optim_params['lr'],
            betas = optim_params['beta'],
            weight_decay = optim_params['weight_decay'])
    elif optim_params['optim_type'] == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
            lr = optim_params['lr'],
            momentum = optim_params['momentum'],
            nesterov = optim_params['nesterov'],
            weight_decay = optim_params['weight_decay'])
    else:
        raise Exception('The selected optimization type is not available.')

    return optimizer

def set_scheduler(optim_params, optimizer):
    if optim_params['lr_schedule_type'] == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, optim_params['step'], optim_params['gamma'])
    elif optim_params['lr_schedule_type'] == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, optim_params['gamma'])
    elif optim_params['lr_schedule_type'] == 'red_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = optim_params['gamma'], patience = optim_params['patience'],
            min_lr = optim_params['end_lr'])
    elif optim_params['lr_schedule_type'] == 'cos_warmup':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs = optim_params['warmup_epochs'], 
            max_epochs = optim_params['num_epochs'], warmup_start_lr = optim_params['warmup_start_lr'], 
            eta_min = optim_params['end_lr'])
    else:
        raise Exception('The selected scheduler type is not available.')
    return scheduler

def set_metric_learning_loss(optim_params):

	if optim_params['loss']['loss_type'] == 'bhtl':
		loss_func = BHTL(optim_params['loss']['squared'], optim_params['loss']['margin'])
		mining = None
		accuracy_calculator = None
	
	elif optim_params['loss']['loss_type'].split('_')[0] == 'pml':
		from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
		from pytorch_metric_learning import losses, miners, distances, reducers
		accuracy_calculator = AccuracyCalculator()
		if optim_params['loss']['loss_type'] == 'pml_bhtl':
			if optim_params['loss']['distance'] == 'LpDistance':
				distance = distances.__dict__[optim_params['loss']['distance']](collect_stats = True, 
					normalize_embeddings = optim_params['loss']['normalize_embedding'], 
					p = optim_params['loss']['p'], 
					power = optim_params['loss']['power'], 
					is_inverted = False)
			elif optim_params['loss']['distance'] == 'DotProductSimilarity':
				distance = distances.__dict__[optim_params['loss']['distance']](collect_stats = True,
					normalize_embeddings = optim_params['loss']['normalize_embedding'],)
			reducer = reducers.AvgNonZeroReducer(collect_stats = True)
			loss_func = losses.TripletMarginLoss(margin = optim_params['loss']['margin'], distance = distance, reducer = reducer)

		mining = miners.TripletMarginMiner(collect_stats = True,
			margin = optim_params['loss']['margin'],
			distance = distance,
			type_of_triplets = optim_params['loss']['mining'])

	if optim_params['loss']['ia_c_var_min']:
			loss_func = IaCVM(loss_func, optim_params['loss']['iacvm_magnitude'])

	return loss_func, mining, accuracy_calculator

class Trainer_MCC():
    def __init__(self, model, t_set, v_set, DEVICE, optim_params, model_path, history_path, test_ID, num_epochs = 300):
        self.model = model
        self.t_set = t_set
        self.v_set = v_set
        self.DEVICE = DEVICE
        self.optim_params = optim_params
        self.model_path = model_path
        self.history_path = history_path
        self.test_ID = test_ID
        self.test_name = 'Test_' + self.test_ID + '_MCC_'
        self.num_epochs = num_epochs
        self.checkpoint_path = os.path.join(self.model_path, 'checkpoints_' + self.test_ID)

    def compute_minibatch_accuracy(self, output, label):
        max_index = output.max(dim = 1)[1]
        return (max_index == label).sum().cpu().item(), (max_index == label).sum().cpu().item()/label.size()[0] 

    def train_model(self):
        
        from torch.nn import CrossEntropyLoss

        optimizer = set_optimizer(self.optim_params, self.model)
        scheduler = set_scheduler(self.optim_params, optimizer)

        criterion = CrossEntropyLoss()

        sr = save_results(self.history_path, self.checkpoint_path, self.test_name)

        self.model.train()
        
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []        
        min_loss_t = 1000.0
        min_loss_v = 1000.0                
        
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch} / {self.num_epochs}')
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0
            for data, target in tqdm(self.t_set, 'Training'):
                data = data.to(self.DEVICE)
                target = target.to(self.DEVICE)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output,target)
                true, _ = self.compute_minibatch_accuracy(output, target)
                epoch_loss += float(loss.item())*len(data)
                epoch_acc += true
                loss.backward()
                optimizer.step()

            epoch_loss /= len(self.t_set.dataset)
            epoch_acc /= len(self.t_set.dataset)
            print(f'train_loss: {epoch_loss} - train_accuracy: {epoch_acc}')
            print()
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

            min_loss_t = sr.save_checkpoints(epoch_loss, min_loss_t, self.model, optimizer, 'train')

            epoch_val_loss = 0.0
            epoch_val_acc = 0
            optimizer.zero_grad()
            self.model.eval()
            for data, target in tqdm(self.v_set, 'Validation'):
                data = data.to(self.DEVICE)
                target = target.to(self.DEVICE)
                with torch.no_grad():
                    output_val = self.model(data)                    
                validation_loss = criterion(output_val,target)
                val_true, _ = self.compute_minibatch_accuracy(output_val, target)                    
                epoch_val_loss += validation_loss.item()*len(data)
                epoch_val_acc += val_true
            
            epoch_val_loss /= len(self.v_set.dataset)
            epoch_val_acc /= len(self.v_set.dataset)
            print(f'val_loss: {epoch_val_loss} - val_accuracy: {epoch_val_acc}')
            print()
            val_loss.append(epoch_val_loss)
            val_acc.append(epoch_val_acc)
            
            min_loss_v = sr.save_checkpoints(epoch_val_loss, min_loss_v, self.model, optimizer, 'val')

            if self.optim_params['lr_schedule_type'] == 'red_on_plateau':
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

            sr.save_pkl('train_losses', train_loss)
            sr.save_pkl('val_losses', val_loss)
            sr.save_pkl('train_accuracy', train_acc)
            sr.save_pkl('val_accuracy', val_acc)

    def __call__(self):
        self.train_model()                               

class Trainer_TL(): 
    def __init__(self, model, tds, vds, t_dl, v_dl, DEVICE, optim_params, model_path, history_path, test_ID, num_epochs = 300):
        self.model = model
        self.tds = tds
        self.vds = vds                  
        self.t_dl = t_dl
        self.v_dl = v_dl          
        self.DEVICE = DEVICE
        self.optim_params = optim_params
        self.model_path = model_path
        self.history_path = history_path
        self.test_ID = test_ID
        self.test_name = 'Test_' + self.test_ID + '_TL_'
        self.num_epochs = num_epochs
        self.checkpoint_path = os.path.join(self.model_path, 'checkpoints_' + self.test_ID)

    def get_all_embeddings(self, dataset, model):
        from pytorch_metric_learning import testers
        tester = testers.BaseTester(dataloader_num_workers = 0)
        return tester.get_all_embeddings(dataset, model)

    def test(self, training_set, validation_set, model, accuracy_calculator, same_authors):
        
        if model.emb_type == 'sampling':
            return 0
        
        else:
            
            embeddings_tr, labels_tr = self.get_all_embeddings(training_set, model)
            labels_tr = labels_tr.squeeze()
            print("Computing accuracy")
            accuracies_tr = accuracy_calculator.get_accuracy(embeddings_tr,
                                                    embeddings_tr,
                                                    labels_tr, 
                                                    labels_tr,
                                                    embeddings_come_from_same_source = True)

            print("Training set accuracy (Mean Average Precision) = {}".format(accuracies_tr["mean_average_precision"]))
            
            embeddings_v, labels_v = self.get_all_embeddings(validation_set, model)
            labels_v = labels_v.squeeze()

            if same_authors:

                accuracies_v = accuracy_calculator.get_accuracy(embeddings_v,
                                                        embeddings_tr,
                                                        labels_v,
                                                        labels_tr,
                                                        embeddings_come_from_same_source = False)
                print("Validation set accuracy (Mean Average Precision) = {}".format(accuracies_v["mean_average_precision"]))
            
            else:

                accuracies_v = accuracy_calculator.get_accuracy(embeddings_v,
                                                        embeddings_v,
                                                        labels_v,
                                                        labels_v,
                                                        embeddings_come_from_same_source = True)
                print("Validation set accuracy (Mean Average Precision) = {}".format(accuracies_v["mean_average_precision"]))
            
            return accuracies_tr["mean_average_precision"], accuracies_v["mean_average_precision"]

    def train_model(self):
        loss_func, mining, accuracy_calculator = set_metric_learning_loss(self.optim_params)
        
        optimizer = set_optimizer(self.optim_params, self.model)
        scheduler = set_scheduler(self.optim_params, optimizer)
        
        sr = save_results(self.history_path, self.checkpoint_path, self.test_name)
        
        self.model.train()
        
        train_loss = []
        val_loss = []
        train_accs = []
        val_accs = []
        min_loss_t = 1000.0
        min_loss_v = 1000.0
        
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch} / {self.num_epochs}')
            self.model.train()
            epoch_loss = 0.0
            for data, target in tqdm(self.t_dl, 'Training'):
                data = data.to(self.DEVICE)
                if self.model.emb_type == 'sampling':
                    batch_size = len(data)
                    target = target.repeat(self.model.samples,1).t().reshape(batch_size*self.model.samples,1)
                target = target.to(self.DEVICE)
                optimizer.zero_grad()
                output = self.model(data)
                
                if self.optim_params['loss']['loss_type'].split('_')[0] == 'pml':
                    indices_tuple = mining(output, target)
                    loss = loss_func(output, target, indices_tuple)
                    
                else:
                    loss = loss_func(output, target)

                epoch_loss += float(loss.item())*len(data)
                loss.backward()
                optimizer.step()

            epoch_loss /= len(self.t_dl.dataset)
            print(f'train_loss: {epoch_loss}')
            print()
            train_loss.append(epoch_loss)

            min_loss_t = sr.save_checkpoints(epoch_loss, min_loss_t, self.model, optimizer, 'train')

            epoch_val_loss = 0.0
            optimizer.zero_grad()
            self.model.eval()
            for data, target in tqdm(self.v_dl, 'Validation'):
                data = data.to(self.DEVICE)
                if self.model.emb_type == 'sampling':
                    batch_size = len(data)
                    target = target.repeat(self.model.samples,1).t().reshape(batch_size*self.model.samples,1)
                target = target.to(self.DEVICE)
                with torch.no_grad():
                    output_val = self.model(data)

                if self.optim_params['loss']['loss_type'].split('_')[0] == 'pml':
                    indices_tuple_val = mining(output_val, target)
                    validation_loss = loss_func(output_val, target, indices_tuple_val)
                else:
                    validation_loss = loss_func(output_val, target)

                epoch_val_loss += validation_loss.item()*len(data)

            epoch_val_loss /= len(self.v_dl.dataset)
            print(f'val_loss: {epoch_val_loss}')
            print()
            val_loss.append(epoch_val_loss)

            min_loss_v = sr.save_checkpoints(epoch_val_loss, min_loss_v, self.model, optimizer, 'val')
            
            if not accuracy_calculator == None:
                train_acc, val_acc = self.test(self.tds, self.vds, self.model, accuracy_calculator, self.optim_params['same_authors'])
                train_accs.append(train_acc)
                val_accs.append(val_acc)

                sr.save_pkl('train_MAPs', train_accs)
                sr.save_pkl('val_MAPs', val_accs)

            if self.optim_params['lr_schedule_type'] == 'red_on_plateau':
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

            with torch.no_grad():
                if self.model.alpha < 1.0:
                    self.model.alpha.clamp_(self.optim_params['alpha_min'], self.optim_params['alpha_max'])

            sr.save_pkl('train_losses', train_loss)
            sr.save_pkl('val_losses', val_loss)

    def __call__(self):
        self.train_model()

class Trainer_SN():
    def __init__(self, model, t_set, v_set, DEVICE, optim_params, model_path, history_path, test_ID, num_epochs = 300):
        self.model = model
        self.t_set = t_set
        self.v_set = v_set
        self.DEVICE = DEVICE
        self.optim_params = optim_params
        self.model_path = model_path
        self.history_path = history_path
        self.test_ID = test_ID
        self.test_name = 'Test_' + self.test_ID + '_SN_'
        self.num_epochs = num_epochs
        self.checkpoint_path = os.path.join(self.model_path, 'checkpoints_' + self.test_ID)

    def train_model(self):
        
        from torch.nn import BCELoss

        optimizer = set_optimizer(self.optim_params, self.model)
        scheduler = set_scheduler(self.optim_params, optimizer)

        sr = save_results(self.history_path, self.checkpoint_path, self.test_name)
        
        self.model.train()
        
        train_loss = []
        val_loss = []     
        min_loss_t = 1000.0
        min_loss_v = 1000.0        

        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch} / {self.num_epochs}')
            self.model.train()
            epoch_loss = 0.0
            for anchor, positive, negative in tqdm(self.t_set, "Training"):
                
                anchor = anchor.repeat(2,1,1,1)
                anchor = anchor.to(self.DEVICE)

                compared = torch.cat((positive, negative), 0)
                compared = compared.to(self.DEVICE)

                target = torch.cat((torch.ones(positive.shape[0]),torch.zeros(negative.shape[0])),0)
                target = target.to(self.DEVICE)
                target = target.unsqueeze(1)

                optimizer.zero_grad()
                output = self.model(anchor, compared)

                BCE_loss = BCELoss()
                loss = BCE_loss(output, target.detach())
                epoch_loss += float(loss.item())*(len(anchor)/2)
                loss.backward()
                optimizer.step()

            epoch_loss /= len(self.t_set.dataset)
            print(f'train_loss: {epoch_loss}')
            print()
            train_loss.append(epoch_loss)

            min_loss_t = sr.save_checkpoints(epoch_loss, min_loss_t, self.model, optimizer, 'train')

            epoch_val_loss = 0.0
            optimizer.zero_grad()
            self.model.eval()
            for anchor, positive, negative in tqdm(self.v_set, 'Validation'):

                anchor = anchor.repeat(2,1,1,1)
                anchor = anchor.to(self.DEVICE)

                compared = torch.cat((positive, negative), 0)
                compared = compared.to(self.DEVICE)

                target = torch.cat((torch.ones(positive.shape[0]),torch.zeros(negative.shape[0])),0)
                target = target.to(self.DEVICE)
                target = target.unsqueeze(1)

                with torch.no_grad():
                    output_val = self.model(anchor, compared)

                BCE_loss = BCELoss()
                validation_loss = BCE_loss(output_val, target)                   
                epoch_val_loss += validation_loss.item()*(len(anchor)/2)
            
            epoch_val_loss /= len(self.v_set.dataset)
            print(f'val_loss: {epoch_val_loss}')
            print()
            val_loss.append(epoch_val_loss)

            min_loss_v = sr.save_checkpoints(epoch_val_loss, min_loss_v, self.model, optimizer, 'val')

            if self.optim_params['lr_schedule_type'] == 'red_on_plateau':
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

            sr.save_pkl('train_losses', train_loss)
            sr.save_pkl('val_losses', val_loss)

    def __call__(self):
        self.train_model()