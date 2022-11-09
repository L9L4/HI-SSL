import os, torch
t = torch
import numpy as np
import itertools
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas as pd
from copy import deepcopy

from utils.torch_utils import *
from utils.data import *

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
import umap

def print_losses(root, test_ID, test_type):

    data_path = root + os.sep + 'data/'
    losses = {'train': [], 'val': []}

    for loss in list(losses.keys()):
        with open(data_path + f'Test_{test_ID}_{test_type}_{loss}_losses.pkl', 'rb') as f:
            losses[loss] = pkl.load(f)

    with open(data_path + f'Test_{test_ID}_{test_type}_losses.txt', 'w') as f:
        f.write('The optimal value of loss for the training set is: {:01.3f}\n'.format(np.min(losses['train'])))
        f.write('The optimal value of loss for the validation set is: {:01.3f}\n'.format(np.min(losses['val'])))
        best_epoch_train = np.where(np.array(losses['train']) == min(losses['train']))[0][0] + 1
        best_epoch = np.where(np.array(losses['val']) == min(losses['val']))[0][0] + 1
        f.write(f"Epoch corresponding to the optimal value of the training loss: {best_epoch_train}\\{len(losses['train'])}\n")
        f.write(f"Epoch corresponding to the optimal value of the validation loss: {best_epoch}\\{len(losses['val'])}\n")

    plt.plot(losses['train'])
    plt.plot(losses['val'])
    plt.title('Model loss')
    plt.ylabel('Loss [-]')
    plt.xlabel('Epoch [-]')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.savefig(data_path + f'Test_{test_ID}_{test_type}_losses.png')
    plt.close()

def print_accs(root, test_ID, test_type):

    data_path = root + os.sep + 'data/'
    accs = {'train': [], 'val': []}

    if test_type == 'MCC':
        metric = 'accuracy'
        plot_title = 'Model Accuracy'
        y_label = 'Accuracy [-]'
    else:
        metric = 'MAPs'
        plot_title = 'Model MAP'
        y_label = 'MAP [-]'

    for acc in list(accs.keys()):
        with open(data_path + f'Test_{test_ID}_{test_type}_{acc}_{metric}.pkl', 'rb') as f:
            accs[acc] = pkl.load(f)

    with open(data_path + f'Test_{test_ID}_{test_type}_{metric}.txt', 'w') as f:
        f.write('The optimal value of accuracy for the training set is: {:01.3f}\n'.format(np.max(accs['train'])))
        f.write('The optimal value of accuracy for the validation set is: {:01.3f}\n'.format(np.max(accs['val'])))
        best_epoch_train = np.where(np.array(accs['train']) == max(accs['train']))[0][0] + 1
        best_epoch = np.where(np.array(accs['val']) == max(accs['val']))[0][0] + 1
        f.write(f"Epoch corresponding to the optimal value of the training accuracy: {best_epoch_train}\\{len(accs['train'])}\n")
        f.write(f"Epoch corresponding to the optimal value of the validation accuracy: {best_epoch}\\{len(accs['val'])}\n")

    plt.plot(accs['train'])
    plt.plot(accs['val'])
    plt.title(f'{plot_title}')
    plt.ylabel(f'{y_label}')
    plt.xlabel('Epoch [-]')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.savefig(data_path + f'Test_{test_ID}_{test_type}_{metric}.png')
    plt.close()  

class feature_analysis():

    def __init__(self, root, train_dir, data_dir, model, batch_size, transforms, device, test_ID, test_type, test_params, phase = 'train'):

        self.data_path = root + os.sep + 'data/'
        self.data_dir = data_dir
        self.authors = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        self.model = model
        self.mean_, self.std_ = load_rgb_mean_std(train_dir)
        self.batch_size = batch_size
        self.transforms = transforms
        self.device = device
        self.phase = phase
        self.test_ID = test_ID
        self.test_type = test_type
        self.test_params = test_params

    def compute_features(self):

        self.model.eval()
        
        dl = Standard_DataLoader(self.data_dir, self.transforms, 32, False, 'test', self.mean_, self.std_, True)
        dataset = dl.generate_dataset()
        _, set_ = dl.load_data()

        
        c_to_idx = dataset.class_to_idx
        idx_to_c = {c_to_idx[k]: k for k in list(c_to_idx.keys())}

        dict_aut = {k: [] for k in range(len(self.authors))}

        for data, target in tqdm(set_):
            data = data.to(self.device)
            
            if self.transforms['rc_p'] == 1.0:
                bs, ncrops, c, h, w = data.size()
                with torch.no_grad():
                    output = self.model(data.view(-1, c, h, w))
                    output = output.view(bs, ncrops, -1).mean(1)
                    output = output.cpu().detach().numpy()
            else:
                with torch.no_grad():
                    output = self.model(data)
                    if self.model.emb_type == 'sampling':
                        bs = len(data)
                        output = output.view(bs, self.model.samples, -1).mean(1)
                        output = output.cpu().detach().numpy()
                    else:
                        output = output.cpu().detach().numpy()

            for item_ in range(output.shape[0]):
                dict_aut[target[item_].item()].append(output[item_])

        new_dict_aut = {idx_to_c[k]: dict_aut[k] for k in list(idx_to_c.keys())}

        all_dataframes = {}
        for author in tqdm(range(len(self.authors))):
            all_dataframes[self.authors[author]] = pd.DataFrame.from_dict(new_dict_aut[str(self.authors[author])], orient='columns')
            name = [self.authors[author]]*len(os.listdir(self.data_dir + os.sep + self.authors[author]))
            all_dataframes[self.authors[author]].insert (0, "Nome", name)

        for author in tqdm(range(len(self.authors))):
            if author == 0:
                df = all_dataframes[self.authors[author]]
            else:
                df = df.append(all_dataframes[self.authors[author]], ignore_index=True)

        cat_list = []
        cat_dict = {self.authors[num]: num for num in range(len(self.authors))}
        for i in range(len(df["Nome"])):
            cat_list.append(cat_dict[df["Nome"][i]])

        embeddings_to_save = deepcopy(df)
        embeddings_to_save.pop('Nome')
        embeddings_to_save.to_csv(self.data_path + os.sep + 'Test_' + self.test_ID + '_' + self.test_type + '_embeddings_' + self.phase + '.tsv', 
            index=False, 
            sep="\t", 
            header = False)

        labels = list(df['Nome'].values)
        with open(self.data_path + os.sep + 'Test_' + self.test_ID + '_' + self.test_type + '_metadata_' + self.phase + '.tsv', "w") as f:
            for label in labels:
                f.write("{}\n".format(label))

        return cat_list, df

    def produce_plots(self, df, cat_list, dim_red):

        assert dim_red in ['PCA', 't-SNE', 'UMAP'], 'Wrong dimensionality reduction technique; insert either "PCA" or "t-SNE" or "UMAP"'
        
        df_values = df.drop(columns=['Nome'])
        x = df_values.values

        if dim_red == 'PCA':
            pca = PCA(n_components=2)
            x_red = pca.fit_transform(x)
        elif dim_red == 't-SNE':
            x_red = TSNE(n_components = 2).fit_transform(x)
        elif dim_red == 'UMAP':
            reducer = umap.UMAP()
            x_red = reducer.fit_transform(x)

        principalDf = pd.DataFrame(data = x_red, columns = ['x1', 'x2'])
        principalDf.insert (0, 'Nome', df['Nome'])
        principalDf.insert (1, 'Categoria', cat_list)

        centroid_name = []
        centroids_x = []
        centroids_y = []
        for i in range(len(self.authors)):
            name = self.authors[i]
            p1 = np.mean(principalDf[principalDf['Nome'] == name]['x1'])
            p2 = np.mean(principalDf[principalDf['Nome'] == name]['x2'])
            centroid_name.append(name)
            centroids_x.append(p1)
            centroids_y.append(p2)

        cmap_dict = {'train': 'twilight', 'val': 'viridis'}

        fig, ax = plt.subplots(figsize = (15,15))
        sc = plt.scatter(list(principalDf['x1']), list(principalDf['x2']), c = list(principalDf['Categoria']),
            s = 50, cmap = cmap_dict[self.phase], edgecolors='none')

        size=81
        lp = lambda i: plt.plot([],color=sc.cmap(sc.norm(i)), ms=np.sqrt(size), mec='none',
            label = list(dict.fromkeys(list(principalDf['Nome'])))[i], ls='', marker='o')[0]

        handles = [lp(i) for i in range(len(list(set(principalDf['Categoria']))))]
        plt.legend(handles=handles, fontsize = 15, loc = 'best')

        plt.scatter(centroids_x, centroids_y, c='black', s=50, marker='x')
        plt.xlabel('x1', fontsize = 20)
        plt.ylabel('x2', fontsize = 20)

        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)

        for j in range(len(centroids_x)):
            ax.annotate(centroid_name[j], xy = (centroids_x[j], centroids_y[j]), xytext = (-5, 5),
                textcoords = 'offset points',ha = 'right', va = 'bottom', color = 'black', fontsize = 20)

        plt.savefig(self.data_path + os.sep + 'Test_' + self.test_ID + '_' + self.test_type + '_' + dim_red + '_' + self.phase + '.png')

    def compute_mean_average_precision(self, df):

        if self.phase == 'train':
            RATIO = self.test_params['ratio_train']/100
        elif self.phase == 'val':
            RATIO = self.test_params['ratio_val']/100
        else:
            raise Exception('Phase not included: select either "train" or "val"')

        df = df.sample(frac=RATIO, replace=False)
        occurrencies = dict(df['Nome'].value_counts())
        AP_per_class = {key: 0 for key in list(occurrencies.keys())}
        
        APs = []
        for i in tqdm(list(df.index)):
            label_q = df.loc[i, :].values.tolist()[0]
            features_q = np.array(df.loc[i, :].values.tolist()[1:])
            backup_df = deepcopy(df)
            backup_df = backup_df.drop(i)
            backup_df = backup_df.reset_index(drop = True)
            distances = []
            for j in range(len(backup_df)):
                features_c = np.array(backup_df.loc[j, :].values.tolist()[1:])
                dist = np.linalg.norm(features_q - features_c)
                distances.append(dist)
            backup_df.insert(1, 'Distances', distances)
            backup_df = backup_df.sort_values(by='Distances', ascending=True)
            av_pr = average_precision(label_q, list(backup_df['Nome']), occurrencies[label_q]-1)
            AP_per_class[label_q] += av_pr
            APs.append(av_pr)

        APs = np.array(APs)
        MAP = np.mean(APs)
        AP_per_class = {key: AP_per_class[key]/occurrencies[key] for key in list(AP_per_class.keys())}

        with open(self.data_path + os.sep + 'Test_' + self.test_ID + '_' + self.test_type + '_Mean_Average_Precision.txt', 'a') as f:
            f.write(f'Mean Average Precision ({RATIO*100} % of the {self.phase} set) = {round(MAP*100,2)} %\n\n')
            for key in list(occurrencies.keys()):
                f.write(f'Class {key} MAP is {round(AP_per_class[key]*100,2)}, based on {occurrencies[key]} samples\n')

    def __call__(self):
        
        cat_list, df = self.compute_features()

        for dim_red in ['PCA', 't-SNE', 'UMAP']:
            self.produce_plots(df, cat_list, dim_red)

        self.compute_mean_average_precision(df)


class mcc_accuracy():
    
    def __init__(self, root, train_dir, data_dir, model, transforms, device, test_ID, test_type, phase = 'train'):

        self.data_path = root + os.sep + 'data/'
        self.data_dir = data_dir
        self.model = model
        self.mean_, self.std_ = load_rgb_mean_std(train_dir)
        self.transforms = transforms
        self.device = device
        self.test_ID = test_ID
        self.test_type = test_type
        self.phase = phase

    def produce_output(self):
        
        self.model.eval()
        
        dl = Standard_DataLoader(self.data_dir, self.transforms, 32, False, 'test', self.mean_, self.std_, True)
        dataset = dl.generate_dataset()
        _, set_ = dl.load_data()

        labels = []
        preds = []
        target_names = list(dataset.class_to_idx.keys())
        c_to_idx = dataset.class_to_idx
        idx_to_c = {c_to_idx[k]: k for k in list(c_to_idx.keys())}

        for data, target in set_:
            data = data.to(self.device)
            labels += list(target.numpy())
            target = target.to(self.device)
            

            with torch.no_grad():
                output = self.model(data)
                max_index = output.max(dim = 1)[1]
                max_index = max_index.cpu().detach().numpy()
                preds += list(max_index)

        label_class_names = [idx_to_c[id_] for id_ in labels]
        pred_class_names = [idx_to_c[id_] for id_ in preds]

        return labels, preds, label_class_names, pred_class_names, target_names 

    def plot_confusion_matrix(self, 
                          lcn,
                          pcn,
                          tn,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

        cm = confusion_matrix(lcn, pcn, labels=tn)

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(20, 20))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if tn is not None:
            tick_marks = np.arange(len(tn))
            plt.xticks(tick_marks, tn, rotation=45)
            plt.yticks(tick_marks, tn)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy = {:0.4f}; misclass = {:0.4f}'.format(accuracy, misclass))
        plt.savefig(self.data_path + os.sep + 'Test_' + self.test_ID + '_' + self.test_type + '_confusion_matrix_' + self.phase + '.png')

    def produce_report(self, 
                          lab,
                          preds,
                          tn):
    
        with open(self.data_path + os.sep + 'Test_' + self.test_ID + '_' + self.test_type + '_classification-report_' + self.phase + '.txt', 'w') as f:
            f.write(classification_report(lab, preds, target_names=tn))

    def __call__(self):
        
        labels, preds, label_class_names, pred_class_names, target_names = self.produce_output()

        self.plot_confusion_matrix(label_class_names, pred_class_names, target_names)

        self.produce_report(labels, preds, target_names)