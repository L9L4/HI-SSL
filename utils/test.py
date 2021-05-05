import os, torch
t = torch
import numpy as np
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

def print_losses(root, test_ID, test_type):

    data_path = root + os.sep + 'data/'
    losses = {'train': [], 'val': []}

    for loss in list(losses.keys()):
        with open(data_path + f'Prova_{test_ID}_{test_type}_{loss}_losses.pkl', 'rb') as f:
            losses[loss] = pkl.load(f)

    with open(data_path + f'Prova_{test_ID}_{test_type}_losses.txt', 'w') as f:
        f.write('The optimal value of loss for the background set is: {:01.3f}\n'.format(np.min(losses['train'])))
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
    plt.savefig(data_path + f'Prova_{test_ID}_{test_type}_losses.png')
    plt.close()

def print_accs(root, test_ID, test_type):

    data_path = root + os.sep + 'data/'
    accs = {'train': [], 'val': []}

    for acc in list(accs.keys()):
        with open(data_path + f'Prova_{test_ID}_{test_type}_{acc}_MAPs.pkl', 'rb') as f:
            accs[acc] = pkl.load(f)

    with open(data_path + f'Prova_{test_ID}_{test_type}_MAPs.txt', 'w') as f:
        f.write('The optimal value of accuracy for the background set is: {:01.3f}\n'.format(np.max(accs['train'])))
        f.write('The optimal value of accuracy for the validation set is: {:01.3f}\n'.format(np.max(accs['val'])))
        best_epoch_train = np.where(np.array(accs['train']) == max(accs['train']))[0][0] + 1
        best_epoch = np.where(np.array(accs['val']) == max(accs['val']))[0][0] + 1
        f.write(f"Epoch corresponding to the optimal value of the training accuracy: {best_epoch_train}\\{len(accs['train'])}\n")
        f.write(f"Epoch corresponding to the optimal value of the validation accuracy: {best_epoch}\\{len(accs['val'])}\n")

    plt.plot(accs['train'])
    plt.plot(accs['val'])
    plt.title('Model MAP')
    plt.ylabel('MAP [-]')
    plt.xlabel('Epoch [-]')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.savefig(data_path + f'Prova_{test_ID}_{test_type}_MAPs.png')
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
        
        dl = Standard_DataLoader(self.data_dir, self.transforms, self.batch_size, False, self.mean_, self.std_, True)
        dataset = dl.generate_dataset()
        _, set_ = dl.load_data()

        
        c_to_idx = dataset.class_to_idx
        idx_to_c = {c_to_idx[k]: k for k in list(c_to_idx.keys())}

        dict_aut = {k: [] for k in range(len(self.authors))}

        for data, target in tqdm(set_):
            data = data.to(self.device)
            with torch.no_grad():
                output = self.model(data)
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

        return cat_list, df

    def produce_plots(self, df, cat_list, dim_red):

        assert dim_red in ['PCA', 't-SNE'], 'Wrong dimensionality reduction technique; insert either "PCA" or "t-SNE"'
        
        df_values = df.drop(columns=['Nome'])
        x = df_values.values

        if dim_red == 'PCA':
            pca = PCA(n_components=2)
            x_red = pca.fit_transform(x)
        elif dim_red == 't-SNE':
            x_red = TSNE(n_components = 2).fit_transform(x)

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

        fig, ax = plt.subplots(figsize = (15,15))
        sc = plt.scatter(list(principalDf['x1']), list(principalDf['x2']), c = list(principalDf['Categoria']),
            s = 50, cmap = 'hsv', edgecolors='none')

        size=81
        lp = lambda i: plt.plot([],color=sc.cmap(sc.norm(i + 1)), ms=np.sqrt(size), mec='none',
            label = list(dict.fromkeys(list(principalDf['Nome'])))[i], ls='', marker='o')[0]

        handles = [lp(i-1) for i in np.unique(principalDf['Categoria'])]
        plt.legend(handles=handles)

        plt.scatter(centroids_x, centroids_y, c='black', s=50, marker='x')
        plt.xlabel('x1')
        plt.ylabel('x2')

        for j in range(len(centroids_x)):
            ax.annotate(centroid_name[j], xy = (centroids_x[j], centroids_y[j]), xytext = (-5, 5),
                textcoords = 'offset points',ha = 'right', va = 'bottom')

        plt.savefig(self.data_path + os.sep + 'Prova_' + self.test_ID + '_' + self.test_type + '_' + dim_red + '_' + self.phase + '.png')

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

        with open(self.data_path + os.sep + 'Prova_' + self.test_ID + '_' + self.test_type + '_Mean_Average_Precision.txt', 'a') as f:
            f.write(f'Mean Average Precision ({RATIO*100} % of the {self.phase} set) = {round(MAP*100,2)} %\n\n')
            for key in list(occurrencies.keys()):
            	f.write(f'Manuscript {key} MAP is {round(AP_per_class[key]*100,2)}, based on {occurrencies[key]} samples\n')

    def __call__(self):
        
        cat_list, df = self.compute_features()

        for dim_red in ['PCA', 't-SNE']:
            self.produce_plots(df, cat_list, dim_red)

        self.compute_mean_average_precision(df)