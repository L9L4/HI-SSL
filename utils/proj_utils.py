import os
import tensorflow as tf
from tensorboard.plugins import projector
import pandas as pd
import numpy as np
import random
import yaml
from tqdm import tqdm
from PIL import Image

class TF_Embedding_Projector():

    def __init__(self, config_path, log_root, phase, im_path):
        
        with open(config_path, 'r') as f:
            exp_config = yaml.load(f, Loader=yaml.SafeLoader)

        self.test_ID = exp_config['general']['test_id']
        self.test_type = exp_config['general']['test_type']
        self.log_root = log_root        
        self.phase = phase        
        self.im_path = im_path
        self.log_dir = os.path.join(self.log_root, f'logs/emb_{self.test_ID}_{self.test_type}_{phase}/')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        print(f'Log directory: {self.log_dir}\n')

    def produce_sprite_image(self, samples, optimal_image_size):

        im_names = [os.path.join(root, name) for root, dirs, files in os.walk(self.im_path) for name in files if name.endswith('.jpg')]

        samples_to_add = [im_names[i] for i in samples]
        im_names += samples_to_add

        images = [Image.open(filename).resize((optimal_image_size, optimal_image_size)) for filename in im_names]

        one_square_size = int(np.ceil(np.sqrt(len(images))))
        master_width = optimal_image_size * one_square_size 
        master_height = optimal_image_size * one_square_size

        spriteimage = Image.new(
            mode='RGBA',
            size=(master_width, master_height),
            color=(0,0,0,0))

        for count, image in enumerate(images):
            div, mod = divmod(count,one_square_size)
            h_loc = optimal_image_size*div
            w_loc = optimal_image_size*mod
            spriteimage.paste(image,(w_loc,h_loc))

        spriteimage.convert('RGB').save(os.path.join(self.log_dir, f'Test_{self.test_ID}_{self.test_type}_sprite_{self.phase}.jpg'), transparency=0)

        print('Sprite image: done\n')

    def set_projector(self, x, optimal_image_size):
        
        emb = tf.Variable(x, name='embeddings')
        checkpoint = tf.train.Checkpoint(embedding=emb)
        checkpoint.save(os.path.join(self.log_dir, f'Test_{self.test_ID}_{self.test_type}_embedding_{self.phase}.ckpt'))

        print('Embedding checkpoint: done\n')

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
        embedding.metadata_path = f'Test_{self.test_ID}_{self.test_type}_metadata_{self.phase}.tsv'
        embedding.sprite.image_path = f'Test_{self.test_ID}_{self.test_type}_sprite_{self.phase}.jpg'
        embedding.sprite.single_image_dim.extend([optimal_image_size, optimal_image_size])
        projector.visualize_embeddings(self.log_dir, config)

    def generate_proj_files(self):

        path_to_embeddings = f'./Test_{self.test_ID}_{self.test_type}_embeddings_{self.phase}.tsv'
        path_to_metadata = f'./Test_{self.test_ID}_{self.test_type}_metadata_{self.phase}.tsv'

        embeddings = pd.read_csv(path_to_embeddings, index_col = False, sep = '\t', header = None)

        with open(path_to_metadata, 'r') as f:
            classes = [int(line.strip()) for line in f.readlines()]

        embeddings['Nome'] = classes

        root_rounded = int(np.ceil(np.sqrt(len(embeddings)))**2)

        optimal_image_size = int(np.round(8192/np.sqrt(root_rounded))-1)

        print(f'Optimal shape to resize images for sprite image: {str(optimal_image_size)}x{str(optimal_image_size)}\n')

        n_samples_to_add = root_rounded - len(embeddings)

        print(f'Number of duplicates to add: {n_samples_to_add}\n')

        samples = random.sample(range(0, len(embeddings)), n_samples_to_add)

        for sample in samples:
            embeddings = embeddings.append(embeddings.loc[sample], ignore_index=True)

        emb_values = embeddings.drop(columns=['Nome'])
        x = emb_values.values
        y = embeddings['Nome'].values
        y = [str(value) for value in y]

        with open(os.path.join(self.log_dir, f'Test_{self.test_ID}_{self.test_type}_metadata_{self.phase}.tsv'), 'w') as f:
            for label in y:
                f.write('{}\n'.format(label))

        print('Metadata: done\n')

        return x, samples, optimal_image_size

    def __call__(self):

        x, samples, optimal_image_size = self.generate_proj_files()

        self.produce_sprite_image(samples, optimal_image_size)

        self.set_projector(x, optimal_image_size)

        return self.log_dir