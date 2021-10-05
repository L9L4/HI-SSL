# Self-supervised learning for handwriting identification on medieval manuscripts

This repository allows the user to solve the task of handwriting identification for medieval manuscripts (via minimization of a triplet margin loss) either by:
* fine-tuning a ResNet18 encoder pretrained with
[OBoW: Online Bag-of-Visual-Words Generation for Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Gidaris_OBoW_Online_Bag-of-Visual-Words_Generation_for_Self-Supervised_Learning_CVPR_2021_paper.pdf) on a set of unlabeled manuscripts;
* or fine-tuning a ResNet18 encoder pretrained on the ImageNet dataset;
* or even training a ResNet18 completely from scratch.  

## Contents

1. [Dataset](#dataset)
2. [Installation](#installation)
3. [Test configuration](#test-configuration)
4. [Usage](#usage)
5. [TensorFlow Embedding Projector](#tensorflow-embedding-projector)

## Dataset

## Installation

To setup the dependencies, simply run in a dedicated environment:
```
pip install -r requirements.txt
```

## Test configuration

## Usage

To train a model on the subset `./data_dir/train` while using `./data_dir/val` as validation set, and following the setup defined by the `./config/config.yaml` file, run: 

```
python main.py -dir=./ -td=./data_dir/train -vd=./data_dir/val -c=config
```

To compute the Mean Average Precision at k (MAP@k) and to plot the embeddings via PCA, t-SNE, and UMAP 2D projection for both the training (`./data_dir/train`) and the test set (`./data_dir/test`), according to the configuration file `./config/config.yaml`, run: 

```
python main_test.py -dir=./ -td=./data_dir/train -vd=./data_dir/test -c=config
```

To launch both the processes at once, simply run the bash file `./run.sh` with the two commands:  

```
bash run.sh
```


## TensorFlow Embedding Projector