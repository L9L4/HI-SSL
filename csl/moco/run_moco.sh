export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8888
export WORLD_SIZE=1
pip install -r requirements.txt
python main_moco.py ../Subsets_seed_34 -a resnet18 --epochs 20 -b 64 --lr 0.005 -j 0 --multiprocessing-distributed --dist-url 'env://' --rank 0 --mlp --aug-ms --cos
python main_moco.py ../Subsets_seed_34 -a resnet18 --epochs 20 -b 128 --lr 0.015 -j 0 --multiprocessing-distributed --dist-url 'env://' --rank 0 --mlp --aug-ms --cos
python main_moco.py ../Subsets_seed_34 -a resnet18 --epochs 20 -b 64 --lr 0.015 -j 0 --multiprocessing-distributed --dist-url 'env://' --rank 0 --mlp --aug-ms --cos