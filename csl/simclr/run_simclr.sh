export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8888
export WORLD_SIZE=1
pip install -r requirements.txt
python main_simclr.py -dir ./ -td ../Subsets_seed_34/train -vd ../Subsets_seed_34/val -c simclr_2 -g 4