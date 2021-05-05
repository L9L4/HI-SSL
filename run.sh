pip install -r requirements.txt
python main.py -dir=./ -td=../Subsets_seed_34/train -vd=../Subsets_seed_34/val -c=config
python main_test.py -dir=./ -td=../Subsets_seed_34/train -vd=../Subsets_seed_34/val -c=config