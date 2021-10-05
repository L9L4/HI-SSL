pip install -r requirements.txt
python main.py -dir=./ -td=./data_dir/train -vd=./data_dir/val -c=config
python main_test.py -dir=./ -td=./data_dir/train -vd=./data_dir/test -c=config