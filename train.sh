cd ./train
CONFIG=danet_alpha0.8

python3 train.py --config $CONFIG --gpu_id 1 --gpu_num 1 --config_type train |tee -a $CONFIG.log
# python3 train.py --config $CONFIG --gpu_num 4 --config_type train |tee -a $CONFIG.log