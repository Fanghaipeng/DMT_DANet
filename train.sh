cd ./train
# CONFIG=danet_alpha0.8
# CONFIG=danet_alpha0.8_BayarNoise
CONFIG=danet_alpha0.8_SRMNoise
python3 train.py --config $CONFIG --gpu_id 3 --config_type train |tee -a $CONFIG.log
# python3 train.py --config $CONFIG --gpu_num 1 --config_type train |tee -a $CONFIG.log
# python3 train.py --config $CONFIG --gpu_id -1 --gpu_num 2 --config_type train |tee -a $CONFIG.log