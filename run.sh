cd ./test
CONFIG=danet_alpha0.8
GPU=1
python3 inference.py --gpu_id $GPU --config $CONFIG --test_name psbattle_orisize_yty_test  --gpu_num 1 --config_type test --test_save 0
python3 inference.py --gpu_id $GPU --config $CONFIG --test_name psbattle_orisize_yty_test  --gpu_num 1 --config_type test --test_save 1
