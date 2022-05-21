# DANet_baseline

## Environment

+ Ubuntu 16.04.6 LTS
+ Python 3.6
+ cuda10.1+cudnn7.6.3

## Requirement
+ 安装 [nvidia-apex](https://github.com/NVIDIA/apex)
, 并确保在code目录下
+ pip install requirements.txt

## Usage
将数据集文件夹ps_battles_orisize放到和train.sh同级目录下
### Test
```
bash run.sh
```

### Train
```
bash train.sh
```

请自行在./config的config文件中修改配置参数
将train_path,val_path,test_dir修改为自己的路径
