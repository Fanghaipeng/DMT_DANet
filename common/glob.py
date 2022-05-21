import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="model1")
parser.add_argument("--config_type", type=str, default="train")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
parser.add_argument('--gpu_id', type=int, default=-1, help='GPU ID')
parser.add_argument('--manualSeed', type=int, default=-1)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument('--test_save', type=int, default=0)
parser.add_argument('--test_name', type=str, default=0)
opt = parser.parse_args()

_global_dict = {}


def _init():  # 初始化
    global _global_dict
    # _global_dict['max_anchors_size'] = 320
    # _global_dict['min_anchors_size'] = 320
    # _global_dict['stride'] = 8
    # _global_dict['anchors'] = [(_global_dict['max_anchors_size'], _global_dict['max_anchors_size']),
    #                            (_global_dict['min_anchors_size'], _global_dict['min_anchors_size'])]


def set_value(key, value):
    """ 定义一个全局变量 """
    global _global_dict
    _global_dict[key] = value


def update_global(config, type):
    """ 定义一个全局变量 """
    global _global_dict
    _global_dict['max_anchors_size'] = int(config[type]['imageSize'])
    _global_dict['min_anchors_size'] = int(config[type]['imageSize'])
    _global_dict['stride'] = int(config[type]['stride'])
    _global_dict['anchors'] = [(_global_dict['max_anchors_size'], _global_dict['max_anchors_size']),
                               (_global_dict['min_anchors_size'], _global_dict['min_anchors_size'])]


def get_value(key, defValue=None):
    global _global_dict
    return _global_dict[key]


def update_glob():
    opt = parser.parse_args()

    if not os.path.exists("../config/%s.yaml" % opt.config):
        print("../config/%s.yaml not found." % opt.config)
        exit()
    f = open("../config/%s.yaml" % opt.config, 'r', encoding='utf-8')
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    # print(config)

    # _init()
    update_global(config, opt.config_type)
    print(_global_dict)


if __name__ == "__main__":
    update_glob()