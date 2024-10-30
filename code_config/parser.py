
"""this fucntion parser is used to analyze the configuration file to get an opt"""

import os
import json
from pathlib import Path
from datetime import datetime
from code_util.util import deep_update

config_root = "./file_config"

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    """ convert to NoneDict, which return None for missing key. """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def dict2str(opt, indent_l=1):
    """ dict to string for logger """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

def parse_json_file(json_path):
    json_str = ""
    with open(json_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str)
    return opt

def parse(status = "train"):
    base_config = os.path.join(config_root,"base_config.json")
    base_opt = parse_json_file(base_config)

    if status == "train":
        status_config = os.path.join(config_root,"base_train.json")  
    else:
        status_config = os.path.join(config_root,"base_test.json")
    status_opt = parse_json_file(status_config)
    final_opt = deep_update(base_opt,status_opt)
    experiment_config = os.path.join(config_root,"experiment_config.json")
    experiment_opt = parse_json_file(experiment_config)
    experiment_opt = experiment_opt[experiment_opt["work_now"]]
    final_opt = deep_update(final_opt,experiment_opt["general"])
    final_opt = deep_update(final_opt,experiment_opt[status])
    
    # return dict_to_nonedict(final_opt)
    return final_opt

