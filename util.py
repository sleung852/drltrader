import numpy as np
import json
from pathlib import Path

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def linear(x):
    return np.array(x)/np.array(x).sum()

def str_to_list(x):
    assert isinstance(x, str)
    return x.split('_')

def save_config(config_dict, out):
    assert isinstance(config_dict, dict)
    file = open(out, "w+")
    json.dump(config_dict, file)  
    file.close()
    
def load_config(path):
    file = open(path, "r")
    config_dict = file.read()
    file.close()
    return json.loads(config_dict)

def check_and_create_folder( path):
    Path(path).mkdir(parents=True, exist_ok=True)
    
