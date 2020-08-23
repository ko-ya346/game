import yaml
import os

def read_params():
    dir_path = os.environ["CONFIGS"]

    path = dir_path + "/params.yml"
    with open(path,"r",encoding="utf-8") as f:
        params = yaml.load(stream=f, Loader=yaml.SafeLoader)
    
    return params