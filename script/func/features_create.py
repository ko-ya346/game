import os
import json
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

def load_datasets(dir_name, json_name):
    feature_dir = os.environ["FEATURE"]+f"/{dir_name}"
    configs_dir = os.environ["CONFIGS"]+f"/{json_name}.json"
    with open(configs_dir, "r") as f:
        col_dic = json.load(f)
    feature_col = col_dic["feature"]
    print(feature_col)

    dfs = [pd.read_feather(feature_dir+f"/train/{i}.ftr") for i in feature_col]
    train = pd.concat(dfs, axis=1)
    train = train.set_index("id")
    target_col = pd.read_csv(os.environ["INPUT"]+"/train_data.csv")[["y"]]
    train = pd.concat([train, target_col], axis=1)

    dfs = [pd.read_feather(feature_dir+f"/test/{i}.ftr") for i in feature_col]
    test = pd.concat(dfs, axis=1)
    test = test.set_index("id")
    return train, test


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = os.environ["FEATURE"]
    
    def __init__(self, column, dir_name):
        self.column = column
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.feature_dir = Path(self.dir) / f"{dir_name}"
        if not os.path.exists(Path(self.feature_dir)):
            os.makedirs(self.feature_dir/"train")
            os.makedirs(self.feature_dir/"test")
        self.train_path = Path(self.dir) / f'{dir_name}/train/{self.column}.ftr'
        self.test_path = Path(self.dir) / f'{dir_name}/test/{self.column}.ftr'
    
    def run(self):
        with timer(self.column):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))