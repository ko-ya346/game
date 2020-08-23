import pandas as pd

import os

input_dir = os.environ["INPUT"]
feature_dir = os.environ["FEATURE"]

train = pd.read_csv(input_dir+"/train_data.csv")

train[["y"]].to_feather(feature_dir+"/train/y.ftr")