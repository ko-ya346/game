import os
import sys
import pandas as pd

sys.path.append(os.environ["FUNC"])
from features_create import load_datasets
from lgbm import LGBmodel
from read_yml import read_params


input_dir = os.environ["INPUT"]
output_dir = os.environ["OUTPUT"]

train, test = load_datasets("20200822_2", "feature_col0822_2")

params = read_params()

data = pd.concat([train, test]).reset_index()
print(data.columns)
data = data.drop(["index", "day", "stage", "hour", "month", "year", \
    "lobby-mode", "game-ver", "lobby", "mode"], axis=1)

data.fillna(-1, inplace=True)
data_col = data.columns
# for col in data_col:
#     if "weapon" in col:
#         print(col)
#         data.drop(col, axis=1, inplace=True)

dummy_data = pd.get_dummies(data)

new_train = dummy_data.loc[:train.shape[0]-1]
new_test = dummy_data.loc[train.shape[0]:].drop("y", axis=1)


feature = new_train.drop("y", axis=1)
target = new_train["y"]

for key, par in params.items():
    model = LGBmodel(feature, target, par, f"param_set{key}")
    model.fit()
    model.importance()
    model.curve()
    model.Getpred(new_test)