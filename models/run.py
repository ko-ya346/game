import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.environ["FUNC"])
from convert_data import GetData
from lgbm_ensemble import LGBmodel
from read_yml import read_params
from pred_logistic import get_lr_pred

output_dir = os.environ["OUTPUT"]

train, test = GetData("20200825", "feature_col")
print(train.shape)
print(test.shape)
feature = train.drop("y", axis=1)
target = train["y"]

params = read_params()

#ロジスティック回帰による予測値
lr_pred_df = get_lr_pred()

for key, par in params.items():
    model = LGBmodel(feature, target, par, f"0827_ensemble_{key}")
    model.train()
    # model.importance()
    pred_df = model.getpred(test)

    pred_df = pd.concat([pred_df, lr_pred_df], axis=1)
    pred_df["y"] = np.round(pred_df.drop("id", axis=1).mean(axis=1)).astype("int")
    pred_df[["id", "y"]].to_csv(output_dir + f"/{key}_ensumble.csv", index=False)