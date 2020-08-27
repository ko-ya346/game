import os
import sys

from numpy.core.numeric import asanyarray

sys.path.append(os.environ["FUNC"])
from convert_data import GetData
from lgbm import LGBmodel
from read_yml import read_params

train, test = GetData("20200825", "feature_col")
print(train.shape)
print(test.shape)
feature = train.drop("y", axis=1)
target = train["y"]

params = read_params()

for key, par in params.items():
    model = LGBmodel(feature, target, par, f"0825_{key}")
    model.train()
    model.importance()
    # model.getpred(test)
