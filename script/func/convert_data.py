'''
featuresからlightgbmに突っ込むデータを作る
'''

from features_create import load_datasets
import os
import sys
import pandas as pd

sys.path.append(os.environ["FUNC"])


def GetData(feature_dir, feature_col):
    train, test = load_datasets(feature_dir, feature_col)
    data = pd.concat([train, test]).reset_index()
    # print(data.columns)

    # 不要なカラム削除
    # data = data.drop(["index", "day", "hour", "month", "year", \
    #     "lobby-mode", "game-ver", "lobby", "mode"], axis=1)

    # data.fillna(-1, inplace=True)
    data_col = data.columns

    # weaponはユニーク数が多いので削除
    for col in data_col:
        if "weapon" in col:
            data.drop(col, axis=1, inplace=True)

    # ダミー変数に変換
    dummy_data = pd.get_dummies(data)

    train = dummy_data.loc[:train.shape[0] - 1]
    test = dummy_data.loc[train.shape[0]:].drop("y", axis=1)
    return train, test
