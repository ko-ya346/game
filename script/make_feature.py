

from func.features_create import Feature
import pandas as pd
import numpy as np
import json

import os


class MakeFeature(Feature):
    def create_features(self):
        self.train[self.column] = train[self.column]
        self.test[self.column] = test[self.column]


input_dir = os.environ["INPUT"]
output_dir = os.environ["OUTPUT"]
config_dir = os.environ["CONFIGS"]

train = pd.read_csv(input_dir + "/train_data.csv")
test = pd.read_csv(input_dir + "/test_data.csv")
weapon = pd.read_csv(input_dir + "/statink-weapon2.csv")[
    ["category1", "category2", "key", "splatnet", "mainweapon", "subweapon", "special"]]

# rankのuniqueから作ったウデマエ対応表
# s+は0~9の範囲なので平均の15, xはs+10以上なので最小値の20
rank_dic = {"N": 0, "c-": 1, "c": 2, "c+": 3, "b-": 4, "b": 5,
            "b+": 6, "a-": 7, "a": 8, "a+": 9, "s": 10, "s+": 15, "x": 20}

data = [train, test]
df_tmp = []
epsiron = 0.000000001

for df in data:
    # 対戦した時間帯
    df["period"] = pd.to_datetime(df["period"])
    df["year"] = df["period"].dt.year
    df["month"] = df["period"].dt.month
    df["day"] = df["period"].dt.day
    df["day_of_week"] = df["period"].dt.dayofweek
    df["hour"] = df["period"].dt.hour

    # rank、levelを集計
    for team in ["A", "B"]:
        # 各チームのweapon欠損数（欠員）をカウント
        df_team = df[[f"{team}{i}-weapon" for i in range(1, 5)]]
        df[f"{team}_absence"] = df_team.isnull().sum(axis=1)
        #rankの欠損数をカウント
        df_team = df[[f"{team}{i}-rank" for i in range(1, 5)]]
        df[f"{team}_rank_null"] = df_team.isnull().sum(axis=1)

        for i in range(1, 5):
            # rankの欠損値埋め
            df[f"{team}{i}-rank"] = df[f"{team}{i}-rank"].fillna("N")
            # 数値に変換
            df[f"{team}{i}-rank_num"] = df[f"{team}{i}-rank"].map(rank_dic)

            # 武器情報をmerge
            df = pd.merge(df, weapon, how="left",
                          left_on=f"{team}{i}-weapon", right_on="key",
                          suffixes=(f'_{team}{i-1}', f'_{team}{i}'))

        # 各チームの武器のカテゴリ数
        for cat in ["category1", "category2"]:
            for wep in weapon[cat].unique():
                wep_cat_cnt = np.zeros(df.shape[0])
                for i in range(1, 5):
                    wep_cat_cnt += df[f"{cat}_{team}{i}"] == wep
                df[f"{team}_{cat}_{wep}_cnt"] = wep_cat_cnt

        for col_name in ["rank_num", "level", "splatnet"]:
            # 各チームのcol_nameのdataframe
            if col_name == "splatnet":
                df_col_name = df[[
                    f"{col_name}_{team}{i}" for i in range(1, 5)]]
            else:
                df_col_name = df[[
                    f"{team}{i}-{col_name}" for i in range(1, 5)]]
            df_col_name.fillna(0, inplace=True)

            # 各種統計量
            df[f"{team}_{col_name}_sum"] = df_col_name.sum(axis=1)
            df[f"{team}_{col_name}_mean"] = df_col_name.mean(axis=1)
            df[f"{team}_{col_name}_std"] = df_col_name.std(axis=1)
            df[f"{team}_{col_name}_median"] = df_col_name.median(axis=1)
            df[f"{team}_{col_name}_max"] = df_col_name.max(axis=1)
            df[f"{team}_{col_name}_min"] = df_col_name.min(axis=1)
            df[f"{team}_{col_name}_diff"] = df[f"{team}_{col_name}_max"] - \
                df[f"{team}_{col_name}_min"]

    for col_name in ["rank_num", "level", "splatnet"]:
        # チーム間の統計量比較
        df[f"{col_name}_max_diff"] = df[f"A_{col_name}_max"] - \
            df[f"B_{col_name}_max"]
        df[f"{col_name}_min_diff"] = df[f"A_{col_name}_min"] - \
            df[f"B_{col_name}_min"]
        df[f"{col_name}_max_per"] = df[f"A_{col_name}_max"] / \
            (df[f"B_{col_name}_max"]+epsiron)
        df[f"{col_name}_min_per"] = df[f"A_{col_name}_min"] / \
            (df[f"B_{col_name}_min"]+epsiron)


        df[f"{col_name}_mean_diff"] = df[f"A_{col_name}_mean"] - \
            df[f"B_{col_name}_mean"]
        df[f"{col_name}_mean_per"] = df[f"A_{col_name}_mean"] / \
            (df[f"B_{col_name}_mean"]+epsiron)
        df[f"{col_name}_median_diff"] = df[f"A_{col_name}_median"] - \
            df[f"B_{col_name}_median"]
        df[f"{col_name}_median_per"] = df[f"A_{col_name}_median"] / \
            (df[f"B_{col_name}_median"]+epsiron)

    df.drop(["period", \
             # 'A1-rank_num', 'A2-rank_num', 'A3-rank_num', 'A4-rank_num', \
             # 'B1-rank_num', 'B2-rank_num', 'B3-rank_num', 'B4-rank_num', \
             'A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon', \
             'B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon', \
             #    'A1-rank', 'A1-level', 'A2-rank', 'A2-level',\
             #  'A3-rank', 'A3-level',  'A4-rank', 'A4-level',\
             #      'B1-rank', 'B1-level', 'B2-rank', 'B2-level',\
             #         'B3-rank', 'B3-level', 'B4-rank', 'B4-level',\
             'key_A1', 'key_A2', 'key_A3', 'key_A4', 'key_B1', 'key_B2', 'key_B3', 'key_B4',\
             #         'category2_A1', 'category2_A2', 'category2_A3', 'category2_A4', 'category2_B1', \
             # 'category2_B2', 'category2_B3', 'category2_B4', \
             #     'category1_A1', 'splatnet_A1', 'category1_A2', 'splatnet_A2', 'category1_A3',\
             #             'splatnet_A3', 'category1_A4', 'splatnet_A4', 'category1_B1', \
             #                     'splatnet_B1', 'category1_B2', 'splatnet_B2', 'category1_B3', 'splatnet_B3', 'category1_B4','splatnet_B4' \
             ], axis=1, inplace=True)
    #欠損値を埋める
    for col in ["special", "category1", "category2", "splatnet", "subweapon", "mainweapon", "level"]:
        for i in [3, 4]:
            for team in ["A", "B"]:
                if col=="level":
                    df[f"{team}{i}-{col}"] = df[f"{team}{i}-{col}"].fillna(0)
                else:
                    if df[f"{col}_{team}{i}"].dtypes=="object":
                        df[f"{col}_{team}{i}"] = df[f"{col}_{team}{i}"].fillna("0")
                    else:
                        df[f"{col}_{team}{i}"] = df[f"{col}_{team}{i}"].fillna(0)
    df_tmp.append(df)

train = df_tmp[0]
test = df_tmp[1]

col_dic = {"feature": [col for col in test.columns], "target": "y"}
print(col_dic)

for col in col_dic["feature"]:
    MakeFeature(col, "20200825").run().save()

with open(config_dir + "/feature_col.json", "w") as f:
    json.dump(col_dic, f, indent=4)

print(train.isnull().sum().reset_index().sort_values(0, ascending=False))
print(test.isnull().sum().reset_index().sort_values(0, ascending=False))
