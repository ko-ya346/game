import pandas as pd
import numpy as np
import json

import os
import sys
sys.path.append(os.environ["FUNC"])
from features_create import Feature

class MakeFeature(Feature):
    def create_features(self):
        self.train[self.column] = train[self.column]
        self.test[self.column] = test[self.column]

input_dir = os.environ["INPUT"]
output_dir = os.environ["OUTPUT"]
config_dir = os.environ["CONFIGS"]

train = pd.read_csv(input_dir+"/train_data.csv")
test = pd.read_csv(input_dir+"/test_data.csv")
weapon = pd.read_csv(input_dir+"/statink-weapon2.csv")[["category1", "category2", "key", "splatnet"]]

#rankのuniqueから作ったウデマエ対応表
#s+は0~9の範囲なので平均の15, xはs+10以上なので最小値の20
rank_dic = {"N":0, "c-":1, "c":2, "c+":3, "b-":4, "b":5, \
    "b+":6, "a-":7, "a":8, "a+":9, "s":10, "s+":15, "x":20}

data = [train, test]
df_tmp = []

for df in data:
    #対戦した時間帯
    df["period"] = pd.to_datetime(df["period"])
    df["year"] = df["period"].dt.year
    df["month"] = df["period"].dt.month
    df["day"] = df["period"].dt.day
    df["day_of_week"] = df["period"].dt.dayofweek
    df["hour"] = df["period"].dt.hour
    # df.drop("period", axis=1, inplace=True)

    #rank,、levelを集計
    for team in ["A", "B"]:
        for i in range(1, 5):
            #rankの欠損値埋め
            df[f"{team}{i}-rank"] = df[f"{team}{i}-rank"].fillna("N")
            #数値に変換
            df[f"{team}{i}-rank_num"] = df[f"{team}{i}-rank"].map(rank_dic)
            # df.drop(f"{team}{i}-rank", axis=1, inplace=True)

            df = pd.merge(df, weapon, how="left", \
                left_on=f"{team}{i}-weapon", right_on="key", \
                    suffixes=(f'_{team}{i-1}', f'_{team}{i}'))

        #各チームのrank_numのdataframe
        df_team_rank = df[[f"{team}{i}-rank_num" for i in range(1, 5)]]
        #各種統計量
        # df[f"{team}_rank_num_sum"] = df_team_rank.sum(axis=1)
        # df[f"{team}_rank_num_mean"] = df_team_rank.mean(axis=1)
        # df[f"{team}_rank_num_std"] = df_team_rank.std(axis=1)
        # df[f"{team}_rank_num_median"] = df_team_rank.median(axis=1)
        # df[f"{team}_rank_num_max"] = df_team_rank.max(axis=1)
        # df[f"{team}_rank_num_min"] = df_team_rank.min(axis=1)
        # df[f"{team}_rank_num_diff"] = df[f"{team}_rank_num_max"]-df[f"{team}_rank_num_min"]

        #levelの統計量
        df_team_level = df[[f"{team}{i}-level" for i in range(1, 5)]]
        # df[f"{team}_level_sum"] = df_team_level.sum(axis=1)
        df[f"{team}_level_mean"] = df_team_level.mean(axis=1)
        # df[f"{team}_level_std"] = df_team_level.std(axis=1)
        df[f"{team}_level_median"] = df_team_level.median(axis=1)
        # df[f"{team}_level_max"] = df_team_level.max(axis=1)
        # df[f"{team}_level_min"] = df_team_level.min(axis=1)
        # df[f"{team}_level_diff"] = df[f"{team}_level_max"]-df[f"{team}_level_min"]
        
        # for cat in ["category1"]:
        #     for wep in weapon[cat].unique():
        #         wep_cat_cnt = np.zeros(df.shape[0])
        #         for i in range(1, 5):
        #             wep_cat_cnt += df[f"{cat}_{team}{i}"]==wep
        #         df[f"{team}_{cat}_{wep}_cnt"] = wep_cat_cnt

        df_team_splatnet = df[[f"splatnet_{team}{i}" for i in range(1, 5)]]
        # df[f"{team}_splatnet_sum"] = df_team_splatnet.sum(axis=1)
        df[f"{team}_splatnet_mean"] = df_team_splatnet.mean(axis=1)
        # df[f"{team}_splatnet_std"] = df_team_splatnet.std(axis=1)
        df[f"{team}_splatnet_median"] = df_team_splatnet.median(axis=1)

    df["splatnet_mean_diff"] = df["A_splatnet_mean"]-df["B_splatnet_mean"]
    df["splatnet_median_diff"] = df["A_splatnet_median"]-df["B_splatnet_median"]
    df["level_mean_diff"] = df["A_level_mean"]-df["B_level_mean"]
    df["level_median_diff"] = df["A_level_median"]-df["B_level_median"]

    df.drop(["period", \
        # 'A1-rank_num', 'A2-rank_num', 'A3-rank_num', 'A4-rank_num', \
        # 'B1-rank_num', 'B2-rank_num', 'B3-rank_num', 'B4-rank_num', \
       'A1-weapon', 'A2-weapon', 'A3-weapon', 'A4-weapon', \
    'B1-weapon', 'B2-weapon', 'B3-weapon', 'B4-weapon', \
        #    'A1-rank', 'A1-level', 'A2-rank', 'A2-level',\
        #  'A3-rank', 'A3-level',  'A4-rank', 'A4-level',\
        #      'B1-rank', 'B1-level', 'B2-rank', 'B2-level',\
        #         'B3-rank', 'B3-level', 'B4-rank', 'B4-level',\
                'category2_A1', 'key_A1', 'category2_A2', 'key_A2',
                'category2_A3', 'key_A3', 'category2_A4', 'key_A4', 'category2_B1', 'key_B1',\
                'category2_B2', 'key_B2', 'category2_B3', 'key_B3', 'category2_B4', 'key_B4',
                    # 'category1_A1', 'splatnet_A1', 'category1_A2', 'splatnet_A2', 'category1_A3',\
                    #         'splatnet_A3', 'category1_A4', 'splatnet_A4', 'category1_B1', \
                    #                 'splatnet_B1', 'category1_B2', 'splatnet_B2', 'category1_B3', 'splatnet_B3', 'category1_B4','splatnet_B4' \
                    ], axis=1, inplace=True)
    df_tmp.append(df)

train, test = df_tmp

col_dic = {"feature":[col for col in test.columns], "target":"y"}
print(col_dic)

for col in col_dic["feature"]:
    MakeFeature(col, "20200822_2").run().save()

with open(config_dir+"/feature_col0822_2.json", "w") as f:
    json.dump(col_dic, f, indent=4)
