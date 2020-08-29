from null_importance import cv_mean_test_score, _cross_validate
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold

import lightgbm as lgb


sys.path.append(os.environ["FUNC"])


class LGBmodel():
    def __init__(self, X_train, y_train, params, log_name):
        self.n_splits = 5

        self.X_train = X_train
        self.y_train = y_train
        self.params = params
        self.log_name = log_name
        self.booster = [] #乱数を変えて学習したcv_boosterを入れる

        self.result_path = os.environ["LOG"]

    def train(self):
        for random_num in range(1, 6):
            # kfold
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=random_num)
            
            #lgbmで交差検証
            cv_booster = _cross_validate(self.X_train, self.y_train, kf, self.params)

            # 平均スコア
            test_score = cv_mean_test_score(self.X_train, self.y_train, kf, self.params)
            print(f"平均score:{test_score}")
            self.booster.append(cv_booster)

        # # 平均スコアとパラメータを保存
        # if os.path.exists(self.result_path + "/conditions.csv"):
        #     result_df = pd.read_csv(self.result_path + "/conditions.csv")
        #     params_df = pd.DataFrame.from_dict(self.params, orient="index").T
        #     params_df["log_name"] = self.log_name
        #     params_df["score"] = test_score
        #     conc_df = pd.concat([result_df, params_df])
        #     conc_df.to_csv(self.result_path + "/conditions.csv", index=False)
        # else:
        #     params_df = pd.DataFrame.from_dict(self.params, orient="index").T
        #     params_df["log_name"] = self.log_name
        #     params_df["score"] = test_score
        #     params_df.to_csv(self.result_path + "/conditions.csv", index=False)

    def getpred(self, test):
        # 予測
        output_dir = os.environ["OUTPUT"]
        sub = pd.DataFrame({"id": range(test.shape[0])})
        
        for i in range(1, 6):
            sub[f"y{i}"] = np.array(self.booster[i-1].predict(test)).mean(axis=0)

        sub["y"] = np.round(sub[["y1", "y2", "y3", "y4", "y5"]].mean(axis=1)).astype("int")
        sub[["id", "y"]].to_csv(output_dir + f"/{self.log_name}.csv", index=False)
        print(sub.head())
        print("Finish!")
        return sub

    # def importance(self):
    #     if not os.path.exists(self.result_path+"/"+self.log_name):
    #         os.makedirs(self.result_path+"/"+self.log_name)
    #     # 学習モデルから特徴量の重要度を取り出す
    #     raw_importances = self.cv_booster.feature_importance(
    #         importance_type='gain')

    #     # 特徴量の名前
    #     feature_name = self.cv_booster.boosters[0].feature_name()
    #     importance_df = pd.DataFrame(data=raw_importances,
    #                                  columns=feature_name)
    #     # 平均値でソートする
    #     sorted_indices = importance_df.mean(
    #         axis=0).sort_values(
    #         ascending=False).index
    #     sorted_importance_df = importance_df.loc[:, sorted_indices]

    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     ax.grid()
    #     ax.set_xscale('log')
    #     ax.set_ylabel('Feature')
    #     ax.set_xlabel('Importance')
    #     sns.boxplot(data=sorted_importance_df.iloc[:, :30],
    #                 orient='h',
    #                 ax=ax)
    #     # plt.show()
    #     fig.savefig(self.result_path + f"/{self.log_name}/importance.png")
