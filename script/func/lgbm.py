import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import lightgbm as lgb

sys.path.append(os.environ["FUNC"])
from lgbm_log import lgbm_logger

from base_log import create_logger


class LGBmodel():
    def __init__(self, X_train, y_train, params, log_name):
        self.X_train = X_train
        self.y_train = y_train
        self.log_name = log_name
        self.pred = [] #各predのスコアを格納
        self.score = []
        self.models = [] #学習モデルを格納
        self.evals_result = [{} for _ in range(3)] #学習曲線の出力用
        self.params = params
        self.metric = self.params['metric']
        self.result_path = os.environ["LOG"]

    def fit(self):
        kf = KFold(n_splits=3, shuffle=True, random_state=1)

        for fold_, (train_index, valid_index) in enumerate(kf.split(self.X_train, self.y_train)):
            print(f'fold{fold_ + 1} start')
            create_logger(self.log_name+f"{fold_+1}")

            train_x = self.X_train.iloc[train_index]
            valid_x = self.X_train.iloc[valid_index]
            train_y = self.y_train.iloc[train_index]
            valid_y = self.y_train.iloc[valid_index]
        
            # lab.Datasetを使って、trainとvalidを作っておく
            lgb_train= lgb.Dataset(train_x, train_y)
            lgb_valid = lgb.Dataset(valid_x, valid_y)
            
            gbm = lgb.train(params=self.params, train_set=lgb_train,
                            valid_sets=[lgb_train, lgb_valid],
                            evals_result=self.evals_result[fold_], 
                            early_stopping_rounds=100,
                            verbose_eval=50,
                            callbacks=[lgbm_logger(self.log_name+f"{fold_+1}")]
            )
            y_pred = np.round(gbm.predict(valid_x))
            # print(y_pred)
            # print(valid_y)
            pred_score = accuracy_score(valid_y, y_pred)
            print(f"pred_score: {pred_score}")
            self.score.append(pred_score)
            self.models.append(gbm)
        mean_score = np.mean(self.score)
        print(f"平均score:{mean_score}")

        if os.path.exists(self.result_path+"/conditions.csv"):
            result_df = pd.read_csv(self.result_path+"/conditions.csv")
            params_df = pd.DataFrame.from_dict(self.params, orient="index").T
            params_df["log_name"] = self.log_name
            params_df["score"] = mean_score
            conc_df = pd.concat([result_df, params_df])
            conc_df.to_csv(self.result_path+"/conditions.csv", index=False)
        else:
            params_df = pd.DataFrame.from_dict(self.params, orient="index").T
            params_df["log_name"] = self.log_name
            params_df["score"] = mean_score
            params_df.to_csv(self.result_path+"/conditions.csv", index=False)
    
    def Getpred(self, test):
        output_dir = os.environ["OUTPUT"]
        sub = pd.DataFrame({"id":range(test.shape[0])})
        pred = np.zeros((3, test.shape[0]))

        for i, m in enumerate(self.models):
            pred_ = m.predict(test)
            pred[i, :] = pred_
        print(sub.shape)
        sub["y"] = np.round(np.mean(pred.T, axis=1)).astype("int")

        sub.to_csv(output_dir+f"/{self.log_name}.csv", index=False)
        print("Finish!") 

    def curve(self):
        #学習の様子をプロット
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            ax[i].plot(self.evals_result[i]["training"][self.metric], label="train_loss")
            ax[i].plot(self.evals_result[i]['valid_1'][self.metric], label="valid_loss")
            ax[i].set_title(f"{i+1}fold")
            ax[i].legend()
        fig.savefig(self.result_path+f"/{self.log_name}/curve.png")
        # plt.show()

    
    def importance(self):
        importance = pd.DataFrame(self.models[0].feature_importance(), \
            index=self.X_train.columns, columns=["importance"])
        # print(importance.head())
        importance = importance.sort_values("importance", ascending=1)

        fig = plt.figure(figsize=(20, 50))
        plt.barh(importance.index, importance["importance"])
        fig.savefig(self.result_path+f"/{self.log_name}/importance.png")
        # plt.show()
        importance.to_csv(self.result_path+f"/{self.log_name}/importance.csv")
