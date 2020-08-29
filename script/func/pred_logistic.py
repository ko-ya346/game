import os
import sys
sys.path.append(os.environ["FUNC"])
from features_create import load_datasets
import pandas as pd

from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

from sklearn.metrics import accuracy_score

def get_lr_pred():
    train, test = load_datasets("20200825", "feature_col")
    data = pd.concat([train, test]).reset_index()


    for col in data.columns:
        if data[col].dtypes=="object" or col=="y":
            continue
        data[col] = scale(data[col])

    # weaponはユニーク数が多いので削除
    for col in data.columns:
        if "weapon" in col:
            data.drop(col, axis=1, inplace=True)

    dummy_data = pd.get_dummies(data)

    train = dummy_data.loc[:train.shape[0] - 1]
    test = dummy_data.loc[train.shape[0]:].drop("y", axis=1)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    pred_dic = {}
    X_train = train.drop("y", axis=1)
    y_train = train["y"]

    for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
        print(f'fold{fold_ + 1} start')

        train_x =  X_train.iloc[train_index]
        valid_x =  X_train.iloc[valid_index]
        train_y =  y_train.iloc[train_index]
        valid_y =  y_train.iloc[valid_index]

        lr = LogisticRegression()
        lr.fit(train_x, train_y)

        y_pred = lr.predict(valid_x)
        pred_score = accuracy_score(valid_y, y_pred)
        print(f"pred_score: {pred_score}")
        pred_test = lr.predict_proba(test).T
        pred_dic[f"lr_y{fold_+1}"] = pred_test[1]

    pred_df = pd.DataFrame(pred_dic)
    print(pred_df.shape)
    return pred_df 

if __name__ == "__main__":
    get_lr_pred()