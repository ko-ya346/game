from convert_data import GetData
import logging
import sys
import os

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt

sys.path.append(os.environ["FUNC"])


LOGGER = logging.getLogger(__name__)


class ModelExtractionCallback(object):
    """see: https://blog.amedama.jp/entry/lightgbm-cv-model"""

    def __init__(self):
        self._model = None

    def __call__(self, env):
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            raise RuntimeError('callback has not called yet')

    @property
    def cvbooster(self):
        self._assert_called_cb()
        return self._model


def _cross_validate(train_x, train_y, folds, params):
    """LightGBM で交差検証する関数"""
    lgb_train = lgb.Dataset(train_x, train_y)

    model_extraction_cb = ModelExtractionCallback()
    callbacks = [model_extraction_cb]

    lgb.cv(params, lgb_train, folds=folds,
           early_stopping_rounds=100, callbacks=callbacks, verbose_eval=100)
    return model_extraction_cb.cvbooster


def _predict_oof(cv_booster, train_x, train_y, folds):
    """学習済みモデルから Out-of-Fold Prediction を求める"""
    split = folds.split(train_x, train_y)
    fold_mappings = zip(split, cv_booster.boosters)
    oof_y_pred = np.zeros_like(train_y, dtype=float)
    for (_, val_index), booster in fold_mappings:
        val_train_x = train_x.iloc[val_index]
        y_pred = booster.predict(val_train_x,
                                 num_iteration=cv_booster.best_iteration)
        oof_y_pred[val_index] = y_pred
    return oof_y_pred


def cv_mean_feature_importance(train_x, train_y, folds, params):
    """交差検証したモデルを使って特徴量の重要度を計算する"""
    cv_booster = _cross_validate(train_x, train_y, folds, params)
    importances = cv_booster.feature_importance(importance_type='gain')
    mean_importance = np.mean(importances, axis=0)
    return mean_importance


def cv_mean_test_score(train_x, train_y, folds, params):
    """交差検証で OOF Prediction の平均スコアを求める"""
    cv_booster = _cross_validate(train_x, train_y, folds, params)
    # OOF Pred を取得する
    oof_y_pred = _predict_oof(cv_booster, train_x, train_y, folds)

    test_score = roc_auc_score(train_y, oof_y_pred)
    return test_score


def main():
    train, _ = GetData("20200825", "feature_col")
    train_x = train.drop("y", axis=1)
    train_y = train["y"]

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    n_cols = 100

    lgbm_params = {
        'objective': 'binary',
        'metric': "binary_logloss",
        'verbose': -1}

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    LOGGER.info('Starting base importance calculation')
    base_importance = cv_mean_feature_importance(
        train_x, train_y, folds, lgbm_params)

    LOGGER.info('Starting null importance calculation')
    TRIALS_N = 20
    null_importances = []
    for _ in tqdm(range(TRIALS_N)):
        train_y_permuted = np.random.permutation(train_y)
        null_importance = cv_mean_feature_importance(
            train_x, train_y_permuted, folds, lgbm_params)
        null_importances.append(null_importance)
    null_importances = np.array(null_importances)

    criterion_percentile = 50
    percentile_null_imp = np.percentile(
        null_importances, criterion_percentile, axis=0)
    null_imp_score = base_importance / (percentile_null_imp + 1e-6)
    sorted_indices = np.argsort(null_imp_score)[::-1]

    # 上位 N% の特徴量を使って性能を比較してみる
    use_feature_importance_top_percentages = [
        100, 90, 80, 75, 60, 50, 25, 20, 15, 10, 8, 5, 1]

    mean_test_scores = []
    percentile_selected_cols = []
    for percentage in use_feature_importance_top_percentages:
        sorted_columns = train_x.columns[sorted_indices]
        num_of_features = int(n_cols * percentage / 100)
        selected_cols = sorted_columns[:num_of_features]
        selected_train_x = train_x[selected_cols]
        LOGGER.info(f'Null Importance score TOP {percentage}%')
        LOGGER.info(f'selected features: {list(selected_cols)}')
        LOGGER.info(f'selected feature len: {len(selected_cols)}')
        percentile_selected_cols.append(selected_cols)

        mean_test_score = cv_mean_test_score(
            selected_train_x, train_y, folds, lgbm_params)
        LOGGER.info(f'mean test_score: {mean_test_score}')
        mean_test_scores.append(mean_test_score)

    _, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(mean_test_scores, color='b', label='mean test score')
    ax1.set_xlabel('Importance TOP n%')
    ax1.set_ylabel('mean test score')
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot([len(cols) for cols in percentile_selected_cols],
             color='r', label='selected features len')
    ax2.set_ylabel('selected features len')
    ax2.legend()
    plt.xticks(range(len(use_feature_importance_top_percentages)),
               use_feature_importance_top_percentages)
    plt.show()


if __name__ == "__main__":
    main()
