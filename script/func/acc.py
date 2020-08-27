import numpy as np

def accuracy(preds, data):
    """精度 (Accuracy) を計算する関数"""
    # 正解ラベル
    y_true = data.get_label()
    y_pred = np.round(preds)
    acc = np.mean(y_true == y_pred)
    
    return 'accuracy', acc, True