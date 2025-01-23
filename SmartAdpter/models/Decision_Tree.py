import time
import os
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime

import numpy as np
sys.path.append("..")

from utils.load_MMdata import get_train_data
from utils.load_MMdata import get_test_data

from utils.SSpMV_setting import *

from compute_metrics import get_acc_new
from compute_metrics import get_precision_new


def train_decision_tree(features, labels):
    model = DecisionTreeClassifier(
        criterion='gini',              # 使用基尼指数进行划分
        max_depth=15,                  # 最大深度设置为15
        min_samples_split=2,           # 分割内部节点所需的最小样本数
        min_samples_leaf=1,            # 叶节点的最小样本数
        min_impurity_decrease=0.005,   # 不纯度减少的最小阈值
        max_leaf_nodes=None            # 最大叶节点数，None表示无限制
    )
    model.fit(features, labels)
    return model


def evaluate_model(model, features, labels):
    # 使用模型进行预测
    predictions = model.predict(features)

    # 计算宏观精确率、宏观召回率和宏观F1分数
    macro_precision = precision_score(labels, predictions, average='macro')
    macro_recall = recall_score(labels, predictions, average='macro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    return accuracy, macro_precision, macro_recall, macro_f1

def main():
    training_data_list = "train_genlist.txt"
    val_data_list = "val_genlist.txt"
    test_data_list = "test_list.txt"
    
    settings_idx = 1
    label_class = [".format_label", ".det_format_label"]
    print ("Running D-Tree with the setting: [{}]".format(settings_idx))
    # 读取train data
    image_array, RB_array, CB_array, feat_array, label_array = get_train_data(training_data_list, label_class[settings_idx])  # 确保此函数返回正确的数据
    
    # 读取validation data
    val_image_array, val_RB_array, val_CB_array, val_feat_array, val_label_array = get_train_data(val_data_list, label_class[settings_idx])

    # 训练决策树模型
    tree_model = train_decision_tree(feat_array, label_array)

    # 评估模型
    accuracy, precision, recall, f1 = evaluate_model(tree_model, val_feat_array, val_label_array)
    
    # 打印或记录评估结果
    print(f"Model Accuracy: {accuracy}")
    print(f"Macro Precision: {precision}")
    print(f"Macro Recall: {recall}")
    print(f"Macro F1 Score: {f1}")

if __name__ == "__main__":
    main()
