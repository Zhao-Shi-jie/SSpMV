import time
import os
import sys

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten

import numpy as np

sys.path.append("..")

from utils.load_data import get_train_data
from utils.load_data import get_test_data
from utils.data_setting import *

from wide_and_deep_com_concat import train_model
from wide_and_deep_com_concat import train_model_concat
from wide_and_deep_com_concat import train_model_add
from new_wide_deep_add import train_new_model_add
from new_wide_deep_concat import train_new_model_concat
from new_wide import train_new_wide_model
from new_wide_deep import train_new_model_wide_deep


from baseline import train_baseline_model
from deep import train_deep_model
from wide import train_wide_model
from train_xgboost import train_xbg

from evaluate import evaluate_new_wide_deep_add_model
from evaluate import evaluate_new_wide_deep_concat_model
from evaluate import evaluate_model
from evaluate import evaluate_xgboost
from evaluate import evaluate_wide_model
from evaluate import evaluate_deep_model
from evaluate import evaluate_baseline_hart
from evaluate import evaluate_new_wide_model
from evaluate import evaluate_new_wide_deep_model

from compute_metrics import get_acc
from compute_metrics import get_precision

def train_and_get_res():
  training_data_list = "/home/xionghantao/codes/works/JPDC_special_issue/data/training_list/train_list_0.txt"
  test_list = "/home/xionghantao/codes/works/JPDC_special_issue/data/testing_list/test_list_0.txt"

  image_array, feat_array, label_array, train_data = get_train_data(training_data_list, image_data, feat_data, label_data, label_file_suffix)
  
  model_path = "/data/xionghantao/HUAWEI_second_stage_data/models/wide_and_deep_model_com"
  train_model(model_path, image_array, feat_array, label_array)

  image_array, feat_array, label_array, test_data = get_test_data(test_list, image_data, feat_data, label_data, label_file_suffix)
  eva_path = "/home/xionghantao/codes/works/JPDC_special_issue/data/predictive_result/evaluate_acc.txt"
  res_path = "/home/xionghantao/codes/works/JPDC_special_issue/data/predictive_result/predict_result.txt"
  evaluate_model(model_path, image_array, feat_array, label_array, test_data, eva_path, res_path)

  metric_path = "/home/xionghantao/codes/works/JPDC_special_issue/data/predictive_result/metrics.res"
  get_acc(test_list, label_data, res_path, metric_path)
  get_precision(test_list, label_data, res_path, num_of_labels, metric_path)

def train_and_get_res_ten_fold():
  train_list_dir = "/home/xionghantao/codes/works/JPDC_special_issue/data/training_list"
  test_list_dir = "/home/xionghantao/codes/works/JPDC_special_issue/data/testing_list"
  res_dir = "/home/xionghantao/codes/works/JPDC_special_issue/data/predictive_result"
  fold_num = 10
  for i in range(fold_num):
 
    training_list = train_list_dir + "/train_list_" + str(i) + ".txt"
    testing_list = test_list_dir + "/test_list_" + str(i) + ".txt"

    image_array, feat_array, label_array, train_data = get_train_data(training_list, image_data, feat_data, label_data, label_file_suffix)
    model_path = "/data/xionghantao/HUAWEI_second_stage_data/models/new_wide_deep_add_withoutBN" + str(i)
    #train_model_add(model_path, image_array, feat_array, label_array)
    #train_xbg(model_path, feat_array, label_array)
    #train_wide_model(model_path, feat_array, label_array)
    #train_deep_model(model_path, image_array, label_array)
    #train_baseline_model(model_path, feat_array, label_array)
    #train_new_model_add(model_path,image_array,feat_array,label_array)
    train_new_model_wide_deep(model_path, image_array, feat_array, label_array)

    image_array, feat_array, label_array, test_data = get_test_data(testing_list, image_data, feat_data, label_data, label_file_suffix)
    eva_path = res_dir + "/evaluate_acc_" + str(i) + ".txt"
    res_path = res_dir + "/predict_result_" + str(i) + ".txt"
    #evaluate_model(model_path, image_array, feat_array, label_array, test_data, eva_path, res_path)
    #evaluate_xgboost(model_path, feat_array, res_path)
    #evaluate_wide_model(model_path, feat_array, label_array, test_data, eva_path, res_path)
    #evaluate_deep_model(model_path, image_array, label_array, test_data, eva_path, res_path)
    #evaluate_baseline_hart(model_path, feat_array, label_array, test_data, eva_path, res_path)
    evaluate_new_wide_deep_model(model_path, image_array, feat_array, label_array, test_data, eva_path, res_path)

    metric_path = res_dir + "/metrics_" + str(i) + ".res"
    get_acc(testing_list, label_data, res_path, metric_path)
    get_precision(testing_list, label_data, res_path, num_of_labels, metric_path)

if __name__ == "__main__":
  train_and_get_res_ten_fold()

  
 

  
