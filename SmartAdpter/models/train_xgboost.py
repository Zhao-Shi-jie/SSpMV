import time
import os
import sys
import numpy as np
import xgboost as xgb

sys.path.append("..")

from utils.load_data import get_train_data
from utils.load_data import get_test_data

from utils.data_setting import *

from compute_metrics import get_acc
from compute_metrics import get_precision




def train_xbg(model_path, feat_array, label_array):
  #data = np.random.rand(5,10) # 5 entities, each contains 10 features
  #label = np.random.randint(3, size=5) # binary target
  print(feat_array.shape)
  print(label_array.shape) 
  data_train = xgb.DMatrix(feat_array, label=label_array)
  params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 19,               # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 12,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
  }
  num_round = 1
  model = xgb.train(params, data_train, num_round)
  model.save_model(model_path)


def evaluate_xgboost(model_path, feat_array_test, res_path):
  
  data_test = xgb.DMatrix(feat_array_test)
  load_model = xgb.Booster({'nthread':4})
  load_model.load_model(model_path)
  results = load_model.predict(data_test)

  f_predict = open(res_path, "w")

  for result in results:
    f_predict.write(str(int(result)) + "\n")



def train_and_get_res_ten_fold():
  train_list_dir = "/home/xionghantao/codes/works/JPDC_special_issue/data/training_list"
  test_list_dir = "/home/xionghantao/codes/works/JPDC_special_issue/data/testing_list"
  res_dir = "/home/xionghantao/codes/works/JPDC_special_issue/data/xgboost"
  fold_num = 10
  for i in range(fold_num):
 
    training_list = train_list_dir + "/train_list_" + str(i) + ".txt"
    testing_list = test_list_dir + "/test_list_" + str(i) + ".txt"

    image_array, feat_array, label_array, train_data = get_train_data(training_list, image_data, feat_data, label_data, label_file_suffix)
    model_path = "/data/xionghantao/HUAWEI_second_stage_data/models/xgboost/xgboost_" + str(i)

    train_xbg(model_path, feat_array, label_array)
   

    image_array, feat_array, label_array, test_data = get_test_data(testing_list, image_data, feat_data, label_data, label_file_suffix)
    eva_path = res_dir + "/evaluate_acc_" + str(i) + ".txt"
    res_path = res_dir + "/predict_result_" + str(i) + ".txt"
   
    evaluate_xgboost(model_path, feat_array, res_path)
   

    metric_path = res_dir + "/metrics_" + str(i) + ".res"
    get_acc(testing_list, label_data, res_path, metric_path)
    get_precision(testing_list, label_data, res_path, num_of_labels, metric_path)

if __name__ == "__main__":
  train_and_get_res_ten_fold()


 

  
