import sys
import os
sys.path.append("..")
from utils.data_setting import *


def get_acc(test_list, label_data, res_path, metric_path):
  file_list = []
  with open(test_list, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
  label_array = []
  # The real label
  for file_ in file_list:
    label_path = os.path.join(label_data, file_ + label_file_suffix)
    with open(label_path, "r") as f_read:
      label = f_read.readline().strip().split(" ")[1]
    label_array.append(int(label))
  cnt_equal = 0
  cnt_not = 0
  # compare with predictive label
  with open(res_path, "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
      if label_array[i] == int(lines[i].strip()):
        cnt_equal = cnt_equal + 1
      else:
        cnt_not = cnt_not + 1
  #print(cnt_equal)
  #print(cnt_not)
  acc = cnt_equal / (cnt_not + cnt_equal)
  f = open(metric_path, "a+")
  f.write("Acc:" + str(acc) + "\n")
  f.close()
  


def get_precision(test_list, label_data, res_path, label_num, metric_path):
  file_list = []
  with open(test_list, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
      
  label_array = []
  for file_ in file_list:
    label_path = os.path.join(label_data, file_ + label_file_suffix)
    with open(label_path, "r") as f_read:
      label = f_read.readline().strip().split(" ")[1]
    label_array.append(int(label))
  
  predict_label_array = []
  with open(res_path, "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
      predict_label_array.append(int(lines[i].strip()))
  

  sum_p = 0.0
  sum_r = 0.0
  sum_f1 = 0.0
  for label in range(label_num):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    precision = 0.0
    recall = 0.0
    F1 = 0.0

    for i in range(len(label_array)):
      if label_array[i] == label and predict_label_array[i] == label:
        TP = TP + 1
      if label_array[i] == label and predict_label_array[i] != label:
        FN = FN + 1
      if label_array[i] != label and predict_label_array[i] == label:
        FP = FP + 1
      if label_array[i] != label and predict_label_array[i] != label:
        TN = TN + 1
    
    if TP + FP == 0:
      precision = 0.0
    else:
      precision = TP / (TP + FP)
  
    if TP + FN == 0:
      recall = 0.0
    else:
      recall = TP / (TP + FN)
    
    if (precision + recall)  < 0.000000000001:
      F1 = 0.0
    else:
      F1 = (2 * precision * recall) / (precision + recall)
   
    sum_p = sum_p + precision
    sum_r = sum_r + recall
    sum_f1 = sum_f1 + F1

  Macro_p = sum_p / label_num
  Macro_r = sum_r / label_num
  Macro_f1 = sum_f1 / label_num
  
  f = open(metric_path, "a+")
  f.write("macro_precision:" + str(Macro_p) + "\n")
  f.write("macro_recall:" + str(Macro_r) + "\n")
  f.write("macro_f1:" + str(Macro_f1) + "\n")

  f.close()
    
def aver_metric(metrics_dir="/home/xionghantao/codes/works/JPDC_special_issue/data/predictive_result_wide_deep/"):
  fold_num = 10
  Accs = []
  macro_p = []
  macro_r = []
  macro_f = []
  for i in range(fold_num):
    dir = metrics_dir + "metrics_" + str(i) + ".res"
    with open(dir, "r") as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        if "Acc" in line:
          acc = line.split(":")[1]
          Accs.append(float(acc))
    
    with open(dir, "r") as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        if "macro_precision" in line:
          m_p = line.split(":")[1]
          macro_p.append(float(m_p))

    with open(dir, "r") as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        if "macro_recall" in line:
          m_r = line.split(":")[1]
          macro_r.append(float(m_r))

    with open(dir, "r") as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip()
        if "macro_f1" in line:
          m_f = line.split(":")[1]
          macro_f.append(float(m_f))

  aver_acc = sum(Accs) / len(Accs)
  aver_macro_p = sum(macro_p) / len(macro_p)
  aver_macro_r = sum(macro_r) / len(macro_r)
  aver_macro_f = sum(macro_f) / len(macro_f)

  

  print(aver_acc)
  print(aver_macro_p)
  print(aver_macro_r)
  print(aver_macro_f)




def get_precision_each_label(test_list, label_data, res_path, label_num):
  file_list = []
  with open(test_list, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
      
  label_array = []
  for file_ in file_list:
    label_path = os.path.join(label_data, file_ + label_file_suffix)
    with open(label_path, "r") as f_read:
      label = f_read.readline().strip().split(" ")[1]
    label_array.append(int(label))
  
  predict_label_array = []
  with open(res_path, "r") as f:
    lines = f.readlines()
    for i in range(len(lines)):
      predict_label_array.append(int(lines[i].strip()))
  

  for label in range(label_num):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    precision = 0.0
    recall = 0.0
    F1 = 0.0

    for i in range(len(label_array)):
      if label_array[i] == label and predict_label_array[i] == label:
        TP = TP + 1
      if label_array[i] == label and predict_label_array[i] != label:
        FN = FN + 1
      if label_array[i] != label and predict_label_array[i] == label:
        FP = FP + 1
      if label_array[i] != label and predict_label_array[i] != label:
        TN = TN + 1
    
    if TP + FP == 0:
      precision = 0.0
    else:
      precision = TP / (TP + FP)
  
    if TP + FN == 0:
      recall = 0.0
    else:
      recall = TP / (TP + FN)
    
    if (precision + recall)  < 0.000000000001:
      F1 = 0.0
    else:
    #  pass
      F1 = (2 * precision * recall) / (precision + recall)
    print("label:", label)
    print("number:",TP + FN)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", F1)
    print("\n")






def recompute_metric_ten_fold():
  train_list_dir = "/home/xionghantao/codes/works/JPDC_special_issue/data/training_list"
  test_list_dir = "/home/xionghantao/codes/works/JPDC_special_issue/data/testing_list"
  res_dir = "/home/xionghantao/codes/works/JPDC_special_issue/data/new_wide_deep_concat"
  fold_num = 10
  for i in range(fold_num):

    training_list = train_list_dir + "/train_list_" + str(i) + ".txt"
    testing_list = test_list_dir + "/test_list_" + str(i) + ".txt"

    #image_array, feat_array, label_array, train_data = get_train_data(training_list, image_data, feat_data, label_data, label_file_suffix)
    #model_path = "/data/xionghantao/JPDC_models/baseline_hart_wig_" + str(i)
  
    #train_baseline_model(model_path, feat_array, label_array)

    #image_array, feat_array, label_array, test_data = get_test_data(testing_list, image_data, feat_data, label_data, label_file_suffix)
    eva_path = res_dir + "/evaluate_acc_" + str(i) + ".txt"
    res_path = res_dir + "/predict_result_" + str(i) + ".txt"

    #evaluate_baseline_model(model_path, feat_array, label_array, test_data, eva_path, res_path)

    metric_path = res_dir + "/metrics_" + str(i) + ".res"
    get_acc(testing_list, label_data, res_path, metric_path)
    get_precision(testing_list, label_data, res_path, num_of_labels, metric_path)

  

if __name__ == "__main__":
  test_list = "/home/xionghantao/codes/works/JPDC_special_issue/data/testing_list/test_list_1.txt"
  res_path = "/home/xionghantao/codes/works/JPDC_special_issue/data/baseline_hart_wig/predict_result_1.txt"
  #metric_path = "/home/xionghantao/codes/works/JPDC_special_issue/data/xgboost/metrics.res"
  #get_acc(test_list, label_data, res_path, metric_path)
  #get_precision_each_label(test_list, label_data, res_path, 19)

  aver_metric("/data1/xionghantao/data/JPDC/prediction_result/baseline_hart_wig/")

  