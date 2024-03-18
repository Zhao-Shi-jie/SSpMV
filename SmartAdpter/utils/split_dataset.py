import time
import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten

from sklearn.preprocessing import StandardScaler
import numpy as np

label_file_suffix = ".final_com_label"
image_data_test = "/data1/xionghantao/data/JPDC/training_and_testing_data/image"
test_data_test = "/data1/xionghantao/data/JPDC/training_and_testing_data/feature/"
label_data_test = "/data1/xionghantao/data/JPDC/training_and_testing_data/label"

st = StandardScaler()


def get_data_list(fold_num=10, data_path="/data1/xionghantao/data/JPDC/data_list"):
 
  src_path = "/data1/xionghantao/data/JPDC/training_and_testing_data/feature/"
  data_files = os.listdir(src_path)
  # select half of data as the selected_data_list, approximately 5000+
  selected_data_list = []
  for i in range(len(data_files)):
    #if i % 2 == 0:
    selected_data_list.append(data_files[i].split(".")[0])
  # the number of total samples, including both training and testing dataset
  total_data_num = len(selected_data_list)
  test_data_size = total_data_num // fold_num
  for i in range(fold_num):
    f_0 = open(os.path.join(data_path, "train_list_" + str(i) + ".txt"), "w")
    f_1 = open(os.path.join(data_path, "test_list_" + str(i) + ".txt"), "w")
    test_list = selected_data_list[i * test_data_size:i * test_data_size + test_data_size]
    train_list = []
    for name in selected_data_list:
      if name not in test_list:
        train_list.append(name)
    #print(test_list)
    
    for name in train_list:
      f_0.write(name + "\n")
    for name in test_list:
      f_1.write(name + "\n")
    
    f_0.close()
    f_1.close()

def label_statistics(path="/home/xionghantao/codes/works/JPDC_special_issue/data"):
  train_path = os.path.join(path,"training_list", "train_list_1.txt")
  test_path = os.path.join(path,"testing_list/", "test_list_1.txt")
  names = []
  label_path = "/data1/xionghantao/HUAWEI_data/train_data/label"
  with open(train_path, "r") as f_0:
    for line in f_0.readlines():
      line = line.strip()
      names.append(line)
  #with open(test_path, "r") as f_1:
  #  for line in f_1.readlines():
  #    line = line.strip()
  #    names.append(line)
  label_res = {}
  for name in names:
    label_dir = os.path.join(label_path, name + ".final_com_label")
    with open(label_dir, "r") as f:
      line = f.readline()
      combination = line.strip().split(" ")[0]
      if combination not in label_res.keys():
        label_res[combination] = 1
      else:
        label_res[combination] = label_res[combination] + 1
  num = 0
  for key in label_res.keys():
    num = num + int(label_res[key])
  print(num)
  print(label_res)



if __name__ == "__main__":
  get_data_list(fold_num=10)
  #label_statistics()


  