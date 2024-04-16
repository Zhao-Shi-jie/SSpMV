import os
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
st = StandardScaler()

image_file_suffix = ".image"
image_dim = 128
feat_file_suffix = ".features"

def read_images(data_list_path, image_array_path):
  file_list = []
  with open(data_list_path, "r") as f:   #开存有文件名字的data_list_path
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())

  image_array = []    
  for file_ in file_list:           # 按照 image_dim 维度将image_data内存的数据存成2维张量
    image_tmp = []
    image_path = os.path.join(image_array_path, file_ + image_file_suffix)
    with open(image_path, "r") as f_read:
      while True:
        line = f_read.readline()
        if not line:
          break
        image_tmp.append([float(i) for i in line.split()])
    image_tmp = np.array(image_tmp).reshape((image_dim, image_dim))
    image_array.append(image_tmp)
  return np.array(image_array)


def read_features(data_list_path, feature_array_path):
  file_list = []
  with open(data_list_path, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
  feature_array = []
  for file_ in file_list:
    feature_path = os.path.join(feature_array_path, file_ + feat_file_suffix)
    feature = []
    with open(feature_path, "r") as f_read:
      feats = f_read.readlines()
      for feat in feats:
        # for each each feature: feat_name feat_value
        if len(feat.split(" ")) < 2:
          feat = float(feat.strip().split(" ")[0])
        else:
          feat = float(feat.strip().split(" ")[1])

        feature.append(feat)    
    feature_array.append(feature)

  return np.array(feature_array)

def read_labels(data_list_path, label_array_path, label_file_suffix):
  file_list = []
  with open(data_list_path, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
  label_array = []
  for file_ in file_list:
    label_path = os.path.join(label_array_path, file_ + label_file_suffix)
    with open(label_path, "r") as f_read:
      label = f_read.readline().strip().split(" ")[1]
    label_array.append(int(label))
  return np.array(label_array)


def get_train_data(data_list, image_data, feat_data, label_data, label_file_suffix):
  image_array = read_images(data_list, image_data)    # data list name, image data path
  feat_array = read_features(data_list, feat_data)
  label_array = read_labels(data_list, label_data, label_file_suffix)
  feat_array = st.fit_transform(feat_array)
  train_data = tf.data.Dataset.from_tensor_slices((image_array, feat_array, label_array)).shuffle(10000).batch(32)
  return image_array,feat_array,label_array,train_data


def get_test_data(test_list, image_data, feat_data, label_data, label_file_suffix):
  image_array_test = read_images(test_list, image_data)
  feat_array_test = read_features(test_list, feat_data)
  label_array_test = read_labels(test_list, label_data, label_file_suffix)
  feat_array_test = st.fit_transform(feat_array_test)
  test_data = tf.data.Dataset.from_tensor_slices((image_array_test, feat_array_test, label_array_test))
  
  return image_array_test,feat_array_test,label_array_test,test_data