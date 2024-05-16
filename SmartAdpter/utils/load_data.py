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

# data_list 保存的是数据集中的 matrix name
# image_data 存储为 128 *128 = 16384 行
# feat_data  存储为 18 行，每行: 特征名字 <空格> 特征值 
# label_data 存储为 1  行，每行: 预条件名 <空格> 对应算法label号
def get_train_data(data_list, image_data, feat_data, label_data, label_file_suffix):
  # 读取
  image_array = read_images(data_list, image_data)    # data list name, image data path
  feat_array = read_features(data_list, feat_data)
  label_array = read_labels(data_list, label_data, label_file_suffix)
  # 对特征数据进行标准化处理。标准化是机器学习预处理中常用的方法，
  # 目的是将特征数据规范到一个标准的范围内，通常是一个均值为0，标准差为1的分布。
  # 这有助于模型更好地学习和收敛。
  feat_array = st.fit_transform(feat_array)
  # 这一行创建了一个 TensorFlow 数据集。from_tensor_slices 方法可以将给定的元组（或者是其他形式的组合数据）
  # 转换为一个 tf.data.Dataset 对象，其中每个元素对应于元组中相应位置的切片。
  # .shuffle(10000)：这一方法将数据集的条目进行随机打乱，以避免模型在训练过程中对数据顺序产生依赖。这里的 10000 表示随机缓冲区的大小。
  # .batch(32)：这一方法将数据集中的元素分成大小为32的批次。这是训练神经网络时的标准做法，因为一次处理整个数据集通常是不可行的，而且批量处理可以帮助优化梯度下降过程。
  train_data = tf.data.Dataset.from_tensor_slices((image_array, feat_array, label_array)).shuffle(10000).batch(32)
  return image_array,feat_array,label_array,train_data


def get_test_data(test_list, image_data, feat_data, label_data, label_file_suffix):
  image_array_test = read_images(test_list, image_data)
  feat_array_test = read_features(test_list, feat_data)
  label_array_test = read_labels(test_list, label_data, label_file_suffix)
  feat_array_test = st.fit_transform(feat_array_test)
  test_data = tf.data.Dataset.from_tensor_slices((image_array_test, feat_array_test, label_array_test))
  
  return image_array_test,feat_array_test,label_array_test,test_data