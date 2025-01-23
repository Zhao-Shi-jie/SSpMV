import time
import os
import sys

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, BatchNormalization, Concatenate

import numpy as np
sys.path.append("..")

from utils.load_MMdata import get_train_data_new
from utils.load_MMdata import get_test_data_new

from utils.SSpMV_setting import *

from compute_metrics import get_acc_new
from compute_metrics import get_precision_new

"""
Self defined wide model: should be FFNN : Feed-Forward Neural Network
The standard FFNN is a multi-layer feedforward network with an input, a hidden, and an output layer. 
It can be used to learn and store a large number of mappings between the input and output layers.
"""
class WideModel(Model):
  def __init__(self):
    super(WideModel, self).__init__()
    # self.input1 = InputLayer(input_shape=(features_dim,))   # human designed features size
    self.bn = BatchNormalization()
    self.dense1 = Dense(512, activation='relu')
    self.dense2 = Dense(1024, activation='relu')
    self.dense3 = Dense(32)            # should be the number of algorithms

  def call(self, x):
    x = self.bn(x)
    # x = self.input1(x)
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    return x

"""
Self defined deep model: should be CNN : Convolutional Neural Network
Block count is related to non-zero elements on the matrix, 
and normalization restricts their number to within a reasonable range (0~255).
"""
class DeepModel(Model):
  def __init__(self):
    super(DeepModel, self).__init__()
    # self.input1 = InputLayer(input_shape=(256, 256, 3))     # Input shape: 256 x 256
    self.conv1 = Conv2D(16, (3, 3), padding='same',activation='relu')      # 16 filters with (3,3) shape
    self.conv2 = Conv2D(16, (5, 5), strides=(2, 2), padding='same',  activation='relu')
    self.pool = MaxPooling2D(pool_size=(2, 2))
    self.flatten = Flatten()
    self.dense = Dense(32)

  def call(self, x):
    # x = self.input1(x)
    x = self.conv1(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.pool(x)
    x = self.flatten(x)
    x = self.dense(x)       # 2-layer CNN and flatten, outputsize: should be the number of algorithms
    return x

class Conv1DModel(Model):
  def __init__(self):
    super(Conv1DModel, self).__init__()
    # filters = 16, kernel_size = 3
    self.conv1 = Conv1D(16, 3, activation='relu', padding='same')
    self.conv2 = Conv1D(16, 5, strides=2, activation='relu', padding='same')
    self.pool = MaxPooling1D(2)
    self.flatten = Flatten()
    self.dense = Dense(32)
  
  def call(self, x):
    x = self.conv1(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.pool(x)
    x = self.flatten(x)
    x = self.dense(x)
    return x

@tf.keras.utils.register_keras_serializable()
class SpMV_Adapter(Model):
  def __init__(self, num_of_labels, *args, **kwargs):
    super(SpMV_Adapter, self).__init__(*args, **kwargs)
    self.num_of_labels = num_of_labels
    self.wide = WideModel()
    self.deep = DeepModel()
    self.conv1d_rb = Conv1DModel()
    self.conv1d_cb = Conv1DModel()
    self.concatenate = Concatenate()
    self.dense1 = Dense(512, activation='relu')
    self.dropout = Dropout(0.5)
    self.dense2 = Dense(256, activation='relu')
    self.final_dense = Dense(self.num_of_labels, activation='softmax')  # 输出每个类别的概率

  def call(self, inputs):
    x0 = self.wide(inputs[0])
    x1 = self.deep(inputs[1])
    x2 = self.conv1d_rb(inputs[2])
    x3 = self.conv1d_cb(inputs[3])
    x =  self.concatenate([x0, x1, x2, x3])
    x = self.dense1(x)
    x = self.dropout(x)
    x = self.dense2(x)
    x = self.dropout(x)
    x = self.final_dense(x)
    return x
  
  def get_config(self):
    # 需要包括所有初始化参数
    config = super(SpMV_Adapter, self).get_config()
    config.update({'num_of_labels': self.num_of_labels})
    return config

def train_MM_model(model_path, image_array, RB_array, CB_array, feat_array, label_array, label_nums):
  model = SpMV_Adapter(label_nums)
  Optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=Optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  if len(image_array.shape) == 3:
    image_array = np.expand_dims(image_array, -1)
    
  model.fit([feat_array, image_array, RB_array, CB_array], label_array, batch_size=64, epochs=512)
  # tf.keras.models.save_model(model, model_path)
  model.save(model_path)  # 推荐使用这种方式
  print ("Finish MM-Adapter(prob) Model Saving")


def train_and_get_res(training_data_list, label_suffix, label_nums):
  image_array, RB_array, CB_array, feat_array, label_array = get_train_data_new(training_data_list, label_suffix)
  
  print("Image  shape   : ", image_array.shape)
  print("RB_arr shape   : ", RB_array.shape)
  print("CB_arr shape   : ", CB_array.shape)
  print("Features shape : ", feat_array.shape)
  print("Label    shape : ", label_array.shape)
  
  # 保存模型的目录
  model_path = "/data/lsl/SSpMV/models/SpMV_Adapter_prob.keras"
  
  train_MM_model(model_path, image_array, RB_array, CB_array, feat_array, label_array, label_nums)

def evaluate_MM_Adapter(model_path, image_array_test, Row_Block_array_test, Col_Block_array_test, feat_array_test, label_array_test, test_data, eva_path, res_path):
  
  loaded_model = tf.keras.models.load_model(model_path)
  
  test_loss, test_acc = loaded_model.evaluate([feat_array_test, image_array_test, Row_Block_array_test, Col_Block_array_test],  label_array_test, verbose=2)
  
  f_eva = open(eva_path, "w")
  f_eva.write("Evaluating Acc:" + str(test_acc) + "\n")
  f_eva.close()

  f_predict = open(res_path, "w")
  for file_ in test_data:
    start = time.time()

    # predictions = loaded_model((tf.expand_dims(file_[1],0), tf.expand_dims(tf.expand_dims(file_[0],0), -1)))
    #  file_ = [feat, img, RB, CB, label]
    # predictions = loaded_model(file_[0], file_[1], file_[2], file_[3])
    # 预处理输入数据：确保数据维度正确
    feat = tf.expand_dims(file_[0], 0)  # 假设feat需要扩展批量维度
    img = tf.expand_dims(file_[1], 0)   # 批次维度
    RB = tf.expand_dims(file_[2], 0)  # 同理
    CB = tf.expand_dims(file_[3], 0)  # 同理
    
    # 模型预测
    predictions = loaded_model([feat, img, RB, CB])

    # 获取最大概率的索引，即预测的类别标签
    idx = tf.math.argmax(predictions[0])

    end = time.time()
    f_predict.write(f"{int(idx.numpy())}\n")
    
  f_predict.close()

def test_model(test_data_list, label_suffix, label_nums):
  
  image_array_test, Row_Block_array_test, Col_Block_array_test, feat_array_test, label_array_test, test_data = get_test_data_new(test_data_list, label_suffix)
  
  eva_path = "/data/lsl/SSpMV/models/prediction_result/MM_Adapter_pb/evaluate_acc.txt"
  res_path = "/data/lsl/SSpMV/models/prediction_result/MM_Adapter_pb/predict_result.txt"
  
  # 保存模型的目录
  model_path = "/data/lsl/SSpMV/models/SpMV_Adapter_prob.keras"
  
  evaluate_MM_Adapter(model_path, image_array_test, Row_Block_array_test, Col_Block_array_test, feat_array_test, label_array_test, test_data, eva_path, res_path)
  
  metric_path = "/data/lsl/SSpMV/models/prediction_result/MM_Adapter_pb/metrics.res"
  base_path = "/data/lsl/MModel-Data"
  label_format_suffix = ".format_label"
  
  get_acc_new(test_data_list, base_path, label_format_suffix, res_path, metric_path)
  get_precision_new(test_data_list, base_path, label_format_suffix, res_path, label_nums, metric_path)

if __name__ == "__main__":
  training_data_list = "train_list.txt"  # 保存的是 dataset matrix name
  test_data_list = "test_list.txt"
  
  settings_idx = 0
  label_class = [".prob_label", ".det_prob_label"]
  print ("Running Prob Model with the setting: [{}]".format(settings_idx))
  
  # train_and_get_res(training_data_list, label_suffix=label_class[settings_idx], label_nums=number_of_labels[settings_idx])
  test_model(test_data_list, label_suffix=label_class[settings_idx], label_nums=number_of_labels[settings_idx])