import time
import os
import sys

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, BatchNormalization, Concatenate

from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime

import numpy as np
sys.path.append("..")

from utils.load_MMdata import get_train_data
from utils.load_MMdata import get_test_data

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
    self.drouout = Dropout(0.2)
    self.dense1 = Dense(512, activation='relu')
    self.dense2 = Dense(4096, activation='relu')
    self.dense3 = Dense(1024, activation='relu')
    self.dense4 = Dense(100, activation='relu')

  def call(self, x):
    x = self.bn(x)
    # x = self.input1(x)
    x = self.dense1(x)
    x = self.drouout(x)
    x = self.dense2(x)
    x = self.drouout(x)
    x = self.dense3(x)
    x = self.drouout(x)
    x = self.dense4(x)
    x = self.drouout(x)
    return x

@tf.keras.utils.register_keras_serializable()
class DIESEL(Model):
  def __init__(self, num_of_labels, *args, **kwargs):
    super(DIESEL, self).__init__(*args, **kwargs)
    self.num_of_labels = num_of_labels
    self.wide = WideModel()
    self.final_dense = Dense(self.num_of_labels)

  def call(self, inputs):
    x0 = self.wide(inputs[0])
    x = self.final_dense(x0)
    return x
  
  def get_config(self):
    # 需要包括所有初始化参数
    config = super(DIESEL, self).get_config()
    config.update({'num_of_labels': self.num_of_labels})
    return config

def train_MM_model(model_path, 
                   image_array, RB_array, CB_array, 
                   feat_array, label_array, 
                   val_image_array, val_RB_array, val_CB_array, val_feat_array, val_label_array,
                   label_nums):
  model = DIESEL(label_nums)
  Optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=Optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  
  if len(image_array.shape) == 3:
    image_array = np.expand_dims(image_array, -1)
  if len(val_image_array.shape) == 3:
        val_image_array = np.expand_dims(val_image_array, -1)
  
  # 获取当前时间并格式化为字符串
  ttime = datetime.now().strftime("%Y%m%d_%H%M%S")
  logger_name = "DIESEL_" + ttime +'.csv'
  csv_logger = CSVLogger(logger_name, append=True, separator=' ')
  
  model.fit([feat_array, image_array, RB_array, CB_array], 
            label_array, 
            batch_size=64, 
            epochs=512,
            validation_data=([val_feat_array, val_image_array, val_RB_array, val_CB_array], val_label_array),
            callbacks=[csv_logger])
  
  # tf.keras.models.save_model(model, model_path)
  model.save(model_path)  # 推荐使用这种方式
  print ("Finish DIESEL Model Saving")

def train_and_get_res(training_data_list, val_data_list, label_suffix, label_nums):
  # training_data_list = "train_list.txt"  # 保存的是 dataset matrix name
  image_array, RB_array, CB_array, feat_array, label_array = get_train_data(training_data_list, label_suffix)
  
  val_image_array, val_RB_array, val_CB_array, val_feat_array, val_label_array = get_train_data(val_data_list, label_suffix)
  
#   print("Image  shape   : ", image_array.shape)
#   print("RB_arr shape   : ", RB_array.shape)
#   print("CB_arr shape   : ", CB_array.shape)
  print("Features shape : ", feat_array.shape)
  print("Label    shape : ", label_array.shape)
  
  # 保存模型的目录
  model_path = "/data/lsl/SSpMV/models/DIESEL.keras"
  
  train_MM_model(model_path, 
                 image_array, RB_array, CB_array, feat_array, label_array, 
                 val_image_array, val_RB_array, val_CB_array, val_feat_array, val_label_array,
                 label_nums)

def evaluate_MM_Adapter(model_path, image_array_test, Row_Block_array_test, Col_Block_array_test, feat_array_test, label_array_test, test_data, eva_path, res_path):
  
  loaded_model = tf.keras.models.load_model(model_path)
  
  test_loss, test_acc = loaded_model.evaluate([feat_array_test, image_array_test, Row_Block_array_test, Col_Block_array_test], label_array_test, verbose=2)
  
  f_eva = open(eva_path, "w")
  f_eva.write("Evaluating DIESEL Acc:" + str(test_acc) + "\n")
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
    
    predictions = loaded_model([feat, img, RB, CB])
    predictions = tf.nn.softmax(predictions)
    idx = tf.math.argmax(predictions[0])

    end = time.time()
    f_predict.write(str(int(idx)) + "\n")
  f_predict.close()

def test_model(test_data_list, label_suffix, label_nums):
  # test_data_list = "test_list.txt"
  image_array_test, Row_Block_array_test, Col_Block_array_test, feat_array_test, label_array_test, test_data = get_test_data(test_data_list, label_suffix)
  
  eva_path = "/data/lsl/SSpMV/models/prediction_result/DIESEL/test_acc.txt"
  res_path = "/data/lsl/SSpMV/models/prediction_result/DIESEL/predict_result.txt"
  
  # 保存模型的目录
  model_path = "/data/lsl/SSpMV/models/DIESEL.keras"
  
  evaluate_MM_Adapter(model_path, image_array_test, Row_Block_array_test, Col_Block_array_test, feat_array_test, label_array_test, test_data, eva_path, res_path)
  
  metric_path = "/data/lsl/SSpMV/models/prediction_result/DIESEL/metrics.res"
  base_path = "/data/lsl/MModel-Data"
  # label_format_suffix = ".format_label"
  
  get_acc_new(test_data_list, base_path, label_suffix, res_path, metric_path)
  get_precision_new(test_data_list, base_path, label_suffix, res_path, label_nums, metric_path)
  

if __name__ == "__main__":
  # training_data_list = "train_list.txt"  # 保存的是 Suite dataset matrix name
  training_data_list = "train_genlist.txt"
  # training_data_list = "train_all.txt"
  
  # val_data_list = "val_list.txt"
  val_data_list = "val_genlist.txt"
  
  test_data_list = "test_list.txt" # 保存的是 Suite dataset matrix name
  # test_data_list = "test_genlist.txt"
  # test_data_list = "test_all.txt"
  
  settings_idx = 1
  label_class = [".format_label", ".det_format_label"]
  print ("Running Model with the setting: [{}]".format(settings_idx))
  
  train_and_get_res(training_data_list, val_data_list, label_suffix=label_class[settings_idx], label_nums=number_of_labels[settings_idx])
  test_model(test_data_list, label_suffix=label_class[settings_idx], label_nums=number_of_labels[settings_idx])