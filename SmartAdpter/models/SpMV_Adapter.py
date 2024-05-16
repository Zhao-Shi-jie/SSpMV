import time
import os
import sys

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, BatchNormalization, Concatenate

import numpy as np

from utils.load_MMdata import get_train_data
from utils.load_MMdata import get_test_data

from utils.SSpMV_setting import *

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

class SpMV_Adapter(Model):
  def __init__(self):
    super(SpMV_Adapter, self).__init__()
    self.wide = WideModel()
    self.deep = DeepModel()
    self.conv1d_rb = Conv1DModel()
    self.conv1d_cb = Conv1DModel()
    self.concatenate = Concatenate()
    # self.dense1 = Dense(512, activation='relu')
    # self.dropout = Dropout(0.5)
    # self.dense2 = Dense(256, activation='relu')
    self.final_dense = Dense(num_of_labels)

  def call(self, inputs):
    x0 = self.wide(inputs[0])
    x1 = self.deep(inputs[1])
    x2 = self.conv1d_rb(inputs[2])
    x3 = self.conv1d_cb(inputs[3])
    x =  self.concatenate([x0, x1, x2, x3])
    # x = self.dense1(x)
    # x = self.dropout(x)
    # x = self.dense2(x)
    x = self.final_dense(x)
    return x

def train_MM_model(model_path, image_array, RB_array, CB_array, feat_array, label_array):
  model = SpMV_Adapter()
  Optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=Optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  if len(image_array.shape) == 3:
    image_array = np.expand_dims(image_array, -1)
  
  # 使用 tf.squeeze 移除大小为1的维度
  RB_1D = tf.squeeze(RB_array, axis=2)  # 结果形状将为 (batch_size, 256, 3)
  CB_1D = tf.squeeze(CB_array, axis=2)
#   print("RB_1D shape   : ",RB_1D.shape)
#   print("RB_1D shape   : ",RB_1D.shape)
  
  model.fit([feat_array, image_array, RB_1D, CB_1D], label_array, batch_size=64, epochs=32)
  # tf.keras.saving.save_model(model,model_path)
  tf.keras.models.save_model(model, model_path)


def train_and_get_res():
  training_data_list = "train_list.txt"  # 保存的是 dataset matrix name
  image_array, RB_array, CB_array, feat_array, label_array = get_train_data(training_data_list)
  
  print("Image  shape   : ", image_array.shape)
  print("RB_arr shape   : ", RB_array.shape)
  print("CB_arr shape   : ", CB_array.shape)
  print("Features array : ", feat_array)
  print("Label    array : ", label_array)
  
  # 保存模型的目录
  model_path = "/data/lsl/SSpMV/models/SpMV_Adapter.keras"
  
  train_MM_model(model_path, image_array, RB_array, CB_array, feat_array, label_array)

if __name__ == "__main__":
  train_and_get_res()