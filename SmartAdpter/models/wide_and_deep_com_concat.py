import time
import os
import sys

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Flatten

import numpy as np

sys.path.append("..")

from utils.load_data import get_train_data
from utils.data_setting import *


class WideModel(Model):
  def __init__(self):
    super(WideModel, self).__init__()
    self.input1 = InputLayer(input_shape=(18,))
    self.dense1 = Dense(128, activation='relu')
    self.dense2 = Dense(num_of_labels)

  def call(self, x):
    x = self.input1(x)
    x = self.dense1(x)
    x = self.dense2(x)
    return x

class DeepModel(Model):
  def __init__(self):
    super(DeepModel, self).__init__()
    self.input1 = InputLayer(input_shape=(128, 128, 1))
    self.conv1 = Conv2D(16, (3, 3), activation='tanh')
    self.conv2 = Conv2D(16, (5, 5), strides=(2, 2), padding='same',  activation='tanh')
    self.maxpool1 = MaxPooling2D(2, 2)
    self.flatten = Flatten()
    self.dense = Dense(num_of_labels)

  def call(self, x):
    x = self.input1(x)
    x = self.conv1(x)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.maxpool1(x)
    x = self.flatten(x)
    x = self.dense(x)
    return x

class WideAndDeepConcat(Model):
  def __init__(self):
    super(WideAndDeepConcat, self).__init__()
    self.wide = WideModel()
    self.deep = DeepModel()
    self.dense = Dense(num_of_labels)


  def call(self, x):
    x0 = self.wide(x[0])
    x1 = self.deep(x[1]) 
    x =  tf.keras.layers.Concatenate()([x0, x1])
    x = self.dense(x)
    return x

class WideAndDeepAdd(Model):
  def __init__(self):
    super(WideAndDeepAdd, self).__init__()
    self.wide = WideModel()
    self.deep = DeepModel()

  def call(self, x):
    x0 = self.wide(x[0])
    x1 = self.deep(x[1]) 
    x =  tf.keras.layers.Add()([x0, x1])
    return x

def train_model(model_path,image_array,feat_array,label_array):
  combined_model = tf.keras.experimental.WideDeepModel(WideModel(), DeepModel())
  combined_model.compile(optimizer="adam",
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
  combined_model.fit([feat_array, tf.expand_dims(image_array, -1)], label_array, epochs=256)  
  #test_loss, test_acc = combined_model.evaluate([feat_array_test, tf.expand_dims(image_array_test, -1)],  label_array_test, verbose=2)
  #print('\nTest accuracy:', test_acc)
  tf.keras.saving.save_model(combined_model, model_path)



def train_model_concat(model_path,image_array,feat_array,label_array):
  combined_model = WideAndDeepConcat()
  combined_model.compile(optimizer="adam",
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
  combined_model.fit([feat_array, tf.expand_dims(image_array, -1)], label_array, epochs=256)  
  #test_loss, test_acc = combined_model.evaluate([feat_array_test, tf.expand_dims(image_array_test, -1)],  label_array_test, verbose=2)
  #print('\nTest accuracy:', test_acc)
  tf.keras.saving.save_model(combined_model, model_path)


def train_model_add(model_path,image_array,feat_array,label_array):
  combined_model = WideAndDeepAdd()
  combined_model.compile(optimizer="adam",
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
  combined_model.fit([feat_array, tf.expand_dims(image_array, -1)], label_array, epochs=256)  
  #test_loss, test_acc = combined_model.evaluate([feat_array_test, tf.expand_dims(image_array_test, -1)],  label_array_test, verbose=2)
  #print('\nTest accuracy:', test_acc)
  tf.keras.saving.save_model(combined_model, model_path)



if __name__ == "__main__":
  training_data_list = "/home/xionghantao/codes/works/JPDC_special_issue/data/training_list/train_list_0.txt"
  image_array, feat_array, label_array, train_data = get_train_data(training_data_list, image_data, feat_data, label_data, label_file_suffix)

  model_path = "/data/xionghantao/HUAWEI_second_stage_data/models/wide_and_deep_model_com"
  train_model(model_path, image_array, feat_array, label_array)

  #model_path_concat = "/data/xionghantao/HUAWEI_second_stage_data/models/wide_and_deep_model_com_concat"
  #train_model_concat(model_path_concat,image_array,feat_array,label_array)
  
  #model_path_add = "/data/xionghantao/HUAWEI_second_stage_data/models/wide_and_deep_model_com_add"
  #train_model_add(model_path_add,image_array,feat_array,label_array)

 

  
