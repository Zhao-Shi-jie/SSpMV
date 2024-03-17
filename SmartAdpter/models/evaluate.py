import time
import os
import sys
import tensorflow as tf
import numpy as np
import xgboost as xgb

sys.path.append("..")

from utils.load_data import get_test_data
from utils.data_setting import *
  
def evaluate_model(model_path, image_array_test, feat_array_test, label_array_test, test_data, eva_path, res_path):
  loaded_model = tf.keras.saving.load_model(model_path)

  test_loss, test_acc = loaded_model.evaluate([feat_array_test, tf.expand_dims(image_array_test, -1)],  label_array_test, verbose=2)
  f_eva = open(eva_path, "w")                                                   
  f_eva.write("Evaluating Acc:" + str(test_acc) + "\n")
  f_eva.close()

  f_predict = open(res_path, "w")                                                   
  

  for file_ in test_data:
    start = time.time()

    predictions = loaded_model((tf.expand_dims(file_[1],0), tf.expand_dims(tf.expand_dims(file_[0],0), -1)))
    predictions = tf.nn.softmax(predictions)
    idx = tf.math.argmax(predictions[0])

    end = time.time()
    f_predict.write(str(int(idx)) + "\n")
  f_predict.close()

def evaluate_xgboost(model_path, feat_array_test, res_path):
  
  data_test = xgb.DMatrix(feat_array_test)
  load_model = xgb.Booster({'nthread':4})
  load_model.load_model(model_path)
  results = load_model.predict(data_test)

  f_predict = open(res_path, "w")

  for result in results:
    f_predict.write(str(int(result)) + "\n")

def evaluate_wide_model(model_path, feat_array_test, label_array_test, test_data, eva_path, res_path):
  loaded_model = tf.keras.saving.load_model(model_path)

  test_loss, test_acc = loaded_model.evaluate(feat_array_test,  label_array_test, verbose=2)
  f_eva = open(eva_path, "w")                                                   
  f_eva.write("Evaluating Acc:" + str(test_acc) + "\n")
  f_eva.close()

  f_predict = open(res_path, "w")                                                   
  

  for file_ in test_data:
    start = time.time()

    predictions = loaded_model(tf.expand_dims(file_[1],0))
    predictions = tf.nn.softmax(predictions)
    idx = tf.math.argmax(predictions[0])

    end = time.time()
    f_predict.write(str(int(idx)) + "\n")
  f_predict.close()

def evaluate_deep_model(model_path, image_array_test, label_array_test, test_data, eva_path, res_path):
  loaded_model = tf.keras.saving.load_model(model_path)

  test_loss, test_acc = loaded_model.evaluate(tf.expand_dims(image_array_test, -1), label_array_test, verbose=2)
  f_eva = open(eva_path, "w")                                                   
  f_eva.write("Evaluating Acc:" + str(test_acc) + "\n")
  f_eva.close()

  f_predict = open(res_path, "w")                                                   
  

  for file_ in test_data:
    start = time.time()

    predictions = loaded_model((tf.expand_dims(tf.expand_dims(file_[0],0), -1)))
    predictions = tf.nn.softmax(predictions)
    idx = tf.math.argmax(predictions[0])

    end = time.time()
    f_predict.write(str(int(idx)) + "\n")
  #  pass
  f_predict.close()



def evaluate_baseline_hart(model_path, feat_array_test, label_array_test, test_data, eva_path, res_path):
  loaded_model = tf.keras.saving.load_model(model_path)

  test_loss, test_acc = loaded_model.evaluate(feat_array_test,  label_array_test, verbose=2)
  f_eva = open(eva_path, "w")                                                   
  f_eva.write("Evaluating Acc:" + str(test_acc) + "\n")
  f_eva.close()

  f_predict = open(res_path, "w")                                                   
  

  for file_ in test_data:
    start = time.time()

    predictions = loaded_model(tf.expand_dims(file_[1],0))
    predictions = tf.nn.softmax(predictions)
    idx = tf.math.argmax(predictions[0])

    end = time.time()
    f_predict.write(str(int(idx)) + "\n")
  f_predict.close()


def evaluate_new_wide_deep_add_model(model_path, image_array_test, feat_array_test, label_array_test, test_data, eva_path, res_path):
  loaded_model = tf.keras.saving.load_model(model_path)

  test_loss, test_acc = loaded_model.evaluate([feat_array_test, tf.expand_dims(image_array_test, -1)],  label_array_test, verbose=2)
  f_eva = open(eva_path, "w")                                                   
  f_eva.write("Evaluating Acc:" + str(test_acc) + "\n")
  f_eva.close()

  f_predict = open(res_path, "w")                                                   
  

  for file_ in test_data:
    start = time.time()

    predictions = loaded_model((tf.expand_dims(file_[1],0), tf.expand_dims(tf.expand_dims(file_[0],0), -1)))
    predictions = tf.nn.softmax(predictions)
    idx = tf.math.argmax(predictions[0])

    end = time.time()
    f_predict.write(str(int(idx)) + "\n")
  f_predict.close()

def evaluate_new_wide_model(model_path, feat_array_test, label_array_test, test_data, eva_path, res_path):
  loaded_model = tf.keras.saving.load_model(model_path)

  test_loss, test_acc = loaded_model.evaluate(feat_array_test,  label_array_test, verbose=2)
  f_eva = open(eva_path, "w")                                                   
  f_eva.write("Evaluating Acc:" + str(test_acc) + "\n")
  f_eva.close()

  f_predict = open(res_path, "w")                                                   
  

  for file_ in test_data:
    start = time.time()

    predictions = loaded_model(tf.expand_dims(file_[1],0))
    predictions = tf.nn.softmax(predictions)
    idx = tf.math.argmax(predictions[0])

    end = time.time()
    f_predict.write(str(int(idx)) + "\n")
  f_predict.close()


def evaluate_new_wide_deep_concat_model(model_path, image_array_test, feat_array_test, label_array_test, test_data, eva_path, res_path):
  loaded_model = tf.keras.saving.load_model(model_path)

  test_loss, test_acc = loaded_model.evaluate([feat_array_test, tf.expand_dims(image_array_test, -1)],  label_array_test, verbose=2)
  f_eva = open(eva_path, "w")                                                   
  f_eva.write("Evaluating Acc:" + str(test_acc) + "\n")
  f_eva.close()

  f_predict = open(res_path, "w")                                                   
  

  for file_ in test_data:
    start = time.time()

    predictions = loaded_model((tf.expand_dims(file_[1],0), tf.expand_dims(tf.expand_dims(file_[0],0), -1)))
    predictions = tf.nn.softmax(predictions)
    idx = tf.math.argmax(predictions[0])

    end = time.time()
    f_predict.write(str(int(idx)) + "\n")
  f_predict.close()

def evaluate_new_wide_deep_model(model_path, image_array_test, feat_array_test, label_array_test, test_data, eva_path, res_path):
  loaded_model = tf.keras.saving.load_model(model_path)

  test_loss, test_acc = loaded_model.evaluate([feat_array_test, tf.expand_dims(image_array_test, -1)],  label_array_test, verbose=2)
  f_eva = open(eva_path, "w")                                                   
  f_eva.write("Evaluating Acc:" + str(test_acc) + "\n")
  f_eva.close()

  f_predict = open(res_path, "w")                                                   
  

  for file_ in test_data:
    start = time.time()

    predictions = loaded_model((tf.expand_dims(file_[1],0), tf.expand_dims(tf.expand_dims(file_[0],0), -1)))
    predictions = tf.nn.softmax(predictions)
    idx = tf.math.argmax(predictions[0])

    end = time.time()
    f_predict.write(str(int(idx)) + "\n")
  f_predict.close()

if __name__ == "__main__":
  test_list = "/home/xionghantao/codes/works/JPDC_special_issue/data/testing_list/test_list_0.txt"
  image_array, feat_array, label_array, test_data = get_test_data(test_list, image_data, feat_data, label_data, label_file_suffix)
  model_path = "/data/xionghantao/HUAWEI_second_stage_data/models/new_wide_deep"
  eva_path = "/home/xionghantao/codes/works/JPDC_special_issue/data/predictive_result/evaluate_acc.txt"
  res_path = "/home/xionghantao/codes/works/JPDC_special_issue/data/predictive_result/predict_result.txt"
  #evaluate_deep_model(model_path, image_array, label_array, test_data, eva_path, res_path)
  #model_path = "/data/xionghantao/HUAWEI_second_stage_data/models/xgboost/model.json"
  #res_path = "/home/xionghantao/codes/works/JPDC_special_issue/data/predictive_result_xgboost/predict_result.txt"
  #evaluate_xgboost(model_path, feat_array, res_path)
  evaluate_new_wide_deep_model(model_path, image_array, feat_array, label_array, test_data, eva_path, res_path)



  