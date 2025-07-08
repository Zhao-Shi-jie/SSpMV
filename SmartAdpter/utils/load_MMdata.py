import os
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
st = StandardScaler()

image_dim    = 256
feat_file_suffix = ".features"
label_format_suffix = ".format_label"
label_prob_suffix = ".prob_label"
RB_suffix = ('.RBave', '.RBmax', '.RBstd')
CB_suffix = ('.CBave', '.CBmax', '.CBstd')

# base_path = "/data/lsl/MModel-Data"
base_path = "/data2/linshengle_data/MModel-Data"
def read_images(data_list, base_path, channel_suffixes=('.ave', '.max', '.std')):
  file_list = []
  with open(data_list, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
      
  # 初始化图像数组列表
  image_list = []
    
  # 遍历数据列表中的每个文件名
  for mtx_name in file_list:
    # 构建每个通道的完整路径
    channel_images = []
    for suffix in channel_suffixes:
        image_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{suffix}")
        # 读取图像数据，假设数据以文本形式保存，每行一个数值
        img_data = np.loadtxt(image_path)
        # 将一维数组重塑为二维图像形状 (256x256)
        img_data = img_data.reshape((image_dim, image_dim))
        channel_images.append(img_data)
    
    # 检查所有通道图像的形状是否一致，以确保能够正确堆叠
    if all(img.shape == channel_images[0].shape for img in channel_images):
        # 沿着最后一个轴将三个通道的数据堆叠起来
        multi_channel_image = np.stack(channel_images, axis=-1)
        image_list.append(multi_channel_image)
    else:
        print(f"Error: Image channels for {mtx_name} have mismatched dimensions.")
  
  # 将图像列表转换为一个numpy数组
  image_array = np.array(image_list)
  return image_array

def read_1D_images(data_list, base_path, channel_suffixes=('.ave', '.max', '.std')):
  file_list = []
  with open(data_list, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
  
  # 初始化图像数组列表
  image_list = []
    
  # 遍历数据列表中的每个文件名
  for mtx_name in file_list:
    # 构建每个通道的完整路径
    channel_images = []
    for suffix in channel_suffixes:
        image_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{suffix}")
        # 读取图像数据，假设数据以文本形式保存，每行一个数值
        img_data = np.loadtxt(image_path)
        # 将一维数组重塑为一维图像形状 (256x1)
        img_data = img_data.reshape((image_dim))
        channel_images.append(img_data)
    
    # 检查所有通道图像的形状是否一致，以确保能够正确堆叠
    if all(img.shape == channel_images[0].shape for img in channel_images):
        # 沿着最后一个轴将三个通道的数据堆叠起来
        multi_channel_image = np.stack(channel_images, axis=-1)
        image_list.append(multi_channel_image)
    else:
        print(f"Error: Image channels for {mtx_name} have mismatched dimensions.")
  # 将图像列表转换为一个numpy数组
  image_array = np.array(image_list)
  return image_array

def read_img_density(data_list, base_path, channel_suffixes=('.nnz',)):
  file_list = []
  with open(data_list, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
  # 初始化图像数组列表
  image_list = []
  # 遍历数据列表中的每个文件名
  for mtx_name in file_list:
    # 构建每个通道的完整路径
    channel_images = []
    for suffix in channel_suffixes:
        image_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{suffix}")
        # Check if the file exists before loading
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue
        
        # Print the file path for debugging
        # print(f"Reading image from: {image_path}")
    
        # 读取图像数据，假设数据以文本形式保存，每行一个数值
        img_data = np.loadtxt(image_path)
        # 将一维数组重塑为二维图像形状 (256x256)
        img_data = img_data.reshape((image_dim, image_dim))
        channel_images.append(img_data)
    # 检查所有通道图像的形状是否一致，以确保能够正确堆叠
    if all(img.shape == channel_images[0].shape for img in channel_images):
        # 沿着最后一个轴将三个通道的数据堆叠起来
        multi_channel_image = np.stack(channel_images, axis=-1)
        image_list.append(multi_channel_image)
    else:
        print(f"Error: Image channels for {mtx_name} have mismatched dimensions.")
  # 将图像列表转换为一个numpy数组
  image_array = np.array(image_list)
  return image_array

def read_features(data_list_path, base_path):
  file_list = []
  with open(data_list_path, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
      
  feature_array = []
  for mtx_name in file_list:
    feature_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{feat_file_suffix}")
    feature = []
    with open(feature_path, "r") as f_read:
      lines = f_read.readlines()
      feats = lines[1:]  # 跳过第一行
      for feat in feats:
          parts = feat.strip().split()
          if len(parts) < 2:
              continue
          feat_value = float(parts[1])
          feature.append(feat_value)

    # print("Feature list before adding to feature_array:", feature)  # 打印每个特征列表
    feature_array.append(feature)
  # 转换列表为NumPy数组前再次检查
  # print("Feature array before conversion to NumPy array:", feature_array)

  # 转换为NumPy数组
  feature_array = np.array(feature_array)
  # print("Final NumPy array:", feature_array)  # 打印最终的NumPy数组
  return feature_array

def read_labels(data_list_path, base_path, label_file_suffix):
  file_list = []
  with open(data_list_path, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
      
  label_array = []
  for mtx_name in file_list:
    # label_path = os.path.join(base_path, file_ + label_file_suffix)
    label_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{label_file_suffix}")
    with open(label_path, "r") as f_read:
      label = f_read.readline().strip().split(" ")[1]
    label_array.append(int(label))
  return np.array(label_array)

def read_labels_prob(data_list_path, base_path, label_file_suffix):
  file_list = []
  with open(data_list_path, "r") as f:
    lines = f.readlines()
    for line in lines:
      file_list.append(line.strip())
      
  label_array = []
  for mtx_name in file_list:
    # label_path = os.path.join(base_path, file_ + label_file_suffix)
    label_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{label_file_suffix}")
    with open(label_path, "r") as f_read:
      # 读取标签文件的第一行
      line = f_read.readline().strip()
      # 分割字符串获取所有概率值，忽略第一个字符串（格式名称）
      probabilities = line.split()[1:]  # 从第二个元素开始是概率值
      # 将字符串概率转换为浮点数
      label = [float(prob) for prob in probabilities]
      label_array.append(label)

  return np.array(label_array)

# data_list 保存的是数据集中的 matrix name
# image_data 存储为 128 *128 = 16384 行
# feat_data  存储为 18 行，每行: 特征名字 <空格> 特征值 
# label_data 存储为 1  行，每行: 格式名 <空格> 对应格式label号
def get_train_data(data_list, label_file_suffix=label_format_suffix, root_dir=base_path):
  # 读取
  image_array = read_images(data_list, root_dir)    # data list name, image data path
  
  Row_Block_array = read_1D_images(data_list, root_dir, RB_suffix)
  Col_Block_array = read_1D_images(data_list, root_dir, CB_suffix)
  
  feat_array = read_features(data_list, root_dir)
  label_array = read_labels(data_list, root_dir, label_file_suffix)
  
  # 对特征数据进行标准化处理。标准化是机器学习预处理中常用的方法，
  # 目的是将特征数据规范到一个标准的范围内，通常是一个均值为0，标准差为1的分布。
  # 这有助于模型更好地学习和收敛。
  feat_array = st.fit_transform(feat_array)

  return image_array, Row_Block_array, Col_Block_array, feat_array, label_array

# 没有多模态， 只考虑读取一个density representation
def get_train_density(data_list, label_file_suffix=label_format_suffix, root_dir=base_path):
  image_density = read_img_density(data_list, root_dir)
  feat_array = read_features(data_list, root_dir)
  label_array = read_labels_prob(data_list, root_dir, label_file_suffix)
  
  feat_array = st.fit_transform(feat_array)
  
  return image_density, feat_array, label_array

# 读取的是 概率向量标签
def get_train_data_new(data_list, label_file_suffix=label_prob_suffix, root_dir=base_path):
  # 读取
  image_array = read_images(data_list, root_dir)    # data list name, image data path
  
  Row_Block_array = read_1D_images(data_list, root_dir, RB_suffix)
  Col_Block_array = read_1D_images(data_list, root_dir, CB_suffix)
  
  feat_array = read_features(data_list, root_dir)
  label_array = read_labels_prob(data_list, root_dir, label_file_suffix)
  
  # 对特征数据进行标准化处理。标准化是机器学习预处理中常用的方法，
  # 目的是将特征数据规范到一个标准的范围内，通常是一个均值为0，标准差为1的分布。
  # 这有助于模型更好地学习和收敛。
  feat_array = st.fit_transform(feat_array)

  return image_array, Row_Block_array, Col_Block_array, feat_array, label_array


def get_test_data(data_list, label_file_suffix=label_format_suffix, root_dir=base_path):
  image_array_test = read_images(data_list, root_dir)
  Row_Block_array_test = read_1D_images(data_list, root_dir, RB_suffix)
  Col_Block_array_test = read_1D_images(data_list, root_dir, CB_suffix)
  
  feat_array_test = read_features(data_list, root_dir)
  label_array_test = read_labels(data_list, root_dir, label_file_suffix)
  feat_array_test = st.fit_transform(feat_array_test)
  
  test_data = tf.data.Dataset.from_tensor_slices((feat_array_test, image_array_test, Row_Block_array_test, Col_Block_array_test,  label_array_test))
  
  return image_array_test, Row_Block_array_test, Col_Block_array_test, feat_array_test, label_array_test, test_data

def get_test_Density(data_list, label_file_suffix=label_format_suffix, root_dir=base_path):
  image_array_test = read_img_density(data_list, root_dir)
  
  feat_array_test = read_features(data_list, root_dir)
  label_array_test = read_labels(data_list, root_dir, label_file_suffix)
  feat_array_test = st.fit_transform(feat_array_test)
  
  test_data = tf.data.Dataset.from_tensor_slices((feat_array_test, image_array_test, label_array_test))
  
  return image_array_test, feat_array_test, label_array_test, test_data

def get_test_data_new(data_list, label_file_suffix=label_prob_suffix, root_dir=base_path):
  image_array_test = read_images(data_list, root_dir)
  Row_Block_array_test = read_1D_images(data_list, root_dir, RB_suffix)
  Col_Block_array_test = read_1D_images(data_list, root_dir, CB_suffix)
  
  feat_array_test = read_features(data_list, root_dir)
  label_array_test = read_labels_prob(data_list, root_dir, label_file_suffix)
  feat_array_test = st.fit_transform(feat_array_test)
  
  test_data = tf.data.Dataset.from_tensor_slices((feat_array_test, image_array_test, Row_Block_array_test, Col_Block_array_test,  label_array_test))
  
  return image_array_test, Row_Block_array_test, Col_Block_array_test, feat_array_test, label_array_test, test_data