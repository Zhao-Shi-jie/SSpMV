import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# 全局配置
IMAGE_DIM = 256
RB_SUFFIX = ('.RBave', '.RBmax', '.RBstd')
CB_SUFFIX = ('.CBave', '.CBmax', '.CBstd')
CHANNEL_SUFFIXES = ('.ave', '.max', '.std')

def read_images(data_list, base_path, channel_suffixes=CHANNEL_SUFFIXES):
    """读取多通道图像数据"""
    image_list = []
    for mtx_name in data_list:
        channel_images = []
        for suffix in channel_suffixes:
            image_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{suffix}")
            image_path = os.path.expanduser(image_path) # 处理 ~ 路径
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"{image_path} not found.")
                
            img_data = np.loadtxt(image_path)
            img_data = img_data.reshape((IMAGE_DIM, IMAGE_DIM))
            channel_images.append(img_data)
        
        # Stack channels: (H, W, C) -> PyTorch needs (C, H, W) later
        multi_channel_image = np.stack(channel_images, axis=-1)
        image_list.append(multi_channel_image)
    
    return np.array(image_list)

def read_1D_images(data_list, base_path, channel_suffixes):
    """读取一维序列数据 (Row/Col Blocks)"""
    image_list = []
    for mtx_name in data_list:
        channel_images = []
        for suffix in channel_suffixes:
            image_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{suffix}")
            image_path = os.path.expanduser(image_path)
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"{image_path} not found.")
                
            img_data = np.loadtxt(image_path)
            # 假设是一维数据，保持原样或根据需要reshape
            channel_images.append(img_data)

        if all(img.shape == channel_images[0].shape for img in channel_images):
        # 沿着最后一个轴将三个通道的数据堆叠起来
            multi_channel_image = np.stack(channel_images, axis=-1)
            image_list.append(multi_channel_image)
        else:
            print(f"Error: Image channels for {mtx_name} have mismatched dimensions.")

    return np.array(image_list)

def read_features(data_list, base_path, feat_suffix=".features"):
    """读取手工特征"""
    feat_list = []
    for mtx_name in data_list:
        feat_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{feat_suffix}")
        feat_path = os.path.expanduser(feat_path)
        
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"{feat_path} not found.")
        
        feature = []
        with open(feat_path, "r") as f_read:
            lines = f_read.readlines()
            feats = lines[1:]  # 跳过第一行
            for feat in feats:
                parts = feat.strip().split()
                if len(parts) < 2:
                    continue
                feat_value = float(parts[1])
                feature.append(feat_value)
        feat_list.append(feature)
    
    feat_array = np.array(feat_list)
    # 标准化特征
    scaler = StandardScaler()
    feat_array = scaler.fit_transform(feat_array)
    return feat_array

def read_labels_prob(data_list, base_path, label_suffix):
    """读取概率标签"""
    label_list = []
    for mtx_name in data_list:
        label_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{label_suffix}")
        label_path = os.path.expanduser(label_path)
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"{label_path} not found.")
            
        label_data = np.loadtxt(label_path)
        label_list.append(label_data)
    return np.array(label_list)

def read_labels_format(data_list, base_path, label_suffix=".format_label"):
    """读取格式标签 (用于评估指标计算)"""
    label_list = []
    for mtx_name in data_list:
        label_path = os.path.join(base_path, f"{mtx_name}/{mtx_name}{label_suffix}")
        label_path = os.path.expanduser(label_path)
        if not os.path.exists(label_path):
             raise FileNotFoundError(f"{label_path} not found.")
        # 假设文件内容是 "FormatName LabelID"
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            parts = line.split()
            if len(parts) >= 2:
                label_list.append(int(parts[1]))
            else:
                label_list.append(int(parts[0]))
    return np.array(label_list)

class SpMVDataset(Dataset):
    def __init__(self, data_list_file, in_dir, out_dir, label_suffix, is_train=True):
        self.in_dir = os.path.expanduser(in_dir)
        self.out_dir = os.path.expanduser(out_dir)
        with open(data_list_file, "r") as f:
            self.data_list = [line.strip() for line in f.readlines()]
        
        # 加载数据
        print(f"Loading images for {len(self.data_list)} samples...")
        self.images = read_images(self.data_list, self.in_dir)
        print("Loading RB data...")
        self.rb_data = read_1D_images(self.data_list, self.in_dir, RB_SUFFIX)
        print("Loading CB data...")
        self.cb_data = read_1D_images(self.data_list, self.in_dir, CB_SUFFIX)
        print("Loading features...")
        self.features = read_features(self.data_list, self.in_dir)
        print("Loading labels...")
        self.labels = read_labels_prob(self.data_list, self.out_dir, label_suffix)
        
        # 转换为 PyTorch Tensor
        # Image: (N, H, W, C) -> (N, C, H, W)
        self.images = torch.FloatTensor(self.images).permute(0, 3, 1, 2)
        # 1D Data: (N, L, C) -> (N, C, L)
        self.rb_data = torch.FloatTensor(self.rb_data).permute(0, 2, 1)
        self.cb_data = torch.FloatTensor(self.cb_data).permute(0, 2, 1)
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(self.labels)

        # 输出数据维度信息
        print(f"Dataset initialized with {len(self.data_list)} samples.")
        print(f"Image shape: {self.images.shape}")
        print(f"RB data shape: {self.rb_data.shape}")
        print(f"CB data shape: {self.cb_data.shape}")
        print(f"Feature shape: {self.features.shape}")
        print(f"Label shape: {self.labels.shape}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'rb': self.rb_data[idx],
            'cb': self.cb_data[idx],
            'feat': self.features[idx],
            'label': self.labels[idx]
        }

def calculate_metrics(pred_file, true_label_file, data_list_file, base_path, label_nums, metric_file):
    """计算准确率和精度"""
    # 读取预测结果
    with open(pred_file, 'r') as f:
        preds = [int(line.strip()) for line in f.readlines()]
    
    # 读取真实标签 (Format Label)
    with open(data_list_file, "r") as f:
        data_list = [line.strip() for line in f.readlines()]
    true_labels = read_labels_format(data_list, base_path)
    
    # 计算 Accuracy
    correct = np.sum(np.array(preds) == np.array(true_labels))
    total = len(true_labels)
    acc = correct / total if total > 0 else 0
    
    # 计算 Precision (按类别)
    precisions = []
    for i in range(label_nums):
        tp = 0
        fp = 0
        for p, t in zip(preds, true_labels):
            if p == i:
                if t == i:
                    tp += 1
                else:
                    fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision)
    
    # 写入结果
    with open(metric_file, "a+") as f:
        f.write(f"Acc: {acc:.4f}\n")
        for i, p in enumerate(precisions):
            f.write(f"Precision Class {i}: {p:.4f}\n")
    
    print(f"Metrics saved to {metric_file}")
    return acc, precisions