import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import SpMVDataset, calculate_metrics
from model import TC_SpMV_Adapter, train_model, evaluate_model

# 配置路径
BASE_PATH = os.path.expanduser("~/computeKernel/lcuda-project/SpLibrary/script/MModel-Data")
TRAIN_LIST = os.path.expanduser("~/computeKernel/lcuda-project/SpLibrary/script/collect/matrices_train.txt")
TEST_LIST = os.path.expanduser("~/computeKernel/lcuda-project/SpLibrary/script/collect/matrices_test.txt")
MODEL_SAVE_PATH = os.path.expanduser("~/computeKernel/lcuda-project/SpLibrary/script/TC_MM_Adapter_pb/12-3/TC_SpMV_Adapter_prob.pth")
RESULT_PATH = os.path.expanduser("~/computeKernel/lcuda-project/SpLibrary/script/TC_MM_Adapter_pb/12-3/predict.res")
METRIC_PATH = os.path.expanduser("~/computeKernel/lcuda-project/SpLibrary/script/TC_MM_Adapter_pb/12-3/metrics.res")

IN_BASE_PATH = os.path.expanduser("~/MModel-Data")
OUT_BASE_PATH = os.path.expanduser("~/computeKernel/lcuda-project/SpLibrary/script/MModel-Data")

# 确保输出目录存在
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# 参数设置
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    settings_idx = 0
    # 注意：PyTorch通常不需要 "just a pad"，这里只取前两个
    label_class = [".prob_label", ".det_prob_label"] 
    # 对应的类别数量，需要根据实际情况填写，这里假设 settings_idx=2 对应某种配置
    # 假设 number_of_labels 是一个列表，这里手动定义一下，你需要根据原代码确认
    number_of_labels = [4, 3, 5] # 示例值，请根据 TC_SpMV_Adapter_prob.py 中的定义修改
    
    current_label_suffix = label_class[settings_idx] # 默认使用 .prob_label，根据 settings_idx 调整

    num_classes = number_of_labels[settings_idx]
    
    print(f"Running Prob Model with setting: [{settings_idx}], Classes: {num_classes}, Device: {DEVICE}")

    # 1. 准备数据
    print("Preparing Training Data...")
    train_dataset = SpMVDataset(TRAIN_LIST, IN_BASE_PATH, OUT_BASE_PATH, current_label_suffix)
    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
      # ============================================================
    # [策略1核心修改] 使用 WeightedRandomSampler 处理类别不平衡
    # ============================================================
    print("Calculating weights for imbalanced dataset...")
    
    # 1. 获取训练集所有样本的类别索引 (假设标签是概率分布，取最大值)
    # train_dataset.labels 是 Tensor (N, num_classes)
    y_train_indices = torch.argmax(train_dataset.labels, dim=1)
    
    # 2. 统计每个类别的样本数量
    class_sample_counts = torch.bincount(y_train_indices)
    print(f"Class counts: {class_sample_counts}")
    
    # 3. 计算每个类别的权重 (样本越少，权重越大)
    # weight = 1.0 / count
    # 注意：如果有类别样本数为0，需要特殊处理，但这里假设都有样本
    class_weights = 1. / class_sample_counts.float()
    
    # 4. 为每个样本分配对应的类别权重
    samples_weights = class_weights[y_train_indices]
    
    # 5. 创建采样器
    # replacement=True 表示允许重复采样 (过采样少数类)，这是关键
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )
    
    # 6. 将采样器传递给 DataLoader
    # 注意：使用 sampler 时，shuffle 必须设置为 False
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,  # <--- 传入采样器
        shuffle=False     # <--- 必须为 False
    )

    print("Preparing Test Data...")
    test_dataset = SpMVDataset(TEST_LIST, IN_BASE_PATH, OUT_BASE_PATH, current_label_suffix)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    model = TC_SpMV_Adapter(num_classes=num_classes)
    
    # 3. 训练模型
    print("Start Training...")
    train_model(model, train_loader, num_epochs=EPOCHS, learning_rate=LEARNING_RATE, device=DEVICE)
    
    # 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # 4. 测试模型
    print("Start Evaluation...")
    # 加载模型 (如果是分开运行，可以取消注释下面这行)
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    evaluate_model(model, test_loader, RESULT_PATH, device=DEVICE)

    # 5. 计算指标
    print("Calculating Metrics...")
    calculate_metrics(RESULT_PATH, None, TEST_LIST, BASE_PATH, num_classes, METRIC_PATH)

if __name__ == "__main__":
    main()
