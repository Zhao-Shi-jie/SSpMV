import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

class TC_SpMV_Adapter(nn.Module):
    def __init__(self, num_classes, feat_dim=40):
        super(TC_SpMV_Adapter, self).__init__()
        
        # ==========================================
        # 1. Image Branch (DeepModel)
        # Input: (Batch, 3, 256, 256)
        # TF: Conv2D(16, 3, same) -> Pool -> Conv2D(16, 5, stride=2, same) -> Pool -> Flatten -> Dense(32)
        # ==========================================
        self.cnn = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # padding=1 <-> 'same' (k=3, s=1)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # -> (16, 128, 128)
            
            # Conv2
            # TF: kernel=5, stride=2, padding='same'. Output size = ceil(Input / Stride)
            # PyTorch: padding=2 for k=5 ensures output size is input/stride
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # -> (16, 32, 32)
            
            nn.Flatten(),                               # -> 16 * 32 * 32 = 16384
            nn.Linear(16384, 32)                        # Output: 32
            # 注意：TF DeepModel 中没有最后的 ReLU，直接输出 Dense(32) 的结果
        )
        
        # ==========================================
        # 2. Row Block Branch (Conv1DModel)
        # TF: Conv1D(16, 3, same) -> Pool -> Conv1D(16, 5, stride=2, same) -> Pool -> Flatten -> Dense(32)
        # ==========================================
        self.rb_cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # TF 使用了 Flatten -> Dense(32)。
            # 如果输入长度不固定，这里会报错。假设输入长度固定（例如 256），则可以计算 Flatten 大小。
            # 为了兼容性，这里保留 AdaptiveAvgPool 方案，或者你需要确认输入长度。
            # 如果要严格对齐 TF 且输入长度固定，需要知道长度 L。
            # 假设使用 AdaptiveAvgPool1d(1) 来模拟全局特征提取，然后映射到 32
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),           # (Batch, 16)
            nn.Linear(16, 32)       # Output: 32
        )
        
        # ==========================================
        # 3. Col Block Branch (Conv1DModel)
        # ==========================================
        self.cb_cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, 32)
        )
        
        # ==========================================
        # 4. Feature Branch (WideModel)
        # TF: BN -> Dense(512) -> Dense(1024) -> Dense(32)
        # ==========================================
        self.feat_mlp = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32) # Output: 32
        )
        
        # ==========================================
        # 5. Fusion & Classifier (SpMV_Adapter)
        # TF: Concatenate([Wide, Deep, RB, CB]) -> Dense(512) -> Dropout -> Dense(256) -> Dropout -> Dense(Num)
        # Inputs: 32 + 32 + 32 + 32 = 128
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, img, rb, cb, feat):
        x_img = self.cnn(img)
        
        # For 1D CNNs with AdaptiveAvgPool, we need to ensure input is (N, C, L)
        x_rb = self.rb_cnn(rb)
        x_cb = self.cb_cnn(cb)
        
        x_feat = self.feat_mlp(feat)
        
        # Concatenate
        combined = torch.cat((x_img, x_rb, x_cb, x_feat), dim=1)
        output = self.classifier(combined)
        return output

def train_model(model, train_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
    model.to(device)
    criterion = nn.KLDivLoss(reduction='batchmean') # 适合概率分布标签
    # 或者使用 MSELoss 如果视作回归: criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            img = batch['image'].to(device)
            rb = batch['rb'].to(device)
            cb = batch['cb'].to(device)
            feat = batch['feat'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(img, rb, cb, feat)
            
            # KLDivLoss expects input in log-space
            loss = criterion(torch.log(outputs + 1e-10), labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, result_path, device='cuda'):
    model.to(device)
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            img = batch['image'].to(device)
            rb = batch['rb'].to(device)
            cb = batch['cb'].to(device)
            feat = batch['feat'].to(device)
            
            outputs = model(img, rb, cb, feat)
            
            # 获取最大概率对应的类别索引
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            
    # 保存预测结果
    with open(result_path, 'w') as f:
        for p in predictions:
            f.write(f"{p}\n")
    print(f"Predictions saved to {result_path}")
