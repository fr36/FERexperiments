import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mae_vit import MAEViT
from get_dataset import UnlabeledDataset, labeledDataset

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def show_images(imgs, title):
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(img.permute(1, 2, 0).clip(0, 1))
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 加载数据集 (这里假设使用FER2013数据集)
train_dataset = labeledDataset(root='D:/FERexperiments/datasets/RAF-DB', phase='train', transform=transform)
val_dataset = labeledDataset(root='D:/FERexperiments/datasets/RAF-DB', phase='val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MAEViT().to(device)

# Stage 1: MAE预训练
print("Starting Stage 1: MAE Pre-training")
optimizer = optim.AdamW(model.parameters(), lr=1.5e-4)
criterion = nn.MSELoss()

for epoch in range(2):  # 预训练100个epoch
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        pred, mask = model(data, mask_ratio=0.75, is_train=True)
        
        # 计算重建损失
        loss = criterion(pred, data.flatten(2).transpose(1, 2))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader)}')

# 保存预训练模型
torch.save(model.state_dict(), 'mae_pretrained.pth')

# 随机选择3张图片进行重建展示
model.eval()
with torch.no_grad():
    val_iter = iter(val_loader)
    images, _ = next(val_iter)
    images = images[:3].to(device)
    
    # 获取重建结果
    pred, mask = model(images, mask_ratio=0.75, is_train=True)
    
    # 反归一化并显示结果
    images = unnormalize(images.cpu())
    pred = pred.reshape(pred.shape[0], 224, 224, 3).permute(0, 3, 1, 2)
    pred = unnormalize(pred.cpu())
    
    show_images(images, "Original Images")
    show_images(pred, "Reconstructed Images")

# Stage 2: 监督微调
print("Starting Stage 2: Supervised Fine-tuning")
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):  # 微调50个epoch
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data, is_train=False)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader)}, '
          f'Accuracy: {accuracy}%')

# 保存最终模型
torch.save(model.state_dict(), 'mae_finetuned.pth')