
from mynet import MAE_ViT

import torch
import torch.nn as nn
from get_dataset import UnlabeledDataset, labeledDataset
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.utils import make_grid



def main():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    unlabeled_dataset = UnlabeledDataset(root='D:/FERexperiments/datasets/FERPlus', phase='train', transform=transform)
    train_dataset = labeledDataset(root='D:/FERexperiments/datasets/RAF-DB', phase='train', transform=transform)
    test_dataset = labeledDataset(root='D:/FERexperiments/datasets/RAF-DB', phase='test', transform=transform)

    pretrain_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    finetune_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    finetune_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAE_ViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Initializing...")



    print("stage 1")
    # 阶段1：无监督预训练
    model.stage = 1
    for epoch in range(4):
        model.train()
        loss = 0.0
        for batch_idx, imgs in enumerate(pretrain_dataloader):
            imgs = imgs.to(device)  # 确保数据在GPU
            
            # 前向传播
            recon = model(imgs)
            loss += nn.MSELoss()(recon, model.patch_embed(imgs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/100] | Batch [{batch_idx+1}/{len(pretrain_dataloader)}] | Loss: {loss.item():.4f}")
            
    
    # 阶段2：监督微调
    model.stage = 2
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    for param in model.encoder.parameters():  
        param.requires_grad = False  

    print("stage 2")
    
    for epoch in range(5):
        model.train()
        train_loss = 0.0
        for imgs, labels in finetune_train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        best_acc = 0.0
        with torch.no_grad():
            for imgs, labels in finetune_test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                loss = criterion(preds, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # 打印统计信息
        train_loss /= len(finetune_train_loader)
        test_loss /= len(finetune_test_loader)
        test_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/50] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Accuracy: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

if __name__ == '__main__':
    main()