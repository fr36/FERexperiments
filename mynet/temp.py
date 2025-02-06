import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from einops import rearrange
from get_dataset import UnlabeledDataset, labeledDataset

# 图像预处理（包含归一化）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 反归一化转换
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return tensor * std + mean

# 模型定义
class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(16*16*3, 768)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=8), 6)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=8), 4)
        self.recon_head = nn.Linear(768, 16*16*3)
        
    def random_masking(self, x, mask_ratio=0.75):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        return x.gather(1, ids_shuffle[:, :len_keep].unsqueeze(-1).expand(-1, -1, D)), ids_shuffle

    def forward(self, x):
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
        x = self.patch_embed(patches)
        x, _ = self.random_masking(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.recon_head(x)

class FineTuneModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.res_blocks = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1)
        )
        self.classifier = nn.Linear(128, 7)
        
    def forward(self, x):
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
        x = self.encoder.patch_embed(patches)
        x = rearrange(self.encoder.encoder(x), 'b (h w) c -> b c h w', h=14)
        x = self.res_blocks(x)
        return self.classifier(x.mean(dim=[2,3]))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    
unlabeled_dataset = UnlabeledDataset(root='D:/FERexperiments/datasets/AffectNet', phase='val', transform=transform)
train_dataset = labeledDataset(root='D:/FERexperiments/datasets/RAF-DB', phase='train', transform=transform)
test_dataset = labeledDataset(root='D:/FERexperiments/datasets/RAF-DB', phase='test', transform=transform)

pretrain_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
finetune_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
finetune_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

def visualize_reconstruction(model, dataloader):
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(dataloader))  # 随机取3张
        recon_patches, _ = model(imgs[:3])
        
        # 将patch转换回图像
        recon_imgs = rearrange(recon_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                              h=14, p1=16, p2=16, c=3)
        
        # 可视化代码（需要matplotlib）
        fig, axes = plt.subplots(3, 2, figsize=(10,15))
        for i in range(3):
            axes[i,0].imshow(imgs[i].permute(1,2,0))
            axes[i,1].imshow(recon_imgs[i].permute(1,2,0))
        plt.show()

def main():
    # 阶段1：无监督预训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae = MAE().to(device)
    optimizer = torch.optim.AdamW(mae.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 预训练循环
    for epoch in range(2):
        for imgs in pretrain_dataloader:
            imgs = imgs.to(device)
            recon = mae(imgs)
            target = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
            loss = criterion(recon, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # 保存预训练模型
    torch.save(mae.state_dict(), "mae_pretrained.pth")
    
    # 阶段2前可视化
    mae.eval()
    with torch.no_grad():
        sample_imgs, _ = next(iter(finetune_test_loader))
        sample_imgs = sample_imgs[:3].to(device)
        recon_patches = mae(sample_imgs)
        
        # 重建图像反归一化
        recon_imgs = rearrange(recon_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                             h=14, p1=16, p2=16, c=3)
        original = denormalize(sample_imgs.cpu())
        reconstructed = denormalize(recon_imgs.cpu())
        
        # 可视化
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        for i in range(3):
            axes[i, 0].imshow(original[i].permute(1, 2, 0))
            axes[i, 1].imshow(reconstructed[i].permute(1, 2, 0))
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
        plt.show()
    
    # 阶段2：监督微调
    finetune_model = FineTuneModel(mae).to(device)
    optimizer = torch.optim.Adam(finetune_model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 微调循环
    for epoch in range(1):
        for imgs, labels in finetune_train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = finetune_model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == '__main__':
    main()



