from pathlib import Path
import random
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
    
class Custom_Dataset(Dataset):
    def __init__(self, root, phase="train", basic_aug=True, transform=None):
        """
        Args:
            root (str): 数据集根目录（如rafdb/）。
            phase (str): 训练或测试阶段（"train"/"test"）。
            basic_aug (bool): 是否启用基础增强（如随机翻转）。
            transform (callable): 主图像预处理/增强。
        """
        self.root = Path(root) / phase
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform

        # 使用ImageFolder自动解析目录结构
        self.image_folder = datasets.ImageFolder(self.root)
        self.samples = self.image_folder.samples  # 文件路径列表
        self.targets = self.image_folder.targets  # 标签列表
        self.label = self.targets.copy()
        self.classes = self.image_folder.classes  # 类别名称
        self.class_to_idx = self.image_folder.class_to_idx  # 类别到索引的映射
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取原始图像和标签
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        # 应用主transform（如ToTensor+Normalize）
        if self.transform:
            image_transformed = self.transform(image)
        else:
            image_transformed = transforms.ToTensor()(image)

        return image_transformed, label, idx