import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random

class CustomImageFolder(ImageFolder):
    def __init__(self, root, phase='train', basic_aug=True, transform=None, clean=True):
        super().__init__(root, transform=transform)
        self.phase = phase
        self.basic_aug = basic_aug
        self.clean = clean
        self.aug_func = [self.flip_image, self.add_g]
        
    @staticmethod
    def flip_image(image):
        return transforms.functional.hflip(image)
        
    @staticmethod
    def add_g(image):
        # Add Gaussian noise
        noise = torch.randn_like(image) * 0.1
        return torch.clamp(image + noise, 0, 1)
        
    def __getitem__(self, idx):
        # Get original ImageFolder output
        image, label = super().__getitem__(idx)
        
        # Create image1 based on clean flag
        if not self.clean:
            image1 = image
            image1 = self.aug_func[0](image1)
            if self.transform is not None:
                image1 = self.transform(image1)
        else:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)
            
        # Apply basic augmentation if in train phase
        if self.phase == 'train' and self.basic_aug and random.uniform(0, 1) > 0.5:
            image = self.aug_func[1](image)
            
        return image, label, idx, image1
