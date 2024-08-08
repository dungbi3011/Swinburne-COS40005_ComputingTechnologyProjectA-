from pathlib import Path
import torchvision
import os
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

class Dataset(torchvision.datasets.VisionDataset):

    def __init__(self, root, transform, target_transform):
        super().__init__(root, transform, target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = []
        self.image_labels = []
        for class_name in os.listdir(root):
            class_path = os.path.join(root, class_name) 
            if os.path.isdir(class_path):
                for image_name in sorted(os.listdir(class_path), key=lambda file_name:int(file_name.split('_')[1].replace(".jpg", ""))):
                    img_path = os.path.join(class_path, image_name)
                    self.image_paths.append(img_path)
                    self.image_labels.append(image_name)
            else:
                raise ValueError('Invalid data folder structure')
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_label = self.image_labels[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            image_tensor = self.target_transform(image_tensor)
        return image, image_label