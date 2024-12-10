from torch.utils.data import Dataset
import cv2
import os
import torch
import numpy as np
from torchvision import transforms

class AugmentedKITTIDataset(Dataset):
    """
    KITTI dataset with image and label directories

    Args:
    image_dir (str): Path to image directory
    label_dir (str): Path to label directory
    transform (albumentations.Compose): Albumentations transforms
    img_height (int): Image height
    img_width (int): Image width

    Returns:
    torch.utils.data.Dataset: Dataset class
    """
    def __init__(self, image_dir, label_dir, transform=None, img_height=224, img_width=224):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        
        self.image_filenames = []
        for f in sorted(os.listdir(image_dir)):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                label_file = os.path.splitext(f)[0] + '.txt'
                if os.path.exists(os.path.join(label_dir, label_file)):
                    self.image_filenames.append(f)
                    
        self.class_mapping = {
            'car': 1, 'cyclist': 2, 'dontcare': 3, 'misc': 4,
            'pedestrian': 5, 'person_sitting': 6, 'tram': 7,
            'truck': 8, 'van': 9
        }

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, 
                                os.path.splitext(self.image_filenames[idx])[0] + '.txt')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_width, self.img_height))
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.transform:
            image = self.transform(image)

        mask = self.create_label_image(label_path, image.shape[-2:])
        return image, mask

    def create_label_image(self, label_path, output_shape):
        """
        Create a class index mask from KITTI format label file

        Args:
        label_path (str): Path to label file
        output_shape (tuple): Output image dimensions
        mask (torch.Tensor): Class index mask
        original_h, original_w (int): Original image dimensions
        class_name (str): Object class name
        class_idx (int): Object class index
        x1, y1, x2, y2 (int): Bounding box coordinates
        
        Returns:
        torch.Tensor: Class index mask
        """
        mask = torch.zeros(output_shape, dtype=torch.long)
        try:
            original_h, original_w = 375, 1242 
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) < 15:
                        continue
                    
                    class_name = parts[0].lower()
                    if class_name not in self.class_mapping:
                        continue
                    
                    class_idx = self.class_mapping[class_name]
                    x1, y1, x2, y2 = map(float, parts[4:8])
                    x1 = int(x1 * self.img_width / original_w)
                    y1 = int(y1 * self.img_height / original_h)
                    x2 = int(x2 * self.img_width / original_w)
                    y2 = int(y2 * self.img_height / original_h)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.img_width-1, x2), min(self.img_height-1, y2)
                    
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    mask[y1:y2+1, x1:x2+1] = class_idx
        except Exception as e:
            print(f"Error processing {label_path}: {e}")
        
        return mask
