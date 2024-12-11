from torch.utils.data import Dataset
import cv2
import os
import torch
import numpy as np

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
        """
        FOr Getting the length of dataset
        """
        return len(self.image_filenames)

    def create_label_image(self, label_path, original_image_shape):
        """
        Create a class index mask from KITTI format label file

        Args:
        label_path (str): Path to label file
        original_image_shape (tuple): Original image dimensions

        Returns:
        np.ndarray: Class index mask
        """
        original_h, original_w = original_image_shape[:2]
        mask = np.zeros((original_h, original_w), dtype=np.int32)
        
        try:
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
                    x1, x2 = max(0, int(x1)), min(original_w, int(x2))
                    y1, y2 = max(0, int(y1)), min(original_h, int(y2))
                    
                    if x1 >= x2 or y1 >= y2:
                        continue
                        
                    mask[y1:y2+1, x1:x2+1] = class_idx
                    
        except Exception as e:
            print(f"Error processing {label_path}: {e}")
            
        return mask

    def __getitem__(self, idx):
        """
        Get image and label at index

        Args:
        idx (int): Index
        img_path (str): Path to image file
        label_path (str): Path to label file
        image (np.ndarray): Image array
        mask (np.ndarray): Label mask array
        transformed (dict): Transformed image and mask

        Returns:
        torch.Tensor: Transformed image
        torch.Tensor: Transformed mask
        """
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir,
                                 os.path.splitext(self.image_filenames[idx])[0] + '.txt')
        

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.create_label_image(label_path, image.shape)
        
        if self.transform:
            try:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image'] 
                mask = transformed['mask']   
                if isinstance(mask, np.ndarray):
                    mask = torch.from_numpy(mask)
            except Exception as e:
                print(f"Transform error on image {img_path}: {e}")
                image = cv2.resize(image, (self.img_width, self.img_height))
                mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            image = cv2.resize(image, (self.img_width, self.img_height))
            mask = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)

        mask = mask.long()
        
        return image, mask
