import os
import numpy as np

class AugmentedKITTIDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
            
        self.image_filenames = []
        self.label_filenames = []
        
        potential_images = [f for f in sorted(os.listdir(image_dir))
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(potential_images)} potential images")
        
        for img_file in potential_images:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(os.path.join(label_dir, label_file)):
                self.image_filenames.append(img_file)
                self.label_filenames.append(label_file)
        
        print(f"Found {len(self.image_filenames)} valid image-label pairs")
        
        if len(self.image_filenames) == 0:
            raise RuntimeError("No valid image-label pairs found!")
    
        self.class_mapping = { 
            'car': 1,
            'cyclist': 2,
            'dontcare': 3,
            'misc': 4,
            'pedestrian': 5,
            'person_sitting': 6,
            'tram': 7,
            'truck': 8,
            'van': 9
        }

    def create_label_image(self, label_path, original_shape):
        mask = np.zeros(original_shape[:2], dtype=np.int64)
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) < 15:
                    continue
                class_name = parts[0].lower()
                if class_name not in self.class_mapping:
                    continue
                    
                class_idx = self.class_mapping[class_name]
                
                x1, y1, x2, y2 = map(float, parts[4:8])
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(original_shape[1]-1, int(x2)), min(original_shape[0]-1, int(y2))
                
                if x1 >= x2 or y1 >= y2:
                    continue
                    
                mask[y1:y2+1, x1:x2+1] = class_idx
                
        except Exception as e:
            print(f"Error processing {label_path}: {e}")
            return np.zeros(original_shape[:2], dtype=np.int64)
            
        return mask