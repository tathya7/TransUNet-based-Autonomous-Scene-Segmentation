import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from networks.vision_transformer import SwinUnet
from config import get_config
import matplotlib.colors as mcolors
import argparse

def parse_args():
    """
    Parse command line arguments
    Args:
        base dir: Base directory for KITTI dataset
        img-height: Input image height
        img-width: Input image width
        num-classes: Number of segmentation classes
        batch-size: Batch size for testing
        result-dir: Directory to save results
        model-path: Path to the trained model
        config-path: Path to model configuration file
    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Test SwinUnet on KITTI dataset")
    parser.add_argument('--base-dir', type=str, required=True, help='Base directory for KITTI dataset')
    parser.add_argument('--img-height', type=int, default=224, help='Input image height')
    parser.add_argument('--img-width', type=int, default=224, help='Input image width')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of segmentation classes')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--result-dir', type=str, default='test_results/swinunet', help='Directory to save results')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--config-path', type=str, default='configs/swinunet_config.yaml', help='Path to model configuration file')
    return parser.parse_args()

class TestKITTIDataset(Dataset):
    """
    Custom dataset class for KITTI dataset
    Args:
        image_dir: Path to the image directory
        img_height: Input image height
        img_width: Input image width
        transform: PyTorch transformations to apply
    
    Returns:
        image: Transformed image tensor
        original_image: Original image
        file_name: Image file name
        original_height: Original image height
        original_width: Original image
    """
    def __init__(self, image_dir, img_height=224, img_width=224, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.img_height = img_height
        self.img_width = img_width
        self.image_files = []
        for f in sorted(os.listdir(image_dir)):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                self.image_files.append(f)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        original_size = (image.shape[0], image.shape[1])  
        image = cv2.resize(image, (self.img_width, self.img_height))


        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        if self.transform:
            image = self.transform(image)
        

        return {
            'image': image,
            'original_image': original_image,
            'file_name': self.image_files[idx],
            'original_height': original_size[0],
            'original_width': original_size[1]
        }

def save_predictions(original_image, images, predictions, NUM_CLASSES, RESULT_DIR, batch_idx):
    """
    Save the original image, predicted mask, and overlay of the original image and predicted mask
    Args:
        original_image: Original image
        images: Transformed image tensor
        predictions: Predicted segmentation mask
        NUM_CLASSES: Number of segmentation classes
        RESULT_DIR: Directory to save results
        batch_idx: Batch index
    
    Returns:
        None
    """
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(NUM_CLASSES)]
    colors[0] = (0, 0, 0, 0)  
    custom_cmap = mcolors.ListedColormap(colors)

    for i in range(images.shape[0]):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        original_height, original_width = original_image.shape[:2]
        img = cv2.resize(img, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        pred = predictions[i].cpu().numpy()
        pred = cv2.resize(pred, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.05)
        ax1.imshow(original_image)
        ax1.set_title("Original Image", {'fontsize' : 8})
        ax1.axis('off')

        ax2.imshow(original_image)
        ax2.imshow(pred, cmap=custom_cmap, vmin=0, vmax=NUM_CLASSES-1)
        ax2.set_title("Predicted Mask", {'fontsize' : 8})
        ax2.axis('off')

        ax3.imshow(original_image)
        ax3.imshow(pred, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=NUM_CLASSES-1)
        ax3.set_title("Segmented Output", {'fontsize' : 8})
        ax3.axis('off')
        
        plt.savefig(f'{RESULT_DIR}/prediction_{batch_idx}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

def main():
    """
    Main function to test SwinUnet on KITTI dataset
    Args:
        config-path: Path to model configuration file   
    Returns
        None
    """
    args = parse_args()

    BASE_DIR = args.base_dir
    TEST_IMAGE = os.path.join(BASE_DIR, "single_image")
    IMG_HEIGHT = args.img_height
    IMG_WIDTH = args.img_width
    NUM_CLASSES = args.num_classes
    BATCHSIZE = args.batch_size
    RESULT_DIR = args.result_dir
    model_path = args.model_path
    CONFIG_PATH = args.config_path

    config = get_config(CONFIG_PATH)
    config.defrost()
    config.MODEL.SWIN.EMBED_DIM = 96
    config.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    config.MODEL.SWIN.DECODER_DEPTHS = [2, 2, 2, 2]
    config.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    config.MODEL.DROP_PATH_RATE = 0.3
    config.freeze()

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestKITTIDataset(
        TEST_IMAGE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        transform=transform
    )

    test_loader = DataLoader(test_dataset,
                             batch_size=BATCHSIZE,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinUnet(config, img_size=IMG_HEIGHT, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("Model Evaluating")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            torch.cuda.synchronize()
            start_time = time.time()
            images = batch['image'].to(device)
            original_image = batch['original_image'][0].numpy()
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            torch.cuda.synchronize()
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            print(f"Inference time (GPU): {inference_time:.2f} ms")
            save_predictions(original_image, images, predictions, NUM_CLASSES, RESULT_DIR, batch_idx)

if __name__ == "__main__":
    main()
