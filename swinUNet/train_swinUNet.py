import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from networks.vision_transformer import SwinUnet
from config import get_config
from AugmentedKITTIDataset import AugmentedKITTIDataset


def calculate_class_weights(dataset, num_classes, batch_size=32):
    """
    Calculate class weights efficiently using batched processing
    """
    print("Calculating class weights (this may take a few minutes)...")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    class_counts = torch.zeros(num_classes)
    for batch in tqdm(loader, desc="Computing class weights"):
        if isinstance(batch, (tuple, list)):
            _, masks = batch
        for i in range(num_classes):
            class_counts[i] += (masks == i).sum().item()
    total = class_counts.sum()
    weights = total / (class_counts + 1e-6)  
    weights = weights / weights.sum()  
    
    print("\nClass distribution:")
    for i in range(num_classes):
        print(f"Class {i}: {class_counts[i]} pixels ({(class_counts[i]/total)*100:.2f}%)") 
    return weights

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, save_dir):

    model.train()
    epoch_loss = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if batch_idx % 50 == 0:
                    save_predictions(images, masks, outputs, epoch, batch_idx, save_dir)
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    return epoch_loss / len(train_loader)

base_dir = "/home/amogha/.cache/kagglehub/datasets/klemenko/kitti-dataset/versions/1"
IMAGE_DIR = os.path.join(base_dir, "data_object_image_2/training/image_2/")
LABEL_DIR = os.path.join(base_dir, "data_object_label_2/training/label_2/")
IMG_SIZE = 224
NUM_CLASSES = 10 
MAX_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset-specific augmentations and preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
  
dataset = AugmentedKITTIDataset(IMAGE_DIR, LABEL_DIR,transform=transform)
trainLoader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=True,num_workers=4,pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "swinUNet_results"
os.makedirs(save_dir, exist_ok=True)
config_path = "configs/swinunet_config.yaml"  # Adjust the path to your config file
config = get_config(config_path)
model = SwinUnet(config, img_size=IMG_SIZE, num_classes=NUM_CLASSES).to(DEVICE)

weights = calculate_class_weights(dataset,NUM_CLASSES)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=MAX_EPOCHS,
        steps_per_epoch=len(trainLoader),
        pct_start=0.3
    )
best_loss = float('inf')
    
for epoch in range(MAX_EPOCHS):
    avg_loss = train_epoch(model, trainLoader, criterion, optimizer, scheduler, device, epoch, save_dir)
    print(f'Epoch {epoch+1}/{MAX_EPOCHS}, Average Loss: {avg_loss:.4f}')
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, f'{save_dir}/best_model.pth')
        print(f'Saved new best model with loss: {best_loss:.4f}')

print("Training completed!")
