import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from networks.UNet import IUNet
import numpy as np
import cv2
import matplotlib.pyplot as plt
from AugmentedKITTIDataset import AugmentedKITTIDataset
from tqdm import tqdm
import os

BASE_DIR = "/afs/glue.umd.edu/home/glue/a/m/amishr17/home/.cache/kagglehub/datasets/klemenko/kitti-dataset/versions/1"
IMAGE_DIR = os.path.join(BASE_DIR, "data_object_image_2/training/image_2/")
LABEL_DIR = os.path.join(BASE_DIR, "data_object_label_2/training/label_2/")
IMG_HEIGHT, IMG_WIDTH = 128, 128
NUM_CLASSES = 10  

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

def save_predictions(images, masks, outputs, epoch, batch_idx, save_dir):
    with torch.no_grad():
        pred = outputs.argmax(dim=1)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        img_display = images[0].cpu().permute(1, 2, 0).numpy()
        img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        axes[0].imshow(np.clip(img_display, 0, 1))
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(masks[0].cpu(), cmap='tab10', vmin=0, vmax=NUM_CLASSES-1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        axes[2].imshow(pred[0].cpu(), cmap='tab10', vmin=0, vmax=NUM_CLASSES-1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.savefig(f'{save_dir}/epoch_{epoch+1}_batch_{batch_idx}.png')
        plt.close()

def main():
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    
    print("Setting up training...")
    train_transform = A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.8),
            A.RandomGamma(p=0.8),
        ], p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    print("Creating dataset...")
    try:
        dataset = AugmentedKITTIDataset(IMAGE_DIR, LABEL_DIR, transform=train_transform)
        print(f"Dataset created successfully with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        return

    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    print(f"DataLoader created with {len(train_loader)} batches")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    save_dir = "training_results"
    os.makedirs(save_dir, exist_ok=True)
    print("Initializing model...")
    model = IUNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

    # Calculate class weights
    weights = calculate_class_weights(dataset, NUM_CLASSES)
    print("Class weights calculated:", weights.numpy())

    # Setup training
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, save_dir)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
            print(f'Saved new best model with loss: {best_loss:.4f}')

    print("Training completed!")

if __name__ == "__main__":
    main()
