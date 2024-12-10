import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from tqdm import tqdm
import os
from networks.vit_seg_modeling import CONFIGS, VisionTransformer
from AugmentedKITTIDataset_unet import AugmentedKITTIDataset
from utils import (
    calculate_class_weights,
    MetricTracker,
    save_metric_values,
    plot_metrics,
    save_per_class_iou,
    save_predictions_with_metrics
)

def parse_args():
    """
    Parse command line arguments, for training SwinUnet on KITTI dataset

    @param
    --base-dir: Base directory for KITTI dataset
    --img-height: Input image height
    --img-width: Input image width
    --num-classes: Number of segmentation classes
    --epochs: Number of training epochs
    --batch-size: Training batch size
    --lr: Learning rate
    --val-split: Validation split ratio
    --config-path: Path to model configuration file
    --embed-dim: Embedding dimension
    --checkpoint-freq: Checkpoint saving frequency in epochs
    --save-dir: Directory to save results
    
    @return
    args: Parsed command line arguments
    
    """
    default_base_dir = "/afs/glue.umd.edu/home/glue/a/m/amishr17/home/.cache/kagglehub/datasets/klemenko/kitti-dataset/versions/1"
    parser = argparse.ArgumentParser(description='TransUNet Training for KITTI Dataset')
    
    parser.add_argument('--base-dir', type=str, default=default_base_dir, help='Base directory for KITTI dataset')
    parser.add_argument('--img-height', type=int, default=224, help='Input image height (default: 224)')
    parser.add_argument('--img-width', type=int, default=224, help='Input image width (default: 224)')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of segmentation classes (default: 10)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio (default: 0.1)')
    parser.add_argument('--save-dir', type=str, default='transunet_results', help='Directory to save results (default: transunet_results)')
    
    return parser.parse_args()

def get_transunet_model(img_height, img_width, num_classes):
    """
    Initialize TransUNet model for segmentation

    @param
    img_height: Input image height
    img_width: Input image width
    num_classes: Number of segmentation classes

    @return
    model: TransUNet model
    """
    config_vit = CONFIGS['R50-ViT-B_16']
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(img_height/16), int(img_width/16))
    model = VisionTransformer(config_vit, img_size=img_height, num_classes=num_classes)
    return model

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, save_dir, metric_tracker):
    """
    Train model for one epoch

    @param
    model: SwinUnet model
    train_loader: DataLoader for training dataset
    criterion: Loss function - CrossEntropyLoss
    optimizer: Optimizer - AdamW
    scheduler: Learning rate scheduler - OneCycleLR
    device: Device to run model on
    epoch: Current epoch number
    save_dir: Directory to save predictions
    metric_tracker: Object to track metrics

    @return
    epoch_loss: Average loss for the epoch
    metric_tracker.get_metrics(): Metrics for the epoch

    """
    model.train()
    epoch_loss = 0
    metric_tracker.reset()
    
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
                
                epoch_loss += loss.item()
                pred = outputs.argmax(dim=1)
                for i in range(pred.shape[0]):
                    pred_mask = torch.zeros(outputs.shape[1], *pred[i].shape, device=device)
                    pred_mask.scatter_(0, pred[i].unsqueeze(0), 1)
                    target_mask = F.one_hot(masks[i], num_classes=outputs.shape[1])
                    target_mask = target_mask.permute(2, 0, 1).float().to(device)
                    
                    metric_tracker.update(pred_mask, target_mask)
                
                current_metrics = metric_tracker.get_metrics()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{current_metrics["dice_mean"]:.4f}',
                    'miou': f'{current_metrics["miou"]:.4f}'
                })
                
                if batch_idx % 500 == 0:
                    save_predictions_with_metrics(
                        images, masks, outputs, current_metrics,
                        epoch, batch_idx, save_dir, outputs.shape[1]
                    )
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
                
        scheduler.step()
    
    return epoch_loss / len(train_loader), metric_tracker.get_metrics()

def validate(model, val_loader, criterion, device, metric_tracker):
    """
     Validate model on validation dataset

    @param
    model: SwinUnet model
    val_loader: DataLoader for validation dataset
    criterion: Loss function - CrossEntropyLoss
    device: Device to run model on
    metric_tracker: Object to track metrics

    @return
    val_loss: Average loss on validation dataset
    metric_tracker.get_metrics(): Metrics on validation dataset

    """
    model.eval()
    val_loss = 0
    metric_tracker.reset()
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            pred = outputs.argmax(dim=1)
            for i in range(pred.shape[0]):
                pred_mask = torch.zeros(outputs.shape[1], *pred[i].shape, device=device)
                pred_mask.scatter_(0, pred[i].unsqueeze(0), 1)
                target_mask = F.one_hot(masks[i], num_classes=outputs.shape[1])
                target_mask = target_mask.permute(2, 0, 1).float().to(device)
                
                metric_tracker.update(pred_mask, target_mask)
    
    return val_loss / len(val_loader), metric_tracker.get_metrics()

def main():
    """
     Main function to train SwinUnet on KITTI dataset

    @param
    args: Command line arguments
    image_dir: Directory containing KITTI images
    label_dir: Directory containing KITTI labels
    device: Device to run model on
    dataset: AugmentedKITTIDataset object
    dataset_size: Size of dataset
    val_size: Size of validation dataset
    train_size: Size of training dataset
    train_loader: DataLoader for training dataset
    val_loader: DataLoader for validation dataset
    model: SwinUnet model
    weights: Class weights for loss function
    criterion: Loss function - CrossEntropyLoss
    optimizer: Optimizer - AdamW
    scheduler: Learning rate scheduler - OneCycleLR
    metric_tracker: Object to track metrics
    metrics_history: Dictionary to store metrics history
    per_class_iou_history: List to store per class IoU history
    best_loss: Best validation loss
    best_miou: Best mIoU

    @return
    None

    """

    args = parse_args()
    IMAGE_DIR = os.path.join(args.base_dir, "data_object_image_2/training/image_2/")
    LABEL_DIR = os.path.join(args.base_dir, "data_object_label_2/training/label_2/")
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Setting up training...")
    train_transform = A.Compose([
        A.Resize(args.img_height, args.img_width),
        A.OneOf([A.RandomBrightnessContrast(p=0.8),A.RandomGamma(p=0.8),], p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(args.img_height, args.img_width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    try:
        full_dataset = AugmentedKITTIDataset(
            IMAGE_DIR, 
            LABEL_DIR, 
            transform=train_transform,
            img_height=args.img_height,
            img_width=args.img_width
        )
        
        dataset_size = len(full_dataset)
        val_size = int(args.val_split * dataset_size)
        train_size = dataset_size - val_size
        
    
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        
        print(f"Dataset created successfully with {train_size} training and {val_size} validation samples")
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing TransUNet model...")
    model = get_transunet_model(args.img_height,  args.img_width, args.num_classes).to(device)
    weights = calculate_class_weights(train_dataset, args.num_classes)
    print("Class weights calculated:", weights.numpy())
    
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR( optimizer,max_lr=args.lr,epochs=args.epochs,steps_per_epoch=len(train_loader),pct_start=0.3)
    
    metric_tracker = MetricTracker(args.num_classes)
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'train_dice_mean': [], 'val_dice_mean': [],
        'train_iou_mean': [], 'val_iou_mean': [],
        'train_f1_mean': [], 'val_f1_mean': [],
        'train_dice_overlap_mean': [], 'val_dice_overlap_mean': [],
        'train_miou': [], 'val_miou': []
    }
    per_class_iou_history = []
    
    print("Starting training...")
    best_loss = float('inf')
    best_miou = 0.0
    
    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer,scheduler, device, epoch, args.save_dir, metric_tracker)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, metric_tracker)
        
        for key in ['dice_mean', 'iou_mean', 'f1_mean', 
                   'dice_overlap_mean', 'miou']:
            metrics_history[f'train_{key}'].append(train_metrics[key])
            metrics_history[f'val_{key}'].append(val_metrics[key])
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        per_class_iou_history.append(val_metrics['per_class_iou'])
        
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Metrics: Dice={train_metrics["dice_mean"]:.4f}, '
              f'IoU={train_metrics["iou_mean"]:.4f}, '
              f'F1={train_metrics["f1_mean"]:.4f}, '
              f'Dice Overlap={train_metrics["dice_overlap_mean"]:.2f}%, '
              f'mIoU={train_metrics["miou"]:.4f}')
        print(f'Val Metrics: Dice={val_metrics["dice_mean"]:.4f}, '
              f'IoU={val_metrics["iou_mean"]:.4f}, '
              f'F1={val_metrics["f1_mean"]:.4f}, '
              f'Dice Overlap={val_metrics["dice_overlap_mean"]:.2f}%, '
              f'mIoU={val_metrics["miou"]:.4f}')
        
        if val_loss < best_loss or val_metrics['miou'] > best_miou:
            best_loss = min(val_loss, best_loss)
            best_miou = max(val_metrics['miou'], best_miou)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'miou': best_miou,
                'metrics': val_metrics
            }, f'{args.save_dir}/best_model.pth')
            print(f'Saved new best model with loss: {best_loss:.4f} '
                  f'and mIoU: {best_miou:.4f}')

    plot_metrics(metrics_history, args.save_dir)
    save_metric_values(metrics_history, args.save_dir)
    save_per_class_iou(per_class_iou_history, args.num_classes, args.save_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
