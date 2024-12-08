
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
    print("Calculating class weights...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    class_counts = torch.zeros(num_classes)
    for batch in tqdm(loader, desc="Computing class weights"):
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

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, save_dir, metric_tracker):
    model.train()
    epoch_loss = 0
    metric_tracker.reset()
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update metrics
            pred = outputs.argmax(dim=1)
            for i in range(pred.shape[0]):
                pred_mask = torch.zeros(outputs.shape[1], *pred[i].shape, device=device)
                pred_mask.scatter_(0, pred[i].unsqueeze(0), 1)
                
                target_mask = torch.zeros(outputs.shape[1], *masks[i].shape, device=device)
                target_mask.scatter_(0, masks[i].unsqueeze(0), 1)
                
                metric_tracker.update(pred_mask, target_mask)
            
            current_metrics = metric_tracker.get_metrics()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{current_metrics["dice_mean"]:.4f}',
                'miou': f'{current_metrics["miou"]:.4f}'
            })
            
            if batch_idx % 500 == 0:
                save_predictions_with_metrics(images, masks, outputs, current_metrics, epoch, batch_idx, save_dir, 10)
    
    scheduler.step()
    return epoch_loss / len(train_loader), metric_tracker.get_metrics()


def validate(model, val_loader, criterion, device, metric_tracker):
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
            
            # Update metrics
            pred = outputs.argmax(dim=1)
            for i in range(pred.shape[0]):
                # Convert predictions to one-hot encoding
                pred_mask = torch.zeros(outputs.shape[1], *pred[i].shape, device=device)
                pred_mask.scatter_(0, pred[i].unsqueeze(0), 1)
                
                # Convert targets to one-hot encoding
                target_mask = torch.zeros(outputs.shape[1], *masks[i].shape, device=device)
                target_mask.scatter_(0, masks[i].unsqueeze(0), 1)
                
                metric_tracker.update(pred_mask, target_mask)
    
    return val_loss / len(val_loader), metric_tracker.get_metrics()

def save_predictions_with_metrics(images, masks, outputs, metrics, epoch, batch_idx, save_dir, num_classes):
    with torch.no_grad():
        pred = outputs.argmax(dim=1)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        

        img_display = images[0].cpu().permute(1, 2, 0).numpy()
        img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        axes[0, 0].imshow(np.clip(img_display, 0, 1))
        axes[0, 0].set_title('Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(masks[0].cpu(), cmap='tab10', vmin=0, vmax=num_classes-1)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(pred[0].cpu(), cmap='tab10', vmin=0, vmax=num_classes-1)
        axes[1, 0].set_title('Prediction')
        axes[1, 0].axis('off')

        axes[1, 1].axis('off')
        metric_text = (
            f"Metrics:\n"
            f"Dice Score: {metrics['dice_mean']:.4f}\n"
            f"IoU Score: {metrics['iou_mean']:.4f}\n"
            f"F1 Score: {metrics['f1_mean']:.4f}\n"
            f"Dice Overlap %: {metrics['dice_overlap_mean']:.2f}%\n"
            f"mIoU: {metrics['miou']:.4f}"
        )
        axes[1, 1].text(0.1, 0.5, metric_text, fontsize=12, va='center')
        
        plt.savefig(f'{save_dir}/epoch_{epoch+1}_batch_{batch_idx}.png')
        plt.close()

class MetricTracker:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        self.f1_scores = []
        self.dice_overlap_scores = []
        self.class_ious = [[] for _ in range(self.num_classes)]
    
    def update(self, pred, target):
        # Handle NaN values
        dice = self.calculate_dice(pred, target)
        iou = self.calculate_iou(pred, target)
        f1 = self.calculate_f1(pred, target)
        dice_overlap = self.calculate_dice_overlap(pred, target)
        
        # Only append non-NaN values
        if not torch.isnan(torch.tensor(dice)):
            self.dice_scores.append(dice)
        if not torch.isnan(torch.tensor(iou)):
            self.iou_scores.append(iou)
        if not torch.isnan(torch.tensor(f1)):
            self.f1_scores.append(f1)
        if not torch.isnan(torch.tensor(dice_overlap)):
            self.dice_overlap_scores.append(dice_overlap)
        
        # Calculate per-class IoU
        for class_idx in range(self.num_classes):
            class_pred = pred[class_idx]
            class_target = target[class_idx]
            class_iou = self.calculate_iou(class_pred, class_target)
            if not torch.isnan(torch.tensor(class_iou)):
                self.class_ious[class_idx].append(class_iou)
    
    def get_metrics(self):
        miou = np.mean([np.mean(class_iou) if len(class_iou) > 0 else 0 for class_iou in self.class_ious])
        return {
            'dice_mean': np.mean(self.dice_scores) if self.dice_scores else 0,
            'iou_mean': np.mean(self.iou_scores) if self.iou_scores else 0,
            'f1_mean': np.mean(self.f1_scores) if self.f1_scores else 0,
            'dice_overlap_mean': np.mean(self.dice_overlap_scores) if self.dice_overlap_scores else 0,
            'miou': miou,
            'per_class_iou': [np.mean(class_iou) if len(class_iou) > 0 else 0 for class_iou in self.class_ious]
        }

    @staticmethod
    def calculate_dice(pred, target):
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()
    
    @staticmethod
    def calculate_iou(pred, target):
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    @staticmethod
    def calculate_f1(pred, target):
        smooth = 1e-5
        true_positives = torch.sum(pred * target)
        false_positives = torch.sum(pred * (1 - target))
        false_negatives = torch.sum((1 - pred) * target)
        
        precision = (true_positives + smooth) / (true_positives + false_positives + smooth)
        recall = (true_positives + smooth) / (true_positives + false_negatives + smooth)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1.item()
    
    @staticmethod
    def calculate_dice_overlap(pred, target):
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        overlap = (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth) * 100
        return overlap.item()

def plot_metrics(metrics_history, save_dir):
    """Plot and save all metrics history."""
    metrics_to_plot = [
        ('loss', 'Loss'),
        ('dice_mean', 'Dice Score'),
        ('iou_mean', 'IoU Score'),
        ('f1_mean', 'F1 Score'),
        ('dice_overlap_mean', 'Dice Overlap %'),
        ('miou', 'Mean IoU')
    ]
    
    for metric_key, metric_name in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_history[f'train_{metric_key}'], label=f'Train {metric_name}')
        plt.plot(metrics_history[f'val_{metric_key}'], label=f'Val {metric_name}')
        plt.title(f'{metric_name} vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{metric_key}_plot.png')
        plt.close()

def save_metric_values(metrics_history, save_dir):
    """Save all metric values to separate text files."""
    metrics_to_save = [
        'loss', 'dice_mean', 'iou_mean', 'f1_mean', 
        'dice_overlap_mean', 'miou'
    ]
    
    for metric in metrics_to_save:
        with open(f'{save_dir}/{metric}_values.txt', 'w') as f:
            f.write(f"Epoch,Train_{metric},Val_{metric}\n")
            for epoch in range(len(metrics_history['train_loss'])):
                f.write(f"{epoch+1},{metrics_history[f'train_{metric}'][epoch]},"
                       f"{metrics_history[f'val_{metric}'][epoch]}\n")

def save_per_class_iou(per_class_iou_history, num_classes, save_dir):
    """Save per-class IoU values."""
    with open(f'{save_dir}/per_class_iou_values.txt', 'w') as f:
        f.write("Epoch," + ",".join([f"Class_{i}" for i in range(num_classes)]) + "\n")
        for epoch, class_ious in enumerate(per_class_iou_history):
            f.write(f"{epoch+1}," + ",".join([f"{iou:.4f}" for iou in class_ious]) + "\n")


def main():
    BASE_DIR = "/afs/glue.umd.edu/home/glue/a/m/amishr17/home/.cache/kagglehub/datasets/klemenko/kitti-dataset/versions/1"
    IMAGE_DIR = os.path.join(BASE_DIR, "data_object_image_2/training/image_2/")
    LABEL_DIR = os.path.join(BASE_DIR, "data_object_label_2/training/label_2/")
    IMG_HEIGHT, IMG_WIDTH, IMG_SIZE = 224, 224, 224
    NUM_CLASSES = 10
    MAX_EPOCHS = 300
    LEARNING_RATE = 1e-4  
    BATCH_SIZE = 8
    VAL_SPLIT = 0.1

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "swinUNet_results"
    os.makedirs(save_dir, exist_ok=True)


    dataset = AugmentedKITTIDataset(
        IMAGE_DIR,
        LABEL_DIR,
        transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH
    )


    dataset_size = len(dataset)
    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


    config_path = "configs/swinunet_config.yaml"
    config = get_config(config_path)
    config.defrost()
    config.MODEL.SWIN.EMBED_DIM = 96
    config.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    config.MODEL.SWIN.DECODER_DEPTHS = [2, 2, 2, 2]
    config.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    config.MODEL.DROP_PATH_RATE = 0.3
    config.freeze()

    model = SwinUnet(config, img_size=IMG_SIZE, num_classes=NUM_CLASSES).to(DEVICE)

    if os.path.exists(config.MODEL.PRETRAIN_CKPT):
        checkpoint = torch.load(config.MODEL.PRETRAIN_CKPT, map_location='cpu')
        model.load_from(checkpoint)

    weights = calculate_class_weights(dataset, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, betas=(0.9, 0.999))
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=MAX_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4
    )

    metric_tracker = MetricTracker(NUM_CLASSES)
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'train_dice_mean': [], 'val_dice_mean': [],
        'train_iou_mean': [], 'val_iou_mean': [],
        'train_f1_mean': [], 'val_f1_mean': [],
        'train_dice_overlap_mean': [], 'val_dice_overlap_mean': [],
        'train_miou': [], 'val_miou': []
    }
    per_class_iou_history = []
    best_loss = float('inf')
    best_miou = 0.0
    checkpoint_freq = 50  
  
    for epoch in range(MAX_EPOCHS):

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer,scheduler, DEVICE, epoch, save_dir, metric_tracker)
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE, metric_tracker)
        
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['train_dice_mean'].append(train_metrics['dice_mean'])
        metrics_history['val_dice_mean'].append(val_metrics['dice_mean'])
        metrics_history['train_iou_mean'].append(train_metrics['iou_mean'])
        metrics_history['val_iou_mean'].append(val_metrics['iou_mean'])
        metrics_history['train_f1_mean'].append(train_metrics['f1_mean'])
        metrics_history['val_f1_mean'].append(val_metrics['f1_mean'])
        metrics_history['train_dice_overlap_mean'].append(train_metrics['dice_overlap_mean'])
        metrics_history['val_dice_overlap_mean'].append(val_metrics['dice_overlap_mean'])
        metrics_history['train_miou'].append(train_metrics['miou'])
        metrics_history['val_miou'].append(val_metrics['miou'])
        per_class_iou_history.append(val_metrics['per_class_iou'])
        
        print(f'Epoch {epoch+1}/{MAX_EPOCHS}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Metrics: Dice={train_metrics["dice_mean"]:.4f}, IoU={train_metrics["iou_mean"]:.4f}, '
              f'F1={train_metrics["f1_mean"]:.4f}, Dice Overlap={train_metrics["dice_overlap_mean"]:.2f}%, '
              f'mIoU={train_metrics["miou"]:.4f}')
        print(f'Val Metrics: Dice={val_metrics["dice_mean"]:.4f}, IoU={val_metrics["iou_mean"]:.4f}, '
              f'F1={val_metrics["f1_mean"]:.4f}, Dice Overlap={val_metrics["dice_overlap_mean"]:.2f}%, '
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
            }, f'{save_dir}/best_model.pth')
            print(f'Saved new best model with loss: {best_loss:.4f} and mIoU: {best_miou:.4f}')
                
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'metrics': val_metrics
            }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pth')
    

    plot_metrics(metrics_history, save_dir)
    save_metric_values(metrics_history, save_dir)
    save_per_class_iou(per_class_iou_history, NUM_CLASSES, save_dir)
    
    print("Training completed! Metrics plots and values have been saved.")

if __name__ == "__main__":
    main()