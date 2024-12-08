import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm



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
        dice = self.calculate_dice(pred, target)
        iou = self.calculate_iou(pred, target)
        f1 = self.calculate_f1(pred, target)
        dice_overlap = self.calculate_dice_overlap(pred, target)
        
        if not torch.isnan(torch.tensor(dice)):
            self.dice_scores.append(dice)
        if not torch.isnan(torch.tensor(iou)):
            self.iou_scores.append(iou)
        if not torch.isnan(torch.tensor(f1)):
            self.f1_scores.append(f1)
        if not torch.isnan(torch.tensor(dice_overlap)):
            self.dice_overlap_scores.append(dice_overlap)
        
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
