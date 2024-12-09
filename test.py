import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.stats import entropy
from networks.vision_transformer import SwinUnet
from config import get_config

class KITTITestDataset(Dataset):
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
def calculate_prediction_metrics(outputs, num_classes):
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(outputs)
    confidence = torch.max(probabilities, dim=1)[0]
    mean_confidence = confidence.mean().item()
    predictions = torch.argmax(outputs, dim=1)
    class_dist = []
    for i in range(num_classes):
        class_pixels = (predictions == i).float().mean().item() * 100
        class_dist.append(class_pixels)
    prob_np = probabilities.cpu().numpy()
    prediction_entropy = entropy(prob_np, axis=1).mean()
    
    return {
        'mean_confidence': mean_confidence,
        'class_distribution': class_dist,
        'entropy': prediction_entropy
    }

def create_side_by_side_visualization(original_image, pred_mask, color_map):
    if torch.is_tensor(original_image):
        original_image = original_image.cpu().numpy()
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[-1] != 3:
        original_image = original_image.transpose(1, 2, 0)
    colored_mask = color_map[pred_mask].astype(np.uint8)
    if colored_mask.shape[:2] != original_image.shape[:2]:
        colored_mask = cv2.resize(colored_mask, 
                                (original_image.shape[1], original_image.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    
    blend = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
    visualization = np.hstack([original_image, colored_mask, blend])
    
    return visualization

def predict_and_save(model, test_loader, device, save_dir, num_classes=10):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    np.random.seed(42) 
    for i in range(num_classes):
        color_map[i] = np.random.randint(0, 255, 3)
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['image'].to(device)
            original_images = batch['original_image']
            file_names = batch['file_name']
            orig_heights = batch['original_height']
            orig_widths = batch['original_width']
            
     
            outputs = model(images)
            batch_metrics = calculate_prediction_metrics(outputs, num_classes)
            predictions = torch.argmax(outputs, dim=1)
            for idx, pred in enumerate(predictions):
                pred_np = pred.cpu().numpy()
                orig_h = int(orig_heights[idx])
                orig_w = int(orig_widths[idx])
                pred_resized = cv2.resize(
                    pred_np.astype(np.uint8), 
                    (int(orig_w), int(orig_h)),
                    interpolation=cv2.INTER_NEAREST
                )
                orig_img = original_images[idx]
                if torch.is_tensor(orig_img):
                    orig_img = orig_img.cpu().numpy()
                visualization = create_side_by_side_visualization(
                    orig_img,
                    pred_resized,
                    color_map
                )

                vis_path = os.path.join(save_dir, f'vis_{file_names[idx]}')
                cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                pred_path = os.path.join(save_dir, f'pred_{file_names[idx]}')
                cv2.imwrite(pred_path, pred_resized)
                image_metrics = {
                    'file_name': file_names[idx],
                    'confidence': batch_metrics['mean_confidence'],
                    'entropy': batch_metrics['entropy'],
                    'class_distribution': batch_metrics['class_distribution']
                }
                all_metrics.append(image_metrics)
    
    save_metrics_summary(all_metrics, save_dir, num_classes)


def save_metrics_summary(metrics, save_dir, num_classes):
    avg_confidence = np.mean([m['confidence'] for m in metrics])
    avg_entropy = np.mean([m['entropy'] for m in metrics])
    avg_class_dist = np.mean([m['class_distribution'] for m in metrics], axis=0)
    with open(os.path.join(save_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(f"Average Confidence: {avg_confidence:.4f}\n")
        f.write(f"Average Entropy: {avg_entropy:.4f}\n")
        f.write("\nAverage Class Distribution:\n")
        for i in range(num_classes):
            f.write(f"Class {i}: {avg_class_dist[i]:.2f}%\n")
        
        f.write("\nPer-Image Metrics:\n")
        for m in metrics:
            f.write(f"\nFile: {m['file_name']}\n")
            f.write(f"Confidence: {m['confidence']:.4f}\n")
            f.write(f"Entropy: {m['entropy']:.4f}\n")

    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), avg_class_dist)
    plt.title('Average Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
    plt.close()

def custom_collate(batch):
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key == 'image':
            collated[key] = torch.stack([item[key] for item in batch])
        elif key == 'original_image':
            collated[key] = [item[key] for item in batch]
        elif key == 'file_name':
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = [item[key] for item in batch]
    
    return collated

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_SIZE = 224
    NUM_CLASSES = 10
    IMAGE_DIR = '/afs/glue.umd.edu/home/glue/a/m/amishr17/home/.cache/kagglehub/datasets/klemenko/kitti-dataset/versions/1/data_object_image_2/testing/image_2'
    SAVE_DIR = '/export/amishr17/703/SwinPred'
    MODEL_PATH = '/export/amishr17/703/swinUNet_results/best_model.pth'
    CONFIG_PATH = "/export/amishr17/703/configs/swinunet_config.yaml"  
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

    test_dataset = KITTITestDataset(image_dir=IMAGE_DIR,img_height=IMG_HEIGHT,img_width=IMG_WIDTH,transform=transform)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,collate_fn=custom_collate)
    model = SwinUnet(config,img_size=IMG_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    

    predict_and_save(model=model,test_loader=test_loader,device=DEVICE,save_dir=SAVE_DIR,num_classes=NUM_CLASSES)
    
    print(f"\nResults saved to {SAVE_DIR}")
    print("- Side-by-side visualizations saved as 'vis_*.png'")
    print("- Raw predictions saved as 'pred_*.png'")
    print("- Metrics summary saved as 'metrics_summary.txt'")
    print("- Class distribution plot saved as 'class_distribution.png'")

if __name__ == "__main__":
    main()
