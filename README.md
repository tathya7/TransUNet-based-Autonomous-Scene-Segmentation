# Automatic Scene Segmentation using UNET architectures. 


## Overview 
This project implements automatic scene segmentation using  deep learning architectures: UNET CNN, Transformer-UNET, and Swin Transformer UNET. It aims to provide accurate pixel-wise segmentation for applications in medical imaging, autonomous vehicles, and urban planning. Our aim is to compare and predict the hypothesis if Transformers are better than Convolution Neural Networks or not.

## Key Features 

- **Implements traditional UNET** for pixel-wise segmentation.
- **Integrates Transformer-UNET** to enhance global feature learning.
- **Leverages Swin Transformer UNET** for hierarchical and efficient segmentation.
- **Supports training and inference** on custom datasets.
- **GPU acceleration** for faster computations.

## File Structure

```
├── data/     
├── networks/              
│   ├── swin_transformer_unet_skip_expand_decoder_sys.py
│   ├── UNet.py
│   ├── UNetparts.py
|   ├── vision_transformer.py
|   ├── vit_segs_config.py
|   ├── vit_segs_modelling.py
|   ├── vit_segs_modelling_resnet_skip.py
├── utils.py
├── train/
|   ├──train_UNET_Cnn.py               # Training UNET CNN script
|   ├──train_Transunet.py              # Training TransUNET script
|   ├── train_sswinUNET.py               # Training SwinTransUNET script
├── train_parallel.sh # Bash Script for traininf
├── inference.py           # Inference script
├── requirements.txt       # Dependency list
└── README.md              # Project documentation

```

## To run the scripts

## Requirements - Libraries
```
python==3.12.0
torch>=1.10.0
albumentations>=1.0.0
albumentations[imgaug]
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
tqdm>=4.60.0
kagglehub
```

## Dataset

```
import kagglehub


path = kagglehub.dataset_download("klemenko/kitti-dataset")
print("Path to model files:", path)
```
**On the Terminal, this code will depict the Path to the dataset:**


## Command Line Instructions to run 

### INSTALL CONDA/MINICONDA and follow the below process -

**1. Activate Conda Environment**

```
conda activate YOUR_ENV
```

**2. Install all the dependencies**

```
pip install requirements.txt
```

**3. For running Bash File**

```
chmod +x train_parallel.sh
./train_parallel.sh
```

**4. Visualizing Bash file logs**

```
tail -f logs/unet/training.log
tail -f logs/transunet/training.log
tail -f logs/swinunet/training.log
```





