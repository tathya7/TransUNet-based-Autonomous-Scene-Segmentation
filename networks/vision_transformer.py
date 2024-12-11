# coding=utf-8
from __future__ import absolute_import, division, print_function

import copy
import logging
import torch
import torch.nn as nn

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

# Set up logging
logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    """
    SwinUnet class that implements a UNet-like architecture with a Swin Transformer as the encoder.
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        """
        Initializes the SwinUnet model.
        
        Parameters:
        - config (dict): Configuration dictionary containing hyperparameters and model settings.
        - img_size (int): The input image dimensions.
        - num_classes (int): Number of output classes.
        - zero_head (bool): Flag to initialize the classification head to zero.
        """
        super(SwinUnet, self).__init__()

        # Store number of classes and configurations
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        # Initialize the Swin Transformer UNet model
        self.swin_unet = SwinTransformerSys(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=self.num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )

    def forward(self, x):
        """
        Forward pass for the SwinUnet model. If the input has a single channel (grayscale), 
        it is repeated 3 times to match the expected input channels for the Swin Transformer.
        
        Parameters:
        - x (Tensor): Input tensor with shape [batch_size, channels, height, width].
        
        Returns:
        - logits (Tensor): Output logits (predictions) from the model.
        """
        # If the input has only one channel (grayscale), repeat it across three channels (RGB)
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Pass through the Swin Transformer UNet
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        """
        Loads pretrained weights into the SwinUnet model if a pretrained checkpoint is provided.
        
        Parameters:
        - config (dict): Configuration dictionary containing path to the pretrained checkpoint.
        
        Notes:
        - If the pretrained checkpoint does not match the current model's structure, 
          it will adjust layer names and shapes where possible.
        """
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Load the pretrained model weights
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            
            # Handle case where 'model' key is not present in the checkpoint
            if "model" not in pretrained_dict:
                print("---start load pretrained model by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                
                # Remove output layer keys if present
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print(f"delete key: {k}")
                        del pretrained_dict[k]
                
                # Load the weights into the Swin Transformer model with strict=False
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                return

            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained model of swin encoder---")

            # Create a copy of the pretrained weights and adjust layer names
            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)

            # Update the full_dict with corrected layer names
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = f"layers_up.{current_layer_num}{k[8:]}"
                    full_dict.update({current_k: v})

            # Remove weights if their shapes do not match
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print(f"delete: {k}; shape pretrain: {v.shape}; shape model: {model_dict[k].shape}")
                        del full_dict[k]

            # Load the adjusted weights into the model
            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")
