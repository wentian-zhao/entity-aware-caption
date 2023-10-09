import os

import numpy as np
import torch

from model.vit.models.modeling import CONFIGS, VisionTransformer

config = CONFIGS["ViT-B_16"]
model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
model.load_from(np.load("/media/wentian/sdb1/work/extract_image_feature/imagenet21k+imagenet2012_ViT-B_16-224.npz"))
model.eval()

input = torch.zeros(3, 224, 224)
output = model.forward(input.unsqueeze(0))