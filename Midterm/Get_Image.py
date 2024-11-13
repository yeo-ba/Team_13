import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
import cv2


class Get_Image:
    def __init__(self, image_path):
        self.image_path = image_path
        self.raw_image = Image.open(image_path).convert("RGB").resize((512,512),Image.LANCZOS)

    def get_bg_source(self, bg_img_path):
        self.bg_mask_image = Image.open(bg_img_path).convert("L").resize((512,512),Image.LANCZOS)
        bg_mask_image_np = self.bg_mask_image.convert('RGB')
        bg_mask_image_np = np.array(bg_mask_image_np)
        self.bg_mask_image_np = bg_mask_image_np.astype(np.float32) / 255.0
        return self.bg_mask_image, self.bg_mask_image_np
    
    def object_canny(self,low_threshold, high_threshold, masked_image):
        masked_image_np = np.array(masked_image)
        image = cv2.Canny(masked_image_np, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        outline_np = image.copy()
        self.canny_image = Image.fromarray(image).resize((512,512), Image.LANCZOS)
        return self.canny_image

    def generate_image(self, bg_mask_np, out):
        one = np.ones(self.bg_mask_image_np.shape)
        fg_mask_np = one-bg_mask_np
        alpha = 0.9

        out_image_np = bg_mask_np * out + alpha*(fg_mask_np*self.raw_image) + (1-alpha)*(fg_mask_np*out)
        out_image_np = np.clip(out_image_np, 0, 255).astype(np.uint8)
        out_image = Image.fromarray(out_image_np)
        return out_image
        
        
