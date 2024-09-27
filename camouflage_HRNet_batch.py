import os
from PIL import Image
import cv2
import numpy as np
import argparse
from importlib import import_module
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, models
import torch
import torch.optim as optim
import torch.nn as nn
import datetime
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold.locally_linear import barycenter_kneighbors_graph

import HRNet
from hidden_recommend import recommend
from utils import scaling, get_features, im_convert, attention_map_cv, gram_matrix_slice

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    camouflage_dir = args.output_dir
    os.makedirs(camouflage_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)

    for parameter in VGG.parameters():
        parameter.requires_grad_(False)

    style_net = HRNet.HRNet()
    style_net.to(device)

    transform = Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
    ])   

    style_weights = args.style_weight_dic
    
    # 批量处理输入文件夹中的所有图像
    for image_file in os.listdir(args.input_folder):
        i_path = os.path.join(args.input_folder, image_file)
        m_path = os.path.join(args.mask_folder, image_file.replace(".png", "_mask.png"))
        bg_path = args.bg_path  # 背景图像

        mask = cv2.imread(m_path, 0)
        mask = scaling(mask, scale=args.mask_scale)

        if args.crop:
            idx_y, idx_x = np.where(mask > 0)
            x1_m, y1_m, x2_m, y2_m = np.min(idx_x), np.min(idx_y), np.max(idx_x), np.max(idx_y)
        else:
            x1_m, y1_m = 0, 0
            y2_m, x2_m = mask.shape
            x2_m, y2_m = 8 * (x2_m // 8), 8 * (y2_m // 8)
        
        x1_m = 8 * (x1_m // 8)
        x2_m = 8 * (x2_m // 8)
        y1_m = 8 * (y1_m // 8)
        y2_m = 8 * (y2_m // 8)
        
        fore_origin = cv2.cvtColor(cv2.imread(i_path), cv2.COLOR_BGR2RGB)
        fore_origin = scaling(fore_origin, scale=args.mask_scale)
        fore = fore_origin[y1_m:y2_m, x1_m:x2_m]
        
        mask_crop = mask[y1_m:y2_m, x1_m:x2_m]
        mask_crop = np.where(mask_crop > 0, 255, 0).astype(np.uint8)
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_crop, kernel, iterations=1)
        
        origin = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
        h_origin, w_origin, _ = origin.shape
        h, w = mask_dilated.shape
        assert h < h_origin, "mask height must be smaller than bg height, and lower mask_scale parameter!!"
        assert w < w_origin, "mask width must be smaller than bg width, and lower mask_scale parameter!!"
        
        print(f"Processing {image_file}: mask size, height:{h}, width:{w}")
        
        if args.hidden_selected is None:
            y_start, x_start = recommend(origin, fore, mask_dilated)
        else:
            y_start, x_start = args.hidden_selected
        
        x1, y1 = x_start + x1_m, y_start + y1_m
        x2, y2 = x1 + w, y1 + h
        if y2 > h_origin:
            y1 -= (y2 - h_origin)
            y2 = h_origin
        if x2 > w_origin:
            x1 -= (x2 - w_origin)
            x2 = w_origin
            
        mat_dilated = fore * np.expand_dims(mask_crop / 255, axis=-1) + origin[y1:y2, x1:x2] * np.expand_dims((mask_dilated - mask_crop) / 255, axis=-1)
        bg = origin.copy()
        bg[y1:y2, x1:x2] = fore * np.expand_dims(mask_crop / 255, axis=-1) + origin[y1:y2, x1:x2] * np.expand_dims(1 - mask_crop / 255, axis=-1)
        
        content_image = transform(image=mat_dilated)["image"].unsqueeze(0)
        style_image = transform(image=origin[y1:y2, x1:x2])["image"].unsqueeze(0)
        content_image = content_image.to(device)
        style_image = style_image.to(device)
        
        # 以下继续损失计算及模型训练流程...

        # 保存每个图像的结果
        result_path = os.path.join(camouflage_dir, f"{os.path.splitext(image_file)[0]}_output.png")
        canvas = origin.copy()
        fore_gen = im_convert(target) * 255.
        canvas[y1:y2, x1:x2] = fore_gen * np.expand_dims(mask_norm, axis=-1) + origin[y1:y2, x1:x2] * np.expand_dims(1.0 - mask_norm, axis=-1)
        canvas = canvas.astype(np.uint8)
        cv2.imwrite(result_path, canvas)
        print(f"Saved output for {image_file} at {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the folder with input images.")
    parser.add_argument('--mask_folder', type=str, required=True, help="Path to the folder with mask images.")
    parser.add_argument('--bg_path', type=str, required=True, help="Path to the background image.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output images.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--mask_scale', type=float, default=0.2, help="Scale factor for the mask.")
    parser.add_argument('--crop', action='store_true', help="Whether to crop the mask region.")
    parser.add_argument('--hidden_selected', type=tuple, default=None, help="Pre-selected hidden region (y_start, x_start).")
    parser.add_argument('--epoch', type=int, default=100, help="Number of epochs.")
    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate.")
    parser.add_argument('--style_weight_dic', type=dict, default={'conv1_1': 1.5, 'conv2_1': 1.5, 'conv3_1': 1.5, 'conv4_1': 1.5}, help="Weights for style loss.")
    
    args = parser.parse_args()
    main(args)