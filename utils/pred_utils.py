import os
import sys
import random
import subprocess
import numpy as np
import rasterio as rio
from PIL import Image

import matplotlib.pyplot as plt
from rasterio.plot import show
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

import subprocess

sys.path.insert(0, "./utils/")
import cnn_utils


def load_model(model_type, exp_dir, n_classes=2, dropout=0.5, mode='RGB'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_file = os.path.join(exp_dir, "best_model.pth")

    if "resnet" in model_type:
        if model_type == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_type == "resnet34":
            model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model_type == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        
        if mode == 'GRAYSCALE':
            #source: https://datascience.stackexchange.com/a/65784
            weights = model.conv1.weight
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.conv1.weight = nn.Parameter(torch.mean(weights, dim=1, keepdim=True))

        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(num_ftrs, n_classes)
            )
        else:
            model.fc = nn.Linear(num_ftrs, n_classes)
    
    if 'inception' in model_type:
        if mode == 'RGB':
            model = models.inception_v3(pretrained=True)
        elif mode == 'GRAYSCALE':
            model = models.inception_v3(pretrained=True, transform_input=False)
            weights = model.Conv2d_1a_3x3.conv.weight.clone()
            model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
            model.Conv2d_1a_3x3.conv.weight = nn.Parameter(torch.mean(weights, dim=1, keepdim=True))
        
        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)

    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()

    print("Model file {} successfully loaded.".format(model_file))
    return model


def generate_predictions(data, model, c, in_file, out_dir, class_name, classes, scale=1.5):
    preds = []
    pbar = tqdm(enumerate(data.iterrows()), total=len(data))
    for index, (_, row) in pbar:
        out_file = os.path.join(out_dir, "temp.tif")
        shape = data[(data.UID == row["UID"])]
        shape.geometry = shape.geometry.apply(lambda x: x.minimum_rotated_rectangle)
        shape.geometry = shape.geometry.scale(scale, scale)

        out_shape = os.path.join(out_dir, "temp.gpkg")
        shape.to_file(out_shape, driver="GPKG")
        if os.path.exists(out_file):
            os.remove(out_file)
        subprocess.call(
            f"gdalwarp -cutline {out_shape} -crop_to_cutline -dstalpha {in_file} {out_file}",
            shell=True,
        )

        if os.path.exists(out_file):
            if c['mode'] == 'RGB':
                image = Image.open(out_file).convert("RGB")
            if c['mode'] == 'GRAYSCALE':
                src = rio.open(out_file)
                image = src.read([1]).squeeze()
                image[image < 0] = 0
                image = Image.fromarray(image, mode='F')
                src.close()
                
            transforms = cnn_utils.get_transforms(c['img_size'], c['mode'])
            input = transforms["test"](image)
            output = model(input.unsqueeze(0))
            _, pred = torch.max(output, 1)
            label = str(classes[int(pred[0])])
            preds.append(label)
        else:
            preds.append(np.nan)

    data[class_name] = preds
    return data
