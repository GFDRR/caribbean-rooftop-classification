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
import geoutils


def load_model(c, exp_dir, n_classes, mode='RGB'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_file = os.path.join(exp_dir, "best_model.pth")
    model = cnn_utils.get_model(c['model'], n_classes, mode, c['dropout'])
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model file {} successfully loaded.".format(model_file))
    return model


def generate_predictions(data, model, c, in_file, out_dir, classes, scale=1.5):
    preds = []
    pbar = tqdm(
        enumerate(data.iterrows()),
        total=len(data),
        bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}",
    )
    for index, (_, row) in pbar:
        out_file = os.path.join(out_dir, "temp.tif")
        shape = data[(data.UID == row["UID"])]
        out_shape = os.path.join(out_dir, "temp.gpkg")
        if os.path.exists(out_file):
            os.remove(out_file) 
        geoutils.crop_shape(shape, out_shape, scale, in_file, out_file)

        if os.path.exists(out_file):
            image = cnn_utils.read_image(out_file, c['mode'])
            transforms = cnn_utils.get_transforms(c['img_size'], c['mode'])
            input = transforms["test"](image)
            output = model(input.unsqueeze(0))
            _, pred = torch.max(output, 1)
            label = str(classes[int(pred[0])])
            preds.append(label)
        else:
            preds.append(np.nan)

    data[c['attribute']] = preds
    return data
