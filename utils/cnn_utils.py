import os
import sys

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import rasterio as rio
import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from torchvision import models, transforms
import torchvision.transforms.functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    Inception_V3_Weights,
    VGG16_Weights,
    EfficientNet_B0_Weights,
)

sys.path.insert(0, "./utils/")
import eval_utils
import geoutils
import clf_utils

SEED = 42


def get_imagenet_mean_std(mode):
    if mode == "RGB":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif mode == "LIDAR":
        # Source: https://stackoverflow.com/a/65717887
        return [
            0.44531356896770125,
        ], [0.2692461874154524]


def read_image(filename, mode):
    if mode == "RGB":
        image = Image.open(filename).convert("RGB")
    elif mode == "LIDAR":
        src = rio.open(filename)
        image = src.read([1]).squeeze()
        image[image < 0] = 0
        image = Image.fromarray(image, mode="F")
        src.close()
    return image


class CaribbeanDataset(Dataset):
    def __init__(
        self, dataset, attribute, classes, mode="RGB", transform=None, prefix=''
    ):
        self.dataset = dataset
        self.attribute = attribute
        self.transform = transform
        self.classes = classes
        self.mode = mode
        self.prefix = prefix

    def __getitem__(self, index):
        item = self.dataset.iloc[index]
        filename = self.prefix + item["filename"]
        image = read_image(filename, self.mode)

        if self.transform:
            x = self.transform(image)

        y = self.classes[item[self.attribute]]
        image.close()
        return x, y

    def __len__(self):
        return len(self.dataset)


class SquarePad:
    # Source: https://www.grepper.com/answers/353879/pytorch+pad+to+square
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, "constant")


def visualize_data(data, data_loader, phase="TEST", mode="RGB", n=4):
    imagenet_mean, imagenet_std = get_imagenet_mean_std(mode)
    inputs, classes = next(iter(data_loader[phase]))
    fig, axes = plt.subplots(n, n, figsize=(6, 6))

    key_list = list(data[phase].classes.keys())
    val_list = list(data[phase].classes.values())

    for i in range(n):
        for j in range(n):
            image = inputs[i * n + j].numpy().transpose((1, 2, 0))
            title = key_list[val_list.index(classes[i * n + j])]
            if mode == "RGB":
                image = np.clip(
                    np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1
                )
                axes[i, j].imshow(image)
            elif mode == "LIDAR":
                axes[i, j].imshow(image, cmap="viridis", norm="linear")
            axes[i, j].set_title(title, fontdict={"fontsize": 7})
            axes[i, j].axis("off")


def get_resampled_dataset(data, phase, config):
    data = data[data.dataset == phase]
    if phase == "train" and config["resampler"] != None:
        resampler = clf_utils.get_resampler(config["resampler"])
        data, _ = resampler.fit_resample(data, data[config["attribute"]])
    return data


def load_dataset(config, phases, prefix=''):
    mode = config['data'].split("_")[0]
    csv_path = os.path.join(config["csv_dir"], f"{config['data']}.csv")
    data_dir = os.path.join(config['tile_dir'], mode)
    dataset = pd.read_csv(csv_path)

    transforms = get_transforms(size=config["img_size"], mode=mode)
    classes = list(dataset[config["attribute"]].unique())
    classes = {class_: index for index, class_ in enumerate(classes)}
    logging.info(classes)

    data = {
        phase: CaribbeanDataset(
            get_resampled_dataset(dataset, phase, config)
            .sample(frac=1, random_state=SEED)
            .reset_index(drop=True),
            config["attribute"],
            classes,
            mode,
            transforms[phase],
            prefix
        )
        for phase in phases
    }

    data_loader = {
        phase: torch.utils.data.DataLoader(
            data[phase],
            batch_size=config["batch_size"],
            num_workers=config["n_workers"],
            shuffle=True,
        )
        for phase in phases
    }

    return data, data_loader, classes


def train(data_loader, model, criterion, optimizer, device, logging, wandb=None):
    model.train()

    y_actuals, y_preds = [], []
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            y_actuals.extend(labels.cpu().numpy().tolist())
            y_preds.extend(preds.data.cpu().numpy().tolist())

    epoch_loss = running_loss / len(data_loader)
    epoch_results = eval_utils.evaluate(y_actuals, y_preds)
    epoch_results["loss"] = epoch_loss

    learning_rate = optimizer.param_groups[0]["lr"]
    logging.info(f"Train Loss: {epoch_loss} {epoch_results} LR: {learning_rate}")

    if wandb is not None:
        wandb.log({"train_" + k: v for k, v in epoch_results.items()})
    return epoch_results


def evaluate(data_loader, class_names, model, criterion, device, logging, wandb=None):
    model.eval()

    y_actuals, y_preds = [], []
    running_loss = 0.0
    confusion_matrix = torch.zeros(len(class_names), len(class_names))

    for inputs, labels in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            probs, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        y_actuals.extend(labels.cpu().numpy().tolist())
        y_preds.extend(preds.data.cpu().numpy().tolist())

    epoch_loss = running_loss / len(data_loader)
    epoch_results = eval_utils.evaluate(y_actuals, y_preds)
    epoch_results["loss"] = epoch_loss

    confusion_matrix, cm_metrics, cm_report = eval_utils.get_confusion_matrix(
        y_actuals, y_preds, class_names
    )
    logging.info(f"Val Loss: {epoch_loss} {epoch_results}")

    if wandb is not None:
        wandb.log({"val_" + k: v for k, v in epoch_results.items()})
    return epoch_results, (confusion_matrix, cm_metrics, cm_report)


def get_transforms(size, mode="RGB"):
    imagenet_mean, imagenet_std = get_imagenet_mean_std(mode)

    return {
        "TRAIN": transforms.Compose(
            [
                SquarePad(),
                transforms.Resize(size),
                transforms.RandomRotation((-90, 90)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        ),
        "TEST": transforms.Compose(
            [
                SquarePad(),
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        ),
    }


def get_model(model_type, n_classes, mode, dropout=0):
    if "resnet" in model_type:
        if model_type == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_type == "resnet34":
            model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model_type == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        if mode == "LIDAR":
            # source: https://datascience.stackexchange.com/a/65784
            weights = model.conv1.weight
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            model.conv1.weight = nn.Parameter(torch.mean(weights, dim=1, keepdim=True))

        num_ftrs = model.fc.in_features
        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(num_ftrs, n_classes)
            )
        else:
            model.fc = nn.Linear(num_ftrs, n_classes)

    if "inception" in model_type:
        if mode == "RGB":
            model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        elif mode == "LIDAR":
            model = models.inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
            )
            weights = model.Conv2d_1a_3x3.conv.weight.clone()
            model.Conv2d_1a_3x3.conv = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, bias=False
            )
            model.Conv2d_1a_3x3.conv.weight = nn.Parameter(
                torch.mean(weights, dim=1, keepdim=True)
            )

        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)

    if "vgg" in model_type:
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        if mode == "LIDAR":
            weights = model.features[0].weight.clone()
            model.features[0] = nn.Conv2d(
                1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
            model.features[0].weight = nn.Parameter(
                torch.mean(weights, dim=1, keepdim=True)
            )

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, n_classes)

    if "efficientnet" in model_type:
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        if mode == "LIDAR":
            weights = model.features[0][0].weight.clone()
            model.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
            model.features[0][0].weight = nn.Parameter(
                torch.mean(weights, dim=1, keepdim=True)
            )
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, n_classes)

    return model


def load_model(
    model_type,
    n_classes,
    pretrained,
    scheduler_type,
    optimizer_type,
    label_smoothing=0.0,
    mode="RGB",
    lr=0.001,
    momentum=0.9,
    gamma=0.1,
    step_size=7,
    patience=7,
    dropout=0,
    device="cpu",
):
    model = get_model(model_type, n_classes, mode, dropout)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler_type == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=patience, mode='max'
        )

    return model, criterion, optimizer, scheduler
