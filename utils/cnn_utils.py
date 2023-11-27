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

# Add temporary fix for hash error: https://github.com/pytorch/vision/issues/7744
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict


def get_imagenet_mean_std(mode):
    """
    Get the mean and standard deviation values for normalization based on the mode.

    Args:
    - mode (str): The mode of the data. Can be "RGB" or "LIDAR".

    Returns:
    - tuple: A tuple containing two lists - the mean and standard deviation values.
      For "RGB" mode, returns ([R_mean, G_mean, B_mean], [R_std, G_std, B_std]).
      For "LIDAR" mode, returns ([LIDAR_mean], [LIDAR_std]).
    """
    
    if mode == "RGB":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif mode == "LIDAR":
        # Source: https://stackoverflow.com/a/65717887
        return [
            0.44531356896770125,
        ], [0.2692461874154524]
    
    
def read_image(filename, mode):
    """
    Read an image from file based on the specified mode.

    Args:
    - filename (str): The path to the image file.
    - mode (str): The mode of the data. Can be "RGB" or "LIDAR".

    Returns:
    - Image: An image object (PIL.Image) for "RGB" mode or lidar data transformed 
    to an image for "LIDAR" mode.
    """
        
    if mode == "RGB":
        image = Image.open(filename).convert("RGB")
    elif mode == "LIDAR":
        src = rio.open(filename)
        image = src.read([1]).squeeze() 
        image[image < 0] = 0
        image = image % 255
        image = Image.fromarray(image)
        src.close()
    return image


class CaribbeanDataset(Dataset):
    def __init__(
        self, dataset, attribute, classes, mode="RGB", transform=None, prefix=''
    ):
        """
        Custom dataset for Caribbean images.

        Args:
        - dataset (pandas.DataFrame): The dataset containing image information.
        - attribute (str): The column name specifying the attribute for classification.
        - classes (dict): A dictionary mapping attribute values to classes.
        - mode (str, optional): The mode of the data. Defaults to "RGB".
        - transform (callable, optional): Optional transformations to apply to the image. 
        Defaults to None.
        - prefix (str, optional): Prefix to append to file paths. Defaults to an empty string.
        """
        
        self.dataset = dataset
        self.attribute = attribute
        self.transform = transform
        self.classes = classes
        self.mode = mode
        self.prefix = prefix

    def __getitem__(self, index):
        """
        Retrieves an item (image and label) from the dataset based on index.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - tuple: A tuple containing the transformed image (if transform is specified)
        and its label.
        """
        
        item = self.dataset.iloc[index]
        filepath= self.prefix + item["filepath"]
        filepath = filepath.replace('\\', '/')
        image = read_image(filepath, self.mode)

        if self.transform:
            x = self.transform(image)

        y = self.classes[item[self.attribute]]
        image.close()
        return x, y

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: Length of the dataset.
        """
        
        return len(self.dataset)


class SquarePad:
    """
    A callable class to pad an image to make it square.
    Source: https://www.grepper.com/answers/353879/pytorch+pad+to+square

    Attributes:
    - None

    Methods:
    - __call__(image): Pads the input image to make it square.
    """
    
    def __call__(self, image):
        """
        Pads the input image to make it square.

        Args:
        - image (PIL.Image): The input image to be padded.

        Returns:
        - PIL.Image: The padded image.
        """
        
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, "constant")


def visualize_data(data, data_loader, phase="TEST", mode="RGB", n=4):
    """
    Visualize a sample of data from a DataLoader.

    Args:
    - data (dict): A dictionary containing data split into different phases 
    (TRAIN, VALIDATION, TEST).
    - data_loader (torch.utils.data.DataLoader): DataLoader containing the data.
    - phase (str, optional): The phase of data to visualize. Defaults to "TEST".
    - mode (str, optional): The mode of the data. Can be "RGB" or "LIDAR". Defaults to "RGB".
    - n (int, optional): Number of images to visualize in a grid. Defaults to 4.
    """
    
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
                axes[i, j].imshow(image, cmap="viridis")
            axes[i, j].set_title(title, fontdict={"fontsize": 7})
            axes[i, j].axis("off")


def get_resampled_dataset(data, phase, config):
    """
    Resample the dataset based on the specified phase and configuration.

    Args:
    - data (pandas.DataFrame): The entire dataset.
    - phase (str): The phase of the data. E.g., "train", "test", etc.
    - config (dict): Configuration settings including the resampling strategy and attribute.

    Returns:
    - pandas.DataFrame: The resampled dataset based on the specified phase and configuration.
    """
    
    data = data[data.dataset == phase]
    if phase == "train" and config["resampler"] != None:
        resampler = clf_utils.get_resampler(config["resampler"])
        data, _ = resampler.fit_resample(data, data[config["attribute"]])
    return data


def load_dataset(config, phases, prefix=''):
    """
    Load dataset based on configuration settings and phases.

    Args:
    - config (dict): Configuration settings including data directories, attributes, etc.
    - phases (list): List of phases for which to load the dataset (e.g., ["train", "test"]).
    - prefix (str, optional): Prefix to be added to file paths. Defaults to an empty string.

    Returns:
    - tuple: A tuple containing:
        - dict: A dictionary containing datasets for each phase.
        - dict: A dictionary containing data loaders for each phase.
        - dict: A dictionary containing classes and their mappings.
    """
    
    mode = config['data'].split("_")[0]
    csv_path = os.path.join(config["csv_dir"], f"{config['data']}.csv")
    data_dir = os.path.join(config['tile_dir'], mode)
    dataset = pd.read_csv(csv_path)

    transforms = get_transforms(size=config["img_size"], mode=mode)
    valid_classes = list(dataset[config["attribute"]].unique())
    classes = geoutils.get_classes_dict(config['attribute'])
    classes = dict((v,k) for k,v in classes.items() if v in valid_classes)
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
            drop_last=True
        )
        for phase in phases
    }

    return data, data_loader, classes


def train(data_loader, model, criterion, optimizer, device, logging, wandb=None):
    """
    Train the model on the provided data.

    Args:
    - data_loader (torch.utils.data.DataLoader): DataLoader containing training data.
    - model (torch.nn.Module): The neural network model.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - device (str): Device to run the training on (e.g., 'cuda' or 'cpu').
    - logging: Logging object to record training information.
    - wandb: Weights & Biases object for logging if available. Defaults to None.

    Returns:
    - dict: Results of the training including loss and evaluation metrics.
    """
    
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
    """
    Evaluate the model using the provided data.

    Args:
    - data_loader (torch.utils.data.DataLoader): DataLoader containing validation/test data.
    - class_names (list): List of class names.
    - model (torch.nn.Module): The neural network model.
    - criterion: Loss function.
    - device (str): Device to run evaluation on (e.g., 'cuda' or 'cpu').
    - logging: Logging object to record evaluation information.
    - wandb: Weights & Biases object for logging if available. Defaults to None.

    Returns:
    - tuple: A tuple containing:
        - dict: Results of the evaluation including loss and evaluation metrics.
        - tuple: A tuple containing confusion matrix, metrics, and report.
    """
    
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
    """
    Get image transformations for training and testing phases.

    Args:
    - size (int): Size of the transformed images.
    - mode (str, optional): The mode of the data. Can be "RGB" or "LIDAR". Defaults to "RGB".

    Returns:
    - dict: A dictionary containing transformation pipelines for "TRAIN" and "TEST" phases.
    """
    
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
    """
    Get a neural network model based on specified parameters.

    Args:
    - model_type (str): The type of model architecture to use.
    - n_classes (int): The number of output classes.
    - mode (str): The mode of the data. Can be "RGB" or "LIDAR".
    - dropout (float, optional): Dropout rate if applicable. Defaults to 0.

    Returns:
    - torch.nn.Module: A neural network model based on the specified architecture and mode.
    """
    
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
    """
    Load a neural network model with specified configurations.

    Args:
    - model_type (str): The type of model architecture to use.
    - n_classes (int): The number of output classes.
    - pretrained (bool): Whether to use pre-trained weights.
    - scheduler_type (str): The type of learning rate scheduler to use.
    - optimizer_type (str): The type of optimizer to use.
    - label_smoothing (float, optional): Label smoothing parameter. Defaults to 0.0.
    - mode (str, optional): The mode of the data. Defaults to "RGB".
    - lr (float, optional): Learning rate. Defaults to 0.001.
    - momentum (float, optional): Momentum factor for SGD optimizer. Defaults to 0.9.
    - gamma (float, optional): Gamma factor for learning rate scheduler. Defaults to 0.1.
    - step_size (int, optional): Step size for learning rate scheduler. Defaults to 7.
    - patience (int, optional): Patience for ReduceLROnPlateau scheduler. Defaults to 7.
    - dropout (float, optional): Dropout rate if applicable. Defaults to 0.
    - device (str, optional): Device to run the model on. Defaults to "cpu".

    Returns:
    - tuple: A tuple containing the loaded model, criterion, optimizer, and scheduler.
    """
    
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
