import os
import sys
from tqdm import tqdm
import rasterio as rio
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, Subset

import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F
from torchvision.models import (
    ResNet18_Weights, 
    ResNet34_Weights, 
    ResNet50_Weights, 
    Inception_V3_Weights, 
    VGG16_Weights
)
from sklearn.preprocessing import minmax_scale

sys.path.insert(0, "./utils/")
import eval_utils

SEED = 42


def get_imagenet_mean_std(mode):
    if mode == 'RGB':
         return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif mode == 'GRAYSCALE':
        # Source: https://stackoverflow.com/a/65717887
        return [0.44531356896770125, ], [0.2692461874154524]
    
    
def read_image(filename, mode):
    if mode == 'RGB':
        image = Image.open(filename).convert("RGB")
    elif mode == 'GRAYSCALE':
        src = rio.open(filename)
        image = src.read([1]).squeeze()
        image[image < 0] = 0
        image = Image.fromarray(image, mode='F')
        src.close()
    return image


class CaribbeanDataset(Dataset):
    def __init__(self, dataset, image_folder, attribute, classes, mode='RGB', transform=None):
        self.dataset = dataset
        self.image_folder = image_folder
        self.attribute = attribute
        self.transform = transform
        self.classes = classes 
        self.mode = mode

    def __getitem__(self, index):
        item = self.dataset.iloc[index]
        filename = item["filename"]
        filename = os.path.join(
            self.image_folder, self.attribute, item["label"], filename
        )
        image = read_image(filename, self.mode)
            
        if self.transform:
            x = self.transform(image)
            
        y = self.classes[item["label"]]
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


def visualize_data(data, data_loader, phase="test", mode='RGB', n=4):
    imagenet_mean, imagenet_std = get_imagenet_mean_std(mode)
    inputs, classes = next(iter(data_loader[phase]))
    fig, axes = plt.subplots(n, n, figsize=(6, 6))

    key_list = list(data[phase].classes.keys())
    val_list = list(data[phase].classes.values())

    for i in range(n):
        for j in range(n):
            image = inputs[i * n + j].numpy().transpose((1, 2, 0))
            title = key_list[val_list.index(classes[i * n + j])]
            if mode == 'RGB': 
                image = np.clip(np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1)
                axes[i, j].imshow(image)
            elif mode == 'GRAYSCALE': 
                axes[i, j].imshow(image, cmap='viridis', norm='linear')
            axes[i, j].set_title(title, fontdict={"fontsize": 7})
            axes[i, j].axis("off")


def load_dataset(config, phases):
    csv_file = os.path.join(config["csv_dir"], f"{config['attribute']}.csv")
    dataset = pd.read_csv(csv_file)
    transforms = get_transforms(size=config["img_size"], mode=config["mode"])
    classes = list(dataset.label.unique())
    classes = {class_: index for index, class_ in enumerate(classes)}
    print(classes)

    data = {
        phase: CaribbeanDataset(
            dataset[dataset.dataset == phase].sample(
                frac=1, random_state=SEED
            ).reset_index(drop=True),
            config["data_dir"],
            config["attribute"],
            classes,
            config["mode"],
            transforms[phase],
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


def train(data_loader, model, criterion, optimizer, device, wandb=None):
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
    epoch_results['loss'] = epoch_loss
    
    learning_rate = optimizer.param_groups[0]["lr"]
    print(f"Loss: {epoch_loss} {epoch_results} LR: {learning_rate}")

    if wandb is not None:
        wandb.log({"train_" + k: v for k, v in epoch_results.items()})
    return epoch_results


def evaluate(data_loader, class_names, model, criterion, device, wandb=None):
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
    epoch_results['loss'] = epoch_loss
    
    confusion_matrix, cm_metrics, cm_report = eval_utils.get_confusion_matrix(
        y_actuals, y_preds, class_names
    )
    print(f"Loss: {epoch_loss} {epoch_results}")

    if wandb is not None:
        wandb.log({"val_" + k: v for k, v in epoch_results.items()})
    return epoch_results, (confusion_matrix, cm_metrics, cm_report)


def get_transforms(size, mode='RGB'):
    imagenet_mean, imagenet_std = get_imagenet_mean_std(mode)

    return {
        "train": transforms.Compose(
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
        "test": transforms.Compose(
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
            
        if mode == 'GRAYSCALE':
            #source: https://datascience.stackexchange.com/a/65784
            weights = model.conv1.weight
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.conv1.weight = nn.Parameter(torch.mean(weights, dim=1, keepdim=True))
            
        num_ftrs = model.fc.in_features
        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(num_ftrs, n_classes)
            )
        else:
            model.fc = nn.Linear(num_ftrs, n_classes)
            
    if 'inception' in model_type:
        if mode == 'RGB':
            model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        elif mode == 'GRAYSCALE':
            model = models.inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1, 
                transform_input=False
            )
            weights = model.Conv2d_1a_3x3.conv.weight.clone()
            model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
            model.Conv2d_1a_3x3.conv.weight = nn.Parameter(torch.mean(weights, dim=1, keepdim=True))
        
        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        
    if 'vgg' in model_type:
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        if mode == 'GRAYSCALE':
            weights = model.features[0].weight.clone()
            model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            model.features[0].weight = nn.Parameter(torch.mean(weights, dim=1, keepdim=True))
        
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, n_classes)
        
    return model


def load_model(
    model_type,
    n_classes,
    pretrained,
    scheduler_type,
    optimizer_type,
    mode='RGB',
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
    criterion = nn.CrossEntropyLoss()

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler_type == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=patience
        )

    return model, criterion, optimizer, scheduler


def generate_train_test(
    folder_path,
    column,
    out_dir,
    test_size,
    test_iso=None,
    stratified=True,
    verbose=True,
):
    folder_dir = os.path.join(folder_path, column)
    folders = [folder.name for folder in os.scandir(folder_dir)]
    data = []

    for row, folder in enumerate(folders):
        filepath = os.path.join(folder_dir, folder)
        files = os.listdir(filepath)

        for file in files:
            iso = file.split("_")[0]
            data.append([iso, file, folder])

    data = pd.DataFrame(data, columns=["iso", "filename", "label"])
    data["dataset"] = None

    total_size = len(data)
    test_size = int(total_size * test_size)

    test = data.copy()
    if test_iso != None:
        test = data[data.iso == test_iso]

    if stratified:
        value_counts = data.label.value_counts().items()
        for label, count in value_counts:
            subtest = test[test.label == label]
            subtest_size = int(test_size * (count / total_size))
            subtest_files = subtest.sample(
                subtest_size, random_state=SEED
            ).filename.values
            data.loc[data["filename"].isin(subtest_files), "dataset"] = "test"
    data.dataset = data.dataset.fillna("train")

    if verbose:
        value_counts = pd.concat([
            data.label.value_counts(), 
            data.label.value_counts(normalize=True)
        ], axis=1, keys=["counts", "percentage"])
        print(value_counts)

        subcounts = pd.DataFrame(
            data.groupby(["dataset", "label"]).size().reset_index()
        )
        subcounts.columns = ["dataset", "label", "count"]
        subcounts["percentage"] = (
            subcounts[subcounts.dataset == "test"]["count"] / test_size
        )
        subcounts = subcounts.set_index(["dataset", "label"])
        print(subcounts)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, f"{column}.csv")
    data.to_csv(out_file, index=False)

    return data