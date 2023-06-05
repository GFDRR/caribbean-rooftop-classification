from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, Subset

import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights
)

import sys
sys.path.insert(0, './utils/')
import eval_utils


class CaribbeanDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)

def load_dataset(data_dir, config, phases, size=224, train_size=0.8):
    
    # Apply transforms to the (full) dataset
    dataset = datasets.ImageFolder(data_dir)
    transforms = get_transforms(size=size)
    classes = dataset.classes
    
    print(classes)
    print(dataset.class_to_idx)
    
    data = {
        phase: CaribbeanDataset(dataset, transforms[phase])
        for phase in phases
    }
    
    # Randomly split the dataset into 80% train / 20% test 
    # by subsetting the transformed datasets into train and test sets
    
    indices = list(range(len(dataset)))
    split = int(train_size * len(dataset))
    np.random.shuffle(indices)
    
    split_indices = {'train': indices[:split], 'test': indices[split:]}
    data = {
        phase: Subset(data[phase], indices=split_indices[phase])
        for phase in phases
    }
    
    data_loader = {
        phase: torch.utils.data.DataLoader(
            data[phase], 
            batch_size=config['batch_size'], 
            num_workers=config['n_workers'],
            shuffle=True
        )
        for phase in phases
    }
    
    return data, data_loader, classes
    
def train(data_loader, model, criterion, optimizer, scheduler, device, wandb=None):
    model.train()

    y_actuals, y_preds = [], []
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

            y_actuals.extend(labels.cpu().numpy().tolist())
            y_preds.extend(preds.data.cpu().numpy().tolist())
    
    epoch_results = eval_utils.evaluate(y_actuals, y_preds)
    learning_rate = optimizer.param_groups[0]["lr"]
    scheduler.step(epoch_results['f1_score'])
    print(epoch_results, learning_rate)

    if wandb is not None:
        wandb.log({'train_' + k: v for k, v in epoch_results.items()})
    return epoch_results


def evaluate(data_loader, class_names, model, criterion, device, wandb=None):
    model.eval()

    y_actuals, y_preds = [], []
    confusion_matrix = torch.zeros(len(class_names), len(class_names))

    for inputs, labels in tqdm(data_loader, total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            probs, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        y_actuals.extend(labels.cpu().numpy().tolist())
        y_preds.extend(preds.data.cpu().numpy().tolist())
    
    epoch_results = eval_utils.evaluate(y_actuals, y_preds)
    confusion_matrix, cm_metrics, cm_report = eval_utils.get_confusion_matrix(
        y_actuals, y_preds, class_names
    )
    print(epoch_results)

    if wandb is not None:
        wandb.log({'val_' + k: v for k, v in epoch_results.items()})
    return epoch_results, (confusion_matrix, cm_metrics, cm_report)


def get_transforms(size):
    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation((0,360)),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)
            ]
        ),
    }


def load_model(
    model_type,
    n_classes,
    pretrained,
    scheduler_type,
    optimizer_type,
    lr=0.001,
    momentum=0.9,
    gamma=0.1,
    step_size=7,
    patience=7,
    dropout=0,
    device="cpu",
):
    if "resnet" in model_type:
        if model_type == "resnet18":
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_type == "resnet34":
            model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model_type == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        
        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_ftrs, n_classes)
            )
        else:
            model.fc = nn.Linear(num_ftrs, n_classes)
        
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
