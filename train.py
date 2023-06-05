import os
import json_fix 
import json
import time
import random
import argparse
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

import sys
sys.path.insert(0, "./utils/")
import config
import cnn_utils

import wandb
#from codecarbon import EmissionsTracker

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
json.fallback_table[np.ndarray] = lambda array: array.tolist()

def main(c):
    
    # Create experiment directory
    # This folder will contain the trained model
    # and records of the results
    
    exp_name = c['exp_name']
    exp_dir = f'./exp/{exp_name}/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        
    wandb.init(project="GFDRR")
    wandb.run.name = exp_name
    wandb.config = c
    
    # Load dataset    
    phases = ["train", "test"]
    data, data_loader, classes = cnn_utils.load_dataset(
        c['data_dir'], config=c, phases=phases, train_size=0.8
    )   
    print("Train/test sizes: {}/{}".format(len(data['train']), len(data['test'])))
    for phase in phases:
        print(f'{phase.title()} distribution')
        n_classes = [label for _, label in data[phase]]
        print(Counter(n_classes))
    
    # Load model, optimizer, and scheduler
    model, criterion, optimizer, scheduler = cnn_utils.load_model(
        n_classes=len(classes),
        model_type=c['model'],
        pretrained=c['pretrained'],
        scheduler_type=c['scheduler'],
        optimizer_type=c['optimizer'],
        lr=c['lr'],
        momentum=c['momentum'],
        gamma=c['gamma'],
        step_size=c['step_size'],
        patience=c['patience'],
        dropout=c['dropout'],
        device=device,
    )
    print(model)
    
    # Instantiate wandb tracker
    # and codecarbon emissions tracker
    wandb.watch(model)
    #tracker = EmissionsTracker(output_dir=exp_dir)
    #tracker.start()
    
    # Commence model training
    n_epochs = c['n_epochs']
    since = time.time()
    best_score = -1
    
    for epoch in range(1, n_epochs + 1):    
        print("\nEpoch {}/{}".format(epoch, n_epochs))
        print("-" * 10)
        
        train_results = cnn_utils.train(
            data_loader["train"],
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            wandb=wandb
        )
        val_results, _ = cnn_utils.evaluate(
            data_loader['test'],
            classes,
            model,
            criterion,
            device,
            wandb=wandb
        )
        
        # Save best model so far
        if val_results["f1_score"] > best_score:
            best_score = val_results["f1_score"]
            best_weights = model.state_dict()
            model.load_state_dict(best_weights)
            torch.save(model.state_dict(), exp_dir + "best_model.pth")
        
        # Terminate if learning rate becomes too low
        learning_rate = optimizer.param_groups[0]["lr"]
        if learning_rate < 1e-7:
            break
    
    # Terminate trackers
    #emissions = tracker.stop()
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    
    # Load best model
    model.load_state_dict(best_weights)
    
    # Calculate test performance using best model
    print('\nTest Results')
    test_results, cm_results = cnn_utils.evaluate(
        data_loader['test'],
        classes,
        model,
        criterion,
        device,
        wandb=wandb
    )
    confusion_matrix, cm_metrics, cm_report = cm_results
    
    # Save results in experiment directory
    with open(exp_dir + "results.json", "w") as f:
        json.dump(test_results, f)
    confusion_matrix.to_csv(exp_dir + "confusion_matrix.csv")
    cm_metrics.to_csv(exp_dir + "cm_metrics.csv")
    file = open(exp_dir + "cm_report.log", "a").write(cm_report)
    
    # Save best model
    torch.save(model.state_dict(), exp_dir + "best_model.pth")
    
    
if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Example Description")
    parser.add_argument("--exp_config", help="Config file")
    args = parser.parse_args()

    # Load config
    c = config.create_config(args.exp_config)
    
    main(c)
