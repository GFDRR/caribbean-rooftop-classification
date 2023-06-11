import os
import sys
import time
import random
import argparse
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, "./utils/")
import config
import cnn_utils
import eval_utils
import wandb

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(c):
    exp_name = c["exp_name"]
    exp_dir = f"./exp/{exp_name}/"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    wandb.init(project="GFDRR")
    wandb.run.name = exp_name
    wandb.config = c
    print(c)

    # Load dataset
    phases = ["train", "test"]
    data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases)
    print("Train/test sizes: {}/{}".format(len(data["train"]), len(data["test"])))
    for phase in phases:
        print(f"{phase.title()} distribution")
        n_classes = [label for _, label in data[phase]]
        print(Counter(n_classes))

    # Load model, optimizer, and scheduler
    model, criterion, optimizer, scheduler = cnn_utils.load_model(
        n_classes=len(classes),
        model_type=c["model"],
        pretrained=c["pretrained"],
        scheduler_type=c["scheduler"],
        optimizer_type=c["optimizer"],
        lr=c["lr"],
        momentum=c["momentum"],
        gamma=c["gamma"],
        step_size=c["step_size"],
        patience=c["patience"],
        dropout=c["dropout"],
        mode=c['mode'],
        device=device,
    )
    print(model)

    # Instantiate wandb tracker
    wandb.watch(model)

    # Commence model training
    n_epochs = c["n_epochs"]
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
            device,
            wandb=wandb,
        )
        val_results, val_cm = cnn_utils.evaluate(
            data_loader["test"], 
            classes,
            model, 
            criterion, 
            device, 
            wandb=wandb
        )
        scheduler.step(val_results['loss'])

        # Save best model so far
        if val_results["f1_score"] > best_score:
            best_score = val_results["f1_score"]
            best_weights = model.state_dict()
            model.load_state_dict(best_weights)
            
            eval_utils.save_results(val_results, val_cm, exp_dir)
            model_file = os.path.join(exp_dir, "best_model.pth")
            torch.save(model.state_dict(), model_file)

        # Terminate if learning rate becomes too low
        learning_rate = optimizer.param_groups[0]["lr"]
        if learning_rate < 1e-10:
            break

    # Terminate trackers
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Load best model
    model.load_state_dict(best_weights)

    # Calculate test performance using best model
    print("\nTest Results")
    test_results, test_cm = cnn_utils.evaluate(
        data_loader["test"], classes, model, criterion, device, wandb=wandb
    )

    # Save results in experiment directory
    eval_utils.save_results(test_results, test_cm, exp_dir)

    # Save best mode
    model_file = os.path.join(exp_dir, "best_model.pth")
    torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Example Description")
    parser.add_argument("--exp_config", help="Config file")
    args = parser.parse_args()

    # Load config
    c = config.create_config(args.exp_config)

    main(c)
