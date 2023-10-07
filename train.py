import os
import sys
import time
import argparse
from collections import Counter
import torch

sys.path.insert(0, "./utils/")
import config
import cnn_utils
import eval_utils
import wandb
import logging

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(c):    
    # Create experiment folder
    mode = c['mode']
    exp_name = c['config_name']
    exp_dir = os.path.join(c["exp_dir"], exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    logname = os.path.join(exp_dir, f"{exp_name}.log")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.FileHandler(logname)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    logging.info(device)
    logging.info(exp_name)

    # Set wandb configs
    wandb.init(project="GFDRRv2", config=c)
    wandb.run.name = exp_name
    wandb.config = c
    logging.info(c)
    
    # Load dataset
    phases = ["TRAIN", "TEST"]
    data, data_loader, classes = cnn_utils.load_dataset(config=c, phases=phases)
    logging.info(f"Train/test sizes: {len(data['TRAIN'])}/{len(data['TEST'])}")
    for phase in phases:
        logging.info(f"{phase.title()} distribution")
        n_classes = [label for _, label in data[phase]]
        logging.info(Counter(n_classes))

    # Load model, optimizer, and scheduler
    model, criterion, optimizer, scheduler = cnn_utils.load_model(
        n_classes=len(classes),
        model_type=c["model"],
        pretrained=c["pretrained"],
        scheduler_type=c["scheduler"],
        optimizer_type=c["optimizer"],
        label_smoothing=c["label_smoothing"],
        lr=c["lr"],
        momentum=c["momentum"],
        gamma=c["gamma"],
        step_size=c["step_size"],
        patience=c["patience"],
        dropout=c["dropout"],
        mode=mode,
        device=device,
    )
    logging.info(model)

    # Instantiate wandb tracker
    wandb.watch(model)

    # Commence model training
    n_epochs = c["n_epochs"]
    since = time.time()
    best_score = -1

    for epoch in range(1, n_epochs + 1):
        logging.info("\nEpoch {}/{}".format(epoch, n_epochs))

        # Train model
        cnn_utils.train(
            data_loader["TRAIN"],
            model,
            criterion,
            optimizer,
            device,
            wandb=wandb,
            logging=logging
        )
        # Evauate model
        val_results, val_cm = cnn_utils.evaluate(
            data_loader["TEST"], classes, model, criterion, device, wandb=wandb, logging=logging
        )
        scheduler.step(val_results["f1_score"])

        # Save best model so far
        if val_results["f1_score"] > best_score:
            best_score = val_results["f1_score"]
            best_weights = model.state_dict()

            eval_utils.save_results(val_results, val_cm, exp_dir)
            model_file = os.path.join(exp_dir, f"{exp_name}.pth")
            torch.save(model.state_dict(), model_file)
        logging.info(f"Best F1 score: {best_score}")

        # Terminate if learning rate becomes too low
        learning_rate = optimizer.param_groups[0]["lr"]
        if learning_rate < 1e-10:
            break

    # Terminate trackers
    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # Load best model
    model_file = os.path.join(exp_dir, f"{exp_name}.pth")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)

    # Calculate test performance using best model
    logging.info("\nTest Results")
    test_results, test_cm = cnn_utils.evaluate(
        data_loader["TEST"], classes, model, criterion, device, wandb=wandb, logging=logging
    )

    # Save results in experiment directory
    eval_utils.save_results(test_results, test_cm, exp_dir)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--exp_config", help="Config file")
    args = parser.parse_args()

    # Load config
    c = config.load_config(args.exp_config)

    main(c)
