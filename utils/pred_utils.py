import os
import sys

from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import numpy as np

import torch
import torch.nn.functional as nn

sys.path.insert(0, "./utils/")
import cnn_utils
import geoutils
import config

import logging
logging.basicConfig(level = logging.INFO)

def predict(bldgs, in_file, exp_config, prefix=''):
    c = config.create_config(exp_config, prefix=prefix)
    exp_dir = os.path.join(c['exp_dir'], c['version'], c['exp_name'])
    classes = geoutils.get_classes_dict(c['attribute'])
    logging.info(f"Config: {c}")

    model = load_model(c, exp_dir, n_classes=len(classes))
    return generate_predictions(bldgs, model, c, in_file, exp_dir, classes=classes)   


def load_model(c, exp_dir, n_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_file = os.path.join(exp_dir, "best_model.pth")
    model = cnn_utils.get_model(c["model"], n_classes, c["mode"], c["dropout"])
    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()
    logging.info("Model file {} successfully loaded.".format(model_file))
    return model


def generate_predictions(data, model, c, in_file, out_dir, classes, scale=1.5):
    preds, probs = [], []
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
            image = cnn_utils.read_image(out_file, c["mode"])
            transforms = cnn_utils.get_transforms(c["img_size"], c["mode"])
            output = model(transforms["test"](image).unsqueeze(0))
            probs.append(nn.softmax(output, dim=1).detach().numpy()[0])
            _, pred = torch.max(output, 1)
            label = str(classes[int(pred[0])])
            preds.append(label)
        else:
            probs.append([np.nan] * len(classes))
            preds.append(np.nan)
    
    probs_col = [f"{class_}_PROB" for class_ in classes] 
    probs = pd.DataFrame(probs, columns=probs_col)
    data[c["attribute"]] = preds
    return gpd.GeoDataFrame(pd.concat([data, probs], axis=1))
