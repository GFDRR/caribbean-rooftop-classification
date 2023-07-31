import os
import sys

from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as nn

sys.path.insert(0, "./utils/")
import geoutils
import cnn_utils


def extract_probs_embedding(model, c, input_file):
    """Extracts deep feature embeddings from the trained CNN model.
    Source: https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c

    Args:
        model (nn.Module): Neural network to be used for feature extraction.
        c (dict): Config file.
        input_file (str): The image file for which to extract the feature embeddings.

    Returns:
        array: An array containing the class probabilities.
        array: An array containing the feature embedding.
    """

    def copy_data(m, i, o):
        embedding.copy_(o.data.reshape(o.data.size(1)))

    image = cnn_utils.read_image(input_file, c["mode"])
    transforms = cnn_utils.get_transforms(c["img_size"], c["mode"])

    embedding_size = c["embed_size"]
    embedding = torch.zeros(embedding_size)
    layer = model._modules.get("avgpool")
    h = layer.register_forward_hook(copy_data)

    output = model(transforms["test"](image).unsqueeze(0))
    probs = nn.softmax(output, dim=1).detach().numpy()[0]
    embedding = embedding.detach().numpy()

    h.remove()
    return probs, embedding


def predict(data, c1, c2, model1, model2, source1=None, source2=None, scale=1.5):
    output = []
    pbar = tqdm(
        enumerate(data.iterrows()),
        total=len(data),
        bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}",
    )
    for index, (_, row) in pbar:
        classes = geoutils.classes_dict[c1["attribute"]]

        if "file1" in data.columns:
            file1 = data.iloc[index].file1
        else:
            file1 = "temp.tif"
            if os.path.exists(file1):
                os.remove(file1)
            shape = data[(data.UID == row["UID"])]
            geoutils.crop_shape(shape, "temp.shp", scale, source1, file1)

        model1_probs, model1_embeds = extract_probs_embedding(model1, c1, file1)
        model1_pred = str(classes[np.argmax(model1_probs)])

        if "file2" in data.columns:
            file2 = data.iloc[index].file2
        else:
            file2 = "temp.tif"
            os.remove(file2)
            shape = data[(data.UID == row["UID"])]
            geoutils.crop_shape(shape, "temp.shp", scale, source2, file2)

        model2_probs, model2_embeds = extract_probs_embedding(model2, c2, file2)
        model2_pred = str(classes[np.argmax(model2_probs)])

        mean_probs = np.mean([model1_probs, model2_probs], axis=0)
        mean_pred = str(classes[np.argmax(mean_probs)])

        probs = list(model1_probs) + list(model2_probs) + list(mean_probs)
        embeds = list(model1_embeds) + list(model2_embeds)
        preds = [model1_pred, model2_pred, mean_pred]
        output.append(probs + embeds + preds)

    columns = (
        [f"model1_prob_{x}" for x in range(len(model1_probs))]
        + [f"model2_prob_{x}" for x in range(len(model2_probs))]
        + [f"mean_prob_{x}" for x in range(len(mean_probs))]
        + [f"model1_embedding_{x}" for x in range(len(model1_embeds))]
        + [f"model2_embedding_{x}" for x in range(len(model2_embeds))]
        + ["model1_pred", "model2_pred", "mean_pred"]
    )
    output = pd.DataFrame(output, columns=columns)
    return output


def get_features(c, data):
    """Returns the list of feature names depending on the fusion strategy.

    Args:
        c (dict): Config file.
        data (Pandas DataFrame): The dataframe of feature embeddings and class probabilities.

    Returns:
        list: Contains a list of string value indicating the feature names.
    """

    if c["mode"] == "fusion_probs":
        features = [
            x for x in data.columns if ("model1_prob" in x) or ("model2_prob" in x)
        ]
    elif c["mode"] == "fusion_embeds":
        features = [
            x
            for x in data.columns
            if ("model1_embedding" in x) or ("model2_embedding" in x)
        ]
    return features
