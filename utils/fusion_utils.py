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

embedding_sizes = {'inceptionv3': 2048, 'resnet50': 2048, 'efficientnetb0': 1280}


def extract_probs_embedding(model, c, input_file):
    """Extracts class probabilities and feature embeddings from the given image 
    using the provided CNN model.
    Source: https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c

    Args:
        model (nn.Module): The neural network model for feature extraction.
        c (dict): Configuration file.
        input_file (str): Path to the image file for feature extraction.

    Returns:
        tuple: A tuple containing the class probabilities array and the feature embedding array.
    """

    def copy_data(m, i, o):
        embedding.copy_(o.data.reshape(o.data.size(1)))

    image = cnn_utils.read_image(input_file, c["mode"])
    transforms = cnn_utils.get_transforms(c["img_size"], c["mode"])

    embedding_size = embedding_sizes[c['model']]
    embedding = torch.zeros(embedding_size)
    layer = model._modules.get("avgpool")
    h = layer.register_forward_hook(copy_data)

    output = model(transforms["TEST"](image).unsqueeze(0))
    probs = nn.softmax(output, dim=1).detach().numpy()[0]
    embedding = embedding.detach().numpy()

    h.remove()
    return probs, embedding


def predict(data, c1, c2, model1, model2, source1=None, source2=None, scale=1.5):
    """Predicts using two models and generates output based on specified data.

    Args:
        data (Pandas DataFrame): The data to perform predictions on.
        c1 (dict): Configuration for the first model.
        c2 (dict): Configuration for the second model.
        model1 (nn.Module): The first neural network model.
        model2 (nn.Module): The second neural network model.
        source1 (str, optional): Path to the source for the first model. Defaults to None.
        source2 (str, optional): Path to the source for the second model. Defaults to None.
        scale (float, optional): Scaling factor. Defaults to 1.5.

    Returns:
        Pandas DataFrame: Predicted results based on the specified data.
    """
    
    output = []
    pbar = tqdm(
        enumerate(data.iterrows()),
        total=len(data),
        bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}",
    )
    for index, (_, row) in pbar:
        classes = geoutils.get_classes_dict(c1["attribute"])

        if "file1" in data.columns:
            file1 = data.iloc[index].file1
        else:
            file1 = "temp.tif"
            if os.path.exists(file1):
                os.remove(file1)
            shape = data[(data.UID == row["UID"])]
            geoutils.crop_shape(shape, scale, source1, file1)

        if "file2" in data.columns:
            file2 = data.iloc[index].file2
        else:
            file2 = "temp.tif"
            os.remove(file2)
            shape = data[(data.UID == row["UID"])]
            geoutils.crop_shape(shape, scale, source2, file2)
        
        model1_probs, model1_embeds = extract_probs_embedding(model1, c1, file1)
        model1_pred = str(classes[np.argmax(model1_probs)])
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
    """Returns a list of feature names based on the fusion strategy.

    Args:
        c (dict): Configuration file.
        data (Pandas DataFrame): The DataFrame containing feature embeddings and 
        class probabilities.

    Returns:
        list: A list of strings indicating the feature names based on the fusion strategy.
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
