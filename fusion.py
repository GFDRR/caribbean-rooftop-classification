import os
import sys
import argparse
import joblib

import rasterio as rio
import pandas as pd
import numpy as np
import geopandas as gpd

import torch
import torchsummary
from tqdm import tqdm

sys.path.insert(0, "./utils/")
import geoutils
import cnn_utils
import config
import pred_utils
import eval_utils
import model_utils

import torch.nn.functional as nn


cnn_arch = {'resnet50': 2048, 'inceptionv3': 2048, 'efficientnetb0': 1280}

def extract_probs_embedding(model, c, input_file):
    # Source: https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
    def copy_data(m, i, o):
        embedding.copy_(o.data.reshape(o.data.size(1)))
    image = cnn_utils.read_image(input_file, c["mode"])
    transforms = cnn_utils.get_transforms(c["img_size"], c["mode"])
    
    embedding_size = cnn_arch[c['model']]
    embedding = torch.zeros(embedding_size)
    layer = model._modules.get('avgpool')
    h = layer.register_forward_hook(copy_data)
    
    output = model(transforms["test"](image).unsqueeze(0))
    probs = nn.softmax(output, dim=1).detach().numpy()[0]
    embedding = embedding.detach().numpy()
    
    h.remove()
    return probs, embedding


def predict(data, c1, c2, model1, model2, source1=None, source2=None):
    output = []
    for index in tqdm(range(len(data)), total=len(data)):
        classes = geoutils.classes_dict[c1['attribute']]
        
        file1 = data.iloc[index].file1
        model1_probs, model1_embeds = extract_probs_embedding(model1, c1, file1)
        model1_pred = str(classes[np.argmax(model1_probs)])

        file2 = data.iloc[index].file2
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


def main(c):    
    out_dir = os.path.join(c['exp_dir'], c['exp_name'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file = os.path.join(out_dir, "output.csv")
    
    c1 = config.create_config(c["config1"])
    c2 = config.create_config(c["config2"])
    classes = geoutils.classes_dict[c1["attribute"]]
    
    if not os.path.exists(output_file):                
        exp_dir = os.path.join(c['exp_dir'], c1['exp_name'])
        model1 = pred_utils.load_model(c1, exp_dir=exp_dir, n_classes=len(classes))
        exp_dir = os.path.join(c['exp_dir'], c2['exp_name'])
        model2 = pred_utils.load_model(c2, exp_dir=exp_dir, n_classes=len(classes))

        csv_file = os.path.join(c1["csv_dir"], f"{c1['attribute']}.csv")
        data = pd.read_csv(csv_file)
        
        def f(x, c): 
            return os.path.join(c["data_dir"], c["attribute"], x.label, x.filename)
        data['file1'] = data.apply(lambda x: f(x, c1), axis=1)
        data['file2'] = data.apply(lambda x: f(x, c2), axis=1)
        output = predict(data, c1, c2, model1, model2)
        
        output['UID'] = data.filename.values
        output['dataset'] = data.dataset.values
        output['label'] = data.label.values
    
        output.to_csv(output_file, index=False)
    
    output = pd.read_csv(output_file)
    test = output[output.dataset == 'test']
    train = output[output.dataset == 'train']
    
    results_dir = os.path.join(out_dir, c['mode'])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Get results for mean of softmax probabilities
    if c['mode'] == 'fusion_mean':
        preds = test["mean_pred"]
    else:
        results_dir = os.path.join(results_dir, c['model'])
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if c['mode'] == 'fusion_probs':
            features = [
                x for x in output.columns
                if ('model1_prob' in x) or ('model2_prob' in x)
            ]
        elif c['mode'] == 'fusion_embeds':
            features = [
                x for x in output.columns
                if ('model1_embedding' in x) or ('model2_embedding' in x)
            ]
        target = "label"
        
        cv = model_utils.model_trainer(c, train, features, target)
        print(cv.best_estimator_)
        print(cv.best_score_)
        
        model = cv.best_estimator_
        model.fit(train[features], train[target].values)
        preds = model.predict(test[features])
            
        model_file = os.path.join(results_dir, "best_model.pkl")
        joblib.dump(model, model_file)
    
    results = eval_utils.evaluate(test["label"], preds)
    cm = eval_utils.get_confusion_matrix(test["label"], preds, classes)
    eval_utils.save_results(results, cm, results_dir)
            
        
if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Data Fusion")
    parser.add_argument("--exp_config", help="Config file")
    args = parser.parse_args()

    # Load config
    c = config.create_config(args.exp_config)
    print(c)

    main(c)