import os
import sys
import argparse
import joblib
import pandas as pd

sys.path.insert(0, "./utils/")
import geoutils
import config
import pred_utils
import eval_utils
import model_utils
import fusion_utils

import logging
logging.basicConfig(level = logging.INFO)


def main(c):
    out_dir = os.path.join(c["exp_dir"], c["exp_name"])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file = os.path.join(out_dir, "output.csv")

    c1 = config.create_config(c["config1"])
    c2 = config.create_config(c["config2"])
    logging.info(c1)
    logging.info(c2)
    classes = geoutils.classes_dict[c1["attribute"]]

    if not os.path.exists(output_file):
        exp_dir = os.path.join(c["exp_dir"], c1["exp_name"])
        model1 = pred_utils.load_model(c1, exp_dir=exp_dir, n_classes=len(classes))
        exp_dir = os.path.join(c["exp_dir"], c2["exp_name"])
        model2 = pred_utils.load_model(c2, exp_dir=exp_dir, n_classes=len(classes))

        csv_file = os.path.join(c1["csv_dir"], f"{c1['attribute']}.csv")
        data = pd.read_csv(csv_file)

        def f(x, c):
            return os.path.join(c["data_dir"], c["attribute"], x.label, x.filename)

        data["file1"] = data.apply(lambda x: f(x, c1), axis=1)
        data["file2"] = data.apply(lambda x: f(x, c2), axis=1)
        output = fusion_utils.predict(data, c1, c2, model1, model2)

        output["UID"] = data.filename.values
        output["dataset"] = data.dataset.values
        output["label"] = data.label.values

        output.to_csv(output_file, index=False)

    output = pd.read_csv(output_file)
    test = output[output.dataset == "test"]
    train = output[output.dataset == "train"]

    results_dir = os.path.join(out_dir, c["mode"])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Get results for mean of softmax probabilities
    if c["mode"] == "fusion_mean":
        preds = test["mean_pred"]
    else:
        results_dir = os.path.join(results_dir, c["model"])
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        features = fusion_utils.get_features(c, output)
        target = "label"

        cv = model_utils.model_trainer(c, train, features, target)
        logging.info(cv.best_estimator_)
        logging.info(cv.best_score_)

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
    logging.info(c)

    main(c)
