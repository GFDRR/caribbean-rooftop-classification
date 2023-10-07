import collections
import numpy as np

from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    StandardScaler,
    RobustScaler,
)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from utils import clf_utils
from utils import eval_utils

import logging

logging.basicConfig(level=logging.INFO)

SEED = 42


def _get_scalers(scalers):
    """Returns a list of scalers for hyperparameter optimization.

    Args:
        scalers (list): A list of strings indicating the scalers
            to include in the hyperparameter search space.

    Returns:
        list: A list of sclaer instances.
    """

    scalers_list = []

    if "MinMaxScaler" in scalers:
        scalers_list.append(MinMaxScaler())
    if "StandardScaler" in scalers:
        scalers_list.append(StandardScaler())
    if "RobustScaler" in scalers:
        scalers_list.append(RobustScaler())
    if "MaxAbsScaler" in scalers:
        scalers_list.append(MaxAbsScaler())
    return scalers_list


def _get_pipeline(model, selector):
    """Instantiates and returns a pipeline based on
    the input configuration.

    Args:
        model (object): The model instance to include in the pipeline.
        selector (object): The selector instance to include in the pipeline.

    Returns:
        sklearn pipeline instance.
    """

    if model in clf_utils.MODELS:
        model = clf_utils.get_model(model)

    if selector in clf_utils.SELECTORS:
        selector = clf_utils.get_selector(selector)

    return Pipeline(
        [
            ("scaler", "passthrough"),
            ("selector", selector),
            ("model", model),
        ]
    )


def _get_params(scalers, model_params, selector_params):
    """Instantiates the parameter grid for hyperparameter optimization.

    Args:
        scalers (dict): A dictionary indicating the the list of scalers.
        model_params (dict): A dictionary containing the model parameters.
        selector_params (dict): A dictionary containing the feature
            selector parameters.

    Returns
        dict: Contains the parameter grid, combined into a single dictionary.
    """

    def _get_range(param):
        if param[0] == "np.linspace":
            return list(np.linspace(*param[1:]).astype(int))
        elif param[0] == "range":
            return list(range(*param[1:]))
        return param

    scalers = {"scaler": _get_scalers(scalers)}

    if model_params:
        model_params = {
            "model__" + name: _get_range(param) for name, param in model_params.items()
        }
    else:
        model_params = {}

    if selector_params:
        selector_params = {
            "selector__" + name: _get_range(param)
            for name, param in selector_params.items()
        }
    else:
        selector_params = {}

    params = [model_params, selector_params, scalers]

    return dict(collections.ChainMap(*params))


def get_cv(c):
    """Returns a model selection instance.

    Args:
        c (dict): The config dictionary indicating the model,
            selector, scalers, parameters, and model selection
            instance.

    Returns:
        object: The model selector instance.
    """

    pipe = _get_pipeline(c["model"], c["selector"])
    params = _get_params(c["scalers"], c["model_params"], c["selector_params"])
    cv, cv_params = c["cv"], c["cv_params"]

    assert cv in [
        "RandomizedSearchCV",
        "GridSearchCV",
    ]

    scoring = eval_utils.get_scoring()
    if cv == "RandomizedSearchCV":
        return RandomizedSearchCV(
            pipe, params, scoring=scoring, random_state=SEED, **cv_params
        )
    elif cv == "GridSearchCV":
        return GridSearchCV(pipe, params, scoring=scoring, **cv_params)


def model_trainer(c, data, features, target):
    logging.info("Features: {}, Target: {}".format(features, target))

    X = data[features]
    y = data[target].values

    cv = get_cv(c)
    logging.info(cv)
    cv.fit(X, y)

    logging.info("Best estimator: {}".format(cv.best_estimator_))
    return cv
