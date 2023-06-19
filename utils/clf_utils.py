from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    VarianceThreshold,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

SEED = 42


def get_selector(selector):
    """Instantiates and returns a selector instance.

    Args:
        selector (str): Indicates the selector to instantiate.

    Returns:
        object: The selector for feature selection.
    """

    SELECTORS = {
        "SelectKBest": SelectKBest(),
        "SelectKBest_chi2": SelectKBest(chi2),
        "SelectKBest_f_classif": SelectKBest(f_classif),
        "SelectKBest_mutual_info_classif": SelectKBest(mutual_info_classif),
        "RFE": RFE(),
        "VarianceThreshold": VarianceThreshold(),
    }
    assert selector in SELECTORS
    return SELECTORS[selector]



def get_model(model):
    """Instantiates and returns a model instance.

    Args:
        model (str): Indicates the model to instantiate.

    Returns:
        object: The model instance for development.
    """

    MODELS = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=SEED),
        "SGDClassifier": SGDClassifier(random_state=SEED),
        "RidgeClassifier": RidgeClassifier(random_state=SEED),
        "LinearSVC": LinearSVC(max_iter=1000, verbose=1, random_state=SEED),
        "SVC": SVC(random_state=SEED),
        "NuSVC": NuSVC(random_state=SEED),
        "RandomForestClassifier": RandomForestClassifier(verbose=1, n_jobs=-1, random_state=SEED),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=SEED),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=SEED),
        "MultinomialNB": MultinomialNB(),
        "GaussianProcessClassifier": GaussianProcessClassifier(random_state=SEED),
        "MLPClassifier": MLPClassifier(random_state=SEED),
        "LGBMClassifier": LGBMClassifier(random_state=SEED),
        "XGBClassifier": XGBClassifier(random_state=SEED)
    }
    assert model in MODELS
    return MODELS[model]
