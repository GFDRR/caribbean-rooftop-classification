"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import yaml
from easydict import EasyDict


def create_config(config_file_exp, prefix=None):
    """Loads YAML config file as dictionary.

    Args:
        config_file_exp (str): Path to config file.
        prefix (str): Config file prefix.

    Returns:
        dict: The config as a dictionary.
    """

    with open(config_file_exp, "r") as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()
    for k, v in config.items():
        cfg[k] = v
    if prefix is not None:
        for key in cfg:
            if "dir" in key:
                cfg[key] = prefix + cfg[key]

    return cfg
