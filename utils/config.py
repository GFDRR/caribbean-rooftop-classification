"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict


def create_config(config_file_exp, prefix=""):
    """Loads a YAML config file and converts it into a dictionary.

    Args:
        config_file_exp (str): Path to the configuration file.
        prefix (str): Prefix to be added to file paths in the configuration.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    
    with open(config_file_exp, "r") as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()
    for k, v in config.items():
        cfg[k] = v
        if 'data' in k:
            cfg['mode'] = cfg['data'].split("_")[0]
        if "dir" in k:
            if len(prefix) > 0:
                cfg[k] = prefix + cfg[k]  
            if not os.path.exists(cfg[k]):
                os.makedirs(cfg[k])
    return cfg


def load_config(config_file_exp, prefix=""):
    """Loads configuration files with system configurations included.

    Args:
        config_file_exp (str): Path to the experiment-specific configuration file.
        prefix (str): Prefix to be added to file paths in the configuration.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    
    sys_config_file = prefix + "./configs/config.yaml"
    sys_config = create_config(sys_config_file, prefix=prefix)
    config = create_config(config_file_exp, prefix=prefix)
    config['config_name'] = os.path.basename(config_file_exp).split('.')[0]
    config.update(sys_config)
    
    
    return config
