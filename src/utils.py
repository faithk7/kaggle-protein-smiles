import argparse
import time
from datetime import datetime

import yaml

from config import config


def timeit(func):
    """A decorator that prints the execution time of the function it decorates."""

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Capture the end time
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def parse_yaml(relative_path: str) -> None:
    yaml_path = config.CONFIG_DIR / relative_path

    with open(yaml_path, "r") as file:
        yaml_config = yaml.safe_load(file)

    for key, value in yaml_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Invalid config key: {key} when loading from yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Update config values at runtime.")
    parser.add_argument(
        "--config_path", "-c", type=str, help="Path to YAML config file"
    )

    return parser.parse_args()


def extract_modelname(modelname: str):
    return modelname.split("/")[-1]


def append_curtimestr(filename: str) -> str:
    cur_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{filename}_{cur_time}"
