import logging
from typing import Union, Dict
from pathlib import Path
import yaml

from easydict import EasyDict


def parse_yaml(yaml_file: Union[Path, str]) -> Dict:
    data = yaml.load(
        open(yaml_file, 'r'), Loader=yaml.Loader
    )
    return EasyDict(data)


def get_logger(log_file_path, name='ssl-pose'):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s line:%(lineno)d]:%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
