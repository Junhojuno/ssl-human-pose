from typing import Union, Dict
from pathlib import Path
import yaml

from easydict import EasyDict


def parse_yaml(yaml_file: Union[Path, str]) -> Dict:
    data = yaml.load(
        open(yaml_file, 'r'), Loader=yaml.Loader
    )
    return EasyDict(data)
