"""Helper functions for loading configuration files."""

import json
from pprint import pprint
from easydict import EasyDict


def get_config_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as config_file:
        try:
            config_dict = json.load(config_file)
        except ValueError:
            print("Invalid json file format.")
            exit(-1)

    config = EasyDict(config_dict)
    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    print("Configurations:")
    pprint(config)
    print(r"---------------------------------")
    return config
