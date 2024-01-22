import json
from argparse import ArgumentParser

import numpy as np


def parse_command_line_args():
    """
    Parse command line arguments.
    Return
    ------
    config_path: str
        configurations file path.
    """
    parser = ArgumentParser()
    parser.add_argument("-cfg", "--config-file", type=str, required=True)
    args = parser.parse_args()
    return args.config_file


def get_value(config_dict, searched_key):
    """
    Takes a configuration dict and a searched key as arguments, searches
    recursively for the searched_key in the configuration dict and returns
    the corresponding value.

    Parameters
    -----------
    config_dict : dictionary
        Usually from configuration json file / is usually nested dict

    searched_key : str
        String key name

    Returns
    -------
    value: str
        Value associated to the searched key
    """
    for key, value in config_dict.items():
        # if key found in the first level of dict return it
        if key == searched_key:
            return value
        # else if value is a dict --> nested dict --> need to check into it
        elif isinstance(value, dict):
            # --> call again the function (recursively)
            value_found = get_value(config_dict[key], searched_key)
            # return value if not None
            if value_found is not None:
                return value_found
    return None


def load_json(file_path):
    """
    Takes a json filepath as input, opens it and returns the content
    :param file_path: string path json file
    :return: the content of the json file (a dictionary)
    """
    with open(file_path, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(dict_to_save, file_path):
    """Save a dictionary to a json file."""
    dict_to_save = json.dumps(dict_to_save, cls=NpEncoder)
    with open(file_path, "w", encoding="utf8") as fp:
        json.dump(dict_to_save, fp, indent=4)
    with open(file_path, "r", encoding="utf8") as fp:
        dict_to_save = fp.read().replace("\\", "").strip('"')
    with open(file_path, "w", encoding="utf8") as fp:
        fp.write(dict_to_save)


class NpEncoder(json.JSONEncoder):
    """
    Class that defines a JSON encoder which converts
    numpy objects to standard types for saving to json.
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)
