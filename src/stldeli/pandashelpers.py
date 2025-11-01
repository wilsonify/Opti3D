#!/usr/bin/env python
# coding: utf-8

"""
This script automates the process of slicing a single STL file with multiple
combinations of command-line arguments for Slic3r. Each configuration is
applied, executed, and logged, and the resulting metadata is aggregated into
a CSV file.
"""

import itertools as it
import logging
import os
from logging.config import dictConfig
from subprocess import CalledProcessError, check_output

import pandas as pd

from stldeli import config


def flag2placeholder(flag: str) -> str:
    """
    Convert a flag string into a valid command-line placeholder.

    Example:
        --layer-height -> layer_height[layer_height]
    """
    logging.debug("flag2placeholder(%s)", flag)
    flag_str_clean = str(flag).strip("-").replace("-", "_")
    return f"{flag_str_clean}[{flag_str_clean}]"


def get_combinations_from_configurations(configurations: dict):
    """
    Generate all possible combinations from a configuration dictionary.

    Example:
        configurations = {"--layer-height": [0.1, 0.2], "--infill": [20, 50]}
        -> yields all combinations of those options.
    """
    logging.info("get_combinations_from_configurations")
    return it.product(*(configurations[name] for name in configurations))


def get_series_from_gcode(gcode_file_path: str) -> pd.Series:
    """
    Extract metadata from a G-code file into a pandas Series.

    Lines starting with ';' are treated as comments possibly containing
    key-value pairs in the form "KEY=VALUE".
    """
    logging.debug("get_series_from_gcode(%s)", gcode_file_path)
    metarow = pd.Series(dtype=object)

    try:
        with open(gcode_file_path, encoding="utf-8") as gcode_file:
            for line in gcode_file:
                if line.startswith(';'):
                    datum = line.strip('; \n').split('=', maxsplit=1)
                    if len(datum) == 2:
                        metarow[datum[0].strip()] = datum[1].strip()
    except FileNotFoundError:
        logging.error("G-code file not found: %s", gcode_file_path)

    return metarow


# pylint: disable=too-many-locals
def main() -> pd.DataFrame:
    """
    Main workflow:
    1. Generate all slic3r argument combinations.
    2. Run slic3r for each combination.
    3. Collect metadata into a single DataFrame.
    """
    logging.info("Starting main process")

    combinations = list(get_combinations_from_configurations(config.slic3r_configurations))
    total = len(combinations)
    logging.info("%d possible slice combinations", total)

    count = 0
    metadata = pd.DataFrame()
    input_file = os.path.abspath("stl_files/largecube.stl")

    for configuration in combinations:
        logging.debug("configuration = %s", configuration)
        metarow = pd.Series(configuration, index=config.slic3r_configurations.keys())
        output_file_format = "[input_filename_base]"
        print(f"{count + 1} out of {total}")

        cmd = ["slic3r"]
        for key, value in zip(config.slic3r_configurations.keys(), configuration):
            logging.debug("adding %s = %s to cmd", key, value)
            metarow[key] = value
            if value:
                cmd.append(str(key))
                if not isinstance(value, bool):
                    cmd.append(str(value))
            output_file_format += "_" + flag2placeholder(key)

        cmd.append("--output-filename-format")
        gcode_file_path = f"{count}_{output_file_format}_.gcode"
        cmd.extend([gcode_file_path, input_file])

        metarow["filenumber"] = count
        cmd_str = " ".join(map(str, cmd))
        print(cmd_str)

        try:
            check_output(cmd)
            gcode_series = get_series_from_gcode(gcode_file_path)
            combined_row = pd.concat([metarow, gcode_series])
            metadata = pd.concat([metadata, combined_row.to_frame().T], ignore_index=True)
            os.remove(gcode_file_path)
            count += 1
        except CalledProcessError as error:
            logging.error("Unable to slice configuration %s: %s", configuration, error)
            continue

    return metadata


if __name__ == "__main__":
    os.makedirs(config.LOG_DIR, exist_ok=True)
    dictConfig(config.LOG_DICT_CONFIG)

    logging.info("Launching slic3r automation")
    df_metadata = main()  # pylint: disable=invalid-name
    output_path = "metadata.csv"
    df_metadata.to_csv(output_path, index=False)
    logging.info("Saved metadata to %s", output_path)
