import itertools as it
import logging
import os
from logging.config import dictConfig
from subprocess import CalledProcessError, check_output
import pandas as pd
import math

from stldeli import config


def flag2placeholder(flag):
    """Convert a flag into a valid command-line placeholder."""
    flag_str = str(flag)
    flag_str_clean = flag_str.strip("-").replace("-", "_")
    return f"{flag_str_clean}[{flag_str_clean}]"


def get_combinations_from_configurations(configurations):
    """Return a generator of all possible parameter tuples."""
    logging.info("get_combinations_from_configurations")
    return it.product(*(configurations[name] for name in configurations))


def get_series_from_gcode(gcode_file_path):
    """Extract metadata from G-code file as a pandas Series."""
    metarow = pd.Series()
    with open(gcode_file_path) as gcode_file:
        for line in gcode_file:
            if line.startswith(';'):
                datum = line.strip('; \n').split('=')
                if len(datum) == 2:
                    metarow[datum[0]] = datum[1]
    return metarow


# pylint: disable=too-many-locals
def main():
    """Main function: generate all slicing combinations and collect metadata."""
    logging.info("main")

    # Prepare configuration iterator
    configurations = config.slic3r_configurations

    # Compute total combinations without materializing the iterator
    total = math.prod(len(v) for v in configurations.values())
    logging.info(f"{total} possible slices")

    combinations = get_combinations_from_configurations(configurations)

    count = 0
    _metadata = pd.DataFrame()
    input_file = os.path.abspath("stl_files/largecube.stl")

    for configuration in combinations:
        logging.debug(f"configuration = {configuration}")
        metarow = pd.Series(configuration, index=configurations.keys())
        output_file_format = "[input_filename_base]"
        print(f"{count + 1} out of {total}")
        cmd = ["slic3r"]

        for key, value in zip(configurations.keys(), configuration):
            logging.debug(f"adding {key} with value {value} to cmd")
            metarow[key] = value
            if value:
                cmd.append(str(key))
                if not isinstance(value, bool):
                    cmd.append(str(value))
            output_file_format += "_" + flag2placeholder(key)

        gcode_file_path = f"{count}_{output_file_format}_.gcode"
        cmd += ["--output-filename-format", gcode_file_path, input_file]

        metarow = pd.concat([metarow, pd.Series({"filenumber": count})])
        print(" ".join(str(arg) for arg in cmd))

        try:
            check_output(cmd)
            metarow = pd.concat([metarow, get_series_from_gcode(gcode_file_path)], axis=1)
            os.remove(gcode_file_path)
            _metadata = pd.concat([_metadata, metarow.to_frame().T], ignore_index=True)
            count += 1
        except CalledProcessError as error_message:
            print(f"Unable to slice: {error_message}")
            continue

    return _metadata


if __name__ == '__main__':
    os.makedirs(config.LOG_DIR, exist_ok=True)
    dictConfig(config.LOG_DICT_CONFIG)
    metadata = main()
    metadata.to_csv('metadata.csv', index=False)
