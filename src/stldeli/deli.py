# !/usr/bin/env python
# coding: utf-8

"""
this script automates the process of slicing the same stl file with many possible combinations of command line arguments
that can be passed to slic3r
"""
import glob
import itertools as it
import logging
import os
from logging.config import dictConfig  # pylint: disable=ungrouped-imports
from subprocess import CalledProcessError, check_output

import pandas as pd
from stldeli import config


def flag2placeholder(flag):
    """
    convert a flag into valid commandline argument
    :param flag:
    :return:
    """
    logging.debug("flag2placeholder")
    flag_str = str(flag)
    flag_str_clean = flag_str.strip("-").replace("-", "_")
    return flag_str_clean + "[" + flag_str_clean + "]"


def get_combinations_from_configurations(configurations):
    """
    convert configured dict into a generator of all possible tuples
    :param configurations:
    :return:
    """
    logging.info("get_combinations_from_configurations")
    return it.product(*(configurations[Name] for Name in configurations))


# pylint: disable=too-many-locals
def main():
    """
    main function
    :return: dataframe of metadata
    """
    logging.info("main")
    combinations = get_combinations_from_configurations(config.slic3r_configurations)
    total = len(list(combinations))
    logging.info("{} possible slices".format(total))

    count = 0
    _metadata = pd.DataFrame()
    input_file = os.path.abspath("stl_files/largecube.stl")
    for configuration in list(it.product(*config.slic3r_configurations.values())):
        logging.debug("configuration  = {}".format(configuration))
        metarow = pd.Series(configuration, index=config.slic3r_configurations.keys())
        output_file_format = "[input_filename_base]"
        print("{} out of {}".format(count + 1, total))
        cmd = ["slic3r"]

        for key, value in zip(config.slic3r_configurations.keys(), configuration):
            logging.debug("adding {} with value of {} to cmd".format(key, value))
            metarow[key] = value
            if value:
                cmd.append(str(key))
                if not isinstance(value, bool):
                    cmd.append(str(value))
            output_file_format += "_" + flag2placeholder(key)

        cmd.append("--output-filename-format")
        cmd.append("{count}_{output_file_format}_.gcode".format(count=count,
                                                                output_file_format=output_file_format
                                                                )
                   )
        cmd.append(input_file)
        metarow = metarow.append(pd.Series(count, index=["filenumber"]))
        cmd_str = ''
        for arg in cmd:
            cmd_str += ' ' + str(arg)
        print(cmd_str)
        try:
            check_output(cmd)
            for gcode_file_path in glob.glob('stl_files//*.gcode'):
                with open(gcode_file_path) as gcode_file:
                    for line in gcode_file.readlines():
                        if line.startswith(';'):
                            datum = line.strip('; \n').split('=')
                            if len(datum) == 2:
                                metarow[datum[0]] = datum[1]
                os.remove(gcode_file_path)
            _metadata = _metadata.append(metarow, ignore_index=True)
            count += 1
        except CalledProcessError as error_message:
            print("unable to slice with error: {}".format(error_message))
            continue

    return _metadata


if __name__ == '__main__':
    os.makedirs(config.LOG_DIR, exist_ok=True)
    dictConfig(config.LOG_DICT_CONFIG)
    metadata = main()  # pylint: disable=invalid-name
    metadata.to_csv('metadata.csv')
