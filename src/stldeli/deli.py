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
from logging.config import dictConfig
from subprocess import check_output, CalledProcessError

import pandas as pd

import config


def flag2placeholder(flag):
    logging.info("flag2placeholder")
    flag_str = str(flag)
    flag_str_clean = flag_str.strip("-").replace("-", "_")
    return flag_str_clean + "[" + flag_str_clean + "]"


def get_combinations_from_configurations(configurations):
    logging.info("get_combinations_from_configurations")
    return it.product(*(configurations[Name] for Name in configurations))


def main():
    logging.info("main")
    combinations = get_combinations_from_configurations(config.configurations)
    total = len(list(combinations))
    logging.info("{} possible slices".format(total))

    count = 0
    _metadata = pd.DataFrame()
    input_file = os.path.abspath("stl_files/largecube.stl")
    for configuration in list(it.product(*config.configurations.values())):
        logging.debug("configuration  = ".format(configuration))
        metarow = pd.Series(configuration, index=config.configurations.keys())
        output_file_format = "[input_filename_base]"
        print("{} out of {}".format(count + 1, total))
        cmd = ["slic3r"]

        for key, value in zip(config.configurations.keys(), configuration):
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
        except CalledProcessError as e:
            print("unable to slice with error: {}".format(e))
            continue

    return _metadata


if __name__ == '__main__':
    os.makedirs(config.log_dir, exist_ok=True)
    dictConfig(config.log_dict_config)
    metadata = main()
    metadata.to_csv('metadata.csv')
