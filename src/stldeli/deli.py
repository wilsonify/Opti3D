# !/usr/bin/env python
# coding: utf-8

"""
this script automates the process of slicing the same stl file with many possible combinations of command line arguments
that can be passed to slic3r
"""

import glob
import itertools as it
import os
import logging
from logging.config import dictConfig
from subprocess import check_output, CalledProcessError
import pandas as pd
import stldeli.config

def flag2placeholder(flag):
    flag_str = str(flag)
    flag_str_clean = flag_str.strip("-").replace("-", "_")
    return flag_str_clean + "[" + flag_str_clean + "]"


def main():
    configurations = {
        # "--nozzle-diameter":[0.5,1.0],
        # "--use-firmware-retraction":[True,False],
        # "--use-volumetric-e":[True,False],
        # "--vibration-limit":[0,5,10],
        # "--filament-diameter":[2,3,4],
        # "--extrusion-multiplier":[0.9,1,1.1],
        # "--bed-temperture":[60, 65, 70, 75, 80],
        "--temperature": [200,
                          # 220,
                          250],
        "--layer-height": [0.02,
                           # 0.1,
                           0.2],
        "--infill-every-layers": [1,
                                  # 5,
                                  10],
        "--perimeters": [0,
                         1],
        "--solid-layers": [1,
                           5,
                           10],
        "--fill-density": [10,
                           # 50,
                           90],
        "--fill-angle": [30,
                         # 45,
                         60],
        "--fill-pattern": [
            # "octagram-spiral",
            "rectilinear",
            # "line",
            "honeycomb",
            # "concentric",
            # "hilbert-curve",
            # "archimedean-chords"
        ],
        '--solid-infill-speed': [40,
                                 # 60,
                                 120],
        # "--external-fill-pattern":[],
        # "--end-gcode":[],
        # "--before-layer-gcode":[],
        # "--layer-gcode":[],
        # "--toolchange-gcode":[],
        # "--seam-position":[],
        # "--external-perimeters-first":[],
        # "--spiral-vase":[],
        # "--only-retract-when-crossing-perimeters":[],
        # "--solid-infill-below-area":[],
        # "--infill-only-where-needed":[True,False],
        # "--infill-first":[],
        # "--extra-perimeters":[],
        # "--avoid-crossing-perimeters":[],
        # "--thin-walls":[],
        # "--overhangs":[],
        # "--support-material":[],
        # "--support-material-threshold":[],
        # "--support-material-pattern":[],
        # "--support-material-spacing":[],
        # "--support-material-angle":[],
        # "--support-material-contact-distance":[],
        # "--support-material-interface-layers":[],
        # "--support-material-interface-spacing":[],
        # "--raft-layers":[],
        # "--support-material-enforce-layers":[],
        # "--dont-support-bridges":[],
        # "--retract-length":[],
        # "--retract-speed":[],
        # "--retract-restart-extra":[],
        # "--retract-before-travel":[],
        # "--retract-lift":[],
        # "--retract-layer-change":[],
        # "--wipe":[],
        # "--cooling":[],
        # "--min-fan-speed":[],
        # "--max-fan-speed":[],
        # "--bridge-fan-speed":[],
        # "--fan-below-layer-time":[],
        # "--slowdown-below-layer-time":[],
        # "--min-print-speed":[],
        # "--disable-fan-first-layers":[],
        # "--fan-always-on":[],
        # "--skirts":[],
        # "--skirt-distance":[],
        # "--skirt-height":[],
        # "--min-skirt-length":[],
        # "--brim-width":[],
        # "--scale":[],
        # "--rotate":[],
        # "--duplicate":[],
        # "--duplicate-grid":[],
        # "--duplicate-distance":[],
        # "--xy-size-compensation":[],
        # "--complete-objects":[],
        # "--extruder-clearance-radius":[],
        # "--extruder-clearance-height":[],
        # "--notes":[],
        # "--resolution":[],
        # "--extrusion-width":[],
        # "--first-layer-extrusion-width":[],
        # "--perimeter-extrusion-width Set a different extrusion width for perimeters":[],
        # "--external-perimeter-extrusion-width":[],
        # "--infill-extrusion-width":[],
        # "--solid-infill-extrusion-width":[],
        # "--top-infill-extrusion-width":[],
        # "--support-material-extrusion-width":[],
        # "--infill-overlap":[],
        # "--bridge-flow-ratio Multiplier for extrusion when bridging (> 0, default: 1)":[],
        # "--extruder-offset":[],
        # "--perimeter-extruder":[],
        # "--infill-extruder":[],
        # "--solid-infill-extruder":[],
        # "--support-material-extruder":[],
        # "--support-material-interface-extruder Extruder to use for support material interface (1+, default: 1)":[],
        # "--ooze-prevention":[],
        # "--standby-temperature-delta":[],
        # "--ooze-prevention is enabled (default: -5)":[],
        # "--retract-length-toolchange":[],
        # "--retract-restart-extra-toolchange":[]
    }

    combinations = it.product(*(configurations[Name] for Name in configurations))
    total = len(list(combinations))
    logging.info("{} possible slices".format(total))

    count = 0
    metadata = pd.DataFrame()
    input_file = os.path.abspath("stl_files/largecube.stl")
    for configuration in list(it.product(*configurations.values())):
        metarow = pd.Series(configuration, index=configurations.keys())
        output_file_format = "[input_filename_base]"
        print("{} out of {}".format(count + 1, total))
        cmd = ["slic3r"]

        for key, value in zip(configurations.keys(), configuration):
            # print("adding {} with value of {} to cmd".format(key,value))
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
            metadata = metadata.append(metarow, ignore_index=True)
            count += 1
        except CalledProcessError as e:
            print("unable to slice with error: {}".format(e))
            continue

    # In[8]:

    [_ for _ in combinations]

    # In[10]:

    metadata.to_csv('metadata.csv')

    # In[ ]:

    for file_path in glob.glob('gcode_files/*.gcode'):
        # print(file_path)
        with open(file_path) as gcode_file:
            for line in gcode_file.readlines():
                if line.startswith(';'):
                    datum = line.strip('; \n').split('=')
                    if len(datum) == 2:
                        file_data[datum[0]] = [datum[1]]
        datum_row = pd.DataFrame.from_dict(data=file_data, orient='columns')


if __name__ == '__main__':
    os.mkdir(config.log_dir,exists_ok=True)
    dictConfig(config.log_dict_config)
    main()
