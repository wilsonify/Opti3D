"""
configuration script imported by other modules
"""

import os
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 22})

LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'deli.log')

LOG_DICT_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s | %(levelname)s | %(filename)s | %(name)s | %(lineno)d | %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': 'DEBUG',
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'default',
            'filename': LOG_FILE,
            'level': 'DEBUG',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG',
    },
}

# pylint: disable=invalid-name
slic3r_configurations = {
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

important_features = [
    'layer_height',
    'infill_density',
    'nozzle_temperature',
    'wall_thickness'
]
strength_controllable_parameters = [
    '--layer-height',  # layer_height
    '--fill-density',  # infill_density
    '--temperature',  # nozzle_temperature
    '--solid-layers'  # wall_thickness
]

filament_controllable_parameters = [
    '--infill-every-layers',
    '--fill-density',
    '--layer-height',
]
