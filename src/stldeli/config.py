"""
Configuration script imported by other modules.

Contains:
- Matplotlib font configuration
- Logging configuration
- Slic3r parameter dictionary (active subset with documented unused parameters)
- Lists of important and controllable parameters
"""

import os

from matplotlib import pyplot as plt

# ----------------------------------------------------------------------
# Visualization defaults
# ----------------------------------------------------------------------
plt.rcParams.update({'font.size': 22})

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Slic3r configuration space
# ----------------------------------------------------------------------
slic3r_configurations = {
    "--temperature": [200, 250],
    "--layer-height": [0.02, 0.2],
    "--infill-every-layers": [1, 10],
    "--perimeters": [0, 1],
    "--solid-layers": [1, 5, 10],
    "--fill-density": [10, 90],
    "--fill-angle": [30, 60],
    "--fill-pattern": ["rectilinear", "honeycomb"],
    "--solid-infill-speed": [40, 120],
}

"""
Additional parameters available in Slic3r but not currently used here include:
  - Extrusion parameters: --extrusion-multiplier, --filament-diameter
  - Retraction and cooling: --retract-length, --retract-speed, --fan-always-on
  - Support and adhesion: --support-material, --raft-layers, --brim-width
  - Geometric modifiers: --rotate, --scale, --duplicate-grid
  - Pattern options: --external-fill-pattern, --top-infill-extrusion-width
  - Miscellaneous: --notes, --resolution
These can be added later if a broader configuration sweep is needed.
"""

# ----------------------------------------------------------------------
# Parameter metadata
# ----------------------------------------------------------------------
important_features = [
    'layer_height',
    'infill_density',
    'nozzle_temperature',
    'wall_thickness',
]

strength_controllable_parameters = [
    '--layer-height',  # layer_height
    '--fill-density',  # infill_density
    '--temperature',  # nozzle_temperature
    '--solid-layers',  # wall_thickness
]

filament_controllable_parameters = [
    '--infill-every-layers',
    '--fill-density',
    '--layer-height',
]
