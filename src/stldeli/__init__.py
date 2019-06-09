"""
initialize the stldeli
"""
import logging

import pandas as pd

logging.getLogger(__name__).addHandler(logging.NullHandler())

global evolution_df

evolution_df = pd.DataFrame(columns=['layer_height',
                                     'fill_density',
                                     'infill_every_layers',
                                     'wall_thickness',
                                     'nozzle_temperature',
                                     'iteration',
                                     'cost'
                                     ])

global iteration
iteration = 1
