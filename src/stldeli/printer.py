#!/usr/bin/env python
# coding: utf-8

"""
analysis and optimization
"""
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor  # pylint: disable=ungrouped-imports
from sklearn.linear_model import LinearRegression  # pylint: disable=ungrouped-imports


# pylint: disable=too-many-statements,too-many-locals
def main():
    """
    main
    :return:
    """
    plt.rcParams.update({'font.size': 22})

    data = pd.read_csv("data.csv").clean_column_names()

    for column in data.columns:
        logging.info(column)
        logging.info(np.sort(data[column].unique()))

    important_features = ['layer_height',
                          'infill_density',
                          'nozzle_temperature',
                          'wall_thickness'
                          ]

    y_df = data[data.columns.intersection(['tension_strength'])]
    # x = data[data.columns.difference(['tension_strength','elongation','roughness'])]
    x_df = data[important_features]
    # x = pd.get_dummies(x_nonnumeric).join(x_numeric)

    logging.info(x_df.columns)

    strength_regressor_rf = RandomForestRegressor()
    strength_regressor_rf.fit(x_df, y_df.values.reshape(-1))

    strength_regressor_linear = LinearRegression()
    strength_regressor_linear.fit(x_df, y_df.values.reshape(-1))

    feature_importance = pd.concat([pd.Series(strength_regressor_rf.feature_importances_, name='importance'),
                                    pd.Series(x_df.columns, name='feature')
                                    ], axis=1
                                   ).sort_values('importance')

    _, axis = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    feature_importance.plot.barh(x='feature',
                                 y='importance',
                                 color='grey',
                                 legend=False,
                                 ax=axis
                                 )
    axis.set_xlabel('relative importance')
    axis.set_ylabel('')
    axis.set_title('Tensile Strength')

    strength_predicted = pd.Series(strength_regressor_rf.predict(x_df), name='predicted')
    strength_actual = pd.Series(y_df.values.reshape(-1), name='actual')
    strength_residual = strength_actual - strength_predicted
    pd.concat([strength_predicted,
               strength_actual]
              , axis=1
              ).plot.scatter(x='actual', y='predicted', title='Tensile Strength (MPa)')

    strength_residual.name = 'residual'

    pd.concat([strength_predicted,
               strength_residual]
              , axis=1
              ).plot.scatter(x='predicted', y='residual', title='Tensile Strength (MPa)')

    np.sqrt(sklearn.metrics.mean_squared_error(y_true=strength_actual,
                                               y_pred=strength_predicted
                                               )
            )

    stats.probplot(strength_residual, dist="norm", plot=plt)

    pd.concat([pd.Series(strength_regressor_linear.predict(x_df), name='predicted'),
               pd.Series(y_df.values.reshape(-1), name='actual')
               ], axis=1
              ).plot.scatter(x='actual', y='predicted')

    metadata = pd.read_csv('metadata.csv')

    metadata.info()

    strength_controllable_parameters = [
        '--layer-height',  # layer_height
        '--fill-density',  # infill_density
        '--temperature',  # nozzle_temperature
        '--solid-layers'  # wall_thickness
    ]

    newx = metadata[strength_controllable_parameters]

    tensile = strength_regressor_rf.predict(newx)
    tensile_series = pd.Series(tensile, name='tensile_strength_predicted')

    tensile_series.plot.hist()

    metadata_enriched = metadata.join(tensile_series)

    filament = metadata['filament used '].str.strip(' ').str.split(' ', expand=True).rename(
        columns={0: 'filament_used_mm',
                 1: 'filament_used_cm3'
                 }
    )

    metadata_enriched['filament_used_mm'] = filament['filament_used_mm'] \
        .str.replace('mm', '') \
        .apply(float)

    metadata_enriched['filament_used_cm3'] = filament['filament_used_cm3'] \
        .str.strip('()') \
        .str.replace('cm3', '') \
        .apply(float)

    metadata_enriched['infill extrusion width (mm)'] = metadata['infill extrusion width '] \
        .str.replace('mm', '') \
        .apply(float)

    figure = sns.lmplot(
        x='filament_used_cm3',
        y='tensile_strength_predicted',
        hue='--infill-every-layers',
        # col='--fill-density',
        # row='--layer-height',
        data=metadata_enriched
    )
    figure.axes[0, 0].set_xlabel('filament used ($cm^3$)')
    figure.axes[0, 0].set_ylabel('tensile strength (MPa)')

    filament_controllable_parameters = ['--infill-every-layers',
                                        '--fill-density',
                                        '--layer-height',
                                        ]

    y_df = metadata_enriched[metadata_enriched.columns.intersection(['filament_used_cm3'])]
    x_df = metadata_enriched[filament_controllable_parameters]

    y_df.plot.hist(legend=False)

    filament_regressor_rf = RandomForestRegressor()

    filament_regressor_rf.fit(x_df, y_df.values.reshape(-1))
    # filament_regressor_linear.fit(x_numeric.dropna(1),y)

    feature_importance = pd.concat([pd.Series(filament_regressor_rf.feature_importances_, name='importance'),
                                    pd.Series(x_df.columns, name='feature')
                                    ], axis=1
                                   ).sort_values('importance')

    feature_importance.dropna()

    _, axis = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    feature_importance.dropna()[-10:].plot.barh(x='feature',
                                                y='importance',
                                                color='grey',
                                                legend=False,
                                                ax=axis
                                                )

    axis.set_xlabel('relative importance')
    axis.set_ylabel('')
    axis.set_title('Filament Usage')

    filament_predicted = pd.Series(filament_regressor_rf.predict(x_df), name='predicted')
    filament_actual = pd.Series(y_df.values.reshape(-1), name='actual')
    filament_residual = filament_actual - filament_predicted
    filament_residual.name = 'residual'
    pd.concat([filament_predicted, filament_actual], axis=1).plot.scatter(x='actual', y='predicted',
                                                                          title='Filament Used ($cm^3$)')

    np.sqrt(sklearn.metrics.mean_squared_error(y_pred=filament_predicted, y_true=filament_actual))

    stats.probplot(filament_residual.sample(n=100), dist="norm", plot=plt)

    average_filament = metadata_enriched['filament_used_cm3'].mean()
    average_strength = metadata_enriched['tensile_strength_predicted'].mean()

    def cost_function(input_array, progress_df):
        """
        layer_height,
        fill_density,
        infill_every_layers,
        wall_thickness,
        nozzle_temperature
        """
        (layer_height,
         fill_density,
         infill_every_layers,
         wall_thickness,
         nozzle_temperature) = input_array

        x_strength = pd.Series({'layer_height': layer_height,
                                'fill_density': fill_density,
                                'wall_thickness': wall_thickness,
                                'nozzle_temperature': nozzle_temperature}
                               )

        x_filament = pd.Series({'layer_height': layer_height,
                                'fill_density': fill_density,
                                'infill_every_layers': infill_every_layers,
                                })

        _strength = strength_regressor_rf.predict(x_strength.values.reshape(1, -1))
        _filament = filament_regressor_rf.predict(x_filament.values.reshape(1, -1))
        cost = 0.5 * _filament / average_filament - 0.5 * _strength / average_strength

        row = pd.Series(np.append(input_array, cost),  # pylint: disable=used-before-assignment
                        index=('layer_height',
                               'fill_density',
                               'infill_every_layers',
                               'wall_thickness',
                               'nozzle_temperature',
                               'iteration',
                               'cost'
                               ))
        progress_df.append(row)

        return cost

    cost_function(np.array([0.02, 10.00, 1, 1, 200]), pd.DataFrame())

    bounds = [(0.02, 0.8),  # layer_height
              (10.0, 90.0),  # fill_density
              (1.0, 10.0),  # infill_every_layers
              (1.0, 10.0),  # wall_thickness
              (200, 300)  # nozzle_temperature
              ]

    iterations_df = pd.DataFrame()
    result = differential_evolution(cost_function,
                                    bounds,
                                    args=(iterations_df)
                                    # strategy='rand2exp',
                                    # maxiter=1000,
                                    # popsize=25,
                                    # tol=0.001,
                                    # mutation=(0.5, 1),
                                    # recombination=0.7,
                                    # seed=None,
                                    # disp=False,
                                    # polish=True,
                                    # init='latinhypercube',
                                    # atol=0
                                    )

    logging.info(result.x, result.fun)

    cost_function(np.array([0.57153236,
                            46.12917446,
                            4.29097592,
                            4.92265957,
                            276.59808607]
                           ), iterations_df
                  )

    metadata_enriched.to_csv('metadata_enriched.csv')


if __name__ == '__main__':
    main()
