#!/usr/bin/env python
# coding: utf-8

"""
Analysis and optimization of 3D printing parameters.

This script trains two RandomForest models:
  1. Predicting tensile strength.
  2. Predicting filament usage.

It then performs a cost optimization over selected parameters
using differential evolution.
"""

import logging

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from stldeli import config


# pylint: disable=too-many-statements,too-many-locals
def main():
    """Main analysis and optimization routine."""
    logging.info("Starting analysis...")

    # --- Load and clean input data ---
    data = pd.read_csv("data.csv").clean_column_names()

    y_df = data[data.columns.intersection(["tension_strength"])]
    x_df = data[config.important_features]
    logging.info("Input features: %s", list(x_df.columns))

    # --- Train model for tensile strength prediction ---
    strength_regressor_rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    strength_regressor_rf.fit(x_df, y_df.values.reshape(-1))

    feature_importance = (
        pd.DataFrame({
            "feature": x_df.columns,
            "importance": strength_regressor_rf.feature_importances_,
        })
        .sort_values("importance")
        .reset_index(drop=True)
    )

    _, axis = plt.subplots(figsize=(8, 8))
    feature_importance.plot.barh(x="feature", y="importance", color="grey", legend=False, ax=axis)
    axis.set_xlabel("relative importance")
    axis.set_ylabel("")
    axis.set_title("Tensile Strength Feature Importance")

    # --- Strength residual analysis ---
    strength_predicted = pd.Series(strength_regressor_rf.predict(x_df), name="predicted")
    strength_actual = pd.Series(y_df.values.reshape(-1), name="actual")
    strength_residual = strength_actual - strength_predicted

    pd.concat([strength_actual, strength_predicted], axis=1).plot.scatter(
        x="actual", y="predicted", title="Tensile Strength (MPa)"
    )
    pd.concat([strength_predicted, strength_residual], axis=1).plot.scatter(
        x="predicted", y="residual", title="Tensile Strength Residuals"
    )

    rmse_strength = np.sqrt(mean_squared_error(y_true=strength_actual, y_pred=strength_predicted))
    logging.info("Tensile strength RMSE: %.4f", rmse_strength)
    stats.probplot(strength_residual, dist="norm", plot=plt)

    # --- Load and enrich metadata ---
    metadata = pd.read_csv("metadata.csv")
