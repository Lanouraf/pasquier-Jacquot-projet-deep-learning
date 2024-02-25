#streamlit

import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from viz import mnist_like_viz, training_curves
import dl

def main():
    """The main function of the app.

    Calls the appropriate mode function, depending on the user's choice
    in the sidebar. The mode function that can be called are
    `regression`, `sinus`, `mnist_viz`, and `fashionmnist`.

    Returns
    -------
    None
    """
    st.title("Batch and Layer Normalization")

    home_data = get_data()

    app_mode = st.sidebar.selectbox(
        "Choose the experiment",
        [
            "Homemade Batch Normalization",
            "Internal Covariate Shift",
            "BN Avantages",
            "BN before or after activation",
            "Homemade Layer Normalization",
            "LN in ConvNets",
            "BN or LN"
        ],
    )  # , "Show the source code"])
    if app_mode == "Homemade Layer Normalization":
        st.write("To continue select a mode in the selection box to the left.")
    #elif app_mode == "Internal Covariate Shift": 
    #elif app_mode == "BN Avantages":     
    #elif app_mode == "BN before or after activation":   
    #elif app_mode == "Homemade Layer Normalization":
    #elif app_mode == "LN in ConvNets":
    #elif app_mode == "BN or LN":


def get_data():
    """Loads the home training data from Kaggle.

    Returns
    -------
    home_data: pd.DataFrame
        The home training data.

    """
    url = "https://www.kaggle.com/datasets/seriousran/appletwittersentimenttexts/download?datasetVersionNumber=1"
    home_data = pd.read_csv(url)
    return home_data

if __name__ == "__main__":
    main()