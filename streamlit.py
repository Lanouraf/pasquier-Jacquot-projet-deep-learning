#streamlit

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io
import requests
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
    if app_mode == "Home":
        st.write("To continue select a mode in the selection box to the left.")
    elif app_mode == "Homemade Layer Normalization":

    #elif app_mode == "Internal Covariate Shift": 
    #elif app_mode == "BN Avantages":     
    #elif app_mode == "BN before or after activation":   
    #elif app_mode == "Homemade Layer Normalization":
    #elif app_mode == "LN in ConvNets":
    #elif app_mode == "BN or LN":

@st.cache
def get_data():
    """Loads the Apple Twitter sentiment texts data from Kaggle.

    Returns
    -------
    df: pd.DataFrame
        The Apple Twitter sentiment texts data.

    Notes
    -----
    This is the dataset dowloaded from
        "https://www.kaggle.com/datasets/seriousran/appletwittersentimenttexts/download?datasetVersionNumber=1"

    """
    directory = "/Users/maudjacquot/Desktop/pasquier-Jacquot-projet-deep-learning/apple-twitter-sentiment-texts.csv"
    home_data = pd.read_csv(directory)
    return home_data

def homemade_layernorm(home_data):
    st.text(
        "This is the head of the dataframe"
    )
    st.write(home_data.head())



if __name__ == "__main__":
    main()