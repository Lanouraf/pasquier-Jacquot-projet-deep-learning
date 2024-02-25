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
    if app_mode == "Homemade Layer Normalization":
        st.write("To continue select a mode in the selection box to the left.")
    #elif app_mode == "Internal Covariate Shift": 
    #elif app_mode == "BN Avantages":     
    #elif app_mode == "BN before or after activation":   
    #elif app_mode == "Homemade Layer Normalization":
    #elif app_mode == "LN in ConvNets":
    #elif app_mode == "BN or LN":


def get_data():
    """Loads the Apple Twitter sentiment texts data from Kaggle.

    Returns
    -------
    df: pd.DataFrame
        The Apple Twitter sentiment texts data.

    """
    url = "https://www.kaggle.com/datasets/seriousran/appletwittersentimenttexts/download?datasetVersionNumber=1"
    
    # Télécharger le fichier CSV depuis l'URL
    response = requests.get(url)
    
    # Vérifier si la requête a réussi
    if response.status_code == 200:
        # Lire le contenu du fichier CSV
        content = response.content.decode("utf-8")
        # Convertir le contenu en DataFrame pandas
        df = pd.read_csv(StringIO(content))
        return df
    else:
        print("Failed to download data.")
        return None

if __name__ == "__main__":
    main()