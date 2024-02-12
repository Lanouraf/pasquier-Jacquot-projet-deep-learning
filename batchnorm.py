

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


def main():
    """The main function of the app.

    Calls the appropriate mode function, depending on the user's choice
    in the sidebar. The mode function that can be called are
    `regression`, `sinus`, `mnist_viz`, and `fashionmnist`.

    Returns
    -------
    None
    """
    st.title("manipulation de la batch normalization  et layer normalization")

    home_data = get_data()

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            "Show instructions",
            "Home data regression",
            "Sinus regression",
            "Show MNIST",
            "Deep Learning",
        ],
    )  # , "Show the source code"])
    if app_mode == "Show instructions":
        st.write("To continue select a mode in the selection box to the left.")
    # elif app_mode == "Show the source code":
    #     st.code(get_file_content_as_string("./app.py"))
    elif app_mode == "Home data regression":
        regression(home_data)
    elif app_mode == "Sinus regression":
        sinus()
    elif app_mode == "Show MNIST":
        mnist()
    elif app_mode == "Deep Learning":
        fashionmnist()


@st.cache
def get_data():
    """Loads the home training data.

    Returns
    -------
    home_data: pd.DataFrame
        The home training data.

    Notes
    -----
    This is the dataset dowloaded from https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality.

    """
    apple_quality_file_path = "/home/pasquier/Bureau/projet deep L /pasquier-Jacquot-projet-deep-learning/data/apple_quality /apple_quality.csv"
    home_data = pd.read_csv(apple_quality_file_path)
    return home_data



if __name__ == "__main__":
    main()