#streamlit

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, tqdm_notebook
import os
import sys

# Importez d'autres bibliothèques nécessaires

# Ajoutez ce bloc au début de votre code
class NullWriter(object):
    def write(self, arg):
        pass

# Redirigez la sortie standard vers os.devnull
sys.stdout = NullWriter()

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
            "Home",
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
        homemade_layernorm(home_data)

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
    directory = "apple-twitter-sentiment-texts.csv"
    home_data = pd.read_csv(directory)
    return home_data

def homemade_layernorm(home_data):
    st.text(
        "This is the head of the dataframe where text contains Apple Twitter texts and sentiment contains -1, 0 or 1 correspoding to negative, neutral or positive"
    )
    st.write(home_data.head())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class Sequences(Dataset):
        def __init__(self, home_data, vectorizer):
            ### create tokens for your dataset
            self.vectorizer = vectorizer
            # Create a list containing all the texts from df
            list_texts = home_data.text.tolist()
            # Generate your sequences using your vectorizer
            self.sequences = self.vectorizer.transform(list_texts)
            ###
            # We convert the labels to a list of labels (before it was within a dataframe)
            self.sentiments = home_data.sentiment.tolist()

        def __getitem__(self, i):
            ### TODO: self.sequences is a sparse matrix, where the rows contain the texts and the columns the unique words within the dataset
            # Select the sequence at the index i
            sequence_i = self.sequences[i]
            # Select the label at the index i
            sentiment_i = self.sentiments[i]
            ###
            # We return here the sequence and the label at the index i. We convert the sparse matrix to a numpy array.
            return sequence_i.toarray(), sentiment_i

        def __len__(self):
            return self.sequences.shape[0]
        
    # Create your vectorizer
    vectorizer = CountVectorizer(stop_words="english", max_df=0.99, min_df=0.005)
    # Create a list containing all the texts from df
    list_texts = home_data.text.tolist()
    # Fit your vectorizer on this list
    vectorizer.fit(list_texts)
    ###

    # We divide the dataset into a train and a test set
    train_home_data= home_data.iloc[:int(0.7*home_data.shape[0])]
    test_home_data = home_data.iloc[int(0.7*home_data.shape[0]):]

    ### TODO: create a train and a test Sequences datasets
    train_dataset = Sequences(train_home_data, vectorizer)
    test_dataset = Sequences(test_home_data, vectorizer)
    # Create the associated DataLoaders, with a batch size of 4096
    train_loader = DataLoader(dataset=train_dataset,batch_size=4096)
    test_loader = DataLoader(dataset=test_dataset,batch_size=4096)

    
    class BagOfWordsClassifier(nn.Module):
        ### TODO: implement your model with three hidden layers
        def __init__(self, vocab_size, hidden1, hidden2, out_shape):
            super(BagOfWordsClassifier, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(vocab_size,hidden1),
                nn.ReLU()
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden1,hidden2),
                nn.ReLU()
            )
            self.layer3 = nn.Sequential(
                nn.Linear(hidden2,out_shape),
            )
        def forward(self, x):
            ### TODO

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = x.squeeze()
            x = nn.Sigmoid()(x)

            return x
    ###
    
    class LayerNorm(nn.Module):
        def __init__(self, features, eps=1e-6):
            super(LayerNorm, self).__init__()
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
    class BagOfWordsClassifierLayerNorm(nn.Module):
        def __init__(self, vocab_size, hidden1, hidden2, out_shape):
            super(BagOfWordsClassifier, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Linear(vocab_size,hidden1),
                LayerNorm(hidden1),
                nn.ReLU()
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden1,hidden2),
                LayerNorm(hidden2),
                nn.ReLU()
            )
            self.layer3 = nn.Sequential(
                nn.Linear(hidden2,out_shape),
            )

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = x.squeeze()
            x = nn.Sigmoid()(x)
            return x
    



    vocab_size=len(train_dataset.vectorizer.vocabulary_)
    hidden1=128
    hidden2=64
    output_shape=1
    model=BagOfWordsClassifier(vocab_size, hidden1,hidden2,output_shape)
    model2=BagOfWordsClassifierLayerNorm(vocab_size, hidden1,hidden2,output_shape)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    optimizer2 = optim.Adam(model2.parameters(), lr = 0.001)
    ###

    model.train()
    train_losses = []
    for epoch in range(10):
        progress_bar = tqdm.notebook.tqdm(train_loader, leave=False)
        losses = []
        total = 0
        for inputs, target in progress_bar:
            # Inputs are of shape (bs, 1, voc size), we remove the 1 with squeeze() and we convert them to floats
            inputs = inputs.squeeze().float()
            targets = target.float()
            ### TODO: implement the training loop as usual.
            ###
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, targets)
            # Backpropagation
            loss.backward()
            optimizer.step()
            #progress_bar.set_description(f'Loss: {loss.item():.3f}')

            losses.append(loss.item())
            total += 1

        epoch_loss = sum(losses) / total
        train_losses.append(epoch_loss)

    model2.train()
    train2_losses = []
    for epoch in range(10):
        progress_bar = tqdm.notebook.tqdm(train_loader, leave=False)
        losses = []
        total = 0
        for inputs, target in progress_bar:
            # Inputs are of shape (bs, 1, voc size), we remove the 1 with squeeze() and we convert them to floats
            inputs = inputs.squeeze().float()
            targets = target.float()
            ### TODO: implement the training loop as usual.
            ###
            optimizer2.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, targets)
            # Backpropagation
            loss.backward()
            optimizer2.step()
            #progress_bar.set_description(f'Loss: {loss.item():.3f}')

            losses.append(loss.item())
            total += 1

        epoch_loss = sum(losses) / total
        train2_losses.append(epoch_loss)

    #st.write(train_losses)
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(test_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features.squeeze().float())
            ### TODO: compute the predicted sentiments
            predicted_sentiments = torch.round(logits)
            ###

            num_examples += targets.size(0)
            correct_pred += (predicted_sentiments == targets).sum()
    st.text("Accuracy without Layer Norm: ")
    st.write(np.round(float((correct_pred.float()/num_examples)),4) * 100)

    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(test_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model2(features.squeeze().float())
            ### TODO: compute the predicted sentiments
            predicted_sentiments = torch.round(logits)
            ###

            num_examples += targets.size(0)
            correct_pred += (predicted_sentiments == targets).sum()
    st.text("Accuracy with Layer Norm: ")
    st.write(np.round(float((correct_pred.float()/num_examples)),4) * 100)

    



if __name__ == "__main__":
    main()