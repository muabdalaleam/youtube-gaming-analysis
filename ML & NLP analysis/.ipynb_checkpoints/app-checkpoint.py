# ==================Import the packeges=======================
import streamlit as st
import tensorflow as tf
import keras
import pickle
import pandas as pd
import numpy as np
# ============================================================



# ===========Loading the models & preprocessors===============
features_labels =  {"X_train_labels": None,
                   "X_test_labels": None,
                   "y_train_labels": None,
                   "y_test_labels": None}

models = {"Random Forest": None, "XGB Classifier": None,
          "NN regressor": None}

preprocessors = {"PCA": None, "Scaler": None, "Vectorizer": None,
                 "Encoder": None}


for label, val in features_labels.items():
    with open(f"features labels/{label}.pickle", "rb") as f:
        val = pickle.load(f)

for model_name, model in models.items():
    with open(f"models/{model_name}.pickle", "rb") as f:
        model = pickle.load(f)
        
for processor_name, processor in preprocessors.items():
    with open(f"preprocessors/{processor_name}.pickle", "rb") as f:
        processor = pickle.load(f)
# ============================================================



# ==================Getting user interacts====================
channel_name = st.text_input("Input your **YouTube :red[Channel]** name:", "AboFlah")

video_title = st.text_input(
    "Enter the title of the **:red[Video] name** name you want to create: ")
# ============================================================



# ==================Returning basic outputs===================

# ============================================================

option = st.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)
