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
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
local_css("style.css")

channel_name = st.text_input("Input your **YouTube :red[Channel]** name: ", "AboFlah")

video_title = st.text_input(
    "Enter the title of the **:red[Video] name** name you want to create: ")

duration_in_minutes = st.text_input(
    "Enter Your **Video Duration in :red[Minutes]:**", 0)

try:
    duration_in_seconds = int(st.text_input(
        "(Optional) Enter Your **Video Duration in :red[Seconds]:**",
        int(duration_in_minutes) * 60))
    
except:
    st.error("Enter duration as integer.")

thumbnail = st.file_uploader("Upload or drag & drop your **Video :red[Thumbnail]** image: ")


with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)"))
    

# ============================================================



# ==================Returning basic outputs===================

# ============================================================

option = st.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)
