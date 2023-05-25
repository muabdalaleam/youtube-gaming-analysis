# ==================Import the packeges=======================
import streamlit as st
import re
import tensorflow as tf
import keras
import nltk
import pickle
import pandas as pd
import numpy as np
import pandasql as ps
import google_auth_oauthlib.flow
import googleapiclient.discovery
from nltk.corpus import wordnet
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googleapiclient.discovery import build
import googleapiclient.errors

TEXT_COLUMNS = ["title", "description", "channelTitle", "about"]
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64',
            'uint16', 'uint32', 'uint64', float, int]

# ============================================================



# ===================Settting up YouTube build================
API_KEY: str = "AIzaSyANcOOmvv5fs6Gx7vKXucSelmScjx3V3Qg"
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"


youtube = build(
    API_SERVICE_NAME, API_VERSION, developerKey= API_KEY)
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

st.header("Enter the following inputs:")

channel_name = st.text_input("Input your **YouTube :red[Channel]** name: ", "Ali Abdaal")

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



# ==============Gettign the channel & videos data=============

# Getting the channel's id
channel_id_response = youtube.search().list(
    q= channel_name,
    type='channel',
    part='id',
    maxResults= 1
).execute()

channel_id = channel_id_response['items'][0]['id']['channelId']


# Getting channel's stats
channel_stats_response = youtube.channels().list(
        part= "snippet,contentDetails,statistics",
        id= channel_id).execute()

item = channel_stats_response["items"][0]

channel_stats = pd.DataFrame({"channel_name": item["snippet"]["title"],
                              "subscribers": item["statistics"]["subscriberCount"],
                              "playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
                              "total_views": item["statistics"]["viewCount"],
                              "start_date": item["snippet"]["publishedAt"],
                              "video_count": item["statistics"]["videoCount"],
                              "about": item["snippet"]["description"]}
                              , index=[0])


# Loading 50 videos from the channel
def get_videos_ids(playlist_id, max_results = [50]):

    videos_ids = []
    
    videos_ids_request = youtube.playlistItems().list(
        part= "snippet,contentDetails",
        playlistId= playlist_id,
        maxResults= max_results[0])

    if max_results[0] <= int(channel_stats["video_count"][0]):
        
        videos_ids_response = videos_ids_request.execute()
        # print(JSON(videos_ids_response))
        
        for response in videos_ids_response["items"]:
            videos_ids.append(response["contentDetails"]["videoId"])

    else:
        max_results[0] -= int(channel_stats["video_count"][0])
        videos_ids = get_videos_ids(playlist_id, max_results[0])
        
    return  videos_ids

videos_ids = get_videos_ids(channel_stats["playlist_id"][0])


# Loading the videos stats
def get_video_stats(videos_ids: list) -> pd.DataFrame:

    """This function takes the videos IDs list and request
       for the statistics of the videos then save them into
       a DataFrame."""

    all_video_info = []
    videos_count = len(videos_ids)

    for i in range(0, videos_count, 50):
        
        chunk = videos_ids[i:i+50]
        
        # Giving the request for each 50 video in one time
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(chunk))
        response = request.execute()


        for video in response['items']:
            video_json_encoder = {"snippet": ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
                                  "statistics": ['viewCount', 'likeCount', 'commentCount'],
                                  "contentDetails": ["duration", "definition"]}

            video_info = {}
            video_info['video_id'] = video['id']

            for key in video_json_encoder.keys():
                for val in video_json_encoder[key]:
                    try:
                        video_info[val] = video[key][val]
                    except:
                        video_info[val] = np.nan

            all_video_info.append(video_info)

    df = pd.DataFrame(all_video_info)
    
    return df

videos_stats = get_video_stats(videos_ids)

channel_stats.drop(["playlist_id"], axis= 1, inplace= True)
videos_stats = videos_stats.rename({"channelTitle": "channel_name"})

# Concating the videos and channels data
df = pd.merge(videos_stats, channel_stats, on= "channel_name")
# ============================================================



# ================Text Feature engineering====================

# preparing NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('stopwords-hi')
nltk.download('stopwords-ar')
nltk.download('averaged_perceptron_tagger')


en_stopwords = set(stopwords.words('english')) 
ar_stopwords = set(stopwords.words('arabic')) 
# hi_stopwords = set(stopwords.words('hindi')) 

all_stopwords = en_stopwords.union(ar_stopwords)

# POS tagging

def stopwords_dropper(words: list, stopwords: set) -> list:
    
    # Removing stop words from unalphabetical chars
    filtered_words = [re.sub(r"[\W_]", "", word) for word in words
                      if not word in stopwords]
    
    filtered_words = list(filter(lambda item: item != "", filtered_words))
    return  filtered_words

for col in TEXT_COLUMNS:
    df[f"{col}_pos_tags"] = df[f"{col}_tokens"].apply(lambda words: nltk.pos_tag(words))


# Limmization text

def get_wordnet_pos(treebank_tag: str) -> str:
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    
    else:
        return wordnet.NOUN


lemmatizer = WordNetLemmatizer()
lemmatized_words = []
lemmatized_words_group = []

for col in TEXT_COLUMNS:
    for index, row in df.iterrows():
        for token, pos_tag in zip(row[f"{col}_tokens"], row[f"{col}_pos_tags"]):

            wordnet_pos = get_wordnet_pos(pos_tag[1])
            lemmatized_words_group.append(lemmatizer.lemmatize(token, pos= wordnet_pos))
            lemmatized_words_group = list(set(lemmatized_words_group)) # Dropping duplicates


        lemmatized_words.append(lemmatized_words_group)
        lemmatized_words_group = [] # clearing this list
    
    df[f"{col}_tokens"] = lemmatized_words
    lemmatized_words = []
# ============================================================



# ================Numrical feature engineering================
today = datetime.utcnow().strftime("%Y-%m-%d")
today = datetime.strptime(today, "%Y-%m-%d")

channel_age = today - pd.to_datetime(df["start_date"])
df["channel_age_days"] = channel_age.dt.days.astype(int)

video_age = today - pd.to_datetime(df["publishedAt"]).dt.tz_localize(None)
df["video_age_days"] = video_age.dt.days.astype(np.uint16)


df["language"] = df["language"].astype("category").cat.codes
df["definition"] = df["definition"].astype("category").cat.codes
df["country"] = df["country"].astype("category").cat.codes

cat_cols = ["country", "language", "definition"] # sentimints


df["cat_view_count"] = df["cat_view_count"].replace({"from 1 to 3,000": 1, "from 3,000 to 10,000": 2,
                                                     "from 10,000 to 50,000": 3, "from 50,000 to 100,000": 4,
                                                     "from 100,000 to 300,000": 5, "more than 300,000": 6})

df["cat_like_count"] = df["cat_like_count"].replace({"from 1 to 1,000": 1, "from 1,000 to 5,000": 2,
                                                     "from 5,000 to 10,000": 3, "from 10,000 to 50,000": 4,
                                                     "from 50,000 to 150,000": 5, "more than 150,000": 6})

df["cat_comment_count"] = df["cat_comment_count"].replace({"from 1 to 75": 1, "from 75 to 150": 2,
                                                           "from 150 to 200": 3, "from 200 to 400": 4,
                                                           "from 400 to 600": 5, "more than 600": 6})

accounts: list = ["twitter", "facebook", "instagram", "twitch"]

for account in accounts:
    df[f"have_{account}_account"] = df["about"].str.contains(account)


df["start_date"] = pd.to_datetime(df["start_date"])

df["avg_uploads_per_month"] = df["video_count"] / (df["channel_age_days"] // 30)
df["avg_uploads_per_month"] = df["avg_uploads_per_month"].astype(np.float32)
# ============================================================



# ====================Updating the output=====================

st.text(channel_id)
# ============================================================
option = st.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone'))

st.write('You selected:', option)

print(channel_id)
