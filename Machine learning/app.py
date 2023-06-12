# ==================Import the packeges=======================
import streamlit as st
import os
import io
import re
import dill
import tensorflow as tf
import inspect
import shutil
import pytz
import keras
import nltk
import plotly.io as pio
import pickle
import pandas as pd
import plotly.express as px
import numpy as np
import pandasql as ps
import google_auth_oauthlib.flow
import googleapiclient.discovery
from nltk.corpus import wordnet
from datetime import datetime
from nltk.corpus import stopwords
import datetime as dt
from keras.utils import to_categorical
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from googleapiclient.discovery import build
from streamlit_option_menu import option_menu
import googleapiclient.errors

TEXT_COLUMNS = ["title", "description", "channel_name", "about"]
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64',
            'uint16', 'uint32', 'uint64', float, int]

os.chdir(r"..\Machine learning") # ML & NLP analysis
# ============================================================



# ===================Settting up YouTube build================
API_KEY: str = "AIzaSyAyULusNp9X9wa9fHp_PHYIeeRpdh87YC0"
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"


youtube = build(
    API_SERVICE_NAME, API_VERSION, developerKey= API_KEY)

st.set_page_config(layout="wide", page_title='Youtube gaming analysis ML app.',
                        page_icon = '../imgs/logo.png')
# ============================================================



# ========Loading the models, preprocessor & graphs===========
encoders = {"Targets encoder": None, "Targets decoder": None,
            "y pred decoder": None}

models = {"Random Forest": None, "XGB Classifier": None,
          "Nueral network": None}

preprocessors = {"PCA": None, "Scaler": None, "Vectorizer": None,
                 "Encoder": None}


for encoder_name, encoder in encoders.items():
    with open(f"encoders/{encoder_name}.pickle", "rb") as f:
        encoders[encoder_name] = dill.load(f)

for model_name, model in models.items():

    if model_name == "Nueral network":
        models[model_name] = tf.keras.models.load_model("models/NN regressor")

    else:
        with open(f"models/{model_name}.pickle", "rb") as f:
            models[model_name] = dill.load(f)

for processor_name, processor in preprocessors.items():
    with open(f"preprocessors/{processor_name}.pickle", "rb") as f:
        preprocessors[processor_name] = dill.load(f)

        
subs_vs_channel_name_len = pio.read_json("../Data analysis/plots/json/subs_vs_channel_name_len_line_chart.json")
social_accounts_affect_on_vid_stats = pio.read_json("../Data analysis/plots/json/social_accounts_affect_on_vid_stats.json")
video_stats_per_game = pio.read_json('../Data analysis/plots/json/video_stats_per_game.json')
total_subs_vs_start_date = pio.read_json("../Data analysis/plots/json/total_subs_for_channels_per_start_date.json")
duration_vs_views = pio.read_json("../Data analysis/plots/json/desc_len_vs_views_scatter_plot.json")
# ============================================================



# ==================Getting user inputs=======================
def local_css(file_name):
    
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
local_css("style.css")

st.markdown("<h1 style='text-align: center;'>" + \
            "Youtube <span style= \"color: red\">Gaming</span> analysis ML app</h1>", unsafe_allow_html=True)

subpage = option_menu(
    menu_title= "Main menu",
    options= ["Taking required inputs", "Your Video Predictions",
               "Statistics of your channel"],
    icons= ["youtube", "magic", "graph-up-arrow"],
    menu_icon= "house",
    default_index= 0,
    styles={
    "container": {"background-color": "#0f0f0f"},
    "icon": {"color": "0f0f0f", "font-size": "15px"}, 
    "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#9e9e9e"},
    "nav-link-selected": {"background-color": "red"}
    })


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

inputs = ["channel_name", "video_title", "video_description",
          "duration_in_minutes", "duration_in_seconds", "thumbnail",
          "video_tags", "publish_day", "video_definition"]

if subpage == "Taking required inputs":

    st.header("Enter the following inputs:")

    channel_name = st.text_input("Input your **YouTube :red[Channel]** name: ")
    
    video_title = st.text_input(
        "Enter the title of the **:red[Video] name** name you want to create: ")
    
    video_description = st.text_input(
        "Enter the description of the **:red[Video]** you want to create: ")
    
    video_definition = st.selectbox(
        "What's the **:red[Definition]** of the video you will create: ",
        ('High definition', 'Standard definition'))
    
    publish_day = st.date_input(label='Choose the **:red[Date]** you will upload the video in:',
              value= dt.date(year=2022, month=5, day=20))

    for old, new in {"High definition": "hd", "Standard definition": "sd"}.items():
        video_definition = video_definition.replace(old, new)

    duration_in_minutes = st.number_input(
        "Enter Your **Video Duration in :red[Minutes]:**", 0)

    try:
        duration_in_seconds = int(st.number_input(
            "(Optional) Enter Your **Video Duration in :red[Seconds]:**",
            int(duration_in_minutes) * 60))

    except:
        st.error("Enter duration as integer.")
    
    thumbnail = st.file_uploader("(Optional) Upload or drag & drop your **Video :red[Thumbnail]** image: ")
    
    
    video_tags = st.text_input("What are the **:red[Tags]** of your video" + \
    "(Input them as words between square brackets [] & separated by commas): ")

    # We have to store all varibels in each 'if' scope as temp files so we can acces them
    # outside this 'if' scope.
    
    for input_ in inputs:

        if not globals()[input_] and input_ != "thumbnail":
            st.error(f"Please specfy your: {input_}")
            
        else:
            with open(f"temp/{input_}", "wb") as f:
                pickle.dump(globals()[input_], f)

    
for input_ in inputs:
    
    with open(f"temp/{input_}", "rb") as f:
        
        variable_name = input_
        globals()[variable_name] = pickle.load(f)

duration_in_minutes = duration_in_seconds / 60
# ============================================================



# ==============Gettign the channel & videos data=============

# Getting the channel's id
def get_channel_id(channel):
    
    try:
        channel_id_response = youtube.search().list(
            q= channel,
            type='channel',
            part='id',
            maxResults= 1
        ).execute()
    
    except:
        
        youtube = build(
            API_SERVICE_NAME, API_VERSION, developerKey= "AIzaSyDTjUCfCMzF16_vVasrMDp1GvxlSfUcakA")
    
        channel_id_response = youtube.search().list(
            q= channel,
            type='channel',
            part='id',
            maxResults= 1
        ).execute()
    
    channel_id = channel_id_response['items'][0]['id']['channelId']
    return  channel_id


# Getting channel's stats
def get_channel_stats(channel_id):
    
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
    
    if not "country" in item["snippet"]:
        channel_stats["country"] = np.nan
    
    else:
        channel_stats["country"] = item["snippet"]["country"]

    return  channel_stats


# Loading 50 videos from the channel
def get_videos_ids(channel_stats, max_results = [50]):

    videos_ids = []
    
    videos_ids_request = youtube.playlistItems().list(
        part= "snippet,contentDetails",
        playlistId= channel_stats["playlist_id"][0],
        maxResults= max_results[0])

    if max_results[0] <= int(channel_stats["video_count"][0]):
        
        videos_ids_response = videos_ids_request.execute()
        # print(JSON(videos_ids_response))
        
        for response in videos_ids_response["items"]:
            videos_ids.append(response["contentDetails"]["videoId"])

    else:
        max_results[0] -= int(channel_stats["video_count"])
        videos_ids = get_videos_ids(playlist_id, max_results)
        
    return  videos_ids



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

# Those are just temporary text columns not the main one.
text_cols = ["tags", "about_tokens", "title_tokens",
             "channel_name_tokens", "description_tokens"]

features_to_drop = ["viewCount",
                    "likeCount",
                    "commentCount"]

numeric_cols = ['duration_in_minutes', 'subscribers', 'total_views',
                'video_count', 'title_emojis_count', 'desc_emojis_count',
                'title_length', 'description_length', 'channel_name_length',
                'about_length', 'channel_age_days', 'video_age_days',
                'avg_views', 'avg_uploads_per_month']

target_cols = ["cat_view_count", "cat_like_count", "cat_comment_count"]


cat_cols = ['have_twitter_account', "country",
             'have_facebook_account', "language",
             'have_instagram_account', "definition",
             'have_twitch_account']
# ============================================================



## Preprocessing main function:

# returns X, y, X_input and the DF after preprocessing
def main_preprocess(channel_name: str) -> tuple[np.array,
                                 np.array,
                                 np.array,
                                 pd.DataFrame]:

    # ================Text Feature engineering====================
    
    # preparing NLTK data
    channel_id = get_channel_id(channel_name)
    channel_stats = get_channel_stats(channel_id)
    videos_ids = get_videos_ids(channel_stats)
    videos_stats = get_video_stats(videos_ids)
    
    channel_stats.drop(["playlist_id"], axis= 1, inplace= True)
    
    videos_stats = videos_stats.rename({"channelTitle": "channel_name"}, axis= 1)
    
    videos_stats.loc[-1] = ["video_id_placeholder", channel_name, video_title, video_description, video_tags,
                            publish_day, 1, 1, 1, duration_in_minutes, video_definition]
    
    videos_stats.index = videos_stats.index + 1
    videos_stats = videos_stats.sort_index()
    
    
    # Concating the videos and channels data
    df = pd.merge(videos_stats, channel_stats, on= "channel_name")

    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('stopwords')
    # nltk.download('stopwords-hi')
    # nltk.download('stopwords-ar')
    # nltk.download('averaged_perceptron_tagger')
    
    
    en_stopwords = set(stopwords.words('english')) 
    ar_stopwords = set(stopwords.words('arabic')) 
    # hi_stopwords = set(stopwords.words('hindi')) 
    
    all_stopwords = en_stopwords.union(ar_stopwords)


    # POS tagging
    def stopwords_dropper(words: list, stopwords: set) -> list:
        
        filtered_words = [re.sub(r"[\W_]", "", word) for word in words
                          if not word in stopwords]
        
        filtered_words = list(filter(lambda item: item != "", filtered_words))
        return  filtered_words
    
    for col in TEXT_COLUMNS:
        df[f"{col}_tokens"] = df[col].apply(lambda text: nltk.word_tokenize(text.lower()))
        df[f"{col}_tokens"] = df[f"{col}_tokens"].apply(lambda text: stopwords_dropper(text,
                                                                         all_stopwords))
    
    # df["title_tokens"]
    
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
    
    # Extracting emojies count
    def check_emoji(text) -> bool:
        emoji_regex = re.compile("[\U0001F600-\U0001F64F"
                                "\U0001F300-\U0001F5FF"
                                "\U0001F680-\U0001F6FF"
                                "\U0001F1E0-\U0001F1FF"
                                "\U00002600-\U000027BF"
                                "\U0001F900-\U0001F9FF"
                                "\U0001F100-\U0001F1FF"
                                "\U0001F600-\U0001F64F"
                                "\U00002702-\U000027B0"
                                "\U000024C2-\U0001F251"
                                "\U0001F600-\U0001F6FF]+", flags=re.UNICODE)

        is_emoji = bool(emoji_regex.findall(text))
        return  is_emoji
    
    
    title_emojis_counts: list = []
    desc_emojis_counts: list = []
    
    for title, desc in zip(df["title"], df["description"]):
        
        title_emojis_count: int = 0
        desc_emojis_count: int = 0
        
        for title_char, desc_char in zip(title, desc):
            
                
            if check_emoji(title_char):
                title_emojis_count += 1
                
            if check_emoji(desc_char):
                desc_emojis_count += 1
        
        title_emojis_counts.append(title_emojis_count)
        desc_emojis_counts.append(desc_emojis_count)
        
    df["title_emojis_count"] = title_emojis_counts
    df["desc_emojis_count"] = desc_emojis_counts
    
    
    
    country_languages = {
        'DE': 'German',
        'US': 'English', 'PL': 'Polish',
        'SA': 'Arabic', 'NP': 'Nepali',
        'CA': 'English', 'ES': 'Spanish',
        'TR': 'Turkish', 'IN': 'Hindi',
        'EG': 'Arabic', 'GB': 'English',
        'MX': 'Spanish', 'BR': 'Portuguese',
        'PK': 'Urdu', 'FR': 'French',
        'VN': 'Vietnamese', 'ID': 'Indonesian',
        'AU': 'English', 'HU': 'Hungarian',
        'NL': 'Dutch', 'BG': 'Bulgarian',
        'JP': 'Japanese', 'SG': 'English', 
        'TH': 'Thai', 'PH': 'Tagalog',
        'MT': 'Maltese', 'PE': 'Spanish',
        'SE': 'Swedish', 'IT': 'Italian',
        'KR': 'Korean', 'TW': 'Chinese',
        'FI': 'Finnish', 'DZ': 'Arabic',
        'BD': 'Bengali', 'AR': 'Spanish'}
    
    df["language"] = df["country"].replace(country_languages)
    
    df = df.astype({"commentCount": np.uint16, "viewCount": np.uint32,
                    "likeCount": np.uint32, "subscribers": np.uint32,
                    "video_count": np.uint16})
    
    today = datetime.utcnow().strftime("%Y-%m-%d")
    today = datetime.strptime(today, "%Y-%m-%d")

    channel_age = today - pd.to_datetime(df["start_date"]).apply(lambda x: x.tz_localize(None))
    df["channel_age_days"] = channel_age.dt.days.astype(int)

    video_age = today - pd.to_datetime(df["publishedAt"]).dt.tz_localize(None)
    df["video_age_days"] = video_age.dt.days.astype(np.uint16)


    df["language"] = df["language"].astype("category").cat.codes
    df["definition"] = df["definition"].astype("category").cat.codes
    df["country"] = df["country"].astype("category").cat.codes


    df["cat_view_count"] = pd.cut(df['viewCount'],
                             bins=[0, 3_000, 10_000, 50_000, 100_000, 300_000, 999_999_999_999],
                             labels=["from 1 to 3,000", "from 3,000 to 10,000",
                                     "from 10,000 to 50,000", "from 50,000 to 100,000",
                                     "from 100,000 to 300,000", "more than 300,000"])

    df["cat_comment_count"] = pd.cut(df['commentCount'],
                             bins=[0, 75, 150, 200, 400, 600, 999_999_999_999],
                             labels=["from 1 to 75", "from 75 to 150",
                                     "from 150 to 200", "from 200 to 400",
                                     "from 400 to 600", "more than 600"])

    df["cat_like_count"] = pd.cut(df['likeCount'],
                             bins=[0, 1_000, 5_000, 10_000, 50_000, 150_000, 999_999_999_999],
                             labels=["from 1 to 1,000", "from 1,000 to 5,000",
                                     "from 5,000 to 10,000", "from 10,000 to 50,000",
                                     "from 50,000 to 150,000", "more than 150,000"])

    global views_encoder
    global likes_encoder
    global comments_encoder
                                     
    views_encoder = {"from 1 to 3,000": 1, "from 3,000 to 10,000": 2,
                     "from 10,000 to 50,000": 3, "from 50,000 to 100,000": 4,
                     "from 100,000 to 300,000": 5, "more than 300,000": 6}

    likes_encoder = {"from 1 to 1,000": 1, "from 1,000 to 5,000": 2,
                     "from 5,000 to 10,000": 3,  "from 10,000 to 50,000": 4,
                     "from 50,000 to 150,000": 5,  "more than 150,000": 6}

    comments_encoder = {"from 1 to 75": 1, "from 75 to 150": 2,
                         "from 150 to 200": 3, "from 200 to 400": 4,
                         "from 400 to 600": 5, "more than 600": 6}


    df["cat_view_count"] = df["cat_view_count"].replace(views_encoder)
    df["cat_like_count"] = df["cat_like_count"].replace(likes_encoder)
    df["cat_comment_count"] = df["cat_comment_count"].replace(comments_encoder)
    
    accounts: list = ["twitter", "facebook", "instagram", "twitch"]
    
    for account in accounts:
        df[f"have_{account}_account"] = df["about"].str.contains(account)


    df["start_date"] = pd.to_datetime(df["start_date"])
    
    df["avg_views"] = (df["total_views"].astype(int) / df["video_count"])
    df["avg_views"] = df["avg_views"].astype(int)
    
    df["avg_uploads_per_month"] = df["video_count"] / (df["channel_age_days"] // 30)
    df["avg_uploads_per_month"] = df["avg_uploads_per_month"].astype(np.float32)
    
    
    df["duration_in_minutes"] = float(duration_in_seconds) / 60
    df["publishedAt"] = publish_day
    
    
    for column in TEXT_COLUMNS:
        df[f"{column}_length"] = df[column].str.len()
    
    
    df = df.astype({"total_views": np.uint64, "video_count": np.uint16,
                    "duration_in_minutes": np.float32,
                    "publishedAt": "datetime64[ns]"})
    
    for bool_col in ["have_facebook_account", "have_twitter_account",
                     "have_twitch_account", "have_instagram_account"]:

        df[bool_col] = df[bool_col].astype(np.uint8)
    
    X = df[numeric_cols + cat_cols + text_cols].iloc[1:]
    X_input = df[numeric_cols + cat_cols + text_cols].head(1)

    y = df[target_cols].iloc[1:]

    return  X, y, X_input, df
    # ============================================================



# =================Predicting the video success===============

# Preparing training & testing data

if subpage == "Your Video Predictions":

    X, y, X_input, df = main_preprocess(channel_name)
    
    st.sidebar.markdown("### Advanced options:")
    
    model_name = st.sidebar.selectbox('choose a ML model: ', options=['Random Forest',
                                                                  'XGB Classifier',
                                                                  'Nueral network'], index=0)

    scaler = preprocessors["Scaler"]
    encoder = preprocessors["Encoder"]
    vectorizer = preprocessors["Vectorizer"]
    pca = preprocessors["PCA"]

        
    for col in X[text_cols].columns:
        X[col] = X[col].astype(str)

    for col in X_input[text_cols].columns:
        X_input[col] = X_input[col].astype(str)
    
    X["stacked_text"] = X[text_cols].agg(', '.join, axis=1).astype(str)
    X["stacked_text"] = X["stacked_text"].str.replace("[", "")
    X["stacked_text"] = X["stacked_text"].str.replace("]", "")

    X_input["stacked_text"] = X_input[text_cols].agg(', '.join, axis=1).astype(str)
    X_input["stacked_text"] = X_input["stacked_text"].str.replace("[", "")
    X_input["stacked_text"] = X_input["stacked_text"].str.replace("]", "")
    
    numeric_cols_arr = scaler.transform(X[numeric_cols])
    numeric_cols_arr_input = scaler.transform(X_input[numeric_cols])
    
    cat_cols_arr = X[cat_cols].to_numpy()
    cat_cols_arr_input = X_input[cat_cols].to_numpy()
    
    text_col_arr = vectorizer.transform(X["stacked_text"]).toarray()
    text_col_arr_input = vectorizer.transform(X_input["stacked_text"]).toarray()
    text_cols_max_len = X[text_cols].shape[1]
    
    X = np.asarray(np.hstack(
        (numeric_cols_arr, cat_cols_arr, pca.transform(text_col_arr))))
    
    X_input = np.asarray(np.hstack(
        (numeric_cols_arr_input, cat_cols_arr_input, pca.transform(text_col_arr_input))))
        
    y = y.to_numpy().astype(int)
    y -= 1

    rf = models["Random Forest"]
    clf = models["XGB Classifier"]
    nn_model = models["Nueral network"]

    targets_encoder = encoders["Targets encoder"]
    targets_decoder = encoders["Targets decoder"]
    y_pred_decoder = encoders["y pred decoder"]

    if model_name == "Nueral network":
        nn_model.fit(X,
                     [targets_encoder(y)[0],
                      targets_encoder(y)[1],
                      targets_encoder(y)[2]],
                      verbose= 0,
                      epochs= 350,
                      batch_size= 9)

        y_pred = y_pred_decoder(nn_model.predict(X_input)) + 1
        
    elif model_name == "Random Forest":
        rf.fit(X, y)
        y_pred = rf.predict(X_input) + 1
# e
    elif model_name == "XGB Classifier":
        clf.fit(X, y)
        y_pred = clf.predict(X_input) + 1

    def get_key(val, dict):
   
        for key, value in dict.items():
            if val == value:
                return key
            
    views_class = get_key(y_pred[0][0], views_encoder)
    likes_class = get_key(y_pred[0][1], likes_encoder)
    comments_class = get_key(y_pred[0][2], comments_encoder)

    st.markdown(f"##### The expected **Views** range for your video will be between: :red[{views_class}]")
    st.markdown(f"##### The expected **Likes** range for your video will be between: :red[{likes_class}]")
    st.markdown(f"##### The expected **Comments** range for your video will be between: :red[{comments_class}]")

    st.sidebar.text("The random forest is the only working model for nut that will be fixed later.")
# ============================================================



# ===============Plotting channel statistics==================

if subpage == "Statistics of your channel":

    X, y, X_input, df = main_preprocess(channel_name)
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Videos count", df["video_count"][0])
    col2.metric("Subscribers", df["subscribers"][0])
    col3.metric("AVG Views for one like", f"{np.mean(df['viewCount'] / df['likeCount']):.2f}")
    col4.metric("Views count for one subscriber", f"{np.mean(df['total_views'] / df['subscribers']):.2f}")

    fig = px.bar(df.sort_values(by= "likeCount"), x= "likeCount", y= "title", title="Most liked videos.", width=1600, height= 1000,
                 orientation='h', labels= {"likeCount": "Likes", "title": "Video name"})
    
    st.plotly_chart(fig)
# ============================================================