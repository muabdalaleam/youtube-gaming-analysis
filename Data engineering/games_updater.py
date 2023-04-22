
# Importing packeges
import pandas as pd
import os
import ast
import time
import sqlite3
import pickle
import datetime
import numpy as np
import google_auth_oauthlib.flow
import googleapiclient.discovery
from googleapiclient.discovery import build
import googleapiclient.errors


# Building the youtube build
API_KEY: str = "AIzaSyANcOOmvv5fs6Gx7vKXucSelmScjx3V3Qg"
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"


youtube = build(
    API_SERVICE_NAME, API_VERSION, developerKey= API_KEY)


# Looding the ID's and get the statistics
with open('pickels/games_ids.pickle', 'rb') as f:
    games_ids = pickle.load(f)
    
def get_video_stats(youtube, video_ids: list) -> pd.DataFrame:

    """This function takes the videos IDs list and request
       for the statistics of the videos then save them into
       a DataFrame."""

    all_video_info = []
    videos_count = len(video_ids)

    for i in range(0, videos_count, 50):
        
        chunk = video_ids[i:i+50]
        processed_videos_count = i + len(chunk)
        
        # Giving the request for each 50 video in one time
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(chunk))
        response = request.execute()
        
        # Calculate the progress with updating it.
        print(f"Finished {processed_videos_count / videos_count * 100:.2f}% of loading the videos data")
        os.system('cls' if os.name == 'nt' else 'clear')
        time.sleep(0.001)

        for video in response['items']:
            video_json_encoder = {"statistics": ['viewCount', 'likeCount', 'commentCount']}

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

temp_dfs = []

for game_name, ids_list in games_ids.items():
    
    temp_df = get_video_stats(youtube, ids_list)
    temp_df["game"] = game_name
    temp_dfs.append(temp_df)
    
games_df = pd.concat(temp_dfs)

today = str(datetime.datetime.now().strftime('%Y-%m-%d'))

games_df["Collecting date"] = today


games_df["likeCount"].fillna(1, inplace= True)
games_df["commentCount"].fillna(1, inplace= True)
games_df["viewCount"].fillna(1, inplace= True)

games_df["likeCount"].replace(1, inplace= True)
games_df["commentCount"].replace(1, inplace= True)
games_df["viewCount"].replace(1, inplace= True)


# Optimizing the raw data

games_df.astype({"viewCount": np.uint32, "likeCount": np.uint32(), 
                 "commentCount": np.uint16, "Collecting date": 'datetime64[ns]'})


# Saving the data using .Pickle
try:
    stacked_games_df = pd.read_pickle("data files/stacked_games_df.pickle")
    stacked_games_df = pd.concat([stacked_games_df, games_df], ignore_index=True)
    stacked_games_df.to_pickle("../Cleaned files/stacked_games_df.pickle")
    
except:
    games_df.to_pickle("../Cleaned files/stacked_games_df.pickle")
    
    
# Saving the data using SQLite

conn = sqlite3.connect('../database.db')
games_df.to_sql('stacked_games', conn, if_exists= 'append', index=False)
print("Done succefully!!!")