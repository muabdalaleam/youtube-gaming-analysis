"""
In this python file we will update te videos data and channels data
Usuing the id's that we have saved then save them as new parquet files
just as the first steps in the note book but here we will make it more
automated so we see the changes for videos and channels stats.

and also we will need to not update the time-static data such as 
describtion or video_id they will be justin the base df
"""

# Importing packeges
from datetime import datetime
import sqlite3
import sys
import json
import time
import numpy as np
import os
import pandas as pd
from IPython.display import clear_output
import googleapiclient.discovery
from googleapiclient.discovery import build
import googleapiclient.errors


# Building youyube build
API_KEY: str = "AIzaSyANcOOmvv5fs6Gx7vKXucSelmScjx3V3Qg"
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

youtube = build(
    API_SERVICE_NAME, API_VERSION, developerKey= API_KEY)


print("starting ...")

# Collecting files into python variabels
with open("../videos_ids.txt", "r") as vid:
    videos_ids: list = eval(vid.read())
    
with open("../channels_ids.txt", "r") as vid:
    channels_ids: list = eval(vid.read())
    
    
# Extracting videos stats
def get_video_stats(youtube, video_ids: list) -> pd.DataFrame:

    """This function takes the videos IDs list and request
       for the statistics of the videos then save them into
       a DataFrame."""

    all_video_info = []
    thumbnails = []
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
        print(f"Finished {processed_videos_count / videos_count * 100:.2f}% of loading the videos data",
              end= "\r")
        time.sleep(0.0001)

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

videos_df = get_video_stats(youtube, videos_ids)
    
    
# Getting channels data
def get_channel_stats(youtube, channel_ids: list) -> pd.DataFrame:

    """This function returns the response of requested channels
    into a pandas dataframe and saves it into JSON file"""

    all_channels = []
    chunk_size = 50

    for l, i in enumerate(range(0, len(channel_ids), chunk_size)):

        channel_ids_chunk = channel_ids[i:i + chunk_size]

        request = youtube.channels().list(
            part= "snippet,contentDetails,statistics",
            id= ",".join(channel_ids_chunk),
            maxResults= 50)

        response = request.execute()
        with open(f'../responses/channel_response_{l + 1}.json', 'w') as f:
            json.dump(response, f)

        for item in response["items"]:

            data = {"channel_name": item["snippet"]["title"],
                    "subscribers": item["statistics"]["subscriberCount"],
                    "total_views": item["statistics"]["viewCount"],
                    "video_count": item["statistics"]["videoCount"]}


            all_channels.append(data)

    return  pd.DataFrame(all_channels)

channels_df = get_channel_stats(youtube, channels_ids)
videos_df = videos_df.fillna(1)

lines = [
"videos_df['likeCount'].astype(np.uint32)",
"videos_df['viewCount'].astype(np.uint32)",
"videos_df['commentCount'].astype(np.uint16)"]

for line in lines:
    
    assination_line = line.split(".")[0] + " = " + line
    exec(assination_line)


channels_df["total_views"] = channels_df["total_views"].astype(np.uint64)
channels_df["video_count"] = channels_df["video_count"].astype(np.uint16)
channels_df["subscribers"] = channels_df["subscribers"].astype(np.uint32)


# Saving updated data frames
today = str(datetime.now().strftime('%Y-%m-%d'))


videos_df.to_parquet(f"../data files/videos_{today.replace('-', '_')}.parquet")
channels_df.to_parquet(f"../data files/channels_{today.replace('-', '_')}.parquet")


# Saving the stacked dataframes:

channels_files = []
videos_files = []

directory_path = "../data files/"


for file_name in os.listdir(directory_path):
    
    if "channels" in file_name and not("base" in file_name):
        channels_files.append(directory_path + file_name)
        
    elif ("videos" in file_name) and not("base" in file_name):
        videos_files.append(directory_path + file_name)
     
    

videos_dfs = []
channels_dfs = []

for path in videos_files:
    
    video_df = pd.read_parquet(path)
    start_sympol = path.index("2")
    end_sympol = path.index(".p")
    
    video_df["Collecting date"] = path[start_sympol:end_sympol]
    videos_dfs.append(video_df)

for path in channels_files:
    
    channel_df = pd.read_parquet(path)
    channel_df["Collecting date"] = path[start_sympol +2:end_sympol+2]
    channels_dfs.append(channel_df)
    
    
conn = sqlite3.connect('../../database.db')
    
stacked_videos_df = pd.concat(videos_dfs, ignore_index=True)
stacked_channels_df = pd.concat(channels_dfs, ignore_index=True)

stacked_videos_df.to_sql('stacked_videos', conn, if_exists='replace', index=False)
stacked_channels_df.to_sql('stacked_channels', conn, if_exists='replace', index=False)

stacked_channels_df.to_pickle("../../Cleaned files/stacked_channels.pickle")
stacked_videos_df.to_pickle("../../Cleaned files/stacked_videos.pickle")

print("\nDone ...\n")