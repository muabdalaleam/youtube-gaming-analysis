#!/usr/bin/env python
# coding: utf-8

# ## <center>**<span style="color: red">Wragling</span> steps**

# ## <center> **Importing <span style="color: red">packeges</span>**

# In[1]:


import pandas as pd
import os
import ast
import panel as pn
import sqlite3
import pickle
import numpy as np
import time
import cv2
import urllib.request
import nltk
import seaborn as sns
import matplotlib
from langdetect import detect
import matplotlib.pyplot as plt
import google_auth_oauthlib.flow
from IPython.display import display, clear_output
from IPython.display import set_matplotlib_formats
from IPython.display import JSON
import googleapiclient.discovery
from googleapiclient.discovery import build
import googleapiclient.errors


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
set_matplotlib_formats('pdf', 'svg')


# In[3]:


API_KEY: str = "AIzaSyANcOOmvv5fs6Gx7vKXucSelmScjx3V3Qg"
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"


youtube = build(
    API_SERVICE_NAME, API_VERSION, developerKey= API_KEY)


# ## <center> **Reading The data & simple <span style="color: red">exploring</span>**

# In[4]:


THEME_COLORS = ["#383838", "#ff0000"]
FONT = 13


# In[5]:


channels_files = []
videos_files = []

directory_path = "../Data engineering/data files/"


for file_name in os.listdir(directory_path):
    
    if "channels" in file_name and not("base" in file_name):
        channels_files.append(directory_path + file_name)
        
    elif ("videos" in file_name) and not("base" in file_name):
        videos_files.append(directory_path + file_name)


# In[6]:


videos_files


# In[7]:


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
    
video_df["Collecting date"]


# In[8]:


stacked_videos_df = pd.concat(videos_dfs, ignore_index=True)
stacked_channels_df = pd.concat(channels_dfs, ignore_index=True)

base_videos_df = pd.read_parquet(r"../Data engineering/data files/videos_2023-04-16_base.parquet")
base_channels_df = pd.read_parquet(r"../Data engineering/data files/channels_2023-04-16_base.parquet")


# In[9]:


base_videos_df.head()


# In[10]:


stacked_videos_df.sample(5)


# In[11]:


stacked_videos_df.info() 


# In[12]:


stacked_channels_df.info()


# In[13]:


stacked_videos_df.describe()


# In[14]:


base_channels_df.describe()


# #### **Things to conclude:**
# 
# - The dtypes backed again to object dtype instead of what we have done
# - The optimizations we have done compressed the data very well from 180 MB to just 6
# - The parquet conversation copmressed a lot of strorage but in real it has more than that.
# - The outliers needs to be treated so we can do the visualizations as best as we cans
# - There are lots of **Feature extraction & engineering** work we shoild do
# - We should check for any bias in the data

# ## **<center> Creating <span style="color: red">new DFs</span> for deffrint tasks**

# Until now we have **two types** if you did't understand that first one is the base one which have<br>
# the data that **doesn 't update with time** and the second one is the **changes in time for some<br>
# vars** But the both for each video have the a **primary key** which is the video or channel ID

# - **Comments df** using 100 sample of the videos ID
# - **A df for ML models** that will be normally distrbuted and ROS <ins>(We will not do this now)
# - **Videos-thubmnails df** for CNN analysis using just 200 sampels
# - **Videos per game df for** analyzig each game performance in view, likes etc..

# #### **1- Comments df**

# We will try to get the **20** comments for **200** random videos each of them has more<br>
# than **25** comments and we will need them to be all <ins>english</ins> comments.

# In[15]:


languages = []
temp_df = base_videos_df[base_videos_df["commentCount"].astype(float) > 100]

for description in temp_df["description"].sample(500):
    try:
        languages.append(detect(description))
    except:
        languages.append(None)
        
temp_df["language"] = pd.Series(languages)
temp_df = temp_df[temp_df["language"] == "en"]

videos_ids = temp_df.sample(100)["video_id"]


# In[16]:


videos_ids


# In[17]:


def get_comments(youtube, videos_ids):
    
    comments = []
    like_counts = []
    comments_ids = []
    reply_counts = []
    
    for video_id in videos_ids:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=20)
        
        response = request.execute()

        for item in response["items"]:
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            like_counts.append(item["snippet"]["topLevelComment"]["snippet"]["likeCount"])
            comments_ids.append(video_id)
            reply_counts.append(item["snippet"]["totalReplyCount"])

    df = pd.DataFrame({"Comments": np.array(comments),
                       "LikeCounts": like_counts,
                       "ReplyCounts": reply_counts})
    
    df["video_id"] = comments_ids
    
    return df


# In[18]:


comments_df = get_comments(youtube, videos_ids)


# In[19]:


comments_df.sample(7)


# #### **2-  Videos thubnails df**

# In[20]:


stacked_videos_df["thumbnails"] = "https://img.youtube.com/vi/" + stacked_videos_df["video_id"] + "/default.jpg"
print("Here 's an example for the the thumbnails URLs: " + stacked_videos_df["thumbnails"][99])
stacked_videos_df = stacked_videos_df.drop("thumbnails", axis= 1)


# In[21]:


# # Create a new dataframe with video ids
# thumbnails_df = base_videos_df[["video_id"]].sample(1000).copy().reset_index()
# thumbnails = []


# for index, row in thumbnails_df.iterrows():
#     url = "https://img.youtube.com/vi/" + row["video_id"] + "/default.jpg"

#     with urllib.request.urlopen(url) as response:
#         array = np.asarray(bytearray(response.read()), dtype=np.uint8)
#         img = cv2.imdecode(array, cv2.COLOR_BGR2RGB)
#         img = np.array(img)
        
#     time.sleep(0.001)
#     plt.show()
    
#     thumbnails.append(img)
#     progress = (index + 1) / len(thumbnails_df) * 100
#     clear_output(wait=True)
#     display(f"Processing thumbnails... {progress:.2f}% complete")
    
    
# thumbnails_df["thumbnails"] = thumbnails
# thumbnails_df["thumbnails"] = thumbnails_df["thumbnails"]


# In[22]:


# with open(r"pickels/thumbnails_df.pickle", "wb") as f:
#     pickle.dump(thumbnails_df, f, protocol= pickle.HIGHEST_PROTOCOL)


# In[23]:


with open("pickels/thumbnails_df.pickle", "rb") as f:
    thumbnails_df = pickle.load(f)


# In[24]:


plt.imshow(thumbnails_df["thumbnails"][421])
plt.title("BGR thumbnail of random video")

# plt.savefig("plots/random thumbnail.svg")
plt.show()


# #### **3- Specific games df**

# We will try here to analyze 60 videos data from about **15** games and ther data and we<br>
# will also create base df and stacked df to track each game growth and also we will<br>
# save the result videos ids into a **.txt** file.

# The games we will analyze are:
# 1. Minecraft
# 3. Fortnite
# 5. Roblox
# 4. Among Us
# 5. Grand Theft Auto V
# 6. Call of Duty: Warzone
# 7. Apex Legends
# 8. Valorant
# 9. Genshin Impact
# 10. Rust
# 11. League of Legends
# 12. Pubg mobile
# 13. Counter strike
# 14. ARK survival evolved
# 15. Overwatch

# In[25]:


games = [
    "Minecraft", "Fortnite", "Roblox",
    "Among Us", "Grand Theft Auto V", "Call of Duty: Warzone",
    "Apex Legends", "Valorant", "Genshin Impact", "Rust",
    "League of Legends", "Pubg mobile", "Counter strike",
    "ARK survival evolved", "Overwatch"]


# In[26]:


def search_videos_ids(content_type, count):
    
    next_page_token = None
    all_videos_ids = []

    # using next page tokens because limited max results
    while len(all_videos_ids) < count:
        
        search_response = youtube.search().list(
            q=content_type,
            type= "video",
            part=  "id",
            maxResults= min(50, count - len(all_videos_ids)),
            pageToken= next_page_token
        ).execute()

        # Extract the channel IDs from the search results
        channel_ids = [item["id"]["videoId"] for item in search_response["items"]]

        # Add the channel IDs to the list of all channel IDs
        all_videos_ids.extend(channel_ids)

        next_page_token = search_response.get('nextPageToken')
        if not next_page_token:
            break

    return all_videos_ids


# In[27]:


# games_ids = dict()

# for game in games:
#     games_ids[game] = search_videos_ids(game, 100)


# In[28]:


# with open(r"pickels/games_ids.pickle", "wb") as f:
#     pickle.dump(games_ids, f, protocol= pickle.HIGHEST_PROTOCOL)


# In[29]:


with open('pickels/games_ids.pickle', 'rb') as f:
    games_ids = pickle.load(f)
    
print("Here's an example for a ids snippet from Minecraft videos: \n" + \
      str(games_ids["Minecraft"][0:5]))


# In[30]:


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
        clear_output(wait= True)
        time.sleep(0.00001)

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


# In[31]:


temp_dfs = []

for game_name, ids_list in games_ids.items():
    
    temp_df = get_video_stats(youtube, ids_list)
    temp_df["game"] = game_name
    temp_dfs.append(temp_df)
    
base_games_df = pd.concat(temp_dfs)


# In[32]:


print("The length of the data frame is: " + str(len(base_games_df)))


# In[33]:


base_games_df.sample(5)


# ## <center>**<span style="color: red">Visualizing</span> data issues**

# In[34]:


fig, axes = plt.subplots(1, 3)

for i, col in enumerate(["viewCount", "commentCount", "likeCount"]):
    
    axes[i].get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    sns.boxplot(stacked_videos_df[col], ax= axes[i],
               color= THEME_COLORS[1])
    axes[i].set_title(col + "box plot")
    
fig.set_size_inches(15, 5)
# plt.savefig("plots/outliers boxplot.svg")
plt.show()


# In[35]:


fig, axes = plt.subplots(3, 1)
vids_null_count = []
ch_null_count = []
games_null_count = []

for col in base_videos_df.columns:
    vids_null_count.append(base_videos_df[col].isnull().sum())
    
for col in base_channels_df.columns:
    ch_null_count.append(base_channels_df[col].isnull().sum())
    
for col in base_games_df.columns:
    games_null_count.append(base_games_df[col].isnull().sum())

    
axes[0].barh(base_videos_df.columns, vids_null_count, color= THEME_COLORS[1])
axes[0].set_title("Base videos df null counts", fontsize= FONT)

axes[1].barh(base_channels_df.columns, ch_null_count, color= THEME_COLORS[1])
axes[1].set_title("Base channels df null counts", fontsize= FONT)

axes[2].barh(base_games_df.columns, games_null_count, color= THEME_COLORS[1])
axes[2].set_title("Base specific gamse df null counts", fontsize= FONT)


for ax in fig.axes:
    plt.sca(ax)
    plt.ylabel("Columns") 
    plt.xlabel("Null count")

fig.set_size_inches(8, 14)
# plt.savefig("plots/null counts detailed.svg")
plt.show()


# In[36]:


fig, axes = plt.subplots(1, 2)

videos_consumation = stacked_videos_df.memory_usage(index= False, deep= True).values // 1024
videos_cols = stacked_videos_df.memory_usage(index= False, deep= True).index

channels_consumation = stacked_channels_df.memory_usage(index= False, deep= True).values // 1024
channels_cols = stacked_channels_df.memory_usage(index= False, deep= True).index

axes[0].bar(x= videos_cols, height= videos_consumation, color= THEME_COLORS[0])
axes[1].bar(x= channels_cols, height= channels_consumation, color= THEME_COLORS[0])

axes[0].set_title("Stacked videos data consumation")
axes[1].set_title("Stacked channels data consumation")

axes[0].set_ylabel("data consumation in KB")
axes[1].set_ylabel("data consumation in KB")

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.xlabel("Columns")
    
fig.set_size_inches(10, 4)
plt.show()


# In[37]:


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

videos_numerical_cols = ["viewCount",
                         "commentCount",
                         "likeCount"]

for ax, col in zip([ax1, ax2, ax3], videos_numerical_cols):
    stacked_videos_df[col] = stacked_videos_df[col].astype(float)
    ax.hist(stacked_videos_df[col], color= THEME_COLORS[1], bins= 40)
    ax.set_title(col + " hist")
    ax.ticklabel_format(style= "plain")

fig.set_size_inches(10, 12)
fig.suptitle("Stacked videos columns histogram", fontsize= FONT)

plt.show()


# In[38]:


plt.pie(base_videos_df["definition"].value_counts().values,
        autopct='%1.1f%%',
        labels= base_videos_df["definition"].replace(
        {"sd": "standard definition", "hd": "high definition"}).unique(),
        shadow=True, startangle= 70, colors= THEME_COLORS)

centre_circle = plt.Circle((0, 0), 0.8, fc='white')
fig = plt.gcf()
 
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

plt.title("Checking videos definition bias")
plt.show()


# In[39]:


fig, _ = plt.subplots(1)

data_consumation_per_df = {"Base videos df": base_videos_df, "Stacked videos df" : stacked_videos_df,
                           "Thumbnails df" :thumbnails_df, "Comments df" : comments_df,
                           "Base specific games df" : base_games_df}

for key, val in data_consumation_per_df.items():
    data_consumation_per_df[key] = val.memory_usage(index= False, deep= True).values.sum() / 1024 ** 2


plt.bar(x= data_consumation_per_df.keys(),
        height= data_consumation_per_df.values(),
        color= THEME_COLORS[::-1])

plt.title("Data consumation per data frame")
plt.ylabel("Storage in Mega bytes")
fig.set_size_inches(10, 4)


plt.show()


# In[40]:


fig, _ = plt.subplots(1)

nans_per_df = {"Base videos df": base_videos_df, "Stacked videos df" : stacked_videos_df,
               "Thumbnails df" :thumbnails_df, "Comments df" : comments_df,
               "Base specific games df" : base_games_df}

for key, val in nans_per_df.items():
    data_consumation_per_df[key] = val.isna().sum().sum()
    

plt.barh(y= [*data_consumation_per_df.keys()],
         width= [*data_consumation_per_df.values()],
         color= THEME_COLORS)

fig.set_size_inches(h= 3, w= 7)

plt.xlabel("Null counts")
plt.ylabel("Data frame")


plt.title("Null counts per Data frame")
plt.show()


# ## <center> **<span style="color: red">Solving</span> data issues**

# **Things to conclude from the last section**:<br><br>
# The data looks ready for our analysis because of that youtube API provides<br> 
# consistent and tidy data and that's expected from big API as youtube's API<br>
# **but**:
# - The new data frames we made can be a lot more optimized.
# - The base specific games df has many NANs so we need to treat them.
# - There are bias in the definition of videos but we will talk more about that in ML phase.
# - Tags column has NANs that should be treated as empty lists.
# - Some columns such as *total viwes*, *views* have inconsistent data such as zeros.
# - There are strong outliers in channels and some videos data but we will treat them later.
# - The distrbution of the data should be normalized.

# - Getting out of NANS and treating them
# - Optimize the dtypes as possible
# - Dropping useless columns
# - Treating the outliers with smooth function
# - treating inconsistent data

# #### **1-** *Getting out of nulls*

# In[41]:


# We will fill like count and comment count nulls with zero

base_games_df["likeCount"].fillna(0, inplace= True)
base_games_df["commentCount"].fillna(0, inplace= True)

# Now we will set each null tags row to an empty list in str
# format but we will treat this later

base_games_df["tags"].fillna("[]", inplace= True)

base_videos_df["tags"].replace("0", "[]", inplace= True)


# In[42]:


base_channels_df['country'] = base_channels_df['country'].cat.add_categories("N/A")
base_channels_df["country"].fillna("N/A", inplace= True)


# #### **2-** *Optimizing Dtypes*

# In[43]:


print("The dtypes of base games videos df for each column are: \n" + str(base_games_df.dtypes))


# In[44]:


print("The dtypes of thumbnails df for each column are: \n" + str(thumbnails_df.dtypes))


# In[45]:


print("The dtypes of comments df for each column are: \n" + str(comments_df.dtypes))


# In[46]:


print("The dtypes of stacked channels df for each column are: \n" + str(stacked_channels_df.dtypes))


# In[47]:


print("The dtypes of stacked videos df for each column are: \n" + str(stacked_videos_df.dtypes))


# In[48]:


base_games_df.reset_index(inplace= True)
base_videos_df.reset_index(inplace= True)

base_games_df = base_games_df.astype({"game": "category"})

base_games_df["publishedAt"] = pd.to_datetime(base_games_df["publishedAt"])
base_videos_df["publishedAt"] = pd.to_datetime(base_videos_df["publishedAt"])

base_games_df["duration_in_minutes"] = pd.to_timedelta(base_games_df["duration"]).dt.total_seconds() / 60.0
base_videos_df["duration_in_minutes"] = pd.to_timedelta(base_videos_df["duration"]).dt.total_seconds() / 60.0


base_games_df = base_games_df.astype({"viewCount": np.uint32, "likeCount": np.uint32(),
                             "commentCount" : np.uint16(), "definition": "category",
                             "duration_in_minutes": np.float16()})

base_videos_df = base_videos_df.astype({"viewCount": np.uint32, "likeCount": np.uint32(),
                                        "commentCount" : np.uint16(), "definition": "category",
                                        "duration_in_minutes": np.float16()})

base_channels_df["date"] = pd.to_datetime(base_channels_df["date"])


# In[49]:


stacked_channels_df["Collecting date"] = pd.to_datetime(stacked_channels_df["Collecting date"],
                                                        format="%Y_%m_%d")

stacked_videos_df["Collecting date"] = pd.to_datetime(stacked_videos_df["Collecting date"],
                                                      format="%Y_%m_%d")


# In[50]:


comments_df = comments_df.astype({"ReplyCounts": np.uint16,
                                  "LikeCounts": np.uint16})


# #### **3-** *Dropping useless columns*

# In[51]:


print("Before: ", comments_df.columns)
# The columns namming is Camal case which is not consistent with our
# naming format so:

comments_df = comments_df.rename(columns={"Comments" : "comments",
                                          "LikeCounts": "like_counts",
                                          "ReplyCounts": "reply_counts"})
print("After: ", comments_df.columns)


# In[52]:


print(thumbnails_df.columns)
# We can find there are a new index column so we will drop that

thumbnails_df.drop("index", inplace= True, axis= 1)


# In[53]:


print("Before: ", base_games_df.columns, "\n")

# There are 2 more columns that we will not use :
# (index and duration)

base_games_df.drop(["duration", "index"], axis= 1, inplace= True)

print("After: ", base_games_df.columns)


# In[54]:


print("Before: ", base_videos_df.columns, "\n")

# There are 2 more columns that we will not use :
# (index and duration)

base_videos_df.drop(["duration", "index"], axis= 1, inplace= True)

print("After: ", base_videos_df.columns)


# **Important note:** the other dfs doesn't have any wierd cols becuase we looked at them before.

# ##  <center>**Local <span style="color: red">saving</span>**

# We will save the ddata with 3 ways first one is to save the Data frames locally with<br>
# **SQLite** on the local drive and the second way is to save it using a cloud service<br>
# and the third one is saving them using **.pickle** file because it prevents the dtype changes<br>
# better than **.parquet**

# #### **4-** *Treating inconsistent data*

# In[55]:


for df in base_channels_df, stacked_channels_df:
    df["total_views"].replace(0, 1, inplace= True)
    df["video_count"].replace(0, 1, inplace= True)
    
for df in base_videos_df, base_games_df, stacked_videos_df:
    df["viewCount"].replace(0, 1, inplace= True)
    df["likeCount"].replace(0, 1, inplace= True)
    df["commentCount"].replace(0, 1, inplace= True)


# ## <center> **<span style="color: red">Local</span> saving**

# #### **1-** *Saving using **SQLite***

# In[56]:


# We need to transform the tags column to object dtype so it
# can be saved in the data bse and in the pickle file
base_games_df["tags"] = base_games_df["tags"].astype(str)
thumbnails_df["thumbnails"] = thumbnails_df["thumbnails"].astype(str)


# In[57]:


conn = sqlite3.connect('../database.db')

base_videos_df.to_sql('base_videos', conn, if_exists='replace', index=False)
base_games_df.to_sql('base_games', conn, if_exists='replace', index=False)
base_channels_df.to_sql('base_channels', conn, if_exists='replace', index=False)

comments_df.to_sql('comments', conn, if_exists='replace', index=False)
thumbnails_df.to_sql('thumbnails', conn, if_exists='replace', index=False)

stacked_videos_df.to_sql('stacked_videos', conn, if_exists='replace', index=False)
stacked_channels_df.to_sql('stacked_channels', conn, if_exists='replace', index=False)


# In[58]:


df = pd.read_sql_query("""SELECT *
                          FROM base_videos""", conn)


# In[59]:


df.info()


# In[60]:


df = pd.read_sql_query("""SELECT *
                          FROM base_videos
                          WHERE viewCount > 3000000""", conn)

df.head()


# As i thought SQLite don't save the dtypes changes so we will need to save using .pickle<br>
# but we will use the SQLite unoptimized DFs for simple analysis and we will use the .pickle<br>
# on in any thing that need the data to be optimized such as **Regression models**.

# #### **1-** *Saving using **.Pickle***

# In[61]:


comments_df.to_pickle("../Cleaned files/comments.pickle")
thumbnails_df.to_pickle("../Cleaned files/thumbnails.pickle")

base_games_df.to_pickle("../Cleaned files/base_games.pickle")
base_channels_df.to_pickle("../Cleaned files/base_channels.pickle")
base_videos_df.to_pickle("../Cleaned files/base_videos.pickle")

stacked_channels_df.to_pickle("../Cleaned files/stacked_channels.pickle")
stacked_videos_df.to_pickle("../Cleaned files/stacked_videos.pickle")


# In[ ]:




