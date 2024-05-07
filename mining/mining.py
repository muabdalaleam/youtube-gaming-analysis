"""
This function uses Youtube's API key from the environment vars to do the 
following querys:

1- Search for random Gaming channels' ids around the globe
2- Expands those channels data given thier id
3- Collect videos ids using the playlist id in the channels data
4- Expand the data for the videos

And finally all of those data will be stored in the main database with a backup
for raw-data.

Note: some functions in this code is imported by update.py
"""

import os
import ast
import pandas as pd
import sqlite3
import requests
import itertools
import datetime
from dotenv import load_dotenv

load_dotenv()

MAX_RESULTS: int = 50
CHANNELS_COUNT: int = 100
VIDEOS_PER_CHANNEL: int = 20

SEARCH_QUERY = "gaming"
# API_KEY = 


def get_api_key():
	api_keys = [
		os.getenv('YOUTUBE_API_1'),
		os.getenv('YOUTUBE_API_2'),
		os.getenv('YOUTUBE_API_3'),
		os.getenv('YOUTUBE_API_4'),
		os.getenv('YOUTUBE_API_5')
	]

	api_keys_cycle = itertools.cycle(api_keys)

	while True:
		yield next(api_keys_cycle)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

API_KEY_GENERATOR = get_api_key()

def search_channels(save_page_token= False) -> set[str]:
	"""
	Returns a set of channels ids knowing the API_KEY and SEARCH_QUERY global
	constants also this function save pagetoken using bool param sat to False
	by defualt to create a txt file and save the next page token to continue
	the search from where we stopped on deffrint days.

	@params: save_page_token and load_page_token bool params
	@returns: set of channels ids as strings
	"""

	url = 'https://www.googleapis.com/youtube/v3/search'
	params = {
		"key": next(API_KEY_GENERATOR),
		"part": "snippet",
		"q": SEARCH_QUERY,
		"type": "channel",
		"maxResults": MAX_RESULTS
	}

	if os.path.exists('pagetoken.txt'):
		with open('pagetoken.txt', 'r') as page_token:
			params['pageToken'] = page_token.read()

	channels_ids = []

	for _ in range(int(CHANNELS_COUNT / MAX_RESULTS)):

		response = requests.get(url, params=params)

		if response.status_code != 200:
			return channels_ids

		data = response.json()

		for channel in data['items']:
			channels_ids.append(channel['id']['channelId'])
			next_page_token = data.get("nextPageToken")

			if next_page_token:
				params['pageToken'] = next_page_token

			else:
				# this condition should never execute but just in case
				next_page_token = ''
				raise Exception('Scraped all avelaible gaming channels')

	if save_page_token:
		with open('pagetoken.txt', 'w') as f:
			f.write(next_page_token)

	return channels_ids

def get_channels_data(channels_ids: list) -> pd.DataFrame:
	"""
	Requests the data for 50 channels ids a time and add them to pandas df
	and after finishing the looping over all ids returns a pandas df of the
	channels data.

	@params: set of channels ids
	@returns: a pandas dataframe with the channel stats and data
	"""

	df = pd.DataFrame(columns= ['channel_name', 'subscribers', 'total_views',
		'date', 'playlist_id', 'video_count', 'about'])

	for channel_ids_block in chunker(channels_ids, MAX_RESULTS):

		url = 'https://www.googleapis.com/youtube/v3/channels'
		params = {
			"key": next(API_KEY_GENERATOR),
			"part": "snippet,statistics,contentDetails",
			"id": ','.join(channel_ids_block),
			"maxResults": MAX_RESULTS
		}

		response = requests.get(url, params=params)
		if response.status_code != 200:
			return df

		data = response.json()

		for channel in data['items']:
			channel_data = {
					"channel_name":     channel["snippet"]["title"],
					"subscribers":      channel["statistics"]["subscriberCount"],
					"total_views":      channel["statistics"]["viewCount"],
					"date":             channel["snippet"]["publishedAt"],
					"playlist_id":      channel["contentDetails"]["relatedPlaylists"]["uploads"],
					"video_count":      channel["statistics"]["videoCount"],
					"about":            channel["snippet"]["description"]
				}

			df = pd.concat([df, pd.DataFrame([channel_data])], ignore_index=True)

	return df

def get_videos_ids(playlist_ids: list) -> list:
	"""
	This function takes the videos IDs list and request
	for the statistics of the videos then saves them into
	a DataFrame.

	@params: the playlist ids of any channel
	@returns: returns a list of videos ids 
	"""

	videos_ids = []

	for playlist_id in playlist_ids:

		url = 'https://www.googleapis.com/youtube/v3/playlistItems'
		params = {
			"key": next(API_KEY_GENERATOR),
			"part": "id",
			"playlistId":  playlist_id,
			"maxResults": VIDEOS_PER_CHANNEL,
		}

		response = requests.get(url, params= params)

		if response.status_code != 200:
			params['maxResults'] = 10
			response = requests.get(url, params= params)

			if response.status_code != 200:
				videos_ids.append(None)
				continue

		data = response.json()['items']

		for vid in data:
			videos_ids.append(vid['id'])


	return videos_ids

def get_videos_data(videos_ids: list) -> pd.DataFrame:

	df = pd.DataFrame(columns= ['channel_name', 'title', 'description',
		'tags', 'published_at', 'view_count', 'like_count', 'comment_count',
		'duration', 'definition'
	])

	for videos_ids_block in chunker(videos_ids, MAX_RESULTS):

		url = 'https://www.googleapis.com/youtube/v3/videos'
		params = {
			"key": next(API_KEY_GENERATOR),
			"part": "snippet,contentDetails,statistics",
			"id": ','.join(videos_ids_block),
			"maxResults": MAX_RESULTS
		}

		response = requests.get(url, params=params)
		if response.status_code != 200:
			print('An error happened: ', response.status_code)
			return df

		print(response.json())

		data = response.json()

		for video in data['items']:
			video_data = {
				"channel_name":     video["snippet"]["channel_title"],
				"title":            video["snippet"]["title"],
				"description":      video["snippet"]["description"],
				"tags":             video["snippet"]["tags"],
				"published_at":     video["snippet"]["publishedAt"],
				"view_count":       video["statistics"]["viewCount"],
				"like_count":       video["statistics"]["likeCount"],
				"comment_count":    video["statistics"]["commentCount"],
				"definition":       video["contentDetails"]["definition"],
				"duration":         video["contentDetails"]["duration"]
			}

			print('Hi', video_data)

			df = pd.concat([df, pd.DataFrame([video_data])], ignore_index=True)

	return df

# Debugging only
if __name__ == '__main__':
	
	# df = pd.read_csv('temp.csv')
	# playlist_ids = df['playlist_id']

	# vids_ids = get_videos_ids(playlist_ids)

	with open('videos-ids.txt', 'r') as f:
		vid_ids = ast.literal_eval(f.read())

	vids_df = get_videos_data(vid_ids[:50])

	print(vids_df.head())