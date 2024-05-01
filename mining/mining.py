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
import pandas as pd
import sqlite3
import requests
import itertools
import datetime
from dotenv import load_dotenv

load_dotenv()

MAX_RESULTS: int = 50
CHANNELS_COUNT: int = 100
VIDEOS_PER_CHANNEL: int = 10

SEARCH_QUERY = "gaming"
API_KEY = os.getenv('YOUTUBE_API')


# helper function
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


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
		"key": API_KEY,
		"part": "snippet",
		"q": SEARCH_QUERY,
		"type": "channel",
		"maxResults": MAX_RESULTS
	}

	if os.path.exists('pagetoken.txt'):
		with open('pagetoken.txt', 'r') as page_token:
			params['pageToken'] = page_token.read()

	channels_ids = set()

	for _ in range(int(CHANNELS_COUNT / MAX_RESULTS)):

		response = requests.get(url, params=params)
		response.raise_for_status()

		data = response.json()

		for channel in data['items']:
			channels_ids.add(channel['id']['channelId'])
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


def get_channels_data(channels_ids: set[str]) -> pd.DataFrame:
	"""
	Requests the data for 50 channels ids a time and add them to pandas df
	and after finishing the looping over all ids returns a pandas df of the
	channels data.

	@params: set of channels ids
	@returns: a pandas dataframe with the channel stats and data
	"""

	df = pd.DataFrame(columns= ['channel_name', 'subscribers', 'total_views',
		'date', 'playlist_id', 'video_count', 'about'])

	for channel_ids_block in chunker(list(channels_ids), MAX_RESULTS):

		print(channel_ids_block)

		url = 'https://www.googleapis.com/youtube/v3/channels'
		params = {
			"key": API_KEY,
			"part": "snippet,statistics,contentDetails",
			"id": ','.join(channel_ids_block),
			"maxResults": MAX_RESULTS
		}

		response = requests.get(url, params=params)
		response.raise_for_status()

		data = response.json()
		print(data)
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

		df.append(channel_data, ignore_index=True) # tofix

	return df


def get_videos(playlist_ids: list) -> list:
	raise NotImplementedError


def get_videos_data(videos_ids: list) -> pd.DataFrame:
	raise NotImplementedError

# testing only
def main():

	# channels_ids: set = search_channels()

	with open('../data/channels-ids.txt', 'r') as f:
		channels_ids = eval(f.read())

	df = get_channels_data(channels_ids)

if __name__ == '__main__':
	main()