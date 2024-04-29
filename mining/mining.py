"""
This function uses Youtube's API key from the environment vars to do the 
following querys:

1- Search for random Gaming channels' ids around the globe
2- Expands those channels data given thier id
3- Collect videos ids using the playlist id in the channels data
4- Expand the data for the videos

And finally all of those data will be stored in the main database with a backup
for raw-data.
"""

import os
import pandas as pd
import sqlite3
import requests
import datetime
from dotenv import load_dotenv

load_dotenv()

MAX_RESULTS: int = 50
# TODO upscale channels count (have to be doubler to MAX_RESULTS)
CHANNELS_COUNT: int = 100
VIDEOS_PER_CHANNEL: int = 10  # TODO upscale videos per channel

SEARCH_QUERY = "gaming"
API_KEY = os.getenv('YOUTUBE_API')


def search_channels() -> set[str]:
	"""
	Returns a set of channels ids knowing the API_KEY and SEARCH_QUERY global
	constants without needing any param.

	@params: None
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

	channels_ids = set()

	for _ in range(int(CHANNELS_COUNT / MAX_RESULTS)):

		response = requests.get(url, params=params)

		if response.status_code == 200:
			data = response.json()

			for channel in data['items']:
				channels_ids.add(channel['id']['channelId'])
				next_page_token = data.get("nextPageToken")

				if next_page_token:
					params['pageToken'] = next_page_token

	return channels_ids


def get_channels_data(channels_ids: list) -> pd.DataFrame:
	raise NotImplementedError


def get_videos(playlist_ids: list) -> list:
	raise NotImplementedError


def get_videos_data(videos_ids: list) -> pd.DataFrame:
	raise NotImplementedError


def main():
	channels_ids = search_channels()
	print('Channels ids: ', channels_ids)

if __name__ == '__main__':
	main()