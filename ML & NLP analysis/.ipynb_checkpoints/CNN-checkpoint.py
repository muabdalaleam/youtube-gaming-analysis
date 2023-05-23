import tensorflow as tf
import keras
import pandas as pd
import numpy as np


con = sqlite3.connect('../database.db')

thumbnails_df = pd.read_sql_query("""
                        SELECT *
                        FROM thumbnails""", con)

comments_df = pd.read_sql_query("""

                        SELECT *
                        FROM base_videos AS bv

                        INNER JOIN base_channels  AS bc ON
                            bc.channel_name = bv.channelTitle

                        INNER JOIN comments  AS c ON
                            c.video_id = bv.video_id""", con)
con.close()


