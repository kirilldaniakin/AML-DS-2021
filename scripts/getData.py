import pandas as pd
import os
import numpy as np

def get_data(data_path="../data"):
    assert os.path.isfile(data_path+"/train.csv"), f"{os.path.realpath(data_path)} : File does not exist"
    assert os.path.isfile(data_path+"/test.csv"), f"{os.path.realpath(data_path)} : File does not exist"
    # read data
    df_ratings = pd.read_csv(
    'train.csv',
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

    df_ratings_test = pd.read_csv(
    'test.csv',
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
    
  return df_ratings, df_ratings_test

if __name__ == '__main__':
  pass
