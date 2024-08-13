import pandas as pd
import numpy as np
import time

def load_data(path: str = "song_lyrics.csv") -> pd.DataFrame:
    print("Loading Data...")
    start = time.time()
    data = pd.read_csv(path)
    end = time.time()
    print(f"Data Loaded: {start-end}")
    return data

def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    df_ = data.copy()
    
    # Removing data less than 1000 views
    start = time.time()
    df_ = df_[df_.views > 1000]
    end = time.time()
    print(f"Views Filtered: {start - end}")
    
    # Removing misc. Some strange samples including the poetry of William Blake
    start = time.time()
    df_ = df_[(df_.tag != "misc")]
    end = time.time()
    print(f"Misc Filtered: {start - end}")
    
    # Only including decades between 1950 and 2030. This removes erroneous data as well as
    # data unimport to the project
    start = time.time()
    decade = df_.year//10 * 10
    df_[(decade >= 1950) & (decade <= 2030)]
    end = time.time()
    print(f"Decades Filtered: {start - end}")
    
    print(f"Original Samples: {data.shape[0]} \n Filtered Samples: {df_.shape[0]}")
    return df_

def save_data(data: pd.DataFrame, path: str = "filtered_data.csv") -> None:
    data.to_csv(path = path, index=False)
    print(f"Data Saved as {path}")
    return

if __name__ == "__main__":
    data = load_data()
    filtered_data = filter_data(data = data)
    save_data(data=filtered_data)