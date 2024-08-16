import pandas as pd
import numpy as np
import time

def load_data(path: str = "song_lyrics.csv") -> pd.DataFrame:
    print("Loading Data...")
    start = time.time()
    data = pd.read_csv(path)
    print(f"Data Loaded: {time.time()-start}")
    return data

def view_filter(data: pd.DataFrame) -> pd.DataFrame:
    return data[data.views > 1000]

def misc_filter(data: pd.DataFrame) -> pd.DataFrame: 
    return data[(data.tag != "misc")]
    
#Only including decades between 1950 and 2030. This removes erroneous data as well as data unimport to the project
def decade_filter(data: pd.DataFrame) -> pd.DataFrame: 
    decade = data.year//10 * 10
    return data[(decade >= 1950) & (decade <= 2030)]

def language_filter(data: pd.DataFrame) -> pd.DataFrame:
    # Removing non-English samples
    start = time.time()
    return data[(data.language == "en") & (data.language_ft == "en")]
    
def artists_filter(data: pd.DataFrame) -> pd.DataFrame:
    return data[~data.artist.isin(["Genius English Translations", "Glee Cast"])]

def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    data = view_filter(data)
    data = misc_filter(data)
    data = decade_filter(data)
    data = language_filter(data)
    data = artists_filter(data)
    data = data.drop(columns=['language_cld3', 'language_ft', 'language'])
    print(f"Original Samples: {data.shape[0]} \n Filtered Samples: {data.shape[0]}")
    return data

def save_data(data: pd.DataFrame, path: str = "filtered_data.csv") -> None:
    data.to_csv(path, index=False)
    print(f"Data Saved as {path}")
    return

if __name__ == "__main__":
    data = load_data()
    original_samples = data.shape[0]
    filtered_data = filter_data(data = data)
    print(f"Filtering Complete:\n{'-' * 20}\nOriginal Samples: {original_samples}\nSamples Remaining: {filtered_data.shape[0]}\nSamples Filtered: {original_samples - filtered_data.shape[0]}")
    save_data(data=filtered_data)
    