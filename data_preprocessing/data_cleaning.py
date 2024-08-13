import pandas as pd
import numpy as np
import time

def load_data(path: str = "song_lyrics.csv") -> pd.DataFrame:
    print("Loading Data...")
    start = time.time()
    data = pd.read_csv(path)
    print(f"Data Loaded: {time.time()-start}")
    return data

def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    df_ = data.copy()

    # Removing data less than 1000 views
    start = time.time()
    df_ = df_[df_.views > 1000]
    print(f"Views Filtered: {time.time() - start}")
    
    # Removing misc. Some strange samples including the poetry of William Blake
    start = time.time()
    df_ = df_[(df_.tag != "misc")]
    print(f"Misc Filtered: {time.time() - start}")
    
    # Only including decades between 1950 and 2030. This removes erroneous data as well as
    # data unimport to the project
    start = time.time()
    decade = df_.year//10 * 10
    df_ = df_[(decade >= 1950) & (decade <= 2030)]
    print(f"Decades Filtered: {time.time() - start}")

    # Removing non-English samples
    start = time.time()
    df_ = df_[(df_.language == "en") & (df_.language_ft == "en")]
    print(f"Language Filtered: {time.time() - start}")
    
    # Removing genius english translations
    start = time.time()
    df_ = df_[~(df_.artist == "Genius English Translations")]
    print(f"Artists Filtered: {time.time() - start}")
    
    df_ = df_.drop(columns=['language_cld3', 'language_ft', 'language'])
    print(f"Original Samples: {data.shape[0]} \n Filtered Samples: {df_.shape[0]}")

    return df_

def save_data(data: pd.DataFrame, path: str = "filtered_data.csv") -> None:
    data.to_csv(path, index=False)
    print(f"Data Saved as {path}")
    return

if __name__ == "__main__":
    data = load_data()
    filtered_data = filter_data(data = data)
    save_data(data=filtered_data)