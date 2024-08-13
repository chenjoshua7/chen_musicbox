import pandas as pd
import numpy as np

def load_data(path: str = "song_lyrics.csv") -> pd.DataFrame:
    print("Loading Data...")
    data = pd.read_csv(path)
    print("Data Loaded")
    return data

def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    df_ = data.copy()
    
    # Removing data less than 1000 views
    df_ = df_[df_.views > 1000]
    # Removing misc. Some strange samples including the poetry of William Blake
    df_ = df_[(df_.tag != "misc")]
    
    # Only including decades between 1950 and 2030. This removes erroneous data as well as
    # data unimport to the project
    decade = df_.year//10 * 10
    df_[(decade >= 1950) & (decade <= 2030)]
    
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