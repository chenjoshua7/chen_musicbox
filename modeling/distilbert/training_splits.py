import pandas as pd
import re

from sklearn.model_selection import train_test_split

def preprocess_text(x: pd.Series) -> list[str]: 
    x_ = x.copy()
    x_ = x_.apply(lambda x: x.replace('\n', ' '))
    x_ = x_.apply(lambda x: x.replace('\u2005', ' '))
    x_ = [re.sub(r'\[.*?\]\s*', '', line) for line in x_]
    x_ = [re.sub(r'\(.*?\)\s*', '', line) for line in x_]
    return x_

data = pd.read_feather("../../data_preprocessing/filtered_data.feather")
data["processed_lyrics"] = preprocess_text(data.lyrics)

genres = set(data.tag)

# Taking only the top 10000 per genre
top_10000_per_genre = pd.DataFrame()
for genre in genres:
    genre_data = data[data["tag"] == genre]
    top_10000 = genre_data.sort_values("views", ascending=False).iloc[:10000, :]
    top_10000_per_genre = pd.concat([top_10000_per_genre, top_10000])

filtered_data = top_10000_per_genre.reset_index(drop=True)

X_, X_test, y_, y_test = train_test_split(x, y, random_state=1234, test_size=0.75, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_, y_, random_state=1234, test_size=0.8, stratify=y_)
del X_, y_

def save_data_feather(X_train, X_valid, X_test, y_train, y_valid, y_test, path="training_sets.feather") -> None:
    data = {
        'X_train': X_train,
        'X_valid': X_valid,
        'X_test': X_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test
    }
    for key, value in data.items():
        value.reset_index(drop=True, inplace=True)  # To avoid issues with index
    pd.concat(data.values(), keys=data.keys()).reset_index().to_feather(path)
    print(f"Data saved to {path}")

save_data_feather(X_train, X_valid, X_test, y_train, y_valid, y_test)



