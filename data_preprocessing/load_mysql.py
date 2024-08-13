import pymysql
import pandas as pd
import numpy as np
from tqdm import tqdm

print("Starting Data Loading")
data = pd.read_csv("filtered_data.csv")
data = data.where(pd.notnull(data), None)

conn = pymysql.connect(host='localhost',
                       user='root',
                       password='huahua20101',
                       db='song_lyrics')

cursor = conn.cursor()
print("Connected")

# Creating Table
try:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS songs_data (
        id INT PRIMARY KEY,
        title NVARCHAR(50),
        genre NVARCHAR(10),
        artist NVARCHAR(50),
        year YEAR,
        views INT,
        features TEXT,
        lyrics TEXT
        )
        """
    )
    print("Table Created or already exists")
except pymysql.MySQLError as e:
    print(f"Error creating table: {e}")


# Inserting Data
try:
    for row in tqdm(data.itertuples(), desc="Inserting Rows", unit="row"):
        try:
            cursor.execute(
                """
                INSERT INTO songs_data (id, title, genre, artist, year, views, features, lyrics)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, 
                (row.id, row.title, row.tag, row.artist, row.year, row.views, row.features, row.lyrics)
            )
            conn.commit()
        except pymysql.MySQLError as e:
            print(f"Error inserting row with ID {row.id}: {e}")
    print("Data insertion completed.")
except Exception as e:
    print(f"Unexpected error: {e}")

finally:
    cursor.close()
    conn.close()