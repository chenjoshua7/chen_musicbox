import pandas as pd
import numpy as np
import re

import os
import pickle

from sklearn.model_selection import train_test_split
from tqdm import tqdm 

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

## Get rid of linebreaks, short spaces, and lemmatize
def preprocess_text(x: pd.Series) -> list[str]: 
    x_ = x.copy()
    x_ = x_.apply(lambda x: x.replace('\n', ' '))
    x_ = x_.apply(lambda x: x.replace('\u2005', ' '))
    x_ = [re.sub(r'\[.*?\]\s*', '', line) for line in x_]
    x_ = [re.sub(r'\(.*?\)\s*', '', line) for line in x_]
    x_ = pd.Series(x_).apply(lambda x: lemmatize(x))
    return x_

def lemmatize(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join([word for word in lemmatized_words if word not in stop_words])


## Running a Baseline with Logistic Regression

data = pd.read_feather("../data_preprocessing/filtered_data.feather")
print("Data Loaded")

x = data.lyrics
y = data.tag


print("Starting Preprocessing...")
x = preprocess_text(x)
print("Data Preprocessed")

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1234, shuffle = True)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(min_df = 0.01, max_df = 0.7, ngram_range = (1,2), stop_words = "english")),
    ("scale", StandardScaler(with_mean=False)),
    ("lr", OneVsRestClassifier(LogisticRegression(solver='liblinear', multi_class='ovr', penalty = "l1"))),
])

print("Model Training...")
pipeline.fit(X_train, y_train)
print("Model Trained")

print("Predicitng Insample...")
insamp_pred = pipeline.predict(X_train)
insamp_accuracy = np.sum(insamp_pred == y_train)/len(y_train)

print("Predicitng Outsample...")
outsamp_pred = pipeline.predict(X_test)
outsamp_accuracy = np.sum(outsamp_pred == y_test)/len(y_test)

print(f1_score(y_true = y_test, y_pred = outsamp_pred, average = "macro"))

print(f"Insample Accuracy: {insamp_accuracy}")
print(f"Outsample Accuracy: {outsamp_accuracy}")


with open(os.path.join('baseline.pkl'), 'wb') as file:
    pickle.dump(pipeline, file)