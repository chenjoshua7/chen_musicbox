import pandas as pd
import numpy as np

import os
import pickle

from sklearn.model_selection import train_test_split

from utils import preprocess_text, lemmatize, plot_confusion_matrix

from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("chens musicbox")

#################################################
## Running a Baseline with Logistic Regression ##
#################################################

data = pd.read_feather("data_preprocessing/filtered_data.feather")
print("Data Loaded")

x = data.lyrics
y = data.tag

file_path = "processed_lyrics.pkl"

try:
    if os.path.exists(file_path):
        print("File exists. Loading processed data...")
        
        with open(file_path, 'rb') as handle:
            x = pickle.load(handle)
    else:
        print("Starting Preprocessing...")
        x = preprocess_text(x)
        
        with open(file_path, 'wb') as handle:
            pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print("Data Preprocessed and saved to file.")
except Exception as e:
    print(f"An error occurred: {e}")

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1234, shuffle=True)



## Training ##

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=0.01, max_df=0.7, ngram_range=(1,2), stop_words="english")),
    ("scale", StandardScaler(with_mean=False)),
    ("lr", OneVsRestClassifier(LogisticRegression(solver='liblinear', multi_class='ovr', penalty="l1"))),
])

with mlflow.start_run():
    print("Model Training...")
    pipeline.fit(X_train, y_train)
    print("Model Trained")
    
    print("Predicting Insample...")
    insamp_pred = pipeline.predict(X_train)
    insamp_accuracy = np.sum(insamp_pred == y_train)/len(y_train)

    print("Predicting Outsample...")
    outsamp_pred = pipeline.predict(X_test)
    outsamp_accuracy = np.sum(outsamp_pred == y_test)/len(y_test)

    precision = precision_score(y_true=y_test, y_pred=outsamp_pred, average="macro")
    recall = recall_score(y_true=y_test, y_pred=outsamp_pred, average="macro")
    f1 = f1_score(y_true=y_test, y_pred=outsamp_pred, average="macro")

    print(f"Insample Accuracy: {insamp_accuracy}")
    print(f"Outsample Accuracy: {outsamp_accuracy}")
    print(f"F1 Score: {f1}")

    # Log metrics
    mlflow.log_metric("insample_accuracy", insamp_accuracy)
    mlflow.log_metric("outsample_accuracy", outsamp_accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision_score", precision)
    mlflow.log_metric("recall_score", recall)

    # Log model
    mlflow.sklearn.log_model(pipeline, "model")

    # Log parameters
    mlflow.log_param("train_size", 0.3)
    mlflow.log_param("random_state", 1234)
    mlflow.log_param("solver", "liblinear")
    mlflow.log_param("penalty", "l1")
    mlflow.log_param("ngram_range", (1, 2))
    mlflow.log_param("min_df", 0.01)
    mlflow.log_param("max_df", 0.7)
    
    ## Log confusion matrix
    plot_confusion_matrix(y_test, outsamp_pred, classes=y_test.unique())

    # Optionally save the pipeline to a pickle file as an artifact
    with open("baseline.pkl", 'wb') as file:
        pickle.dump(pipeline, file)
    mlflow.log_artifact("baseline.pkl")
