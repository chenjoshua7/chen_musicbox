import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import confusion_matrix
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np


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

def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    # Normalize the confusion matrix by row (i.e., by the number of samples in each actual class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    
    # Save the plot to a file
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()  # Close the plot to avoid displaying it in non-interactive environments
    
    # Log the plot as an artifact in MLflow
    mlflow.log_artifact(plot_path)
    
    # Optionally remove the plot file after logging
    os.remove(plot_path)

    print("Confusion matrix plot logged as an artifact.")
