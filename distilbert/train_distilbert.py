import pandas as pd
import numpy as np
import os
import pickle

import torch
from transformers import DistilBertConfig
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import mlflow
import mlflow.pytorch

import matplotlib.pyplot as plt
import seaborn as sns

from helper_functions.dataset_loader import TextDataset
from helper_functions.distilbert_architecture import DistilBertForSequenceClassification
from helper_functions.pytorch_trainer import PytorchTrainer

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Function to plot and log confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    
    mlflow.log_artifact(plot_path)
    os.remove(plot_path)

    print("Confusion matrix plot logged as an artifact.")

# Set up MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("chens musicbox")

# Load your data
def load_data_feather(path="training_sets.feather"):
    data = pd.read_feather(path)
    datasets = {}
    for name in data['level_0'].unique():
        datasets[name] = np.array(data[data['level_0'] == name].drop(columns=['level_0', 'level_1'])).reshape(-1,)
    return datasets['X_train'], datasets['X_valid'], datasets['X_test'], datasets['y_train'], datasets['y_valid'], datasets['y_test']

X_train, X_valid, X_test, y_train, y_valid, y_test = load_data_feather()
print("Data Acquired")

# Prepare datasets and DataLoader
BATCH_SIZE = 64
MAX_LENGTH = 512

training_set = TextDataset(X_train, y_train, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
print("Training Dataset Loaded...")
validation_set = TextDataset(X_valid, y_valid, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, shuffle=False)
print("Validation Dataset Loaded...")
testing_set = TextDataset(X_test, y_test, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, shuffle=False)
print("Testing Dataset Loaded...")

train_loader = training_set.get_dataloader()
val_loader = validation_set.get_dataloader()
test_loader = testing_set.get_dataloader()

# Initialize model, config, optimizer
config = DistilBertConfig(num_labels=len(set(y_train)))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = DistilBertForSequenceClassification(config=config)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
print("\nModel Initialized\n")

# Start MLflow run
with mlflow.start_run():
    # Log model configuration
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("max_length", MAX_LENGTH)
    mlflow.log_param("learning_rate", 1e-5)
    mlflow.log_param("num_labels", len(set(y_train)))

    trainer = PytorchTrainer(device=device)
    
    #Load model?
    checkpoint = input("Train from checkpoint? (Y/N):")
    if checkpoint.upper() == "Y":
        path = input("Checkpoint path:")
        model, optimizer = trainer.load_model(path = path, model = model, optimizer= optimizer)
        print(f"Currently on Epoch {trainer.cur_epoch}")

    #Select epochs
    epochs = int(input("Number of Training Epochs: "))
    
    # Training process
    trained_model, trained_optimizer = trainer.run(
        model=model,
        optimizer=optimizer,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader


    # Make predictions
    model, optimizer = trainer.load_model(path= "checkpoints/3_val_acc_0.72.pth", model = model, optimizer=optimizer)
    predictions = trainer.predict(testing_loader=test_loader, model = model)
    y_true= testing_set.text_labels
    
    # Calculate accuracy, precision, recall, and f1-score
    accuracy = np.sum(predictions == y_true) / len(predictions)
    precision = precision_score(y_true=y_true, y_pred=predictions, average="macro")
    recall = recall_score(y_true=y_true, y_pred=predictions, average="macro")
    f1 = f1_score(y_true=y_true, y_pred=predictions, average="macro")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision_score", precision)
    mlflow.log_metric("recall_score", recall)

    # Log training history:
    with open("checkpoints/training_history.pkl", 'wb') as file:
        pickle.dump(trainer.history, file)
    mlflow.log_artifact("checkpoints/training_history.pkl")
    
    # Log confusion matrix
    plot_confusion_matrix(y_true, predictions, classes=testing_set.id_to_genre.values())

    # Log the trained model
    #mlflow.pytorch.log_model(trained_model, "distilbert_model")

    print(f"Training complete. Metrics: accuracy={accuracy}, f1={f1}, precision={precision}, recall={recall}")
