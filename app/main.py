import pandas as pd
import io
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, DistilBertConfig
import torch.nn.functional as F
import torch.nn as nn
from transformers import DistilBertModel
from fastapi.middleware.cors import CORSMiddleware

class DistilBertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)
        hidden_state = distilbert_output[0]                    
        pooled_output = hidden_state[:, 0]                   
        pooled_output = self.pre_classifier(pooled_output)   
        pooled_output = nn.ReLU()(pooled_output)             
        pooled_output = self.dropout(pooled_output)        
        logits = self.classifier(pooled_output) 
        
        # Use softmax for multi-class classification
        outputs = F.softmax(logits, dim=-1)
        return outputs

device = torch.device('cpu')

# Loading model and tokenizer
config = DistilBertConfig(num_labels=5)
model = DistilBertForSequenceClassification(config=config)
model.load_state_dict(torch.load("distilbert_model_state.pth", weights_only=False))
model.eval()
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class TextInput(BaseModel):
    text: str

class PredictionInput(BaseModel):
    lyrics: str
    predictions: dict
    actual: str
    
genres = ["rap", "rock", "rb", "pop", "country"]

# Defining App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}

@app.post("/predict/")
async def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    predictions = {}
    logits = logits[0].tolist()

    for genre, score in zip(genres, logits):
        predictions[genre] = str(round(score * 100, 2)) + "%"

    return predictions

# Pull from Blob Storage
def pull_from_blob():
    sas_url = "https://musicblob123.blob.core.windows.net/predictorfeedback?sp=racwdlm&st=2024-09-06T11:49:14Z&se=2025-09-06T19:49:14Z&spr=https&sv=2022-11-02&sr=c&sig=NaHBqIKNFndYKfSSlMVXq2VfCVp%2BZegLfv7ygGD2Kws%3D"
    
    current_week = datetime.now().strftime("%Y-%U")
    file_name = f"lyric_predictions_{current_week}.csv"
    
    blob_service_client = BlobServiceClient(account_url=sas_url)
    blob_client = blob_service_client.get_blob_client(container="predictorfeedback", blob=file_name)
    
    try:
        download_stream = blob_client.download_blob()
        csv_data = download_stream.readall().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        print(f"'{file_name}' downloaded successfully from Azure Blob Storage.")
    except Exception:
        # If blob doesn't exist, return an empty DataFrame
        df = pd.DataFrame(columns=["song_lyrics", "rap", "rock", "rb", "pop", "country", "actual_genre", "timestamp"])
    
    return df, file_name

# Save to Blob Storage function
def save_to_blob(lyrics, predictions, actual):
    # Pull the existing CSV from Azure Blob Storage and get the DataFrame
    df, file_name = pull_from_blob()
    
    # Capture the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a new row to append to the DataFrame
    new_data = {
        'song_lyrics': lyrics,
        'rap': predictions["rap"],
        'rock': predictions["rock"],
        'rb': predictions["rb"],
        'pop': predictions["pop"],
        'country': predictions["country"],
        'actual_genre': actual,
        'timestamp': current_datetime  # Add current date and time
    }
    
    # Append the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    
    # Convert DataFrame to a CSV in memory (using a buffer)
    output = io.StringIO()
    df.to_csv(output, index=False)
    csv_data = output.getvalue().encode('utf-8')
    output.seek(0)  # Reset buffer position

    # Blob Storage configuration
    sas_url = "https://musicblob123.blob.core.windows.net/predictorfeedback?sp=racwdlm&st=2024-09-06T11:49:14Z&se=2025-09-06T19:49:14Z&spr=https&sv=2022-11-02&sr=c&sig=NaHBqIKNFndYKfSSlMVXq2VfCVp%2BZegLfv7ygGD2Kws%3D"
    
    # Create a BlobServiceClient using the container SAS URL
    blob_service_client = BlobServiceClient(account_url=sas_url)
    blob_client = blob_service_client.get_blob_client(container="predictorfeedback", blob=file_name)
    
    # Upload the updated CSV to Azure Blob Storage
    blob_client.upload_blob(csv_data, overwrite=True)
    
    print(f"'{file_name}' uploaded successfully to Azure Blob Storage.")

@app.post("/log/")
async def log_into_blob(data: PredictionInput):
    try:
        save_to_blob(data.lyrics, data.predictions, data.actual)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log data into Blob Storage: {str(e)}")

    return {"message": "Data logged successfully to Blob Storage"}
