from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, DistilBertConfig
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware

import torch.nn as nn
from transformers import DistilBertModel

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

def load_model(path, model):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print("Model Loaded")
    return model


# Loading model and tokenizer
config = DistilBertConfig(num_labels=5)
model = DistilBertForSequenceClassification(config=config)
model.load_state_dict(torch.load("distilbert_model_state.pth", weights_only=False))
model.eval()
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class TextInput(BaseModel):
    text: str

genres = ["rap", "rock", "rb", "pop", "country"]

# Defining App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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
        predictions[genre] = str(round(score * 100,2)) + "%"

    return predictions
