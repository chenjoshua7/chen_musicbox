import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

class TextDataset:
    def __init__(self, text_dataset, labels, max_length=128, batch_size=32, shuffle=True) -> None:
        assert len(text_dataset) == len(labels), "Text samples do not match Labels"
        self.shuffle = shuffle

        self.genre_to_id = {"rap": 0, "rock": 1, "rb": 2, "pop": 3, "country": 4}
        self.id_to_genre = {v: k for k, v in self.genre_to_id.items()}
        
        checkpoint = "distilbert-base-uncased"
        self.text_dataset = list(text_dataset)
        self.text_labels = [self.genre_to_id[label] for label in labels]
        self.num_classes = len(self.genre_to_id)
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        # Tokenize the dataset
        self.input_ids, self.attention_masks = self.tokenize()
        
        # Create the DataLoader and save it internally
        self.dataloader = self.create_dataloader()
        
    def __len__(self):
        return len(self.text_dataset)
    
    def to_dataframe(self):
        return pd.DataFrame({
            "Text": self.text_dataset,
            "Label": [self.id_to_genre[label] for label in self.text_labels]
        })
    
    def __repr__(self):
        return repr(self.to_dataframe())
    
    def tokenize(self):
        inputs = self.tokenizer.batch_encode_plus(
            self.text_dataset,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            return_tensors="np"
        )
        
        input_ids = np.asarray(inputs['input_ids'], dtype='int32')
        attention_masks = np.asarray(inputs['attention_mask'], dtype='int32')
        
        return input_ids, attention_masks
    
    def create_dataloader(self):
        labels_tensor = torch.tensor(self.text_labels, dtype=torch.long)
        input_ids_tensor = torch.tensor(self.input_ids, dtype=torch.long)
        attention_masks_tensor = torch.tensor(self.attention_masks, dtype=torch.long)
        
        dataset = TensorDataset(input_ids_tensor, attention_masks_tensor, labels_tensor)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def get_dataloader(self):
        return self.dataloader
        
    def head(self, n=10):
        df = self.to_dataframe()
        return df.head(n=n)
