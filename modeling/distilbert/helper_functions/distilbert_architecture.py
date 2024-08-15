import torch.nn as nn
from transformers import DistilBertConfig,DistilBertModel

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
        return logits