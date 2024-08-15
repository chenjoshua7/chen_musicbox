from tqdm import tqdm
import torch
import os

class PytorchTrainer:
    def __init__(self, device) -> None:
        self.history = {}
        self.device = device
        self.best_val_accuracy = 0
        self.count = 0
        self.path = "checkpoints"
        self.epochs = 0
        
    def run(self, model, optimizer, epochs, train_loader, val_loader):
        self.epochs = epochs
        
        for epoch in range(1, epochs + 1):
            model, optimizer, (training_loss, training_accuracy) = self.evaluate(model=model, optimizer=optimizer, train_loader=train_loader, epoch=epoch)
            validation_loss, val_accuracy = self.validate(model=model, val_loader=val_loader, epoch=epoch)
            
            self.history[f"Epoch {epoch}"] = [(training_loss, training_accuracy), (validation_loss, val_accuracy)]
            
            self.early_stopping(val_accuracy, model, optimizer, epoch)
            if self.count == 5:
                return model, optimizer
            
            print(f'Epoch {epoch}/{epochs} - Val Accuracy: {val_accuracy:.4f}')   
        return model, optimizer

    def evaluate(self, model, optimizer, train_loader, epoch):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}/{self.epochs}')
        
        model.train()
        running_loss, train_correct, train_total = 0.0, 0, 0

        for step, (input_ids, attention_mask, labels) in progress_bar:
            input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            if loss is not None:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            progress_bar.set_postfix({'train_loss': f'{running_loss / (step + 1):.4f}', 'train_acc': f'{train_correct / train_total:.4f}'})

        train_loss = running_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        return model, optimizer, (train_loss, train_accuracy)

    def validate(self, model, val_loader, epoch):
        model.eval()
        with torch.no_grad():
            valid_loss, valid_correct, valid_total = 0.0, 0, 0
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                
                valid_output = outputs.argmax(dim=1)
                valid_loss += loss.item()
                valid_correct += (valid_output == labels).sum().item()
                valid_total += labels.size(0)
            
            val_accuracy = valid_correct / valid_total
            valid_loss /= len(val_loader)
            print(f'Epoch {epoch} - Validation Loss: {valid_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}')
            return (valid_loss, val_accuracy)

    def early_stopping(self, val_accuracy: float, model, optimizer, epoch) -> None:
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            save_path = os.path.join(self.path, f'{epoch}_val_acc_{val_accuracy:.4f}.pth')
            self.save_model(model=model, optimizer=optimizer, epoch=epoch, path=save_path)
            self.count = 0
        else:
            self.count += 1

    def save_model(self, model, optimizer, epoch, path) -> None:
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()

        torch.save({
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'epoch': epoch
        }, path)
        
    def set_path(self, path: str) -> None:
        self.path = path