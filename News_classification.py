import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


train_texts = train_data['Title'] + " " + train_data['Description']
test_texts = test_data['Title'] + " " + test_data['Description']

train_labels = train_data['Class Index'].values - 1
test_labels = test_data['Class Index'].values - 1

np.random.seed(42)
reduced_train_data = train_data.sample(n=10, random_state=42)
reduced_train_texts = reduced_train_data['Title'] + " " + reduced_train_data['Description']
reduced_train_labels = reduced_train_data['Class Index'].values - 1

reduced_test_data = test_data.sample(n=10, random_state=42)
reduced_test_texts = reduced_test_data['Title'] + " " + reduced_test_data['Description']  # Fixed this line
reduced_test_labels = reduced_test_data['Class Index'].values - 1

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), label


reduced_train_dataset = TextDataset(reduced_train_texts, reduced_train_labels, tokenizer)
reduced_test_dataset = TextDataset(reduced_test_texts, reduced_test_labels, tokenizer)

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=3):
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                # Calculate predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate validation accuracy
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}")

    return train_losses, val_losses, val_accuracies


learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [16, 32]

results = []

for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"\nTraining with Learning Rate: {lr}, Batch Size: {batch_size}")

        # DataLoader for the current batch size
        train_loader = DataLoader(reduced_train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(reduced_test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize a fresh model for each hyperparameter configuration
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # print(device)

        # Define the optimizer with the current learning rate
        optimizer = AdamW(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        # Train and validate
        train_losses, val_losses, val_accuracies = train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5)

        # Store results for visualization
        results.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        })


plt.figure(figsize=(10, 6))

for result in results:
    plt.plot(range(1, len(result['train_losses']) + 1), result['train_losses'], label=f"Train Loss LR={result['learning_rate']}, BS={result['batch_size']}")
    plt.plot(range(1, len(result['val_losses']) + 1), result['val_losses'], label=f"Val Loss LR={result['learning_rate']}, BS={result['batch_size']}")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend(loc='best')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
for result in results:
    plt.plot(range(1, len(result['val_accuracies']) + 1), result['val_accuracies'], label=f"Val Acc LR={result['learning_rate']}, BS={result['batch_size']}")

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy over Epochs")
plt.legend(loc='best')
plt.grid(True)
plt.show()

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

evaluate_model(model, val_loader, device)