import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pymongo
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import json

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["papers"]
paras = db["paragraphs"]

# Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=768):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# Evaluation loss
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation metrics and print misclassified paragraphs
def evaluate_metrics(model, loader, paragraphs, indices, vector_name):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_records = []
    sample_idx = 0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs).squeeze()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).long()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for j in range(len(labels)):
                true_label = labels[j].item()
                pred_label = predicted[j].item()
                if pred_label != true_label:
                    para_idx = indices[sample_idx]
                    paragraph = paragraphs[para_idx]['text']
                    misclassified_records.append({
                        'paragraph': paragraph,
                        'predicted': int(pred_label),
                        'actual': int(true_label)
                    })
                sample_idx += 1

    # Save to JSON
    filename = f"../data/misclassified_{vector_name.replace('_uncased', '')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(misclassified_records, f, ensure_ascii=False, indent=2)

    # Print evaluation metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    accuracy = accuracy_score(all_labels, all_preds)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1



# Main training and evaluation loop
def main(vector_name):
    classified_paras = list(paras.find({'manually_classified': True}))
    X = np.array([para[vector_name] for para in classified_paras])
    y = np.array([1 if para['synthesis'] else 0 for para in classified_paras])

    # Train/val/test split
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, range(len(X)), test_size=0.2, stratify=y, random_state=15)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.5, stratify=y_temp, random_state=15)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    generator = torch.Generator().manual_seed(15)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Initialize model
    model = LogisticRegressionModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.013)

    num_epochs = 500
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, criterion, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Use best model from memory
    torch.save(best_model_state, f"../models/{vector_name.replace('_uncased', '')}.pth")
    model.load_state_dict(best_model_state)

    # Plot losses
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Eval')
    plt.axvline(x=best_epoch + 1, color='r', linestyle='--', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.title(vector_name.replace('_uncased', '').replace('_', ' '))
    plt.savefig(f"training_images/{vector_name.replace('_uncased', '')}_training.png")
    plt.close()

    # Final evaluation
    return evaluate_metrics(model, test_loader, classified_paras, idx_test, vector_name)

# Run all vector types
if __name__ == "__main__":
    vector_names = ['matbert_uncased_cls', 'matbert_uncased_mean', 'scibert_uncased_cls', 'scibert_uncased_mean']
    results = []

    for vector_name in vector_names:
        print(f"\n--- Running for vector: {vector_name} ---")
        acc, prec, rec, f1 = main(vector_name)
        results.append({
            'vector': vector_name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("linear_training_results.csv", index=False)
