import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset
import os

X_train_df = pd.read_csv('data_ready/X_train_ready.csv')
y_train_df = pd.read_csv('data_ready/y_train_ready.csv')
X_test_df = pd.read_csv('data_ready/X_test_ready.csv')
y_test_df = pd.read_csv('data_ready/y_test_ready.csv')

X_train = torch.FloatTensor(X_train_df.values.copy())
y_train = torch.FloatTensor(y_train_df.values.copy()).view(-1, 1)
X_test = torch.FloatTensor(X_test_df.values.copy())
y_test = torch.FloatTensor(y_test_df.values.copy()).view(-1, 1)

class DiabetesRiskNet(nn.Module):
    def __init__(self):
        super(DiabetesRiskNet, self).__init__()
        self.layer1 = nn.Linear(8, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.2)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
        x = self.output(x)
        return x

dataset = TensorDataset(X_train, y_train)
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

epochs = 150
batch_size = 32
fold_results = []

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    train_sub = Subset(dataset, train_ids)
    val_sub = Subset(dataset, val_ids)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

    model = DiabetesRiskNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            y_pred_logits = model(batch_X)
            y_pred_probs = torch.sigmoid(y_pred_logits)
            y_pred_class = (y_pred_probs >= 0.45).float()
            total += batch_y.size(0)
            correct += (y_pred_class == batch_y).sum().item()

    accuracy = 100.0 * correct / total
    fold_results.append(accuracy)
    print(f'Fold {fold + 1} Accuracy: {accuracy:.2f}%')

print(f'Average K-Fold Accuracy: {np.mean(fold_results):.2f}%')

final_model = DiabetesRiskNet()
final_optimizer = optim.Adam(final_model.parameters(), lr=0.001)
final_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    final_model.train()
    for batch_X, batch_y in final_loader:
        y_pred = final_model(batch_X)
        loss = criterion(y_pred, batch_y)
        final_optimizer.zero_grad()
        loss.backward()
        final_optimizer.step()

os.makedirs('saved_models', exist_ok=True)
torch.save(final_model.state_dict(), 'saved_models/diabetes_model.pth')

final_model.eval()
with torch.no_grad():
    y_pred_logits = final_model(X_test)
    y_pred_probs = torch.sigmoid(y_pred_logits)
    threshold = 0.40
    y_pred_class = (y_pred_probs >= threshold).float().numpy()
    y_true = y_test.numpy()

cm = confusion_matrix(y_true, y_pred_class)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted: Healthy', 'Predicted: Sick'],
            yticklabels=['Real: Healthy', 'Real: Sick'],
            annot_kws={"size": 16})

plt.title('Confusion matrix', fontsize=16)
plt.ylabel('Real', fontsize=14)
plt.xlabel('Predicted', fontsize=14)

os.makedirs('plot', exist_ok=True)
plt.savefig('plot/confusion_matrix.png', bbox_inches='tight')