import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

X_train_df = pd.read_csv('X_train_ready.csv')
y_train_df = pd.read_csv('y_train_ready.csv')
X_test_df = pd.read_csv('X_test_ready.csv')
y_test_df = pd.read_csv('y_test_ready.csv')

X_train = torch.FloatTensor(X_train_df.values.copy())
y_train = torch.FloatTensor(y_train_df.values.copy()).view(-1, 1)

X_test = torch.FloatTensor(X_test_df.values.copy())
y_test = torch.FloatTensor(y_test_df.values.copy()).view(-1, 1)

class DiabetesRiskNet(nn.Module):
    def __init__(self):
        super(DiabetesRiskNet, self).__init__()
        self.layer1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

model = DiabetesRiskNet()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 500

for epoch in range(epochs):
    model.train()

    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

with torch.no_grad():
    y_pred_tensor = model(X_test)
    threshold = 0.45
    y_pred_class = (y_pred_tensor >= threshold).float().numpy()
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

plt.savefig('confusion_matrix.png', bbox_inches='tight')