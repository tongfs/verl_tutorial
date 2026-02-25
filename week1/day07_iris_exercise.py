import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

BATCH_SIZE = 64

# load data
X, y = load_iris(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# define model
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 3))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train model
model.train()
step = 0
for epoch in range(10):
    for batch_X, batch_y in train_loader:
        step += 1
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

        # evaluate model
        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            acc = (pred.argmax(1) == y_val).float().mean().item()
            print(
                f"Step {step} (Epoch {epoch + 1}, Batch {len(batch_X)} samples) - Loss: {loss.item():.4f}, Val Accuracy: {acc:.2%}"
            )
