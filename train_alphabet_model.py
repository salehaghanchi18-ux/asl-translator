import os, glob, random
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET_DIR = "gesture_data"
CSV_PATTERN = os.path.join(DATASET_DIR, "gesture_data_*.csv")

# Load all CSVs
files = glob.glob(CSV_PATTERN)
assert files, f"No CSVs found at {CSV_PATTERN}"

X_list, y_list = [], []
for f in files:
    df = pd.read_csv(f, header=None)
    # Expect 64 columns: 63 features + 1 label
    assert df.shape[1] >= 64, f"File {f} has {df.shape[1]} columns; expected >=64"
    X = df.iloc[:, :63].values.astype(np.float32)
    y = df.iloc[:, 63].values.astype(str)
    X_list.append(X)
    y_list.append(y)

X = np.vstack(X_list)
y = np.concatenate(y_list)

# Encode labels (A–D)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)
joblib.dump(le, "label_encoder.pkl")

# Scale features (Standardization)
# Compute mean and std per feature on entire dataset, then split.
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-8
X_scaled = (X - mean) / std
scaler = {"mean": mean, "std": std}
joblib.dump(scaler, "feature_scaler.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Model
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

input_size = X_train_t.shape[1]
num_classes = len(le.classes_)
model = GestureNet(input_size, num_classes)

# Class weights to reduce bias
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-4)  # L2 regularization
epochs = 60
batch_size = 128

# Simple DataLoader-like batching
def iterate_batches(Xt, yt, bs):
    idx = np.arange(Xt.shape[0])
    np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i+bs]
        yield Xt[j], yt[j]

best_acc = 0.0
best_state = None

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in iterate_batches(X_train_t, y_train_t, batch_size):
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.shape[0]

    # Eval
    model.eval()
    with torch.no_grad():
        logits_train = model(X_train_t)
        pred_train = logits_train.argmax(dim=1)
        acc_train = (pred_train == y_train_t).float().mean().item()

        logits_test = model(X_test_t)
        pred_test = logits_test.argmax(dim=1)
        acc_test = (pred_test == y_test_t).float().mean().item()

    if acc_test > best_acc:
        best_acc = acc_test
        best_state = model.state_dict()

    if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
        avg_loss = epoch_loss / X_train_t.shape[0]
        print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f} | Train Acc: {acc_train:.3f} | Test Acc: {acc_test:.3f}")

# Save best model
if best_state is not None:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), "asl_alphabet_model.pt")
print("✅ Model trained and saved as asl_alphabet_model.pt")
print(f"✅ Best Test Accuracy: {best_acc:.3f}")
