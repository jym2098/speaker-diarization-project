import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---- Dataset for precomputed embeddings ----
class PairDataset(Dataset):
    def __init__(self, pair_file):
        self.pairs = torch.load(pair_file)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2, label = self.pairs[idx]
        return x1.float(), x2.float(), label.float()

# ---- Siamese Network for Embeddings ----
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=512):  # adjust input_dim if needed
        super(SiameseNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# ---- Contrastive Loss ----
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# ---- Training Config ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-3
batch_size = 64
num_epochs = 60

# ---- Load Data ----
dataset = PairDataset("pairs.pt")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ---- Model, Loss, Optimizer ----
net = SiameseNetwork(input_dim=dataset[0][0].shape[0]).to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# ---- Training Loop ----
for epoch in range(num_epochs):
    net.train()
    total_loss = 0
    for x1, x2, label in train_loader:
        x1, x2, label = x1.to(device), x2.to(device), label.to(device)
        output1, output2 = net(x1, x2)
        loss = criterion(output1, output2, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ---- Accuracy Evaluation ----
def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x1, x2, label in loader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            out1, out2 = model(x1, x2)
            dist = F.pairwise_distance(out1, out2)
            pred = (dist > 0.5).float()
            correct += (pred == label).sum().item()
            total += label.size(0)
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")

evaluate(train_loader, net)
evaluate(test_loader, net)

# ---- Save model ----
torch.save(net.state_dict(), "trainedsnn.pt")
