import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from torchvision import transforms
from torchvision.datasets import MNIST

from vae_mixin import *


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_enc = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)

    def forward(self, x):
        x = F.relu(self.fc_enc(x))
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.decoder(z)


class VAE(BetaVAEMixin, VAEMixin, nn.Module):
    def __init__(self):
        super().__init__(beta=1)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


def train(model: VAE, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for data, target in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_rec, z = model(data)
        loss = model.loss(x_rec=x_rec, x=data)
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()
    train_loss /= len(train_loader.dataset)
    print(f"Train Epoch: {epoch} Average loss: {train_loss:.4f}")


def test(model: VAE, device, test_loader):
    model.eval()
    test_loss = 0
    z_test = []
    y_true = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_rec, z = model(data)
            loss = model.loss(x_rec=x_rec, x=data)
            z_test.extend(z.tolist())
            y_true.extend(target.tolist())
            test_loss += loss.item() * len(data)
    test_loss /= len(test_loader.dataset)
    y_pred = KMeans(n_clusters=10).fit(z_test).labels_
    ami = adjusted_mutual_info_score(y_true, y_pred)
    print(f"Test set: Average loss: {test_loss:.4f} Clustering AMI: {ami}\n")


torch.manual_seed(42)
device = torch.device("cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten),
    ]
)
dataset1 = MNIST("data", train=True, download=True, transform=transform)
dataset2 = MNIST("data", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
