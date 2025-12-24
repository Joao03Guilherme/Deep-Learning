# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST, INFO
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleCNN(nn.Module):
    def __init__(self, use_softmax=False):
        super(SimpleCNN, self).__init__()
        ### YOUR CODE HERE ###

        self.use_softmax = use_softmax

        # Input: 3 x 28 x 28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Output: 32 x 28 x 28
        # RELU
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Output: 64 x 28 x 28
        # RELU
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # Output: 128 x 28 x 28
        # RELU
        self.linear1 = nn.Linear(in_features=128 * 28 * 28, out_features=256)
        # RELU
        self.linear2 = nn.Linear(in_features=256, out_features=8)  # 8 classes for BloodMNIST


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten to feed into linear layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        if self.use_softmax:
            x = F.softmax(x, dim=1)

        return x
    
class SimpleCNNWithPooling(nn.Module):
    def __init__(self, use_softmax=False):
        super(SimpleCNNWithPooling, self).__init__()
        ### YOUR CODE HERE ###

        self.use_softmax = use_softmax

        # Input: 3 x 28 x 28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Output: 32 x 28 x 28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 32 x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Output: 64 x 14 x 14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 64 x 7 x 7
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # Output: 128 x 7 x 7
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 128 x 3 x 3
        self.linear1 = nn.Linear(in_features=128 * 3 * 3, out_features=256)
        # RELU
        self.linear2 = nn.Linear(in_features=256, out_features=8)  # 8 classes for BloodMNIST


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten to feed into linear layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        if self.use_softmax:
            x = F.softmax(x, dim=1)

        return x

# Data Loading

data_flag = 'bloodmnist'
print(data_flag)
info = INFO[data_flag]
print(len(info['label']))
n_classes = len(info['label'])

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

import time

# --------- Before Training ----------
total_start = time.time()

#Training Function

def train_epoch(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

#Evaluation Function

def evaluate(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.eps' % (name), bbox_inches='tight')

train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_and_save_model(model_name, model, train_loader, val_loader, test_loader, criterion, optimizer, epochs):
    train_losses = []
    val_accs = []
    test_accs = []

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss = train_epoch(train_loader, model, criterion, optimizer)
        val_acc = evaluate(val_loader, model)
        test_acc = evaluate(test_loader, model)

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Time: {epoch_time:.2f} sec")
        
    training_end = time.time()
    training_time = training_end - training_start
    print(f"\nTotal training time: {training_time/60:.2f} minutes "
          f"({training_time:.2f} seconds)")
        
    # Save model
    torch.save(model.state_dict(), model_name + ".pth")
    print(f"Model saved as {model_name}")

    final_test_acc = evaluate(test_loader, model)

    # Save training time and metrics in a file
    with open(model_name + "_metrics.txt", "w") as f:
        f.write(f"Training time: {training_time} seconds\n")
        f.write(f"Final Test Accuracy: {final_test_acc}\n")
        f.write("Epoch,Train Loss,Val Accuracy,Test Accuracy\n")
        for epoch in range(epochs):
            f.write(f"{epoch+1},{train_losses[epoch]:.4f},{val_accs[epoch]:.4f},{test_accs[epoch]:.4f}\n")

    return train_losses, val_accs, test_accs, training_time

epochs_count = 200
epochs = torch.arange(1, epochs_count + 1)

model_configurations = [
    {
        "name": "SimpleCNN-NoPool-NoSoftmax",
        "model": SimpleCNN(use_softmax=False),
        "optimizer": optim.Adam,
        "learning_rate": 0.001,
        "no_maxpool": True,
        "no_softmax": True
    },
    {
        "name": "SimpleCNN-NoPool-Softmax",
        "model": SimpleCNN(use_softmax=True),
        "optimizer": optim.Adam,
        "learning_rate": 0.001,
        "no_maxpool": True,
        "no_softmax": False
    },
    {
        "name": "SimpleCNN-WithPool-NoSoftmax",
        "model": SimpleCNNWithPooling(use_softmax=False),
        "optimizer": optim.Adam,
        "learning_rate": 0.001,
        "no_maxpool": False,
        "no_softmax": True
    },
    {
        "name": "SimpleCNN-WithPool-Softmax",
        "model": SimpleCNNWithPooling(use_softmax=True),
        "optimizer": optim.Adam,
        "learning_rate": 0.001,
        "no_maxpool": False,
        "no_softmax": False
    }
]

for config in model_configurations:
    print(f"\nTraining model: {config['name']}")
    model = config['model'].to(device)
    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    train_losses, val_accs, test_accs, training_time = train_and_save_model(
        model_name=config['name'],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs_count
    )

    # Plotting
    plot(epochs, train_losses, ylabel='Loss', name='{}-training-loss'.format(config['name']))
    plot(epochs, val_accs, ylabel='Accuracy', name='{}-validation-accuracy'.format(config['name']))
    plot(epochs, test_accs, ylabel='Accuracy', name='{}-test-accuracy'.format(config['name']))