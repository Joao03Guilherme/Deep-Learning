#!/usr/bin/env python

# Deep Learning Homework 1 - Feedforward Network

import argparse
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
import pandas as pd

import time
import utils


class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_size, layers,
            activation_type, dropout, **kwargs):
        """ Define a vanilla multiple-layer FFN with `layers` hidden layers 
        Args:
            n_classes (int)
            n_features (int)
            hidden_size (int)
            layers (int)
            activation_type (str)
            dropout (float): dropout probability
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.activation_type = activation_type
        self.dropout = nn.Dropout(dropout)

        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'sparsemax':
            self.activation = nn.Sparsemax(dim=1)
        elif activation_type == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

        # Input layer
        self.layers.append(nn.Linear(n_features, hidden_size))
        self.layers.append(self.activation)
        self.layers.append(self.dropout)

        # Hidden layers
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.activation)
            self.layers.append(self.dropout)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, n_classes)
        

    def forward(self, x, **kwargs):
        """ Compute a forward pass through the FFN
        Args:
            x (torch.Tensor): a batch of examples (batch_size x n_features)
        Returns:
            scores (torch.Tensor)
        """
        for layer in self.layers:
            x = layer(x)
        scores = self.output_layer(x)
        return scores
    
    
def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """ Do an update rule with the given minibatch
    Args:
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        model (nn.Module): a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
    Returns:
        loss (float)
    """
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """ Predict the labels for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
    Returns:
        preds: (n_examples)
    """
    outputs = model(X)
    _, preds = torch.max(outputs, dim=1)
    return preds


@torch.no_grad()
def evaluate(model, X, y, criterion):
    """ Compute the loss and the accuracy for the given input
    Args:
        model (nn.Module): a PyTorch defined model
        X (torch.Tensor): (n_examples x n_features)
        y (torch.Tensor): gold labels (n_examples)
        criterion: loss function
    Returns:
        loss, accuracy (Tuple[float, float])
    """
    outputs = model(X)
    loss = criterion(outputs, y).item()
    _, preds = torch.max(outputs, dim=1)
    accuracy = (preds == y).float().mean().item()
    return loss, accuracy


def plot(epochs, plottables, filename=None, ylim=None):
    """Plot the plottables over the epochs.
    
    Plottables is a dictionary mapping labels to lists of values.
    """
    plt.clf()
    plt.xlabel('Epoch')
    for label, plottable in plottables.items():
        plt.plot(epochs, plottable, label=label)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def question22a():
    """
    Train a one-hidden-layer feedforward neural network for a varying number
    of hidden units. For each width, perform a small grid search
    over all combinations of the following hyperparameters: 
    - 4 learning rates
    - no dropout and dropout with p>0
    - no l2 regularization and l2 regularization with some weight decay >0

    We chose the Adam optimizer and ReLU activations for this exercise.
    """

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    hidden_unit_sizes = [16, 32, 64, 128, 256]
    lr_values = [0.1, 0.01, 0.001, 0.0001]
    dropout_values = [0.0, 0.5]
    l2_values = [0.0, 0.01]
    batch_size = 64
    epochs = 30
    optimizer = torch.optim.Adam
    criterion = nn.CrossEntropyLoss()
    activation = "relu"
    number_of_layers = 1

    data = utils.load_dataset('emnist-letters.npz')
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        generator=torch.Generator().manual_seed(42),
        num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    # Move data to device once
    train_X, train_y = dataset.X.to(device), dataset.y.to(device)
    dev_X, dev_y = dataset.dev_X.to(device), dataset.dev_y.to(device)
    test_X, test_y = dataset.test_X.to(device), dataset.test_y.to(device)

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    # Store results for all configurations
    results = []
    total_configs = len(hidden_unit_sizes) * len(lr_values) * len(dropout_values) * len(l2_values)
    current_config = 0

    for hidden_size in hidden_unit_sizes:
        for lr in lr_values:
            for dropout in dropout_values:
                for l2 in l2_values:
                    current_config += 1
                    print(f"[{current_config}/{total_configs}] Training with hidden_size={hidden_size}, lr={lr}, dropout={dropout}, l2={l2}")

                    model = FeedforwardNetwork(
                        n_classes,
                        n_feats,
                        hidden_size,
                        number_of_layers,
                        activation,
                        dropout
                    ).to(device)

                    optimizer_instance = optimizer(
                        model.parameters(), lr=lr, weight_decay=l2
                    )

                    # training loop
                    for epoch in range(epochs):
                        model.train()
                        for X_batch, y_batch in train_dataloader:
                            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                            train_batch(X_batch, y_batch, model, optimizer_instance, criterion)
                    
                    model.eval()
                    train_loss, train_acc = evaluate(model, train_X, train_y, criterion)
                    val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)
                    
                    print(f'Training accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}')
                    
                    # Store results
                    results.append({
                        'hidden_size': hidden_size,
                        'learning_rate': lr,
                        'dropout': dropout,
                        'l2_decay': l2,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc
                    })
                    
                    # Free up memory
                    del model, optimizer_instance
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Find best configuration for each hidden size
    best_configs = []
    for hidden_size in hidden_unit_sizes:
        subset = df[df['hidden_size'] == hidden_size]
        best_idx = subset['val_accuracy'].idxmax()
        best_configs.append(best_idx)
    
    df['best_for_width'] = False
    df.loc[best_configs, 'best_for_width'] = True
    
    # Sort by hidden_size and then by val_accuracy
    df = df.sort_values(['hidden_size', 'val_accuracy'], ascending=[True, False])
    
    # Create output directory and save CSV
    results_dir = os.path.join("Results", "FFN_results")
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'question22a_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    
    print(f"\nResults saved to: {csv_path}")
    print("\nBest configuration for each hidden size:")
    best_df = df[df['best_for_width'] == True]
    print(best_df.to_string(index=False))


def question22b():
    """
    Plot the training loss and the validation accuracy over epochs for the best 
    configuration found in question 2.2.a
    """

    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 64
    epochs = 30
    optimizer = torch.optim.Adam
    criterion = nn.CrossEntropyLoss()
    activation = "relu"
    number_of_layers = 1
    # Try to load best config from question22a CSV
    results_dir = os.path.join("Results", "FFN_results")
    csv_path = os.path.join(results_dir, 'question22a_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Pick the single best row overall by validation accuracy
        best_row = df.loc[df['val_accuracy'].idxmax()]
        hidden_size = int(best_row['hidden_size'])
        lr = float(best_row['learning_rate'])
        dropout = float(best_row['dropout'])
        l2 = float(best_row['l2_decay'])
        print(f"Loaded best config from CSV: hidden={hidden_size}, lr={lr}, dropout={dropout}, l2={l2}")
    else:
        # Fallback example values if CSV not found
        hidden_size = 16
        lr = 0.001
        dropout = 0.5
        l2 = 0.01
        print("question22a_results.csv not found. Using example fallback config.")

    data = utils.load_dataset('emnist-letters.npz')
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        generator=torch.Generator().manual_seed(42),
        num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    # Move data to device once
    train_X, train_y = dataset.X.to(device), dataset.y.to(device)
    dev_X, dev_y = dataset.dev_X.to(device), dataset.dev_y.to(device)
    test_X, test_y = dataset.test_X.to(device), dataset.test_y.to(device)

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    model = FeedforwardNetwork(
                        n_classes,
                        n_feats,
                        hidden_size,
                        number_of_layers,
                        activation,
                        dropout
                    ).to(device)
    
    optimizer_instance = optimizer(
                        model.parameters(), lr=lr, weight_decay=l2
                    )
    
    # Track metrics for plotting
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    plot_epochs = torch.arange(1, epochs + 1)

    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            loss = train_batch(X_batch, y_batch, model, optimizer_instance, criterion)
            epoch_train_losses.append(loss)
        
        model.eval()
        # Aggregate training loss over minibatches and compute full-set metrics
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        train_loss, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    model.eval()
    final_test_loss, final_test_acc = evaluate(model, test_X, test_y, criterion)
    print(f"Final Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_acc:.4f}")

    # Plot and save
    results_dir = os.path.join("Results", "FFN_results")
    os.makedirs(results_dir, exist_ok=True)

    config = (
        f"batch-{batch_size}-lr-{lr}-epochs-{epochs}-"
        f"hidden-{hidden_size}-dropout-{dropout}-l2-{l2}-"
        f"layers-{number_of_layers}-act-{activation}-opt-adam"
    )

    plot(plot_epochs, {"Train Loss": train_losses, "Valid Loss": val_losses},
         filename=os.path.join(results_dir, f"q22b-training-loss-{config}.pdf"))
    plot(plot_epochs, {"Valid Accuracy": val_accs},
         filename=os.path.join(results_dir, f"q22b-validation-accuracy-{config}.pdf"), ylim=(0, 1))
    
def question22c():
    """
    Produce a plot of the final training accuracy as a function of hidden-layer width.
    """

    # Open the CSV file generated in question 2.2.a
    results_dir = os.path.join("Results", "FFN_results")
    csv_path = os.path.join(results_dir, 'question22a_results.csv')
    df = pd.read_csv(csv_path)

    # Filter for best configurations only per hidden size
    best_df = df[df['best_for_width'] == True]

    # Plot
    plt.clf()
    plt.plot(best_df['hidden_size'], best_df['train_accuracy'], marker='o')
    plt.xlabel('Hidden Layer Width')
    plt.ylabel('Final Training Accuracy')
    plt.title('Final Training Accuracy vs Hidden Layer Width')
    plt.xscale('log', base=2)
    plt.xticks(best_df['hidden_size'])
    plt.ylim(0, 1)
    plt.grid(True)
    plot_path = os.path.join(results_dir, 'question22c-training-accuracy-vs-width.pdf')
    plt.savefig(plot_path, bbox_inches='tight')

def question23a():
    """
    Present a table with the highest validation accuracy during training for each depth
    """

    hidden_size = 32
    hidden_layers = [1, 3, 5, 7, 9]
    # Load best hyperparameters for this hidden_size from question22a CSV
    results_dir = os.path.join("Results", "FFN_results")
    csv_path = os.path.join(results_dir, 'question22a_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_for_width = df[(df['hidden_size'] == hidden_size) & (df['best_for_width'] == True)]
        if not best_for_width.empty:
            best_row = best_for_width.iloc[0]
            lr = float(best_row['learning_rate'])
            dropout = float(best_row['dropout'])
            l2 = float(best_row['l2_decay'])
            print(f"Loaded best config for hidden_size={hidden_size} from CSV: lr={lr}, dropout={dropout}, l2={l2}")
    optimizer = torch.optim.Adam
    criterion = nn.CrossEntropyLoss()
    activation = "relu"
    epochs = 30
    batch_size = 64

    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data = utils.load_dataset('emnist-letters.npz')
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        generator=torch.Generator().manual_seed(42),
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
    
    # Move data to device once
    train_X, train_y = dataset.X.to(device), dataset.y.to(device)
    dev_X, dev_y = dataset.dev_X.to(device), dataset.dev_y.to(device)

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    results = []
    for layer_size in hidden_layers:
        print(f"Training with {layer_size} hidden layers")
        model = FeedforwardNetwork(
            n_classes,
            n_feats,
            hidden_size,
            layer_size,
            activation,
            dropout
        ).to(device)

        optimizer_instance = optimizer(
            model.parameters(), lr=lr, weight_decay=l2
        )

        best_val_acc = 0.0

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                train_batch(X_batch, y_batch, model, optimizer_instance, criterion)

            model.eval()
            val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)
            best_val_acc = max(best_val_acc, val_acc)

        train_loss, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        results.append({
            'hidden_layers': layer_size,
            'best_val_accuracy': best_val_acc,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        print(f"Best validation accuracy for {layer_size} layers: {best_val_acc:.4f}")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    results_dir = os.path.join("Results", "FFN_results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'question23a_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nResults saved to: {csv_path}")


def question23b():
    """
    Plot the training loss curve and the validation accuracy curve over the
    30 epochs for the best depth found in question 23a.
    """

    # Open df generated in question 23a
    results_dir = os.path.join("Results", "FFN_results")
    csv_path = os.path.join(results_dir, 'question23a_results.csv')
    df = pd.read_csv(csv_path)

    # Find best depth
    best_row = df.loc[df['best_val_accuracy'].idxmax()]
    best_depth = best_row['hidden_layers']

    hidden_size = 32
    # Load best hyperparameters for this hidden_size from question22a CSV
    results_dir = os.path.join("Results", "FFN_results")
    csv_path = os.path.join(results_dir, 'question22a_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        best_for_width = df[(df['hidden_size'] == hidden_size) & (df['best_for_width'] == True)]
        if not best_for_width.empty:
            best_row = best_for_width.iloc[0]
            lr = float(best_row['learning_rate'])
            dropout = float(best_row['dropout'])
            l2 = float(best_row['l2_decay'])
            print(f"Loaded best config for hidden_size={hidden_size} from CSV: lr={lr}, dropout={dropout}, l2={l2}")
    optimizer = torch.optim.Adam
    criterion = nn.CrossEntropyLoss()
    activation = "relu"
    epochs = 30
    batch_size = 64

    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data = utils.load_dataset('emnist-letters.npz')
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        generator=torch.Generator().manual_seed(42),
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
    # Move data to device once
    train_X, train_y = dataset.X.to(device), dataset.y.to(device)
    dev_X, dev_y = dataset.dev_X.to(device), dataset.dev_y.to(device)

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        hidden_size,
        int(best_depth),
        activation,
        dropout
    ).to(device)

    optimizer_instance = optimizer(
        model.parameters(), lr=lr, weight_decay=l2
    )

    train_losses, val_losses = [], []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            loss = train_batch(X_batch, y_batch, model, optimizer_instance, criterion)
            epoch_train_losses.append(loss)
        
        model.eval()
        train_loss, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    # Plot using the plot function
    plot_epochs = torch.arange(1, epochs + 1)
    plot(plot_epochs, {"Train Loss": train_losses, "Valid Loss": val_losses},
         filename=os.path.join(results_dir, f"q23b-training-loss-depth-{int(best_depth)}.pdf"))

def question23c():
    """
    Create a plot of training accuracy as a function of depth based on the results from question 23a.
    """ 

    # Import the CSV file generated in question 23a
    results_dir = os.path.join("Results", "FFN_results")
    csv_path = os.path.join(results_dir, 'question23a_results.csv')
    df = pd.read_csv(csv_path)

    # Plot
    plt.clf()
    plt.plot(df['hidden_layers'], df['train_accuracy'], marker='o')
    plt.xlabel('Number of Hidden Layers (Depth)')
    plt.ylabel('Final Training Accuracy')
    plt.title('Final Training Accuracy vs Network Depth')
    plt.xscale('log', base=2)
    plt.xticks(df['hidden_layers'])
    plt.ylim(0, 1)
    plt.grid(True)
    plot_path = os.path.join(results_dir, 'question23c-training-accuracy-vs-depth.pdf')
    plt.savefig(plot_path, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=30, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=64, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_size', type=int, default=32)
    parser.add_argument('-layers', type=int, default=1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-data_path', type=str, default='emnist-letters.npz',)
    parser.add_argument('-model', type=str, default='ffn')
    opt = parser.parse_args()

    utils.configure_seed(seed=42)


    """
    data = utils.load_dataset(opt.data_path)
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    train_X, train_y = dataset.X, dataset.y
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    n_classes = torch.unique(dataset.y).shape[0]  # 26
    n_feats = dataset.X.shape[1]

    print(f"N features: {n_feats}")
    print(f"N classes: {n_classes}")

    # initialize the model
    model = FeedforwardNetwork(
        n_classes,
        n_feats,
        opt.hidden_size,
        opt.layers,
        opt.activation,
        opt.dropout
    )

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    start = time.time()

    model.eval()
    initial_train_loss, initial_train_acc = evaluate(model, train_X, train_y, criterion)
    initial_val_loss, initial_val_acc = evaluate(model, dev_X, dev_y, criterion)
    train_losses.append(initial_train_loss)
    train_accs.append(initial_train_acc)
    valid_losses.append(initial_val_loss)
    valid_accs.append(initial_val_acc)
    print('initial val acc: {:.4f}'.format(initial_val_acc))

    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        model.train()
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            epoch_train_losses.append(loss)

        model.eval()
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        _, train_acc = evaluate(model, train_X, train_y, criterion)
        val_loss, val_acc = evaluate(model, dev_X, dev_y, criterion)

        print('train loss: {:.4f} | val loss: {:.4f} | val acc: {:.4f}'.format(
            epoch_train_loss, val_loss, val_acc
        ))

        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    _, test_acc = evaluate(model, test_X, test_y, criterion)
    print('Final test acc: {:.4f}'.format(test_acc))

    # plot
    config = (
        f"batch-{opt.batch_size}-lr-{opt.learning_rate}-epochs-{opt.epochs}-"
        f"hidden-{opt.hidden_size}-dropout-{opt.dropout}-l2-{opt.l2_decay}-"
        f"layers-{opt.layers}-act-{opt.activation}-opt-{opt.optimizer}"
    )

    losses = {
        "Train Loss": train_losses,
        "Valid Loss": valid_losses,
    }

    # Showing initial loss as well
    plot_epochs = torch.arange(0, opt.epochs + 1)

    # Create output directory
    results_dir = os.path.join("Results", "FFN_results")
    os.makedirs(results_dir, exist_ok=True)

    plot(plot_epochs, losses, filename=os.path.join(results_dir, f'{opt.model}-training-loss-{config}.pdf'))
    print(f"Final Training Accuracy: {train_accs[-1]:.4f}")
    print(f"Best Validation Accuracy: {max(valid_accs):.4f}")
    val_accuracy = { "Valid Accuracy": valid_accs }
    plot(plot_epochs, val_accuracy, filename=os.path.join(results_dir, f'{opt.model}-validation-accuracy-{config}.pdf'))
    """

    question22a()
    question22b()
    question22c()
    question23a()
    question23b()
    question23c()

if __name__ == '__main__':
    main()
