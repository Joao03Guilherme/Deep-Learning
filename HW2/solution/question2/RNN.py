from utils_w_masking import load_rnacompete_data, masked_mse_loss, masked_spearman_correlation, configure_seed
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import random
import numpy as np
import matplotlib.pyplot as plt  # Add matplotlib import

configure_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_d = load_rnacompete_data('RBFOX1', 'train')
val_d   = load_rnacompete_data('RBFOX1', 'val')
test_d  = load_rnacompete_data('RBFOX1', 'test')

train_loader = DataLoader(train_d, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_d, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_d, batch_size=256, shuffle=False)


class RNN(nn.Module):
    def __init__(self, vocab_size=5, embedding_dim=64, hidden_dim=128, output_dim=1, n_layers=2, bidirectional=True, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(embedding_dim,
                            hidden_dim // 2 if bidirectional else hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)

        # More expressive output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        _, hidden = self.rnn(embedded)  # GRU returns (output, hidden) instead of (output, (hidden, cell))

        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)
    

"""Training and Evaluation Functions"""
def train_epoch(loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for sequence, affinity, mask in loader:
        if sequence.dim() == 3: 
            is_ambiguous = (sequence.max(dim=-1).values == sequence.min(dim=-1).values)
            sequence = sequence.argmax(dim=-1)
            sequence[is_ambiguous] = 4

        sequence = sequence.to(device).long()
        affinity = affinity.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        outputs = model(sequence).squeeze()
        loss = masked_mse_loss(outputs, affinity, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader, model):
    model.eval()
    total_loss = 0.0
    spearman_vals = []
    with torch.no_grad():
        for sequence, affinity, mask in loader:
            if sequence.dim() == 3: 
                is_ambiguous = (sequence.max(dim=-1).values == sequence.min(dim=-1).values)
                sequence = sequence.argmax(dim=-1)
                sequence[is_ambiguous] = 4  

            sequence = sequence.to(device).long()
            affinity = affinity.to(device)
            mask = mask.to(device)

            outputs = model(sequence).squeeze()
            total_loss += masked_mse_loss(outputs, affinity, mask).item()
            spearman_vals.append(masked_spearman_correlation(outputs, affinity, mask).item())

    avg_loss = total_loss / len(loader)
    avg_spearman = sum(spearman_vals) / max(len(spearman_vals), 1)
    return avg_loss, avg_spearman


def run_random_search_rnn(n_iter=20):
    # Define hyperparameter search space
    param_space = {
        'learning_rate': (1e-4, 1e-3, 'log'),      # log-uniform
        'embedding_dim': [32, 64, 128],             # discrete choices
        'hidden_dim': [64, 128, 256],               # discrete choices
        'n_layers': [1, 2, 3],                      # discrete choices
        'dropout': (0.0, 0.5, 'uniform'),           # uniform
        'bidirectional': [True, False],             # discrete choices
    }
    
    num_epochs = 10
    best_spearman = -float('inf')
    best_params = None
    results = []
    
    print(f"Running random search with {n_iter} iterations")
    
    for i in range(n_iter):
        # Sample hyperparameters
        params = {}
        for key, value in param_space.items():
            if isinstance(value, list):
                params[key] = random.choice(value)
            elif isinstance(value, tuple):
                low, high, scale = value
                if scale == 'log':
                    params[key] = 10 ** random.uniform(np.log10(low), np.log10(high))
                else:
                    params[key] = random.uniform(low, high)
        
        print(f"\n[{i+1}/{n_iter}] Testing: {params}")
        
        # Create model with current hyperparameters
        model = RNN(
            vocab_size=5,
            embedding_dim=params['embedding_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=1,
            n_layers=params['n_layers'],
            bidirectional=params['bidirectional'],
            dropout=params['dropout'],
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Train for num_epochs
        best_val_spearman_epoch = -float('inf')
        for epoch in range(num_epochs):
            train_loss = train_epoch(train_loader, model, optimizer)
            val_loss, val_spearman = evaluate(val_loader, model)
            
            if val_spearman > best_val_spearman_epoch:
                best_val_spearman_epoch = val_spearman
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Spearman={val_spearman:.4f}")
        
        results.append((params.copy(), best_val_spearman_epoch))
        print(f"  Best Val Spearman: {best_val_spearman_epoch:.4f}")
        
        # Track overall best
        if best_val_spearman_epoch > best_spearman:
            best_spearman = best_val_spearman_epoch
            best_params = params.copy()
    
    # Print summary
    print("\n" + "="*60)
    print("RANDOM SEARCH RESULTS SUMMARY")
    print("="*60)
    for params, spearman in sorted(results, key=lambda x: x[1], reverse=True)[:5]:
        print(f"Spearman={spearman:.4f} | {params}")
    
    print("\n" + "="*60)
    print(f"BEST HYPERPARAMETERS (Val Spearman: {best_spearman:.4f}):")
    print("="*60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    return best_params, best_spearman

def plot_combined(epochs, train_losses, val_losses, name=''):
    """Plot training loss and validation loss"""
    fig, ax1 = plt.subplots()
    
    # Plot both losses on the same axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, train_losses, color='tab:red', label='Training Loss')
    ax1.plot(epochs, val_losses, color='tab:blue', label='Validation Loss')
    ax1.tick_params(axis='y')
    
    # Add legend
    ax1.legend(loc='upper right')
    
    fig.tight_layout()
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')
    plt.clf()

def train_best_model(best_params, num_epochs=20):
    """Train the model with best hyperparameters and generate plots"""
    print("\n" + "="*60)
    print("TRAINING BEST MODEL")
    print("="*60)
    print(f"Hyperparameters: {best_params}")
    print(f"Training for {num_epochs} epochs...")
    
    # Create model with best hyperparameters
    model = RNN(
        vocab_size=5,
        embedding_dim=best_params['embedding_dim'],
        hidden_dim=best_params['hidden_dim'],
        output_dim=1,
        n_layers=best_params['n_layers'],
        bidirectional=best_params['bidirectional'],
        dropout=best_params['dropout'],
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    train_losses = []
    val_losses = []
    val_spearmans = []
    
    best_val_spearman = -float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(train_loader, model, optimizer)
        val_loss, val_spearman = evaluate(val_loader, model)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_spearmans.append(val_spearman)
        
        # Save best model
        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Spearman: {val_spearman:.4f}")
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_loss, test_spearman = evaluate(test_loader, model)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Validation Spearman: {best_val_spearman:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Spearman: {test_spearman:.4f}")
    
    # Plot training curves (now plotting train_losses and val_losses)
    epochs_range = list(range(1, num_epochs + 1))
    plot_combined(epochs_range, train_losses, val_losses, name='RNN_training_curves')
    print("\nPlot saved as 'RNN_training_curves.pdf'")
    
    # Save metrics to file
    with open("RNN_metrics.txt", "w") as f:
        f.write(f"Best Hyperparameters: {best_params}\n")
        f.write(f"Best Validation Spearman: {best_val_spearman:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Spearman: {test_spearman:.4f}\n\n")
        f.write("Epoch,Train Loss,Val Loss,Val Spearman\n")
        for epoch in range(num_epochs):
            f.write(f"{epoch+1},{train_losses[epoch]:.4f},{val_losses[epoch]:.4f},{val_spearmans[epoch]:.4f}\n")
    print("Metrics saved as 'RNN_metrics.txt'")
    
    return model, train_losses, val_losses, test_spearman

# Run random search and then train best model
best_params = {
    'learning_rate': 0.00043594007889544996,
    'embedding_dim': 32,
    'hidden_dim': 256,
    'n_layers': 2,
    'dropout': 0.11160536907441138,
    'bidirectional': True
}
model, train_losses, val_losses, test_spearman = train_best_model(best_params, num_epochs=20)