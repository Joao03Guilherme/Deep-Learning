from utils_w_masking import load_rnacompete_data, masked_mse_loss, masked_spearman_correlation, configure_seed
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

configure_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_d = load_rnacompete_data('RBFOX1', 'train')
val_d   = load_rnacompete_data('RBFOX1', 'val')
test_d  = load_rnacompete_data('RBFOX1', 'test')

train_loader = DataLoader(train_d, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_d, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_d, batch_size=256, shuffle=False)


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Two-layer MLP to calculate attention scores
        # We output 'n_heads' scores for every time step
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_heads, bias=False)
        )

    def forward(self, rnn_outputs, mask=None):
        # rnn_outputs: (batch, seq_len, hidden_dim)
        batch_size, seq_len, _ = rnn_outputs.size()

        # 1. Calculate Attention Scores
        # Output: (batch, seq_len, n_heads)
        scores = self.score_net(rnn_outputs)

        # 2. Apply Masking (ignore padding)
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, seq_len, 1) for broadcasting
            mask_expanded = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # 3. Calculate Weights (Softmax over time dimension)
        # weights: (batch, seq_len, n_heads)
        weights = F.softmax(scores, dim=1)

        # 4. Split RNN outputs into heads to match weights
        # (batch, seq_len, n_heads, head_dim)
        rnn_outputs_split = rnn_outputs.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # 5. Weighted Sum (Pooling)
        # weights needs unsqueeze to match head_dim: (batch, seq_len, n_heads, 1)
        weights_expanded = weights.unsqueeze(-1)

        # Sum over seq_len (dim 1)
        # Result: (batch, n_heads, head_dim)
        context_split = torch.sum(weights_expanded * rnn_outputs_split, dim=1)

        # 6. Concatenate heads back together
        # Result: (batch, hidden_dim)
        context = context_split.view(batch_size, self.n_heads * self.head_dim)

        return context, weights


class RNN(nn.Module):
    def __init__(self, vocab_size=5, embedding_dim=64, hidden_dim=128, output_dim=1, n_layers=2, bidirectional=True, dropout=0.2, n_attention_heads=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(embedding_dim,
                            hidden_dim // 2 if bidirectional else hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)

        # Multi-head attention
        self.attention = MultiHeadAttentionPooling(hidden_dim, n_heads=n_attention_heads)

        # More expressive output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, mask=None):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        rnn_outputs, _ = self.rnn(embedded)  # (batch, seq_len, hidden_dim)

        context, _ = self.attention(rnn_outputs, mask=mask)  # (batch, seq_len, hidden_dim)

        context = self.dropout(context)
        return self.fc(context)


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
        outputs = model(sequence, mask=mask).squeeze()
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

            outputs = model(sequence, mask=mask).squeeze()
            total_loss += masked_mse_loss(outputs, affinity, mask).item()
            spearman_vals.append(masked_spearman_correlation(outputs, affinity, mask).item())

    avg_loss = total_loss / len(loader)
    avg_spearman = sum(spearman_vals) / max(len(spearman_vals), 1)
    return avg_loss, avg_spearman


def plot_combined(epochs, train_losses, val_losses, name=''):
    """Plot training loss and validation loss"""
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, train_losses, color='tab:red', label='Training Loss')
    ax1.plot(epochs, val_losses, color='tab:blue', label='Validation Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')
    plt.clf()


def count_parameters(model):
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_best_model(best_params, num_epochs=20):
    """Train the model with best hyperparameters and generate plots"""
    print("\n" + "="*60)
    print("TRAINING BEST MODEL WITH ATTENTION")
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
        n_attention_heads=best_params.get('n_attention_heads', 1),
    ).to(device)

    print(f"Total trainable parameters: {count_parameters(model):,}")

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
    plot_combined(epochs_range, train_losses, val_losses, name='RNN_attention_training_curves')
    print("\nPlot saved as 'RNN_attention_training_curves.pdf'")

    # Save metrics to file
    with open("RNN_attention_metrics.txt", "w") as f:
        f.write(f"Best Hyperparameters: {best_params}\n")
        f.write(f"Best Validation Spearman: {best_val_spearman:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Spearman: {test_spearman:.4f}\n\n")
        f.write("Epoch,Train Loss,Val Loss,Val Spearman\n")
        for epoch in range(num_epochs):
            f.write(f"{epoch+1},{train_losses[epoch]:.4f},{val_losses[epoch]:.4f},{val_spearmans[epoch]:.4f}\n")
    print("Metrics saved as 'RNN_attention_metrics.txt'")

    return model, train_losses, val_losses, test_spearman

# Best hyperparameters (add n_attention_heads to control attention)
best_params = {
    'learning_rate': 0.0005,
    'embedding_dim': 32,
    'hidden_dim': 256,
    'n_layers': 2,
    'dropout': 0.1,
    'bidirectional': True,
    'n_attention_heads': 2  # Simple attention with 1-2 heads
}
model, train_losses, val_losses, test_spearman = train_best_model(best_params, num_epochs=20)