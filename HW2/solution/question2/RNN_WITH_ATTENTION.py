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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feature_size, n_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        assert feature_size % n_heads == 0, "feature_size must be divisible by n_heads"
        
        self.feature_size = feature_size
        self.n_heads = n_heads
        self.head_dim = feature_size // n_heads

        # Linear transformations for Q, K, V
        self.query = nn.Linear(feature_size, feature_size)
        self.key = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)
        
        # Output projection
        self.out_proj = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Apply linear transformations and reshape for multi-head
        # (batch, seq_len, feature_size) -> (batch, n_heads, seq_len, head_dim)
        queries = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask (if provided) - expand for all heads
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        out = torch.matmul(attention_weights, values)  # (batch, n_heads, seq_len, head_dim)
        
        # Concatenate heads: (batch, seq_len, feature_size)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_size)
        
        # Final projection
        output = self.out_proj(out)

        return output


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
        self.attention = MultiHeadSelfAttention(hidden_dim, n_heads=n_attention_heads)

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
        
        attention_out = self.attention(rnn_outputs, mask=mask)  # (batch, seq_len, hidden_dim)
        
        # Pool over sequence dimension to get (batch, hidden_dim)
        context = attention_out.mean(dim=1)
        
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
    'learning_rate': 0.00043594007889544996,
    'embedding_dim': 32,
    'hidden_dim': 256,
    'n_layers': 2,
    'dropout': 0.11160536907441138,
    'bidirectional': True,
    'n_attention_heads': 1  # Simple attention with 1-2 heads
}
model, train_losses, val_losses, test_spearman = train_best_model(best_params, num_epochs=20)