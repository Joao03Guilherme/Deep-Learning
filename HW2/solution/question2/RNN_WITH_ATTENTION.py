from utils_w_masking import load_rnacompete_data, masked_mse_loss, masked_spearman_correlation, configure_seed
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

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

        # 2. Apply Masking
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


def visualize_attention(model, loader, num_samples=5, name='attention_visualization'):
    """Visualize attention weights for sample sequences with high and low affinity"""
    model.eval()
    
    # Nucleotide mapping for visualization
    idx_to_nuc = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'N'}
    
    # Collect all data from loader to find high/low affinity samples
    all_sequences = []
    all_affinities = []
    all_masks = []
    
    for sequence, affinity, mask in loader:
        if sequence.dim() == 3:
            is_ambiguous = (sequence.max(dim=-1).values == sequence.min(dim=-1).values)
            sequence = sequence.argmax(dim=-1)
            sequence[is_ambiguous] = 4
        # Ensure proper dimensions before appending
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)
        if affinity.dim() == 0:
            affinity = affinity.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        all_sequences.append(sequence)
        all_affinities.append(affinity.flatten())
        all_masks.append(mask)
    
    all_sequences = torch.cat(all_sequences, dim=0)
    all_affinities = torch.cat(all_affinities, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Sort by affinity and select 1 high and 1 low
    sorted_indices = torch.argsort(all_affinities)
    low_indices = sorted_indices[:1]  # 1 lowest affinity
    high_indices = sorted_indices[-1:]  # 1 highest affinity
    selected_indices = torch.cat([high_indices, low_indices])  # High first, then low
    
    # Select the samples
    sequence = all_sequences[selected_indices].to(device).long()
    affinity = all_affinities[selected_indices]
    mask = all_masks[selected_indices].to(device)
    
    # Forward pass to get attention weights
    with torch.no_grad():
        embedded = model.embedding(sequence)
        embedded = model.dropout(embedded)
        rnn_outputs, _ = model.rnn(embedded)
        _, attention_weights = model.attention(rnn_outputs, mask=mask)
    
    # Convert to numpy
    attention_weights = attention_weights.cpu().numpy()
    sequence = sequence.cpu().numpy()
    mask = mask.cpu().numpy()
    affinity = affinity.cpu().numpy()
    
    # Plot attention for each sample
    num_samples = 2  # 1 high + 1 low
    n_heads = attention_weights.shape[2]
    labels = ['High Affinity', 'Low Affinity']
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2.5 * num_samples))
    
    for i in range(num_samples):
        ax = axes[i]
        seq = sequence[i]
        attn = attention_weights[i]
        seq_mask = mask[i]
        aff_value = affinity[i]
        
        # Get actual sequence length
        if seq_mask.ndim == 0 or len(seq_mask) == 1:
            seq_len = len(seq)
        else:
            seq_len = int((seq_mask > 0).sum()) if seq_mask.sum() > 0 else len(seq)
        
        if seq_len == 0:
            seq_len = len(seq)
        
        seq = seq[:seq_len]
        attn = attn[:seq_len]
        
        # Create heatmap
        im = ax.imshow(attn.T, aspect='auto', cmap='viridis', interpolation='nearest')
        
        # Set labels
        ax.set_ylabel('Attention Head', fontsize=10)
        ax.set_xlabel('Sequence Position', fontsize=10)
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f'Head {h+1}' for h in range(n_heads)], fontsize=9)
        
        # Add sequence characters on top
        seq_labels = [idx_to_nuc.get(int(s), '?') for s in seq]
        if seq_len <= 50:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(seq_labels, fontsize=9, fontweight='bold')
        else:
            step = max(1, seq_len // 35)
            ax.set_xticks(range(0, seq_len, step))
            ax.set_xticklabels([seq_labels[j] for j in range(0, seq_len, step)], fontsize=9, fontweight='bold')
        
        ax.set_title(f'{labels[i]} (Affinity: {aff_value:.3f})', fontsize=11, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Attention Weight', shrink=0.8)
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f'{name}.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    print(f"Attention visualization saved as '{name}.pdf'")
    
    # Also plot average attention
    plot_average_attention_selected(attention_weights, sequence, mask, affinity, labels, name=f'{name}_average')


def plot_average_attention_selected(attention_weights, sequence, mask, affinity, labels, name='attention_average'):
    """Plot average attention weights for pre-selected samples"""
    idx_to_nuc = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'N'}
    
    num_samples = len(sequence)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        seq = sequence[i]
        attn = attention_weights[i]
        seq_mask = mask[i]
        aff_value = affinity[i]
        
        # Get actual sequence length
        if seq_mask.ndim == 0 or len(seq_mask) == 1:
            seq_len = len(seq)
        else:
            seq_len = int((seq_mask > 0).sum()) if seq_mask.sum() > 0 else len(seq)
        
        if seq_len == 0:
            seq_len = len(seq)
        
        seq = seq[:seq_len]
        attn_avg = attn[:seq_len].mean(axis=1)
        
        positions = np.arange(seq_len)
        bars = ax.bar(positions, attn_avg, color='steelblue', alpha=0.7)
        
        threshold = np.percentile(attn_avg, 90)
        for j, (bar, weight) in enumerate(zip(bars, attn_avg)):
            if weight >= threshold:
                bar.set_color('crimson')
        
        ax.set_ylabel('Avg Attention', fontsize=10)
        ax.set_xlabel('Position', fontsize=10)
        
        seq_labels = [idx_to_nuc.get(int(s), '?') for s in seq]
        ax.set_title(f'{labels[i]} (Affinity: {aff_value:.3f})', fontsize=11, fontweight='bold')
        
        if seq_len <= 50:
            ax.set_xticks(positions)
            ax.set_xticklabels(seq_labels, fontsize=9, fontweight='bold')
        else:
            step = max(1, seq_len // 35)
            ax.set_xticks(range(0, seq_len, step))
            ax.set_xticklabels([seq_labels[j] for j in range(0, seq_len, step)], fontsize=9, fontweight='bold')
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f'{name}.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    print(f"Average attention visualization saved as '{name}.pdf'")


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

    # Plot training curves
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

# Best hyperparameters
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

# Visualize attention weights
print("\nGenerating attention visualizations...")
visualize_attention(model, test_loader, num_samples=5, name='RNN_attention_visualization')