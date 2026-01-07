from utils_w_masking import load_rnacompete_data, masked_mse_loss, masked_spearman_correlation, configure_seed
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import itertools
import matplotlib.pyplot as plt
import copy
import json

# --- CONFIGURATION ---
MODEL_NAME = "TransformerEncoder"
configure_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PLOTTING & INTERPRETATION TOOLS ---

class AttentionLogger:
    def __init__(self, model):
        self.attn_weights = {}
        self.hooks = []
        for i, layer in enumerate(model.transformer_blocks):
            hook = layer.self_attn.register_forward_hook(self.get_hook(f"Layer_{i}"))
            self.hooks.append(hook)

    def get_hook(self, name):
        def hook(module, input, output):
            self.attn_weights[name] = output[1].detach().cpu()
        return hook

    def remove_hooks(self):
        for h in self.hooks: h.remove()

def plot_scatter(model, loader, name):
    """
    Plots and returns the raw data for external logging.
    """
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for seq, aff, mask in loader:
            if seq.dim() == 3: seq = seq.argmax(dim=-1)
            seq = seq.to(device).long()
            out, _ = model(seq)

            # --- FIX: SAFE FLATTENING ---
            # .view(-1) forces the tensor to be a 1D vector (handling batch size 1 correctly)
            # .tolist() converts it directly to Python floats, preventing numpy shape issues
            all_preds.extend(out.view(-1).cpu().tolist())
            all_trues.extend(aff.view(-1).cpu().tolist())

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(all_trues, all_preds, alpha=0.3, s=10)
    lims = [min(min(all_trues), min(all_preds)), max(max(all_trues), max(all_preds))]
    plt.plot(lims, lims, 'r-', alpha=0.75)
    plt.title(f'{name}: True vs Predicted')
    plt.xlabel("True Affinity"); plt.ylabel("Predicted Affinity")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{name}_scatter.pdf"); plt.close()

    return all_trues, all_preds

def visualize_encoder_attention(model, loader, name):
    """
    Plots and returns the specific attention sample for logging.
    """
    print("Generating attention maps...")
    model.eval()
    logger = AttentionLogger(model)

    sequences, _, _ = next(iter(loader))
    if sequences.dim() == 3: sequences = sequences.argmax(dim=-1)
    sequences = sequences.to(device).long()

    with torch.no_grad():
        _ = model(sequences)

    vocab = ['A', 'C', 'G', 'U', '_']
    seq_idx = sequences[0].cpu().numpy()
    seq_letters = [vocab[i] for i in seq_idx]

    real_len = sum([1 for x in seq_letters if x != '_'])
    seq_letters = seq_letters[:real_len]

    layers = sorted(list(logger.attn_weights.keys()))

    # Data container for logging
    attention_log_data = {
        "sequence": seq_letters,
        "layers": {}
    }

    if not layers:
        logger.remove_hooks()
        return attention_log_data

    # Plotting
    fig, axes = plt.subplots(len(layers), 2, figsize=(10, 5 * len(layers)))
    if len(layers) == 1: axes = np.expand_dims(axes, 0)

    for row, layer_name in enumerate(layers):
        attn_matrix = logger.attn_weights[layer_name][0] # Get batch 0

        if attn_matrix.dim() != 3: continue

        # Save to log (convert tensor -> numpy -> list)
        attention_log_data["layers"][layer_name] = attn_matrix.numpy()[:, :real_len, :real_len].tolist()

        # Average Head Attention Plot
        avg_attn = attn_matrix.mean(dim=0).numpy()[:real_len, :real_len]
        ax1 = axes[row][0]
        ax1.imshow(avg_attn, cmap='viridis', aspect='auto')
        ax1.set_title(f"{layer_name} | Average Attention")
        ax1.set_xticks(range(real_len)); ax1.set_xticklabels(seq_letters, fontsize=8)
        ax1.set_yticks(range(real_len)); ax1.set_yticklabels(seq_letters, fontsize=8)

        # Head 0 Attention Plot
        head_attn = attn_matrix[0].numpy()[:real_len, :real_len]
        ax2 = axes[row][1]
        ax2.imshow(head_attn, cmap='magma', aspect='auto')
        ax2.set_title(f"{layer_name} | Head 0 Only")
        ax2.set_xticks(range(real_len)); ax2.set_xticklabels(seq_letters, fontsize=8)
        ax2.set_yticks(range(real_len)); ax2.set_yticklabels(seq_letters, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{name}_encoder_attention.pdf")
    plt.close()
    logger.remove_hooks()

    return attention_log_data

# --- ARCHITECTURE (TransformerEncoder) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_key_padding_mask=None):
        # average_attn_weights=False is crucial for getting per-head weights
        src2, weights = self.self_attn(src, src, src,
                                       key_padding_mask=src_key_padding_mask,
                                       need_weights=True,
                                       average_attn_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention_projection = nn.Linear(d_model, 1)

    def forward(self, x, padding_mask):
        attn_scores = self.attention_projection(x).squeeze(-1)
        attn_scores = attn_scores.masked_fill(padding_mask, -float('inf'))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        weighted_sum = torch.sum(x * attn_weights, dim=1)
        return weighted_sum, attn_weights.squeeze(-1)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size=5, d_model=128, nhead=8, num_layers=2, dim_feedforward=512, output_dim=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=4)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.attn_pooling = AttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x, return_weights=False):
        padding_mask = (x == 4)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x, src_key_padding_mask=padding_mask)

        x, weights = self.attn_pooling(x, padding_mask)
        x = self.norm(x)
        x = self.fc(x)

        if return_weights: return x.squeeze(), weights
        return x.squeeze(), weights

# --- TRAINING ---

def train_epoch(loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for seq, aff, mask in loader:
        if seq.dim() == 3: seq = seq.argmax(dim=-1)
        seq, aff, mask = seq.to(device).long(), aff.to(device), mask.to(device)
        optimizer.zero_grad()
        out, _ = model(seq)
        loss = masked_mse_loss(out, aff, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader, model):
    model.eval()
    total_loss, spearman_vals = 0.0, []
    with torch.no_grad():
        for seq, aff, mask in loader:
            if seq.dim() == 3: seq = seq.argmax(dim=-1)
            seq, aff, mask = seq.to(device).long(), aff.to(device), mask.to(device)
            out, _ = model(seq)
            total_loss += masked_mse_loss(out, aff, mask).item()
            spearman_vals.append(masked_spearman_correlation(out, aff, mask).item())
    return total_loss / len(loader), sum(spearman_vals) / max(len(spearman_vals), 1)

def run_grid_search(train_loader, val_loader):
    param_grid = {
        'learning_rate': [1e-3, 5e-4],
        'd_model': [64, 128],
        'num_layers': [2],
        'nhead': [4, 8],
        'dim_feedforward': [256],
        'dropout': [0.1]
    }
    keys, combinations = list(param_grid.keys()), list(itertools.product(*param_grid.values()))
    best_spearman, best_params = -float('inf'), None

    print(f"Grid Search: {len(combinations)} configurations.")
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        if params['d_model'] % params['nhead'] != 0: continue

        model = TransformerEncoder(5, **{k:v for k,v in params.items() if k != 'learning_rate'}).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        # INCREASED TO 20 EPOCHS
        best_run_spearman = -float('inf')
        for epoch in range(20):
            train_epoch(train_loader, model, optimizer)
            _, val_spearman = evaluate(val_loader, model)
            best_run_spearman = max(best_run_spearman, val_spearman)

        print(f"Config {i+1}: {params} | Best Val Spearman: {best_run_spearman:.4f}")

        if best_run_spearman > best_spearman:
            best_spearman = best_run_spearman
            best_params = params.copy()

    return best_params

# --- MAIN ---

if __name__ == "__main__":
    print("Loading Data...")
    train_loader = DataLoader(load_rnacompete_data('RBFOX1', 'train'), batch_size=64, shuffle=True)
    val_loader = DataLoader(load_rnacompete_data('RBFOX1', 'val'), batch_size=256, shuffle=False)
    test_loader = DataLoader(load_rnacompete_data('RBFOX1', 'test'), batch_size=256, shuffle=False)

    print(f"\n--- Starting Grid Search for {MODEL_NAME} ---")
    best_params = run_grid_search(train_loader, val_loader)
    print(f"\nWINNING PARAMS: {best_params}")

    print("\n--- Starting Final Training (Extended) ---")
    model = TransformerEncoder(5, **{k:v for k,v in best_params.items() if k != 'learning_rate'}).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    train_losses, val_losses = [], []
    best_val_spearman = -float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    patience = 20
    max_epochs = 200
    counter = 0

    for epoch in range(max_epochs):
        t_loss = train_epoch(train_loader, model, optimizer)
        v_loss, v_spear = evaluate(val_loader, model)
        train_losses.append(t_loss); val_losses.append(v_loss)

        if v_spear > best_val_spearman:
            best_val_spearman = v_spear
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if (epoch+1) % 10 == 0:
            print(f"Ep {epoch+1:03d}: Train Loss={t_loss:.4f}, Val Spear={v_spear:.4f}")

        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("\n--- Generating Plots & Saving Results ---")
    model.load_state_dict(best_model_wts)
    test_loss, test_spear = evaluate(test_loader, model)
    print(f"FINAL TEST SPEARMAN: {test_spear:.4f}")

    # A. Learning Curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title(f"{MODEL_NAME} Training Dynamics")
    plt.legend(); plt.savefig(f"{MODEL_NAME}_loss.pdf"); plt.close()

    # B. Scatter Plot (capture data)
    scatter_trues, scatter_preds = plot_scatter(model, test_loader, MODEL_NAME)

    # C. Attention Maps (capture data)
    attn_data = visualize_encoder_attention(model, test_loader, MODEL_NAME)

    # D. Save Everything to JSON
    results_data = {
        "model_name": MODEL_NAME,
        "best_params": best_params,
        "metrics": {
            "final_test_spearman": test_spear,
            "final_test_loss": test_loss,
            "best_val_spearman": best_val_spearman
        },
        "learning_curve": {
            "train_losses": train_losses,
            "val_losses": val_losses
        },
        "scatter_plot_data": {
            "true_affinities": scatter_trues,
            "predicted_affinities": scatter_preds
        },
        "attention_analysis_sample": attn_data
    }

    json_filename = f"{MODEL_NAME}_results.json"
    with open(json_filename, "w") as f:
        json.dump(results_data, f, indent=4)

    print(f"Results and plot data saved to {json_filename}")