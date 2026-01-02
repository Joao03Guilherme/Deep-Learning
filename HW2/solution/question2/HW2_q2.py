import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import itertools

# Assuming these exist in your utils.py
from utils import load_rnacompete_data, masked_mse_loss, configure_seed, masked_spearman_correlation, plot

# --- CONFIGURATION ---
configure_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {DEVICE}")

# ==========================================
# 1. MODEL DEFINITIONS
# ==========================================

# --- ATTENTION MECHANISM (for RNN) ---
class DotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Project hidden state to query vector
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, rnn_output):
        # rnn_output: [Batch, Seq_Len, Hidden_Dim]
        # Compute attention scores
        # Score = v^T * tanh(W_q * h_i) (simplified Luong/Bahdanau mix for regression)
        # Or simple Dot Product against a learnable context vector
        
        scores = torch.matmul(rnn_output, self.v) # [Batch, Seq_Len]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1) # [Batch, Seq_Len, 1]
        
        # Context vector = Weighted sum
        context = torch.sum(weights * rnn_output, dim=1) # [Batch, Hidden_Dim]
        return context, weights

# --- MODEL A & B: RNN (Base) & RNN + Attention ---
class RnaRNN(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=64, hidden_dim=64, n_layers=1, dropout=0.2, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False # Single direction as per simplicity, can be True
        )
        
        if self.use_attention:
            self.attention = DotProductAttention(hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Seq_Len]
        embedded = self.dropout(self.embedding(x))
        
        # rnn_output: [Batch, Seq_Len, Hidden_Dim]
        rnn_output, (h_n, c_n) = self.rnn(embedded)
        
        if self.use_attention:
            representation, _ = self.attention(rnn_output)
        else:
            # Take last time step
            representation = h_n[-1] 
            
        return self.fc(representation)

# --- MODEL C: TRANSFORMER ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, Emb_Dim]
        return x + self.pe[:, :x.size(1)]

class RnaTransformer(nn.Module):
    def __init__(self, vocab_size=4, embedding_dim=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=n_heads, 
            dim_feedforward=embedding_dim*4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        self.fc = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Seq_Len]
        src = self.embedding(x)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        # output: [Batch, Seq_Len, Emb_Dim]
        output = self.transformer_encoder(src)
        
        # Global Average Pooling
        output = output.mean(dim=1) 
        
        return self.fc(output)

# ==========================================
# 2. TRAINING & TUNING UTILS
# ==========================================

def train_epoch(loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for sequence, affinity, mask in loader:
        if sequence.dim() == 3: 
            sequence = sequence.argmax(dim=-1)
        
        sequence = sequence.to(DEVICE).long()
        affinity = affinity.to(DEVICE)
        mask = mask.to(DEVICE)
        
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
            if sequence.dim() == 3: sequence = sequence.argmax(dim=-1)
            sequence = sequence.to(DEVICE).long()
            affinity = affinity.to(DEVICE)
            mask = mask.to(DEVICE)
            
            outputs = model(sequence).squeeze()
            total_loss += masked_mse_loss(outputs, affinity, mask).item()
            spearman_vals.append(masked_spearman_correlation(outputs, affinity, mask).item())
            
    avg_loss = total_loss / len(loader)
    avg_spearman = sum(spearman_vals) / max(len(spearman_vals), 1)
    return avg_loss, avg_spearman

def hyperparameter_tuning(model_class, param_grid, train_loader, val_loader, fixed_args={}):
    """
    Simple Grid Search.
    param_grid: dict of list, e.g., {'lr': [0.001, 0.01], 'hidden_dim': [32, 64]}
    """
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_score = -1.0 # Spearman is -1 to 1, higher is better
    best_params = None
    best_model_state = None
    
    print(f"\n--- Tuning {model_class.__name__} ---")
    
    for params in combinations:
        # Extract LR for optimizer, rest for model init
        lr = params.pop('lr', 0.001)
        
        # Initialize model with combined fixed args and tuning params
        init_args = {**fixed_args, **params}
        model = model_class(**init_args).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Quick train for tuning (e.g., 5 epochs)
        print(f"Testing params: {params}, lr={lr} ...", end="")
        for _ in range(5):
            train_epoch(train_loader, model, optimizer)
        
        val_loss, val_spear = evaluate(val_loader, model)
        print(f" Val Spearman: {val_spear:.4f}")
        
        if val_spear > best_score:
            best_score = val_spear
            best_params = {**params, 'lr': lr}
            best_model_state = model.state_dict()
            
    print(f"Best Params: {best_params} | Best Val Spearman: {best_score:.4f}")
    return best_params

def full_training(model_class, best_params, train_loader, val_loader, test_loader, fixed_args={}, epochs=20):
    lr = best_params.pop('lr')
    init_args = {**fixed_args, **best_params}
    
    model = model_class(**init_args).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_spear': []}
    
    print(f"\nStarting Full Training for {model_class.__name__}...")
    
    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(train_loader, model, optimizer)
        v_loss, v_spear = evaluate(val_loader, model)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_spear'].append(v_spear)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Train Loss: {t_loss:.4f} | Val Spearman: {v_spear:.4f}")
            
    test_loss, test_spear = evaluate(test_loader, model)
    return model, history, test_spear

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Load Data
    print("Loading Data (RBFOX1)...")
    train_d = load_rnacompete_data('RBFOX1', 'train')
    val_d   = load_rnacompete_data('RBFOX1', 'val')
    test_d  = load_rnacompete_data('RBFOX1', 'test')

    train_loader = DataLoader(train_d, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_d, batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_d, batch_size=256, shuffle=False)

    # ----------------------------------------
    # PART 1: Tune & Train RNN (Baseline)
    # ----------------------------------------
    rnn_grid = {
        'hidden_dim': [32, 64],
        'lr': [0.001, 0.01]
    }
    rnn_fixed = {'vocab_size': 4, 'embedding_dim': 64, 'use_attention': False}
    
    best_rnn_params = hyperparameter_tuning(RnaRNN, rnn_grid, train_loader, val_loader, rnn_fixed)
    _, hist_rnn, score_rnn = full_training(RnaRNN, best_rnn_params, train_loader, val_loader, test_loader, rnn_fixed)

    # ----------------------------------------
    # PART 1: Tune & Train Transformer
    # ----------------------------------------
    trans_grid = {
        'n_layers': [1, 2],
        'n_heads': [2, 4],
        'lr': [0.001, 0.0005]
    }
    trans_fixed = {'vocab_size': 4, 'embedding_dim': 64}
    
    best_trans_params = hyperparameter_tuning(RnaTransformer, trans_grid, train_loader, val_loader, trans_fixed)
    _, hist_trans, score_trans = full_training(RnaTransformer, best_trans_params, train_loader, val_loader, test_loader, trans_fixed)

    # ----------------------------------------
    # PART 2: Tune & Train RNN + Attention
    # ----------------------------------------
    attn_grid = {
        'hidden_dim': [32, 64],
        'lr': [0.001, 0.01]
    }
    attn_fixed = {'vocab_size': 4, 'embedding_dim': 64, 'use_attention': True}
    
    best_attn_params = hyperparameter_tuning(RnaRNN, attn_grid, train_loader, val_loader, attn_fixed)
    _, hist_attn, score_attn = full_training(RnaRNN, best_attn_params, train_loader, val_loader, test_loader, attn_fixed)

    # ----------------------------------------
    # PLOTTING & REPORTING
    # ----------------------------------------
    epochs_x = range(1, 21)
    
    # Q2.1 Plot: RNN vs Transformer
    plot(epochs_x, {
        "RNN Train": hist_rnn['train_loss'],
        "RNN Val": hist_rnn['val_loss'],
        "Transformer Train": hist_trans['train_loss'],
        "Transformer Val": hist_trans['val_loss']
    }, filename="q2_1_models_comparison.png")
    
    # Q2.2 Plot: RNN vs RNN+Attn
    plot(epochs_x, {
        "RNN Base Val Loss": hist_rnn['val_loss'],
        "RNN Attn Val Loss": hist_attn['val_loss']
    }, filename="q2_2_attention_impact.png")

    print("\n=== FINAL TEST RESULTS ===")
    print(f"RNN (Baseline): {score_rnn:.4f}")
    print(f"Transformer:    {score_trans:.4f}")
    print(f"RNN + Attn:     {score_attn:.4f}")