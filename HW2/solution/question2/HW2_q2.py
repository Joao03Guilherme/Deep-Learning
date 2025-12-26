
import torch
from torch.utils.data import DataLoader
from utils import load_rnacompete_data
from utils import masked_mse_loss
from utils import configure_seed
from utils import masked_spearman_correlation
from utils import plot

configure_seed(42)

# 1. Load Data for a specific protein (e.g., 'RBFOX1', 'PTB', 'A1CF')
# This returns a PyTorch TensorDataset ready for training
train_dataset = load_rnacompete_data(protein_name='RBFOX1', split='train')
val_dataset   = load_rnacompete_data(protein_name='RBFOX1', split='val')
test_dataset  = load_rnacompete_data(protein_name='RBFOX1', split='test')

# 2. Wrap in a standard PyTorch DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# MODEL DEFINITION

import torch.nn as nn

class DotProdAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )

    def forward(self, query, context):
        """
        query: batch x tgt_length x hidden_size
        context: batch x src_length x hidden_size
        """
        tgt_batch, tgt_len, tgt_hidden = query.size()
        src_batch, src_len, src_hidden = context.size()
        attn_scores = torch.bmm(query, context.transpose(1, 2))
        alignment = torch.softmax(attn_scores, 2)
        c = torch.bmm(alignment, context)
        attn_h_t = self.mlp(torch.cat([c, query], dim=2))
        return attn_h_t, alignment


import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, attn=None):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(embedding_dim,
                            hidden_dim//2 if bidirectional else hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.attention = attn

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        embedded = self.dropout(embedded)
        enc_output, (hidden, _) = self.rnn(embedded)

        alignment = None
        if self.attention is not None:
            if self.bidirectional:
                hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).unsqueeze(1)
            else:
                hidden_cat = hidden[-1,:,:].unsqueeze(1)
            attn_output, alignment = self.attention(hidden_cat, enc_output)
            hidden = attn_output.squeeze(1)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
                
        return self.fc(hidden)
    
class TransformerNN(torch.nn.Module):
    def __init__(self, input_size=4, d_model=64, nhead=8, num_layers=2, output_size=1):
        raise NotImplementedError("TransformerNN is not implemented yet.")

    def forward(self, x):
        raise NotImplementedError("TransformerNN is not implemented yet.")
    
    
def train_epoch(loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for sequence, affinity, mask in loader:
        sequence = sequence.argmax(dim=-1).to(device).long()  # (B,L)
        affinity = affinity.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        outputs = model(sequence)
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
            sequence = sequence.argmax(dim=-1).to(device).long()  # (B,L)
            affinity = affinity.to(device)
            mask = mask.to(device)
            outputs = model(sequence)
            batch_loss = masked_mse_loss(outputs, affinity, mask)
            total_loss += batch_loss.item()
            spearman = masked_spearman_correlation(outputs, affinity, mask)
            spearman_vals.append(spearman.item())
    avg_loss = total_loss / len(loader)
    avg_spearman = float(sum(spearman_vals) / max(len(spearman_vals), 1))
    return avg_loss, avg_spearman

attn = DotProdAttention(hidden_size=32)
model = RNN(vocab_size=4, embedding_dim=64, hidden_dim=32, output_dim=1, n_layers=2, bidirectional=False, dropout=0.0, attn=attn).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 30

train_losses = []
val_losses = []
val_spearmans = []
test_losses = []
test_spearmans = []

for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch(train_loader, model, optimizer)
    val_loss, val_spear = evaluate(val_loader, model)
    test_loss, test_spear = evaluate(test_loader, model)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_spearmans.append(val_spear)
    test_losses.append(test_loss)
    test_spearmans.append(test_spear)

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_spearman={val_spear:.4f} | test_loss={test_loss:.4f} | test_spearman={test_spear:.4f}")

# Final test evaluation
test_loss, test_spear = evaluate(test_loader, model)
print(f"Test | loss={test_loss:.4f} | spearman={test_spear:.4f}")

# Optional: plot metrics
epochs = list(range(1, num_epochs + 1))
plot(epochs, {"train_loss": train_losses, "val_loss": val_losses, "val_spearman": val_spearmans}, filename="training_metrics.png")

