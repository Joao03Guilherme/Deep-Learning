import torch
import torch.nn as nn


"""
Postitional Encoding
Transformer Encoder
"""

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class RNATransformer(nn.Module):
    """
    Args:
        input_dim: Number of input channels (5 for A, C, G, U, NaN).
        d_model: The internal dimension (embedding size) of the transformer.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Hidden dimension of the feedforward network.
        dropout: Dropout rate.
    """
    def __init__(self, vocab_size=5, d_model=64, nhead=8, num_layers=3, dim_feedforward=256, output_dim=1, dropout=0.1):
        super().__init__()

        # Embedding layers ; up-projection and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout) # Dropout after positional encoding for regularization

        # Transformer Encoder - Attention is all you need layer 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final MLP for regression
        self.fc = nn.Linear(d_model, d_model // 2)
        self.gelu = nn.GELU()
        self.output_layer = nn.Linear(d_model // 2, output_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor : (batch_size, seq_length, 5)
        """
        padding_mask = (x == 4)  # Assuming 4 is the padding index

        # Embedding and positional encoding
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # (batch_size, seq_length, d_model)

        # Masked Mean Pooling

        
        input_mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (batch_size, seq_length, 1)

        # Sum up the vectors of valid nucleotides
        summed = torch.sum(x * input_mask_expanded, dim=1)  # (batch_size, d_model)

        # Count of valid nucleotides for each sequence
        counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)  # (batch_size, 1)

        # Compute mean by dividing summed features by counts
        x = summed / counts  # (batch_size, d_model)

        # Final MLP
        x = self.fc(x)
        x = self.gelu(x)
        x = self.output_layer(x)

        return x