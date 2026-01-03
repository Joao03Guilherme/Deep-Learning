
import torch.nn as nn
import torch

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
        attn_scores = torch.bmm(query, context.transpose(1, 2))
        alignment = torch.softmax(attn_scores, 2)
        c = torch.bmm(alignment, context)
        attn_h_t = self.mlp(torch.cat([c, query], dim=2))
        return attn_h_t, alignment


class RNN(nn.Module):
    def __init__(self, vocab_size=5, embedding_dim=64, hidden_dim=128, output_dim=1, n_layers=2, bidirectional=True, dropout=0.2, use_attention=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Layer normalization for embedding
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.rnn = nn.LSTM(embedding_dim,
                            hidden_dim // 2 if bidirectional else hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)

        # Attention mechanism
        if use_attention:
            self.attention = DotProdAttention(hidden_dim)

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
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)

        enc_output, (hidden, _) = self.rnn(embedded)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        if self.use_attention:
            # Use hidden state as query, enc_output as context
            query = hidden.unsqueeze(1)  # (batch, 1, hidden_dim)
            attn_output, _ = self.attention(query, enc_output)  # (batch, 1, hidden_dim)
            hidden = attn_output.squeeze(1)  # (batch, hidden_dim)
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)