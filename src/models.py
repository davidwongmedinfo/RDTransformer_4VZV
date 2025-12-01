import torch
import torch.nn as nn
import math

# -----------------------------
# Sinusoidal positional encoding  
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, embedding_dim, max_length=2048):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)

        # Register as buffer so it is moved to GPU with model.to(device) but not trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encodings to the input embeddings.
        x: [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        # Ensure dtype & device match input
        pe = self.pe[:, :seq_len].to(dtype=x.dtype, device=x.device)
        x = x + pe
        return x

# -----------------------------
# Transformer encoder model (pretrain)
# -----------------------------
class TransformerEncoderModel_Pretrain(nn.Module):
    """
    Transformer encoder for sequence classification.
    - Uses src_key_padding_mask to prevent attention on padding tokens (index 0).
    - Uses masked mean pooling to produce sequence-level representation.
    """
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes, dropout_rate, max_length, padding_idx=0):
        super(TransformerEncoderModel_Pretrain, self).__init__()
        self.padding_idx = padding_idx

        # Token embedding (padding_idx ignored in gradient updates & masked)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)   

        # Sinusoidal positional encoding 
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim, max_length)
        
        # Transformer encoder stack (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Initialize weights and ensure padding token embedding row is zero
        self.apply(self._init_weights)
        if getattr(self.embedding, "padding_idx", None) is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()
    
    def _init_weights(self, module):
        """Module-wise initialization: linear weights via Xavier, embedding normal then zero padding row."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "padding_idx", None) is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, x):
        """
        x: LongTensor (B, T) of token indices (0 reserved for padding)
        Returns logits: (B, num_classes)
        """
        # Padding mask for transformer 
        padding_mask = (x == self.padding_idx)

        # Token embeddings 
        x = self.embedding(x)  # [B, T, D]
        
        # Add positional encodings  
        x = self.positional_encoding(x)
        
        # Transformer encoder with padding mask
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # (B, T, D)

        # Masked mean pooling: average only over non-padding tokens
        nonpad_mask = (~padding_mask).unsqueeze(-1).type_as(x)  # (B, T, 1)
        denom = nonpad_mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
        pooled = (x * nonpad_mask).sum(dim=1) / denom  # (B, D)  

        # Classifier
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits

# -----------------------------
# Transformer encoder model (finetune)
# -----------------------------
class TransformerEncoderModel_Finetune(nn.Module):
    """
    Transformer encoder for sequence classification.
    - Uses src_key_padding_mask to prevent attention on padding tokens (index 0).
    - Optionally computes a reference embedding (from a reference sequence) and uses differential embeddings.
    - Uses masked max pooling to produce sequence-level representation.
    """
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes, dropout_rate, max_length, reference_indices=None, padding_idx=0):
        super(TransformerEncoderModel_Finetune, self).__init__()
        self.padding_idx = padding_idx

        # Token embedding (padding_idx ignored in gradient updates & masked)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        
        # Sinusoidal positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim, max_length)
        
        # Transformer encoder stack (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Reference sequence indices, stored as a CPU tensor and compute embedding on demand
        if reference_indices is not None:
            # pad/truncate to max_length
            if len(reference_indices) != max_length:
                if len(reference_indices) > max_length:
                    reference_indices = reference_indices[:max_length]
                else:
                    reference_indices = reference_indices + [0] * (max_length - len(reference_indices))
            self._reference_indices = torch.tensor(reference_indices, dtype=torch.long)   
        else:
            self._reference_indices = None

        # cached reference embedding (1, T, D) — start empty
        self._reference_embed = None

    def _compute_reference_embed(self, device):
        """
        Compute and cache the reference embedding on the target device, using the current encoder weights.
        """
        assert self._reference_indices is not None, "No reference indices saved."
        reference_tensor = self._reference_indices.to(device)  # (T,)
        ref_padding_mask = (reference_tensor == 0).unsqueeze(0)  # (1, T)

        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                ref_emb = self.embedding(reference_tensor.unsqueeze(0))  # (1, T, D)
                ref_emb = self.positional_encoding(ref_emb)
                ref_emb = self.transformer_encoder(ref_emb, src_key_padding_mask=ref_padding_mask)
        finally:
            if was_training:
                self.train()
            self._reference_embed = ref_emb

        return self._reference_embed

    def forward(self, x):
        """
        x: LongTensor (B, T) of token indices (0 reserved for padding)
        Returns logits: (B, num_classes)
        """
        idx = x

        # Padding mask for transformer 
        padding_mask = (idx == self.padding_idx)

        # Token embeddings 
        x_emb = self.embedding(x)  # (B, T, D)
        
        # Add positional encodings  
        x_emb = self.positional_encoding(x_emb)

        # Transformer encoder with padding mask
        x_enc = self.transformer_encoder(x_emb, src_key_padding_mask=padding_mask)  # (B, T, D)

        # If reference provided, compute or reuse cached reference embedding (and move to correct device)
        if self._reference_indices is not None:
            device = x_enc.device
            if (self._reference_embed is None) or (self._reference_embed.device != device):
                self._compute_reference_embed(device)
            ref_embed = self._reference_embed.expand(x_enc.size(0), -1, -1)  # (B, T, D)
            diff_embed = x_enc - ref_embed
        else:
            diff_embed = x_enc

        # Mask out positions identical to reference (optional)
        if self._reference_indices is not None:
            # compare indices: idx (B,T) vs reference indices (1,T) -> same_mask (B,T)
            ref_idx = self._reference_indices.to(idx.device).unsqueeze(0)  # (1, T)
            same_mask = (idx == ref_idx)  # (B, T) True where token identical to reference
            diff_embed = diff_embed.masked_fill(same_mask.unsqueeze(-1), 0.0)

        # Masked pooling: use masked max pooling over non-padding positions
        neg_inf = torch.finfo(diff_embed.dtype).min
        diff_masked = diff_embed.masked_fill(padding_mask.unsqueeze(-1), neg_inf)
        pooled, _ = diff_masked.max(dim=1)  # (B, D)
        # Replace -inf (all-masked rows) with zeros to avoid NaNs
        pooled = torch.where(pooled == neg_inf, torch.zeros_like(pooled), pooled)

        out = self.dropout(pooled)
        out = self.classifier(out)  # (B, num_classes)
        return out

    def to(self, *args, **kwargs):
        # Ensure cached reference embedding is cleared when moving the model device
        res = super().to(*args, **kwargs)
        self._reference_embed = None
        return res

# -----------------------------
# Binary classification head (finetune)
# -----------------------------
class Head_SingleLogit(nn.Module):
    """
    A binary classification head with residual connection and layer normalization.
    This module outputs a single logit value for binary classification tasks.
    """
    def __init__(self, in_dim=64, hidden_dim=64, dropout=0.2):
        super().__init__()
        hidden_dim = max(8, min(hidden_dim, in_dim))
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.res_proj = nn.Linear(in_dim, hidden_dim) if hidden_dim != in_dim else nn.Identity()
        self.norm1 = nn.LayerNorm(hidden_dim)    
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        if isinstance(self.res_proj, nn.Linear):
            nn.init.xavier_uniform_(self.res_proj.weight); nn.init.zeros_(self.res_proj.bias)

    def forward(self, x):
        """
        x (torch.Tensor): Input tensor of shape (batch_size, in_dim)
        Returns logits: (B, 1)
        """
        identity = x
        x = self.fc1(x)
        x = x + self.res_proj(identity)    
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x