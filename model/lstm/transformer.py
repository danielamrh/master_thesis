import torch
import torch.nn as nn
import math
from config_amass import * # Use config_amass.py for consistency

class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding, as used in "Attention Is All You Need".
    Made robust to handle inputs longer than the loaded buffer length, 
    which is common when changing SEQUENCE_LENGTH after saving a checkpoint.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Registering as buffer ensures it's saved with the model state_dict
        self.register_buffer('pe', pe) 
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, sequence_length, embedding_dim]
        """
        # x is [batch_size, seq_len, embed_dim]
        current_seq_len = x.size(1)

        # Check if the existing buffer is large enough
        if current_seq_len > self.pe.size(0):
            # This handles the case where a model trained on SEQUENCE_LENGTH=40 (pe.size(0)=41)
            # is now being run with SEQUENCE_LENGTH=80 (current_seq_len=81).
            # We must dynamically create a larger PE for the current batch size.
            print(f"Warning: Positional Encoding buffer size ({self.pe.size(0)}) is smaller than "
                  f"current sequence length ({current_seq_len}). Dynamically extending PE.")
            
            # Recalculate PE up to the required length
            d_model = x.size(2)
            # Ensure the new PE tensor is on the same device as the input tensor 'x'
            position = torch.arange(current_seq_len, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device) * (-math.log(10000.0) / d_model))
            
            new_pe = torch.zeros(current_seq_len, 1, d_model, device=x.device)
            new_pe[:, 0, 0::2] = torch.sin(position * div_term)
            new_pe[:, 0, 1::2] = torch.cos(position * div_term)
            
            # Use the new, larger PE
            pe_to_add = new_pe.transpose(0, 1) # [1, current_seq_len, embed_dim]
        else:
            # Use the registered buffer and slice it to the current sequence length
            # self.pe.transpose(0, 1) is [1, max_len, embed_dim]
            pe_to_add = self.pe.transpose(0, 1)[:, :current_seq_len, :]
        
        # Add the positional encoding
        x = x + pe_to_add
        
        return self.dropout(x)


class ArmPoseTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int = INPUT_SIZE, 
                 output_dim: int = OUTPUT_SIZE, 
                 embed_dim: int = EMBED_DIM, 
                 n_heads: int = NUM_HEADS, 
                 n_layers: int = NUM_TRANSFORMER_LAYERS, 
                 dropout: float = DROPOUT,
                 seq_len: int = SEQUENCE_LENGTH):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.input_fc = nn.Linear(input_dim, embed_dim)

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # --- POSITIONAL ENCODING (max_len is now seq_len + 1 to account for CLS token) ---
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=seq_len + 1)
        
        # Transformer Encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        # Output Head
        self.output_fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, 25)
        
        # 1. Project input
        # -> (batch_size, seq_len, embed_dim)
        x = self.input_fc(x)

        # 2. Prepend the [CLS] token
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # Concatenate: (batch_size, seq_len + 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Add positional encoding
        # -> (batch_size, seq_len + 1, embed_dim)
        x = self.pos_encoder(x)
        
        # 4. Pass through Transformer Encoder
        # -> (batch_size, seq_len + 1, embed_dim)
        x = self.transformer_encoder(x)
        
        # 5. Get the output of the [CLS] token (the first token)
        # -> (batch_size, embed_dim)
        cls_output = x[:, 0, :]
        
        # 6. Pass through the final output head
        # -> (batch_size, 6)
        linear_out = self.output_fc(cls_output)
        output = torch.tanh(linear_out)      
        
        return output