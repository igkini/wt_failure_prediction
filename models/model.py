import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import CodeDataembedding

def pool(enc_out, pooling='mean'):
    """
    enc_out: [B, L, d_model]
    returns: [B, d_model]
    """
    if pooling == 'mean':
        return enc_out.mean(dim=1)
    if pooling == 'last':
        return enc_out[:, -1]
    if pooling == 'cls':
        return enc_out[:, 0]              
    raise ValueError(f'Unknown pooling: {pooling}')

class WTFailureClassifier(nn.Module):

    def __init__(
        self,
        enc_in,  # dim of x_enc
        n_classes=10,
        code_vocab_size=311,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_ff=512,
        dropout=0.1,
        attn='prob',
        embed='timeF',
        freq='h',
        activation='gelu',
        pooling='mean',  # 'mean' | 'cls' | 'last'
        distil=True,
    ):
        super().__init__()
        self.pooling = pooling
        self.enc_embedding = CodeDataembedding(enc_in, d_model, embed, freq, code_vocab_size, dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(e_layers)
            ],
            [
                ConvLayer(d_model)  # keeps distil
                for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )

        self.fc1 = nn.Linear(d_model, d_model//2)
        self.dropout = nn.Dropout(dropout)  
        self.fc2 = nn.Linear(d_model//2, n_classes)

    def forward(self, x_enc, x_code,x_mark_enc=None, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc, x_code)  # [B,L,d]
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)  # [B,L,d]
        pooled = pool(enc_out,self.pooling) # [B,d]
        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)  # Apply dropout here

        logits = self.fc2(hidden)  # [batch, num_error_codes]
        
        return logits
