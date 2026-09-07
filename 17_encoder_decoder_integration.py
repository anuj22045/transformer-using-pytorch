# 17_encoder_decoder_integration.py
"""
WHAT IS ENCODER + DECODER INTEGRATION?
It means connecting the Encoder and Decoder so that the Decoder can use the information produced by the Encoder.

HOW DOES IT WORK?

Let's look at the complete flow.

English Sentence
    ↓
Token IDs
    ↓
Embedding
    ↓
Positional Encoding
    ↓
Encoder
    
Encoder Output
    │
    │
    ▼
    Decoder
    ↑
    │
Target Token IDs
    ↓
Embedding
    ↓
Positional Encoding
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------
# Encoder Block
# ---------------------------------------------------------

class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_ff, num_heads):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        # Self-Attention
        attention_output, _ = self.attention(
            x,
            x,
            x
        )

        # Residual + LayerNorm
        x = self.norm1(
            x + attention_output
        )

        # Feed Forward
        ff_output = self.linear1(x)
        ff_output = torch.relu(ff_output)
        ff_output = self.linear2(ff_output)

        # Residual + LayerNorm
        x = self.norm2(
            x + ff_output
        )

        return x


# ---------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------

class DecoderBlock(nn.Module):

    def __init__(self, d_model, d_ff, num_heads):

        super().__init__()

        # Masked Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # Cross-Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed Forward
        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, decoder_input, encoder_output):

        # -------------------------------------------------
        # 1. Causal Mask
        # -------------------------------------------------

        sequence_length = decoder_input.size(1)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=decoder_input.device
            ),
            diagonal=1
        ).bool()

        # -------------------------------------------------
        # 2. Masked Self-Attention
        # -------------------------------------------------

        self_attention_output, _ = self.self_attention(
            decoder_input,
            decoder_input,
            decoder_input,
            attn_mask=causal_mask
        )

        # Residual + LayerNorm
        x = self.norm1(
            decoder_input + self_attention_output
        )

        # -------------------------------------------------
        # 3. Cross-Attention
        # -------------------------------------------------

        cross_attention_output, _ = self.cross_attention(
            x,
            encoder_output,
            encoder_output
        )

        # Residual + LayerNorm
        x = self.norm2(
            x + cross_attention_output
        )

        # -------------------------------------------------
        # 4. Feed Forward
        # -------------------------------------------------

        ff_output = self.linear1(x)
        ff_output = torch.relu(ff_output)
        ff_output = self.linear2(ff_output)

        # Residual + LayerNorm
        x = self.norm3(
            x + ff_output
        )

        return x


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

batch_size = 2

encoder_sequence_length = 5
decoder_sequence_length = 3

d_model = 8
d_ff = 32
num_heads = 2


# ---------------------------------------------------------
# Create Inputs
# ---------------------------------------------------------

encoder_input = torch.randn(
    batch_size,
    encoder_sequence_length,
    d_model
)

decoder_input = torch.randn(
    batch_size,
    decoder_sequence_length,
    d_model
)


print("Encoder Input Shape:")
print(encoder_input.shape)

print("\nDecoder Input Shape:")
print(decoder_input.shape)


# ---------------------------------------------------------
# Create Encoder
# ---------------------------------------------------------

encoder = EncoderBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads
)


# ---------------------------------------------------------
# Create Decoder
# ---------------------------------------------------------

decoder = DecoderBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads
)


# ---------------------------------------------------------
# Encoder Processing
# ---------------------------------------------------------

encoder_output = encoder(
    encoder_input
)

print("\nEncoder Output Shape:")
print(encoder_output.shape)


# ---------------------------------------------------------
# Decoder Processing
# ---------------------------------------------------------

decoder_output = decoder(
    decoder_input,
    encoder_output
)

print("\nDecoder Output Shape:")
print(decoder_output.shape)