# 16_complete_decoder_block.py
"""
WHAT IS A COMPLETE DECODER BLOCK?
A Transformer Decoder Block is made up of several parts that help the model
generate the output step by step using information from the Encoder.

Example:

English:
I love cats
    ↓
Encoder
    ↓
Encoder Output
    ↓
Decoder
    ↓
French:
Je t'aime les chats

The Decoder has two attention mechanisms:

1. Masked Self-Attention
2. Cross-Attention
And one major processing component:

3. Feed Forward Network
Each is combined with residual connections and LayerNorm.

-------------------------------------------------------------------------------------------------------------------
WHY DO WE NEED IT?

The Encoder understands the English sentence:
I love cats

But the Encoder doesn't generate the French sentence.
That's the Decoder's job.
The Decoder needs to:

1. Understand what it has generated so far
<START> Je t'aime
→ Masked Self-Attention

2. Look at the English sentence
I love cats
→ Cross-Attention

3. Process the combined information
→ Feed Forward Network

---------------------------------------------------------------------------------------------------------------------
COMPLETE DECODER ARCHITECTURE 

This is the most important diagram of this module:

Target Input
    ↓
Embedding
    ↓
Positional Encoding
    ↓
┌────────────────────────────────┐
│      DECODER BLOCK             │
│                                │
│ Masked Self-Attention          │
│          ↓                     │
│     Add + LayerNorm            │
│          ↓                     │
│ Cross-Attention ← Encoder Out  │
│          ↓                     │
│     Add + LayerNorm            │
│          ↓                     │
│ Feed Forward Network           │
│          ↓                     │
│     Add + LayerNorm            │
│          ↓                     │
│       Output                   │
└────────────────────────────────┘

----------------------------------------------------------------------------------------------------------
COMPLETE FLOW ⭐⭐⭐

Put everything together:

Target Input
    
Masked Self-Attention
    ↓
Add + LayerNorm
    ↓
Cross-Attention
    ↑
Encoder Output
    ↓
Add + LayerNorm
    ↓
Feed Forward Network
    ↓
Add + LayerNorm
    ↓
Decoder Output


"""

# -----------------------------------------------------------------------------
# PRACTICAL IMPLEMENTATION
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn


# ---------------------------------------------------------
# Complete Decoder Block
# ---------------------------------------------------------

class DecoderBlock(nn.Module):

    def __init__(self, d_model, d_ff, num_heads):

        super().__init__()

        # -------------------------------------------------
        # 1. Masked Self-Attention
        # -------------------------------------------------

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # -------------------------------------------------
        # 2. Cross-Attention
        # -------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )

        # -------------------------------------------------
        # 3. Feed Forward Network
        # -------------------------------------------------

        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )

        # -------------------------------------------------
        # 4. Layer Normalization
        # -------------------------------------------------

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, decoder_input, encoder_output):

        # -------------------------------------------------
        # 1. Masked Self-Attention
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

        self_attention_output, _ = self.self_attention(
            decoder_input,
            decoder_input,
            decoder_input,
            attn_mask=causal_mask
        )

        print(
            "Masked Self-Attention Shape:",
            self_attention_output.shape
        )

        # -------------------------------------------------
        # 2. Residual + LayerNorm
        # -------------------------------------------------

        x = self.norm1(
            decoder_input + self_attention_output
        )

        print(
            "After First Add + Norm:",
            x.shape
        )

        # -------------------------------------------------
        # 3. Cross-Attention
        # -------------------------------------------------

        cross_attention_output, _ = self.cross_attention(
            x,
            encoder_output,
            encoder_output
        )

        print(
            "Cross-Attention Shape:",
            cross_attention_output.shape
        )

        # -------------------------------------------------
        # 4. Residual + LayerNorm
        # -------------------------------------------------

        x = self.norm2(
            x + cross_attention_output
        )

        print(
            "After Second Add + Norm:",
            x.shape
        )

        # -------------------------------------------------
        # 5. Feed Forward Network
        # -------------------------------------------------

        ff_output = self.linear1(x)

        print(
            "After Linear 1:",
            ff_output.shape
        )

        ff_output = torch.relu(ff_output)

        ff_output = self.linear2(ff_output)

        print(
            "FFN Output Shape:",
            ff_output.shape
        )

        # -------------------------------------------------
        # 6. Residual + LayerNorm
        # -------------------------------------------------

        x = self.norm3(
            x + ff_output
        )

        print(
            "Final Decoder Output:",
            x.shape
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
# Encoder Output
# ---------------------------------------------------------

encoder_output = torch.randn(
    batch_size,
    encoder_sequence_length,
    d_model
)


# ---------------------------------------------------------
# Decoder Input
# ---------------------------------------------------------

decoder_input = torch.randn(
    batch_size,
    decoder_sequence_length,
    d_model
)


print("Encoder Output Shape:")
print(encoder_output.shape)

print("\nDecoder Input Shape:")
print(decoder_input.shape)


# ---------------------------------------------------------
# Create Decoder Block
# ---------------------------------------------------------

decoder_block = DecoderBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads
)


# ---------------------------------------------------------
# Pass Through Decoder Block
# ---------------------------------------------------------

output = decoder_block(
    decoder_input,
    encoder_output
)

print("\nFinal Decoder Output Shape:")
print(output.shape)