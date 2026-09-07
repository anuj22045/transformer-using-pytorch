# 12_complete_encoder_block.py
"""
A Transformer Encoder Block is a combination of several components that work together to understand the input sequence.

So far, we learned these separately:

Token Embedding
        ↓
Positional Encoding
        ↓
Self-Attention
        ↓
Q, K, V
        ↓
Scaled Dot-Product Attention
        ↓
Multi-Head Attention
        ↓
Feed Forward Network
        ↓
Residual Connection
        ↓
Layer Normalization
------------------------------------------------------------------------------------------------------------
WHY DO WE NEED IT?

The Encoder's job is:

Understand the input sentence and create rich contextual representations.
WHY DO WE NEED IT?

The Encoder's job is:

Understand the input sentence and create rich contextual representations.

For example:

Input:

"The cat is sitting on the mat."

The Encoder should understand relationships such as:

cat → sitting
cat → mat
sitting → mat
It does this using Self-Attention.

Then the Feed Forward Network processes each token's information further.
Residual Connections preserve information.
LayerNorm stabilizes the representations.
So each component has a different job:
-------------------------------------------------------------------------------------------------------------------------------
IMPORTANT POINTS ⭐
1. Encoder Block contains TWO major sublayers

The two main computational parts are:

1. Multi-Head Self-Attention
2. Feed Forward Network

And each one is followed by:
Residual Connection
+
Layer Normalization


2. Attention allows tokens to communicate
Self-Attention
        ↓
Tokens interact with each other

For example:
"The animal didn't cross the road because it was tired."
                         ↑
                       "it"
                         ↓
                    "animal"
Attention helps create these contextual relationships.

3. FFN works independently on each token
This is an important difference.
Attention:
Token 1 ←→ Token 2 ←→ Token 3
Tokens communicate.
FFN:
Token 1 → FFN
Token 2 → FFN
Token 3 → FFN
Each token is processed independently.

4. Residual Connections appear twice
Attention → Residual
FFN → Residual
So don't forget:
Attention
   ↓
Residual
   ↓
Norm

FFN
   ↓
Residual
   ↓
Norm

5. LayerNorm appears twice
Attention → Add → LayerNorm
FFN → Add → LayerNorm

6. Shape remains the same
Suppose:
Input = [batch, sequence_length, d_model]

Example:
[2, 5, 8]
After the complete Encoder Block:
Output = [2, 5, 8]
The shape doesn't change.
""" 

# -----------------------------------------------------------------------------
# PRACTICAL IMPLEMENTATION
# -----------------------------------------------------------------------------


import torch
import torch.nn as nn

# ---------------------------------------------------------
# Encoder Block
# ---------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        #Multi-head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = 2,
            batch_first = True
        )

        #Feed Froward Network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        #layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # -------------------------------------------------
        # 1. Multi-Head Self-Attention
        # -------------------------------------------------
        attention_output, attention_weights  = self.attention(
            x,x,x
        )
        # -------------------------------------------------
        # 2. Residual Connection + LayerNorm
        # -------------------------------------------------
        x = self.norm1(x + attention_output)

        # -------------------------------------------------
        # 3. Feed Forward Network
        # -------------------------------------------------

        ff_output = self.linear1(x)
        ff_output = torch.relu(ff_output)
        ff_output = self.linear2(ff_output)

        # -------------------------------------------------
        # 4. Residual Connection + LayerNorm
        # -------------------------------------------------
        x = self.norm2(x+ff_output)
        return x



# ---------------------------------------------------------
# Create Input
# ---------------------------------------------------------

batch_size =2
sequence_length = 3
d_model = 8
d_ff = 32

x = torch.rand(
    batch_size,
    sequence_length,
    d_model
)

print("Input Shape: ", x.shape)

# ---------------------------------------------------------
# Create Encoder Block
# ---------------------------------------------------------

encoder_block = EncoderBlock(
    d_model=d_model,
    d_ff=d_ff
)
# ---------------------------------------------------------
# Pass input through Encoder Block
# ---------------------------------------------------------

output = encoder_block(x)
print("\n Encoder output shape: ",output.shape)

"""
-----------------------------------------------------------------------------
MEMORY NOTE
-----------------------------------------------------------------------------
I need tokens to understand relationships → Multi-Head Self-Attention
I need to preserve original information → Residual Connection
I need stable representations → LayerNorm
I need to process each token → Feed Forward Network

-----------------------------------------------------------------------------
TRANSFORMER CONNECTION
-----------------------------------------------------------------------------
Input
    ↓  
Embedding
    ↓
Positional Encoding
    ↓
┌───────────────────────────┐
│ Complete Encoder Block ★  │
│                           │
│ Multi-Head Attention      │
│ ↓                         │
│ Add + LayerNorm           │
│ ↓                         │
│ Feed Forward              │
│ ↓                         │
│ Add + LayerNorm           │
└───────────────────────────┘
    ↓
Encoder
    ↓
Decoder
    ↓
Linear
    ↓
Output
"""