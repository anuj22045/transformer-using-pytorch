"""
WHAT IS MASKED SELF-ATTENTION?

Masked Self-Attention is a type of self-attention used in the Transformer Decoder where a token is allowed to look only at:

itself and the tokens before it — not future tokens.

For example, suppose the Decoder is generating: Je t'aime les chats
When predicting the next word, it should not be allowed to see words that come later.

Normal Self-Attention
Token 1 → Token 1, 2, 3, 4
Token 2 → Token 1, 2, 3, 4
Token 3 → Token 1, 2, 3, 4
Token 4 → Token 1, 2, 3, 4

Every token can see every token.
Masked Self-Attention
Token 1 → Token 1
Token 2 → Token 1, 2
Token 3 → Token 1, 2, 3
Token 4 → Token 1, 2, 3, 4

Future tokens are blocked.

----------------------------------------------------------------------------------------------------------------------------
WHY DO WE NEED IT?

The Transformer Decoder generates the output one token at a time.
Suppose we're translating:

English:
I love cats

French:
Je t'aime les chats

While generating:
Je

the model doesn't know:
t'aime les chats yet.

If we allowed the Decoder to look at the complete target sentence during training, it could simply look at the future answer.
That would be cheating.

----------------------------------------------------------------------------------------------------------------------------

"""
# PRACTICAL IMPLEMENTATION
import torch
import torch.nn.functional as F


# ---------------------------------------------------------
# Create Query, Key and Value
# ---------------------------------------------------------

Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = torch.tensor([
    [10.0, 20.0],
    [30.0, 40.0],
    [50.0, 60.0]
])


# ---------------------------------------------------------
# Calculate Attention Scores
# ---------------------------------------------------------

scores = Q @ K.T # @ means matrix multiplication

print("Attention Scores:")
print(scores)

print("\nScores Shape:")
print(scores.shape)


# ---------------------------------------------------------
# Scale the Scores
# ---------------------------------------------------------

d_k = Q.size(-1)

scores = scores / torch.sqrt(
    torch.tensor(d_k, dtype=torch.float32)
)

print("\nScaled Scores:")
print(scores)


# ---------------------------------------------------------
# Create Causal Mask
# ---------------------------------------------------------

mask = torch.triu(
    torch.ones(3, 3),
    diagonal=1
)

print("\nMask:")
print(mask)


# ---------------------------------------------------------
# Apply Mask
# ---------------------------------------------------------

scores = scores.masked_fill(
    mask == 1,
    float("-inf")
)

print("\nMasked Scores:")
print(scores)


# ---------------------------------------------------------
# Softmax
# ---------------------------------------------------------

attention_weights = F.softmax(
    scores,
    dim=-1
)

print("\nAttention Weights:")
print(attention_weights)


# ---------------------------------------------------------
# Weighted Values
# ---------------------------------------------------------

output = attention_weights @ V

print("\nOutput:")
print(output)

print("\nOutput Shape:")
print(output.shape)

"""
------------------------------------------
CORE LOGIC 
------------------------------------------
scores = Q @ K.T
scores = scores / torch.sqrt(...)
attention_weights = F.softmax(scores, dim=-1)
output = attention_weights @ V

=============================================================
-----------------------------------------------------------------------------
MEMORY NOTE
-----------------------------------------------------------------------------
I need the Decoder to avoid future tokens → Masked Self-Attention
I need to block future positions → Causal Mask
I need forbidden attention to become zero → -∞ → Softmax → 0
I need autoregressive generation → Masked Self-Attention
"""