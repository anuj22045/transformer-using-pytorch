"""
Module 13 → Masked Self-Attention = the overall mechanism.
Module 14 → Causal Mask = the actual mask that prevents attention to future tokens.

WHAT IS A CAUSAL MASK?

A Causal Mask is a matrix used in the Transformer Decoder to prevent a token from attending to future tokens.
-----------------------------------------------------------------------------------------------------------------

WHY DO WE NEED IT?

The Transformer Decoder generates tokens from left to right.

Suppose we want to generate:
<START> Je t'aime les chats

When predicting:
Je

the model should only know:
<START>

It should NOT know: t'aime les chats

Otherwise, during training the model would already see the answer.
That would be future information leakage.
The causal mask prevents this.
------------------------------------------------------------------------------------------------------------------
IMPORTANT POINTS 
Remember these six points:

1.
Causal Mask blocks future tokens.

2.
It creates an upper-triangular blocked region.

3.
Blocked scores are usually replaced with:
-inf

4.
After Softmax:
-inf → 0 probability

5.
It prevents future information leakage.

6.
It is used by the Decoder's Masked Self-Attention.
"""

# -----------------------------------------------------------------------------
# PRACTICAL IMPLEMENTATION
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# Number of tokens
# ---------------------------------------------------------

sequence_length = 4


# ---------------------------------------------------------
# Create Upper-Triangular Mask
# ---------------------------------------------------------

causal_mask = torch.triu(
    torch.ones(sequence_length, sequence_length),
    diagonal=1
)
# triu means:

# Upper Triangle
# It keeps the values above the main diagonal and changes everything else to 0.

print("Causal Mask:")
print(causal_mask)


# ---------------------------------------------------------
# Attention Scores
# ---------------------------------------------------------

scores = torch.tensor([
    [2.1, 1.5, 0.8, 0.3],
    [1.2, 2.4, 1.1, 0.7],
    [0.5, 1.7, 2.8, 1.0],
    [0.3, 0.9, 1.5, 2.6]
])

print("Original Attention Scores:")
print(scores)


# ---------------------------------------------------------
# Create Causal Mask
# ---------------------------------------------------------

sequence_length = scores.size(0)

causal_mask = torch.triu(
    torch.ones(
        sequence_length,
        sequence_length
    ),
    diagonal=1
)

print("\nCausal Mask:")
print(causal_mask)


# ---------------------------------------------------------
# Apply Causal Mask
# ---------------------------------------------------------

masked_scores = scores.masked_fill(
    causal_mask == 1,
    float("-inf")
)

print("\nMasked Attention Scores:")
print(masked_scores)


# ---------------------------------------------------------
# Apply Softmax
# ---------------------------------------------------------

attention_weights = F.softmax(
    masked_scores,
    dim=-1
)

print("\nAttention Weights:")
print(attention_weights)

"""
-----------------------------------------------------------------------------
MEMORY NOTE
-----------------------------------------------------------------------------
I need to block future tokens → Causal Mask
I need an upper-triangular blocked region → torch.triu()
I need blocked scores to disappear after Softmax → -∞ → 0
I need to prevent future information leakage → Causal Mask

-----------------------------------------------------------------------------------------------------------
VERY IMPORTANT: TRAINING vs INFERENCE 

This is one of the most useful things to understand.

During Training

The entire target sequence may be available:

<START> Je t'aime les chats

But the causal mask prevents each position from seeing the future.

<START> → only <START>
Je      → <START>, Je
t'aime  → <START>, Je, t'aime
les     → <START>, Je, t'aime, les

Therefore, we can train the model on the whole sequence in parallel while still preserving autoregressive behavior.

During Inference

We generate one token at a time:

<START>
   ↓
Je
   ↓
t'aime
   ↓
les
   ↓
chats

There is no future token available yet.

But the same causal principle still applies.
"""