"""
WHAT IS QUERY, KEY, VALUE?

Query (Q), Key (K), and Value (V) are three different representations created from the token embeddings.

They help Self-Attention decide:

Query → What am I looking for?
Key → What information do I contain?
Value → What information should I provide?

In simple words:

Query → asks a question
Key   → tells what each token contains
Value → provides the actual information
----------------------------------------------------------------------------------------------------
WHY DO WE NEED IT?

In the previous module, we compared tokens directly.
But Transformers need a more flexible way to decide:
Which tokens are relevant to the current token?

Q, K, and V separate these three jobs.
This allows the model to learn what to search for, what matches, and what information to collect.
----------------------------------------------------------------------------------------------------

HOW DOES IT WORK?

Suppose:
I love cats

For the token love, we create:
Query for "love"
        ↓
"What information am I looking for?"

Keys of all tokens:
        ↓
I      → Key
love   → Key
cats   → Key
The Query of love is compared with every Key:

Query(love) · Key(I)
Query(love) · Key(love)
Query(love) · Key(cats)
This produces scores.

Then the model uses those scores to decide which Values are important.
Q × K
    ↓
Attention scores
    ↓
Which tokens are important?
    ↓
Use their V
    ↓
New representation
------------------------------------------------------------------------------------------------------

IMPORTANT CONCEPTS:
1. Q, K, V come from the same input

For Self-Attention:

Input Embeddings
      ↓
 ┌────┼────┐
 ↓    ↓    ↓
 Q    K    V

They are usually created using three different learnable weight matrices:

Q = XWQ
K = XWK
V = XWV

Where:

X  = input embeddings
WQ = Query weights
WK = Key weights
WV = Value weights

2. Query
Query represents: What is this token looking for?
Q = XWQ

3. Key
Key represents: What does this token offer for matching?
K = XWK

4. Value
Value represents: What actual information should this token contribute?
V = XWV

"""

# ============================================================
# MODULE 06: QUERY, KEY, VALUE (Q, K, V)
# ============================================================


import torch

# ------------------------------------------------------------
# 1. Input embeddings
# ------------------------------------------------------------

# Sentence:
# "I love cats"
#
# 3 tokens
# embedding dimension = 4

X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # I
    [0.0, 1.0, 0.0, 1.0],   # love
    [1.0, 1.0, 0.0, 0.0]    # cats
])

print("Input X:")
print(X)

print("\nX shape:")
print(X.shape)

# ------------------------------------------------------------
# 2. Create Q, K, V weight matrices
# ------------------------------------------------------------

embedding_dim = 4

WQ = torch.randn(embedding_dim, embedding_dim)
WK = torch.randn(embedding_dim, embedding_dim)
WV = torch.randn(embedding_dim, embedding_dim)

print("\nWQ shape:", WQ.shape)
print("WK shape:", WK.shape)
print("WV shape:", WV.shape)

# ------------------------------------------------------------
# 3. Calculate Query, Key, Value
# ------------------------------------------------------------

Q = torch.matmul(X, WQ)
K = torch.matmul(X, WK)
V = torch.matmul(X, WV)

print("\nQuery (Q):")
print(Q)

print("\nKey (K):")
print(K)

print("\nValue (V):")
print(V)


print("\nShapes:")
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

# ------------------------------------------------------------
# 4. Calculate Query-Key scores
# ------------------------------------------------------------

attention_scores = torch.matmul(
    Q,
    K.T
)

print("\nAttention Scores:")
print(attention_scores)

print("\nAttention Scores Shape:")
print(attention_scores.shape)

# ------------------------------------------------------------
# 5. Convert scores into attention weights
# ------------------------------------------------------------

attention_weights = torch.softmax(
    attention_scores,
    dim=-1
)

print("\nAttention Weights:")
print(attention_weights)

print("\nRow sums:")
print(attention_weights.sum(dim=-1))

# ------------------------------------------------------------
# 6. Weighted combination of Values
# ------------------------------------------------------------

output = torch.matmul(
    attention_weights,
    V
)

print("\nFinal Attention Output:")
print(output)

print("\nOutput Shape:")
print(output.shape)