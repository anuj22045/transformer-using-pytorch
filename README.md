# Transformer Using Pytorch - Complete Course

This repository contains a **module-wise practical study of the Transformer architecture using PyTorch**. It is created purely for learning and understanding how Transformers work internally, from basic components to the complete Encoder and Decoder architecture.

## Modules

### Module 01 — Transformer Architecture Overview
Introduction to the Transformer architecture and its overall Encoder–Decoder flow.

### Module 02 — Encoder vs Decoder
Understanding the different roles of the Encoder and Decoder.

### Module 03 — Token Embedding
Converting token IDs into learnable vector representations.

### Module 04 — Positional Encoding
Adding positional information so the Transformer can understand token order.

### Module 05 — Self-Attention
Understanding how tokens attend to and gather information from other tokens.

### Module 06 — Query, Key, Value (Q, K, V)
Understanding the three representations used to calculate attention.

### Module 07 — Scaled Dot-Product Attention
Calculating attention using Q, K, V, scaling, and Softmax.

### Module 08 — Multi-Head Attention
Using multiple attention heads to learn different relationships between tokens.

### Module 09 — Feed Forward Network
Processing each token representation using Linear layers and an activation function.

### Module 10 — Residual Connection
Adding the original input back to a layer's output to preserve information and improve learning.

### Module 11 — Layer Normalization
Normalizing token representations to keep the Transformer stable during training.

### Module 12 — Complete Encoder Block
Combining Self-Attention, Residual Connections, LayerNorm, and Feed Forward Network.

### Module 13 — Masked Self-Attention
Preventing the Decoder from using information from future tokens.

### Module 14 — Causal Mask
Creating a mask that blocks future positions during Decoder self-attention.

### Module 15 — Cross-Attention
Allowing the Decoder to use information from the Encoder output.

### Module 16 — Complete Decoder Block
Combining Masked Self-Attention, Cross-Attention, Residual Connections, LayerNorm, and FFN.

### Module 17 — Encoder + Decoder Integration
Connecting the Encoder and Decoder through Cross-Attention.

### Module 18 — Linear + Vocabulary Logits
Converting Decoder representations into scores for the vocabulary.

### Module 19 — Teacher Forcing
Understanding how target tokens are provided to the Decoder during training.

### Module 20 — Transformer Training Loop
Building the training process for the Transformer model.

### Module 21 — Padding Mask
Ignoring padding tokens during attention calculations.

### Module 22 — Attention Mask
Understanding different masks used to control attention behavior.

### Module 23 — Complete Transformer
Combining the major Transformer components into one complete architecture.

## How to Run

Install PyTorch:

```bash
pip install torch
```

Run any module independently:

```bash
python 01_transformer_architecture.py
```

Example:

```bash
python 12_complete_encoder_block.py
```

## Learning Order

Follow the modules in order (01 -> 23) for the best learning experience. Each module builds on concepts from previous ones.
