# 20_transformer_training_loop.py
"""
TRANSFORMER TRAINING LOOP
A training loop is the process of repeatedly making predictions, calculating loss, updating model weights, and repeating until the model learns.
WHY DO WE NEED IT?

The Transformer starts with random weights.

We need to repeatedly:

Predict → Calculate Loss → Update Weights → Repeat

so the model gradually improves.
-------------------------------------------------------------------------------------------------------------
HOW DOES IT WORK?

The basic loop is:

Input
  ↓
Transformer
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Optimizer
  ↓
Updated Weights
  ↓
Repeat

The three important lines are:

optimizer.zero_grad() = clears old gradients
loss.backward() = calculates new gradients
optimizer.step() = updates the model weights
"""

import torch
import torch.nn as nn
import torch.optim as optim


# Simple model
model = nn.Linear(1, 1)


# Training data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])


# Loss function
criterion = nn.MSELoss()


# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# Training loop
for epoch in range(100):

    # 1. Clear old gradients
    optimizer.zero_grad()

    # 2. Forward pass
    prediction = model(x)

    # 3. Calculate loss
    loss = criterion(prediction, y)

    # 4. Backpropagation
    loss.backward()

    # 5. Update weights
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.4f}"
        )

"""
Predict → Forward Pass
Measure error → Loss
Calculate gradients → backward()
Update weights → optimizer.step()
Repeat → Training Loop
"""