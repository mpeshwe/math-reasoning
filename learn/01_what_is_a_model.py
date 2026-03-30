"""
LESSON 0.1 — What Is a Model?

A model is just numbers (weights) + a recipe (forward pass).
Let's see this at the lowest level before touching any LLM.
"""

import torch

# ============================================================
# PART 1: A tensor is just a multi-dimensional array of numbers
# ============================================================

# A single number
scalar = torch.tensor(3.14)
print(f"Scalar: {scalar}")
print(f"  Shape: {scalar.shape}")  # no dimensions

# A list of numbers (1D tensor = vector)
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"\nVector: {vector}")
print(f"  Shape: {vector.shape}")  # 4 numbers

# A grid of numbers (2D tensor = matrix)
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(f"\nMatrix:\n{matrix}")
print(f"  Shape: {matrix.shape}")  # 3 rows, 2 columns

# ============================================================
# PART 2: A model's "weights" are just tensors of random numbers
#         that get adjusted during training
# ============================================================

# Imagine a tiny model with just 2 weights
weights = torch.tensor([0.5, -0.3])
bias = torch.tensor([0.1])

print("\n--- Simplest possible model ---")
print(f"Weights: {weights}")
print(f"Bias:    {bias}")

# The "forward pass" = multiply input by weights, add bias
input_data = torch.tensor([2.0, 3.0])
output = torch.dot(weights, input_data) + bias  # (0.5*2) + (-0.3*3) + 0.1 = 0.2

print(f"Input:   {input_data}")
print(f"Output:  {output}")
print(f"Math:    (0.5 × 2) + (-0.3 × 3) + 0.1 = {0.5*2 + (-0.3)*3 + 0.1}")

# ============================================================
# PART 3: A neural network = layers of these operations
# ============================================================

# Let's build a tiny neural network with PyTorch's nn module
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: takes 4 inputs, produces 8 outputs
        self.layer1 = nn.Linear(4, 8)
        # Layer 2: takes 8 inputs, produces 2 outputs
        self.layer2 = nn.Linear(8, 2)
        # Activation: adds non-linearity (lets the model learn curves, not just lines)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)       # multiply by weights + bias
        x = self.activation(x)   # zero out negatives
        x = self.layer2(x)       # multiply by weights + bias again
        return x

model = TinyModel()

print("\n--- A tiny neural network ---")
print(f"Model:\n{model}")

# Let's count the weights
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters (weights): {total_params}")

# Let's SEE the weights — they're just random numbers right now
print(f"\nLayer 1 weights shape: {model.layer1.weight.shape}")
print(f"Layer 1 weights (random!):\n{model.layer1.weight.data}")

# Run a forward pass — feed in 4 numbers, get 2 numbers out
input_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
output = model(input_data)
print(f"\nInput:  {input_data}")
print(f"Output: {output}")
print("(The output is meaningless right now — the weights are random!)")

# ============================================================
# PART 4: Now scale this up in your head
# ============================================================

print("\n--- Scaling up ---")
print(f"Our tiny model:      {total_params:>15,} parameters")
print(f"Llama 3.2 1B:        {1_000_000_000:>15,} parameters")
print(f"Llama 3.2 3B:        {3_000_000_000:>15,} parameters")
print(f"Llama 3.1 70B:       {70_000_000_000:>15,} parameters")

print(f"\nMemory (FP16 = 2 bytes per parameter):")
print(f"Our tiny model:      {total_params * 2:>15,} bytes")
print(f"Llama 3.2 3B:        {3_000_000_000 * 2:>15,} bytes = ~6 GB")
print(f"Llama 3.1 70B:       {70_000_000_000 * 2:>15,} bytes = ~140 GB")

# ============================================================
# KEY TAKEAWAY:
#
# A model = weights (random numbers) + forward pass (multiply recipe)
# Training = adjusting those random numbers until the outputs are useful
#
# An LLM like Llama 3.2 3B is this EXACT same concept,
# just with 3 billion weights instead of 42.
#
# Next: we'll learn how training adjusts those weights.
# ============================================================
