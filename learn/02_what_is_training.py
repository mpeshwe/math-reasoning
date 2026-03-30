"""
LESSON 0.2 — What Is Training?

Training = adjusting weights so the model's output gets closer
to what you want.

We'll teach a tiny model to learn a simple pattern:
    Input: [x, y]  →  Output: x + y

The model starts with random weights and knows NOTHING.
After training, it will learn addition.
"""

import torch
import torch.nn as nn

# ============================================================
# PART 1: The untrained model — knows nothing
# ============================================================

class Adder(nn.Module):
    """A tiny model that we'll teach to add two numbers."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

model = Adder()

# Test it BEFORE training — it will be terrible
print("=== BEFORE TRAINING (random weights) ===")
test_cases = [[3.0, 5.0], [10.0, 20.0], [1.0, 1.0]]
for a, b in test_cases:
    inp = torch.tensor([a, b])
    pred = model(inp).item()
    print(f"  {a} + {b} = {pred:.4f}  (should be {a + b})")

# ============================================================
# PART 2: Training — the 4-step loop
# ============================================================

print("\n=== TRAINING ===")
print("The training loop has 4 steps, repeated thousands of times:\n")
print("  1. FORWARD:   Feed input, get prediction")
print("  2. LOSS:      How wrong was the prediction?")
print("  3. BACKWARD:  Which weights caused the error? (backpropagation)")
print("  4. UPDATE:    Nudge those weights slightly to reduce the error")
print()

# The loss function: measures how wrong the prediction is
# MSE = Mean Squared Error = (prediction - correct_answer)²
loss_fn = nn.MSELoss()

# The optimizer: adjusts weights to reduce loss
# lr = learning rate = how big each adjustment is
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for step in range(2000):

    # Generate random training data: two numbers and their sum
    a = torch.rand(1) * 20  # random number 0-20
    b = torch.rand(1) * 20
    input_data = torch.stack([a, b], dim=-1).squeeze()
    correct_answer = (a + b).squeeze()

    # STEP 1: Forward pass — get prediction
    prediction = model(input_data).squeeze()

    # STEP 2: Compute loss — how wrong is the prediction?
    loss = loss_fn(prediction, correct_answer)

    # STEP 3: Backward pass — figure out which weights caused the error
    optimizer.zero_grad()  # clear previous gradients
    loss.backward()        # compute gradients (∂loss/∂weight for every weight)

    # STEP 4: Update weights — nudge them to reduce loss
    optimizer.step()

    # Print progress every 200 steps
    if step % 200 == 0:
        print(f"  Step {step:>4d} | Loss: {loss.item():.6f}")

# ============================================================
# PART 3: Test the trained model — it should know addition now!
# ============================================================

print("\n=== AFTER TRAINING ===")
for a, b in test_cases:
    inp = torch.tensor([a, b])
    pred = model(inp).item()
    correct = a + b
    error = abs(pred - correct)
    print(f"  {a} + {b} = {pred:.4f}  (should be {correct}, error: {error:.4f})")

# Test on numbers it has NEVER seen
print("\n=== GENERALIZATION (never seen these during training) ===")
new_cases = [[7.5, 2.5], [15.0, 15.0], [0.0, 0.0], [19.9, 0.1]]
for a, b in new_cases:
    inp = torch.tensor([a, b])
    pred = model(inp).item()
    correct = a + b
    error = abs(pred - correct)
    print(f"  {a} + {b} = {pred:.4f}  (should be {correct}, error: {error:.4f})")

# ============================================================
# PART 4: Let's see what the weights look like now
# ============================================================

print("\n=== WHAT THE MODEL LEARNED ===")
print("The weights are no longer random — they've been adjusted")
print("to approximate the function f(x, y) = x + y\n")

# In theory, the perfect weights for addition would be:
#   layer that outputs: 1.0 * x + 1.0 * y + 0.0
# But neural networks find their own (messier) path to the answer.
print(f"Layer 1 weights (first 3 rows):\n{model.layer1.weight.data[:3]}")
print(f"\nThese don't look like [1, 1] because the network found")
print(f"its own internal representation. The BEHAVIOR is correct")
print(f"even if the path is different from what a human would write.")

# ============================================================
# PART 5: The gradient — HOW does training know which direction?
# ============================================================

print("\n=== HOW GRADIENTS WORK ===")

# Fresh simple example
w = torch.tensor([2.0], requires_grad=True)  # a single weight
x = torch.tensor([3.0])                       # input
target = torch.tensor([10.0])                  # we want output to be 10

# Forward: prediction = w * x = 2 * 3 = 6
prediction = w * x
print(f"Weight: {w.item()}, Input: {x.item()}")
print(f"Prediction: w × x = {prediction.item()}")
print(f"Target: {target.item()}")

# Loss: (prediction - target)² = (6 - 10)² = 16
loss = (prediction - target) ** 2
print(f"Loss: (prediction - target)² = ({prediction.item()} - {target.item()})² = {loss.item()}")

# Backward: compute gradient (which direction should w move?)
loss.backward()
print(f"\nGradient of loss with respect to w: {w.grad.item()}")
print(f"Negative gradient means: increase w to reduce loss")
print(f"Positive gradient means: decrease w to reduce loss")
print(f"\nHere gradient = {w.grad.item()}, so we should {'increase' if w.grad.item() < 0 else 'decrease'} w")
print(f"Makes sense: w=2 gives output 6, but we want 10, so w should be bigger")
print(f"Perfect w would be {target.item() / x.item()} (10/3 = 3.333...)")

# ============================================================
# KEY TAKEAWAY:
#
# Training is a loop:
#   1. Predict
#   2. Measure error (loss)
#   3. Figure out which direction to adjust weights (gradients)
#   4. Nudge weights in that direction
#   Repeat thousands of times → model learns the pattern
#
# This is EXACTLY what happens when training an LLM.
# The only differences:
#   - Input/output are text tokens instead of numbers
#   - Loss is "how wrong was the next token prediction"
#   - There are 3 billion weights instead of 50
#   - It takes hours on GPUs instead of seconds on CPU
#
# Next: we'll do this with actual text and a real (tiny) language model.
# ============================================================
