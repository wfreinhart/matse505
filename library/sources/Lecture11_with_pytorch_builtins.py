# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# ---
# id: Lecture11_with_pytorch_builtins
# type: Foundational
# parent_lecture: Lecture11
# ---
#
# ## With `pytorch` builtins
#
# We will be using the `torch.nn.Module` class to interact with `pytorch` builtins for model training:
#
# Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training.
# Subclassing `nn.Module` automatically tracks all fields defined inside your model object, and makes all parameters accessible using your model’s `parameters()` or `named_parameters()` methods.
#
# In this example, we iterate over each parameter, and print its size and a preview of its values.
#
# `pytorch` also has builtin optimizers.
# There is a required positional argument for the parameters to optimize.
# Using `nn.Module` we can simply specify `model.parameters` which will conveniently contain all the learnable weights in the model.
# You can also specify kwargs including `lr` for the learning rate.
#
# We also need to specify a loss function.
# This is often called `criterion` in the tutorials and sample codes.
# The most common loss functions are included in `torch.nn`:
#
# Finally we can implement these in the learning loop.
# The `optimizer` has two methods that need to be called:
# * `zero_grad` to reset gradients in between epochs
# * `step` to apply the update to model parameters managed by the `optimizer`
#
# We can compare this to the `MLPRegressor` by evaluating the RMSE:
#
# This is substantially better than the `rmse = 10.354` from the `LinearRegression` and the `rmse = 39.2` from the `MLPRegressor` at the beginning.

# %%
import torch.nn as nn

torch.manual_seed(0)

# Define the model, inheriting from nn.Module
class ShallowNet(nn.Module):
    # Define a constructor that takes model hyperparameters
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Call the super (nn.Module) constructor
        super(ShallowNet, self).__init__()
        # Define the components of the model
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    # Define a forward method that will be called like model(x)
    def forward(self, x):
        # Canonical naming convention is same variable on every line
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return x

# Initialize the model
model = ShallowNet(input_dim, hidden_dim, output_dim)

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

from torch import optim

# Define the loss function and the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss()

torch.manual_seed(0)
model = ShallowNet(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-1)
# Train the model
epochs = 200
for epoch in range(epochs):
    # Forward pass
    outputs = model(xt_s)
    loss = criterion(outputs, yt)

    # Backward and optimize
    optimizer.zero_grad()  # reset the gradients stored by the optimizer
    loss.backward()
    optimizer.step()

    if (epoch+1)%10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    residual = model(xt_s) - yt
    loss = torch.sqrt( torch.mean( residual**2 ) )
    print(f'RMSE: {loss.item():.4f}')
