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
# id: Lecture11_the_manual_way_with_autograd
# type: Foundational
# parent_lecture: Lecture11
# ---
#
# ## The manual way with autograd
#
# Let's set up a shallow neural network with one hidden layer.
# It will look like this:
#
# <img src="../lectures/assets/lecture11_shallow_nn.jpg" alt="Architecture of a shallow neural network with one hidden layer" width=600>
#
# First we'll set up the inputs again, with `hidden_dim` being the number of neurons in the hidden layer.
#
# Now we'll set up the model parameters:
#
# Note that we need $w$ and $b$ for each of the input and hidden layers.

# %%
# Convert numpy arrays to PyTorch tensors
xt = torch.from_numpy(x.values).float()
yt = torch.from_numpy(y.values).float().unsqueeze(1)

input_dim = xt.shape[1]
hidden_dim = 100
output_dim = yt.shape[1]

# Initialize the weights and bias
wi = torch.ones(input_dim, hidden_dim, requires_grad=True)
bi = torch.zeros(1, hidden_dim, requires_grad=True)

wh = torch.ones(hidden_dim, output_dim, requires_grad=True)
bh = torch.ones(1, output_dim, requires_grad=True)

# Forward pass
xh = torch.relu( xt @ wi + bi )
outputs = torch.relu( xh @ wh + bh )

print(outputs)
