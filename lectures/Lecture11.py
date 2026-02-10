# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -id,-colab,-outputId
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# Today's topics:
# * `pytorch` syntax
# * Linear regression with `pytorch` gradient descent
# * Building a shallow NN with `pytorch`

# %% [markdown]
# # Preliminaries

# %% [markdown]
# ## Dataset
#
# Let's start by loading the concrete data:

# %%
import pandas as pd
import numpy as np
import os

# Set the path to the data file
filename = 'concrete.csv'
local_path = f'../datasets/{filename}'
github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'

# Load the data: try local path first, fallback to GitHub for Colab
if os.path.exists(local_path):
    data = pd.read_csv(local_path)
else:
    data = pd.read_csv(github_url)
data                            # show a view of the data file

# %% [markdown]
# We'll set up the problem to be a regression task using all the features to predict the compressive strength:

# %%
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# %% [markdown]
# ## Baseline with `sklearn`
#
# We already know this dataset has important nonlinearities that make it unsuitable for linear regression.
# Here is the baseline performance with multivariate linear regression using `sklearn.linear_model.LinearRegression`:

# %%
from sklearn import linear_model

model = linear_model.LinearRegression().fit(x, y)
print( f'R2   = {model.score(x, y):.3f}' )

residual = model.predict(x) - y
rmse = np.sqrt( np.mean( (residual-y)**2 ) )
print( f'rmse = {rmse:.1f}' )

# %% [markdown]
# Note that we aren't even considering test performance here, this is purely to evaluate model expressiveness.
# Even with all the data seen during training, the linear regression model cannot capture the nonlinear behavior.
#
# Now let's evaluate the performance of a `sklearn.neural_network.MLPRegressor` object:

# %%
from sklearn import neural_network

model = neural_network.MLPRegressor(random_state=0).fit(x, y)
print( f'R2   = {model.score(x, y):.3f}' )

residual = model.predict(x) - y
rmse = np.sqrt( np.mean( (residual-y)**2 ) )
print( f'rmse = {rmse:.1f}' )

# %% [markdown]
# We see that this NN model is more expressive and can achieve greater performance on the regression task *at least when all the data are available for training*.
# We'll return to the issue of validation and test performance soon, after we get the `pytorch` syntax down.

# %% [markdown]
# # `pytorch` syntax

# %% [markdown]
# ## Why `pytorch`?
#
# We are going to spend a significant amount of time learning to use the `pytorch` framework.
# At the moment it may seem like `sklearn` has everything we need and the additional overhead to use `pytorch` is not worthwhile, but there is simply no way to implement deep learning in `sklearn` or `numpy`.
# In any of these high-level packages, the convenience of having many common models and functions already implemented for you is the very reason why they're limiting.
# We now need lower-level access to the machinery in order to build our own custom models.
#
# The basis of any deep learning package (e.g., `pytorch`, `tensorflow`, or `jax`) is an **automatic differentiation** engine.
# Automatic differentiation is a tool for computing gradients on mathematical operators.
# Here's what it looks like:
#
# <img src="../lectures/assets/lecture11_full_graph.jpg" alt="Computational graph showing operations and gradients" width=600>
#
# Why do we need this?
# Gradient descent is the most practical way to solve high-dimensional optimization problems.
# In this case, the optimization problem is the selection of model parameters (i.e., NN weights) to minimize the model loss function.
# There can easily be millions of parameters to optimize, and the largest models have billions.
# Here's what gradient descent looks like in a 2D parameter space:
#
# <img src="../lectures/assets/lecture11_gradient_descent_2d.jpg" alt="Gradient descent optimization in a 2D parameter space" width=600>
#
# `pytorch` is not unique in its ability to solve these problems.
# However, it has a user-friendly interface and is relatively popular in the community.
# This means there are many well developed features and sample codes for implementing modern ML architectures.
# For this reason, I personally prefer `pytorch` over `tensorflow`, which is a little less polished.
# On the other hand, `keras` is a wrapper built on top of `tensorflow` that abstracts away too many features for my use cases.
# This leaves `pytorch` as a happy medium for me, but you may find a different story in your research.
#
#

# %% [markdown]
# ## Tensors
#
# Pytorch `tensors` are similar to numpy `arrays`.
# However, they are not interoperable -- all calculations performed in pytorch need to be done on `tensors`.

# %%
import torch

xt = torch.from_numpy(x.values).float()
yt = torch.from_numpy(y.values).float()

print(xt.shape, yt.shape)  # shape works just like numpy
print()
print(xt[:5])              # indexing works just like numpy

# %% [markdown]
# These can interact with each other and with the usual operators just like `arrays`:

# %%
print( xt[:1] * 2 )
print( xt[:1]**2 )
print( xt[:1] + yt[:1] )

# %% [markdown]
# They also have methods for common operations just like `arrays`:

# %%
print( xt.mean() )
print( xt.mean(dim=0) )  # using "dim" instead of "axis"

# %% [markdown]
# It is very important to consider the shape and size of the `tensors`.
# Let's explore how these work:

# %%
a = torch.zeros([2, 2, 2])
b = 1
print( a )
print()
print( a + b )

# %% [markdown]
# `tensor` plus `int` adds the value of `int` to all elements in the `tensor`.
# What about lower-dimension `tensor` plus higher-dimension `tensor`?

# %%
b = torch.tensor([1, 2])
print( b )
print()
print( a + b )

# %% [markdown]
# What about a different shape?
# We can change the dimensions using `unsqueeze`:

# %%
print( b.shape )
print( b )
print()
print( b.unsqueeze(0).shape )
print( b.unsqueeze(0) )
print()
print( b.unsqueeze(1).shape )
print( b.unsqueeze(1) )

# %% [markdown]
# Now that we know how to change the dimensions, we can see how they interact with the `+` operator:

# %%
c = b.unsqueeze(0)
print( c )
print()
print( a + c )

# %%
c = b.unsqueeze(1)
print( c )
print()
print( a + c )

# %%
c = b.unsqueeze(1).unsqueeze(2)
print( c )
print()
print( a + c )

# %% [markdown]
# Unlike `arrays`, `tensors` have an additional attribute called `device`:

# %%
xt.device

# %% [markdown]
# This allows for computing on hardware accelerators like Graphics Processing Units (GPUs):
#
# <img src="../lectures/assets/lecture11_computer_parts.jpg" alt="Diagram showing various computer components like CPU and GPU" width=600>
#
# <img src="../lectures/assets/lecture11_hardware_accel.jpg" alt="Comparison between CPU and GPU architectures" width=600>

# %% [markdown]
# ## Linear regression with gradient descent
#
# Let's use `pytorch` to implement linear regression:
#
# $f(x) = w x + b$
#
# In practice, this is a matrix multiply between $w$ and $x$.
# We can retrieve the fitted coefficients from `sklearn`:

# %%
model = linear_model.LinearRegression().fit(x, y)

linear_out = model.predict(x)
residual = linear_out - y
rmse = np.sqrt( np.mean( residual**2 ) )
print( f'rmse = {rmse:.3f}' )

print()
print('parameters:')
print( model.coef_ )
print( model.intercept_ )

# %% [markdown]
# We can use these fitted parameters in `tensor` form to implement the linear regression model:

# %%
xt = torch.from_numpy(x.values).float()
yt = torch.from_numpy(y.values).float().unsqueeze(1)

w = torch.tensor(model.coef_).float().unsqueeze(1)
b = torch.tensor(model.intercept_).float()

out = xt @ w + b  # @ is the symbol for matrix multiplication
residual = out - yt
rmse = torch.sqrt( torch.mean(residual**2) )
print( f'rmse = {rmse:.3f}' )

# %% [markdown]
# Here we see an identical RMSE compared to using `linear_model.LinearRegression()`.
# > Note the use of `unsqueeze`!

# %% [markdown]
# Now we need to try and fit a model from scratch.
# Let's see what happens when we use arbitrary $w, b$ vectors:

# %%
# Initialize the weights and bias
w = torch.ones(xt.shape[1], 1, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)

# Forward pass
outputs = xt @ w + b
residual = outputs - yt
loss = torch.sqrt( torch.mean( residual**2 ) )

print('initial RMSE', loss)

# %% [markdown]
# What's up with this representation of `loss`?
# Well, `pytorch` is reminding us about our use of the `requires_grad` kwarg above.
# Because we forced `w` and `b` to be included in the computational graph, we can now track their influence on the `loss` by referencing their `grad` attribute:

# %%
print( w.grad )
print( b.grad )

# %% [markdown]
# Why do we get `None`?
# Because the gradient is only computed when explicitly requested.
# This operation is called the "backward pass":
#
# <img src="../lectures/assets/lecture11_autograd_forward_backward.jpg" alt="Diagram of forward and backward passes in automatic differentiation" width=600>
#
# Let's force a computation of these gradients using `loss.backward`:

# %%
# Compute gradients
loss.backward()
print( loss )

# %% [markdown]
# While nothing changes with `loss`, we can now access the gradients in variables that contributed to it:

# %%
print( w.grad )
print( b.grad )

# %% [markdown]
# We can utilize this gradient information to make a step in the right direction.
# However, we have to do so without considering the change this will make to the `loss`, so we use a special construction.
#
# By default, all tensors with `requires_grad=True` are tracking their computational history and support gradient computation.
# However, there are some cases when we do not need to do that, for example, when we have trained the model and just want to apply it to some input data, i.e. we only want to do forward computations through the network.
# We can stop tracking computations by surrounding our computation code with `torch.no_grad()` block:

# %%
with torch.no_grad():
    w -= w.grad
    b -= b.grad

# %% [markdown]
# Now we have to reset the gradients:

# %%
w.grad.zero_()
b.grad.zero_()

# %%
w.grad

# %%
outputs = xt @ w + b
residual = outputs - yt
loss = torch.sqrt( torch.mean( residual**2 ) )

loss.backward()

print( w.grad )
print( b.grad )

# %% [markdown]
# What happened?
# We can see that the gradient $\nabla L / \nabla b$ had a value of `4706.7` while the optimal value of $b$ was `-23.16`.
# Therefore when we applied the gradient we greatly overstepped the target.
# This introduces the need for a **learning rate**, an empirical factor that scales the gradients.
# Here's an illustration of the problem:
#
# <img src="../lectures/assets/lecture11_learning_rate.jpg" alt="Impact of different learning rates on gradient descent convergence" width=800>

# %%
# Initialize the weights and bias
w = torch.ones(xt.shape[1], 1, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)

# Forward pass
outputs = xt @ w + b
residual = outputs - yt
loss = torch.sqrt( torch.mean( residual**2 ) )

# Compute gradients
loss.backward()

learning_rate = 1e-3  # set the learning rate to be a small number

# Update weights and reset gradients
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(w)

# %% [markdown]
# Now the values are much more reasonable.
# We can check that the RMSE has gone down compared to the original value of `5543596.5`:

# %%
with torch.no_grad():
    outputs = xt @ w + b
    residual = outputs - yt
    rmse = torch.sqrt( torch.mean( residual**2 ) )
print(f'rmse = {rmse:.1f}')

# %% [markdown]
# Gradient descent works by repeating this process and recomputing the gradients each time we make a step.
# These iterations are called **epochs**.
# The full loop would look something like this:

# %%
# Initialize the weights and bias
w = torch.ones(xt.shape[1], 1, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)

# Train the model
learning_rate = 1e-4
epochs = 20
for epoch in range(epochs):

    # Forward pass
    outputs = xt @ w + b
    residual = outputs - yt
    loss = torch.sqrt( torch.mean( residual**2 ) )

    # Compute gradients
    loss.backward()

    # Update weights and reset gradients
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# %% [markdown]
# We can see here that the loss (RMSE) is decreasing but doesn't reach the level we expect from `sklearn` (`10.4`).
# This might require a very large number of very small steps:

# %%
# Train the model
learning_rate = 1e-5
epochs = 1000
for epoch in range(epochs):

    # Forward pass
    outputs = xt @ w + b
    residual = outputs - yt
    loss = torch.sqrt( torch.mean( residual**2 ) )

    # Compute gradients
    loss.backward()

    # Update weights and reset gradients
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if (epoch+1)%100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# %% [markdown]
# Alternatively, we can condition the problem better.
# 1. more isotropic gradients using normalization
# 2. better behaved gradients using MSE instead of RMSE

# %%
# Normalize the input values
xt_s = (xt - xt.mean(dim=0)) / xt.std(dim=0)

# Initialize the weights and bias
w = torch.ones(xt.shape[1], 1, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)

# Train the model
learning_rate = 4e-1
epochs = 100
for epoch in range(epochs):

    # Forward pass
    outputs = xt_s @ w + b
    residual = outputs - yt
    loss = torch.mean( residual**2 )

    # Compute gradients
    loss.backward()

    # Update weights and reset gradients
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if (epoch+1)%10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# %% [markdown]
# Now that we have a much lower MSE loss, we can evaluate the RMSE:

# %%
with torch.no_grad():
    outputs = xt_s @ w + b
    residual = outputs - yt
    rmse = torch.sqrt( torch.mean( residual**2 ) )
print(f'rmse = {rmse:.3f}')

# %% [markdown]
# This is very marginally higher the RMSE obtained by linear regression with `sklearn`.
# We can compare them graphically:

# %%
from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
_ = ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='reference')
_ = ax.plot(linear_out, y, 's', label='sklearn')
_ = ax.plot(outputs.detach().numpy(), y, '.', label='pytorch')
_ = ax.legend()
_ = ax.set_xlabel('model')
_ = ax.set_ylabel('data')

ax = axes[1]
_ = ax.plot(linear_out, outputs.detach().numpy(), '.')
_ = ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
_ = ax.set_xlabel('sklearn')
_ = ax.set_ylabel('pytorch')

# %% [markdown]
# # Building a shallow NN with `pytorch`

# %% [markdown]
# ## The manual way with autograd
#
# Let's set up a shallow neural network with one hidden layer.
# It will look like this:
#
# <img src="../lectures/assets/lecture11_shallow_nn.jpg" alt="Architecture of a shallow neural network with one hidden layer" width=600>
#

# %% [markdown]
# First we'll set up the inputs again, with `hidden_dim` being the number of neurons in the hidden layer.

# %%
# Convert numpy arrays to PyTorch tensors
xt = torch.from_numpy(x.values).float()
yt = torch.from_numpy(y.values).float().unsqueeze(1)

input_dim = xt.shape[1]
hidden_dim = 100
output_dim = yt.shape[1]

# %% [markdown]
# Now we'll set up the model parameters:

# %%
# Initialize the weights and bias
wi = torch.ones(input_dim, hidden_dim, requires_grad=True)
bi = torch.zeros(1, hidden_dim, requires_grad=True)

wh = torch.ones(hidden_dim, output_dim, requires_grad=True)
bh = torch.ones(1, output_dim, requires_grad=True)

# %% [markdown]
# Note that we need $w$ and $b$ for each of the input and hidden layers.

# %%
# Forward pass
xh = torch.relu( xt @ wi + bi )
outputs = torch.relu( xh @ wh + bh )

print(outputs)

# %% [markdown]
# ## [Check your understanding]
#
# Complete the training loop for the neural network.
# You may need to tune the learning rate, number of epochs, etc.
# How does it compare to the performance of the `MLPRegressor` from `sklearn`?

# %%
# Initialize the weights and bias
# [set up your parameters here]

# Train the model
learning_rate = 4e-1
epochs = 100
for epoch in range(epochs):

    # Forward pass
    # [set up your model transformations here]
    residual = outputs - yt
    loss = torch.mean( residual**2 )

    # Compute gradients
    loss.backward()

    # Update weights and reset gradients
    with torch.no_grad():
        # perform your parameter updates here
        # zero your gradients here

    if (epoch+1)%10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# %% [markdown]
# ## With `pytorch` builtins
#
# We will be using the `torch.nn.Module` class to interact with `pytorch` builtins for model training:

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

# %% [markdown]
# Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training.
# Subclassing `nn.Module` automatically tracks all fields defined inside your model object, and makes all parameters accessible using your model’s `parameters()` or `named_parameters()` methods.
#
# In this example, we iterate over each parameter, and print its size and a preview of its values.

# %%
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# %% [markdown]
# `pytorch` also has builtin optimizers.
# There is a required positional argument for the parameters to optimize.
# Using `nn.Module` we can simply specify `model.parameters` which will conveniently contain all the learnable weights in the model.
# You can also specify kwargs including `lr` for the learning rate.

# %%
from torch import optim

# Define the loss function and the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# %% [markdown]
# We also need to specify a loss function.
# This is often called `criterion` in the tutorials and sample codes.
# The most common loss functions are included in `torch.nn`:

# %%
criterion = nn.MSELoss()

# %% [markdown]
# Finally we can implement these in the learning loop.
# The `optimizer` has two methods that need to be called:
# * `zero_grad` to reset gradients in between epochs
# * `step` to apply the update to model parameters managed by the `optimizer`

# %%
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

# %% [markdown]
# We can compare this to the `MLPRegressor` by evaluating the RMSE:

# %%
# Evaluate the model
with torch.no_grad():
    residual = model(xt_s) - yt
    loss = torch.sqrt( torch.mean( residual**2 ) )
    print(f'RMSE: {loss.item():.4f}')

# %% [markdown]
# This is substantially better than the `rmse = 10.354` from the `LinearRegression` and the `rmse = 39.2` from the `MLPRegressor` at the beginning.

# %% [markdown]
# # Bonus: a more visual example
#
# Let's briefly try working with a more visual example to give you a sense of what's going on.
# We'll use a 1D polynomial function so it's easy to plot.

# %%
from scipy.special import legendre

rng = np.random.default_rng(0)

npoly = 5
a = 2*(rng.random(npoly) - 0.5)
xl = np.linspace(-1, 1, 101)

def f(x):
    y = np.sum([a[i] * legendre(i)(x) for i in range(npoly)], axis=0)
    return y

fig, ax = plt.subplots()
_ = ax.plot(xl, f(xl))

# %% [markdown]
# Now we'll sample the function from across its domain:

# %%
ns = 200
xls = (rng.random(ns) - 0.5)*2  # draws samples uniformly on [-1, 1]
yls = f(xls)

fig, ax = plt.subplots()
_ = ax.plot(xl, f(xl), label='Function')
_ = ax.plot(xls, yls, '.', label='Data')
_ = ax.legend()


# %% [markdown]
# Let's set up a deeper neural network to try:

# %%
class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,10)
        self.fc4 = nn.Linear(10,1)
        self.act = torch.nn.LeakyReLU()
        # self.act = torch.tanh

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.fc4(x)
        return x


# %% [markdown]
# And train this using the same loop structure...

# %%
import tqdm

torch.manual_seed(0)

# make torch tensors from np arrays
xlt = torch.Tensor(xls.reshape(-1, 1))
ylt = torch.Tensor(yls.reshape(-1, 1))

# define the model
model = MultiLayerPerceptron()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)

# do the training
pbar = tqdm.tqdm(np.arange(1000))
model.train()
for epoch in pbar:
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(xlt)
    # Compute Loss
    loss = criterion(y_pred, ylt)

    pbar.set_postfix_str(f'train loss: {loss.item():.3e}')
    # Backward pass
    loss.backward()
    optimizer.step()

# %% [markdown]
# Now we can evaluate the performance:

# %%
with torch.no_grad():
    model_in = torch.tensor(xl.reshape(-1, 1), dtype=torch.float)
    model_out = model(model_in).detach().numpy()

fig, ax = plt.subplots()
_ = ax.plot(xl, f(xl), label='Function')
_ = ax.plot(xls, f(xls), '.', label='Data')
_ = ax.plot(xl, model_out, label='NN')
_ = ax.legend()

# %% [markdown]
# In optimizing this model, you could consider:
# * data sampling (number of samples, coverage/distribution, noise?)
# * model architecture (linear and activation layers, depth, different activation functions)
# * loss functions (including regularization)
# * optimizers (SGD, Adam, RMSprop, LBFGS, and others)
# * metrics: R-squared, MSE, RMSE, MAE, etc.
# * properties of the generating function $f(x)$ (nonlinearity, rate of oscillation)

# %% [markdown]
# ## [Check your understanding]
#
# Try manually tuning some of the different hyperparameters of the model including learning rate, epochs, activation function, and hidden dimension.
# Which appears to have the greatest effect?

# %%

# %% [markdown]
# Try implementing a train/test split and evaluate the test performance in addition to the training performance.

# %%

# %% [markdown]
# Try tuning these hyperparameters using evolutionary algorithm and/or Bayesian optimization.

# %%
