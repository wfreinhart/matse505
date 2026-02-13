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
# id: Lecture11_linear_regression_with_gradient_descent
# type: Foundational
# parent_lecture: Lecture11
# ---
#
# ## Linear regression with gradient descent
#
# Let's use `pytorch` to implement linear regression:
#
# $f(x) = w x + b$
#
# In practice, this is a matrix multiply between $w$ and $x$.
# We can retrieve the fitted coefficients from `sklearn`:
#
# We can use these fitted parameters in `tensor` form to implement the linear regression model:
#
# Here we see an identical RMSE compared to using `linear_model.LinearRegression()`.
# > Note the use of `unsqueeze`!
#
# Now we need to try and fit a model from scratch.
# Let's see what happens when we use arbitrary $w, b$ vectors:
#
# What's up with this representation of `loss`?
# Well, `pytorch` is reminding us about our use of the `requires_grad` kwarg above.
# Because we forced `w` and `b` to be included in the computational graph, we can now track their influence on the `loss` by referencing their `grad` attribute:
#
# Why do we get `None`?
# Because the gradient is only computed when explicitly requested.
# This operation is called the "backward pass":
#
# <img src="../lectures/assets/lecture11_autograd_forward_backward.jpg" alt="Diagram of forward and backward passes in automatic differentiation" width=600>
#
# Let's force a computation of these gradients using `loss.backward`:
#
# While nothing changes with `loss`, we can now access the gradients in variables that contributed to it:
#
# We can utilize this gradient information to make a step in the right direction.
# However, we have to do so without considering the change this will make to the `loss`, so we use a special construction.
#
# By default, all tensors with `requires_grad=True` are tracking their computational history and support gradient computation.
# However, there are some cases when we do not need to do that, for example, when we have trained the model and just want to apply it to some input data, i.e. we only want to do forward computations through the network.
# We can stop tracking computations by surrounding our computation code with `torch.no_grad()` block:
#
# Now we have to reset the gradients:
#
# What happened?
# We can see that the gradient $\nabla L / \nabla b$ had a value of `4706.7` while the optimal value of $b$ was `-23.16`.
# Therefore when we applied the gradient we greatly overstepped the target.
# This introduces the need for a **learning rate**, an empirical factor that scales the gradients.
# Here's an illustration of the problem:
#
# <img src="../lectures/assets/lecture11_learning_rate.jpg" alt="Impact of different learning rates on gradient descent convergence" width=800>
#
# Now the values are much more reasonable.
# We can check that the RMSE has gone down compared to the original value of `5543596.5`:
#
# Gradient descent works by repeating this process and recomputing the gradients each time we make a step.
# These iterations are called **epochs**.
# The full loop would look something like this:
#
# We can see here that the loss (RMSE) is decreasing but doesn't reach the level we expect from `sklearn` (`10.4`).
# This might require a very large number of very small steps:
#
# Alternatively, we can condition the problem better.
# 1. more isotropic gradients using normalization
# 2. better behaved gradients using MSE instead of RMSE
#
# Now that we have a much lower MSE loss, we can evaluate the RMSE:
#
# This is very marginally higher the RMSE obtained by linear regression with `sklearn`.
# We can compare them graphically:

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

xt = torch.from_numpy(x.values).float()
yt = torch.from_numpy(y.values).float().unsqueeze(1)

w = torch.tensor(model.coef_).float().unsqueeze(1)
b = torch.tensor(model.intercept_).float()

out = xt @ w + b  # @ is the symbol for matrix multiplication
residual = out - yt
rmse = torch.sqrt( torch.mean(residual**2) )
print( f'rmse = {rmse:.3f}' )

# Initialize the weights and bias
w = torch.ones(xt.shape[1], 1, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)

# Forward pass
outputs = xt @ w + b
residual = outputs - yt
loss = torch.sqrt( torch.mean( residual**2 ) )

print('initial RMSE', loss)

print( w.grad )
print( b.grad )

# Compute gradients
loss.backward()
print( loss )

print( w.grad )
print( b.grad )

with torch.no_grad():
    w -= w.grad
    b -= b.grad

w.grad.zero_()
b.grad.zero_()

w.grad

outputs = xt @ w + b
residual = outputs - yt
loss = torch.sqrt( torch.mean( residual**2 ) )

loss.backward()

print( w.grad )
print( b.grad )

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

with torch.no_grad():
    outputs = xt @ w + b
    residual = outputs - yt
    rmse = torch.sqrt( torch.mean( residual**2 ) )
print(f'rmse = {rmse:.1f}')

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

with torch.no_grad():
    outputs = xt_s @ w + b
    residual = outputs - yt
    rmse = torch.sqrt( torch.mean( residual**2 ) )
print(f'rmse = {rmse:.3f}')

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
