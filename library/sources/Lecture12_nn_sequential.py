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
# id: Lecture12_nn_sequential
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## `nn.Sequential`
#
# nn.Sequential is a PyTorch module that allows you to stack several layers and create a neural network. This is particularly useful for creating shallow networks, where all the layers have a similar structure and purpose.
#
# Here is an example of how to use `nn.Sequential` to create a fully connected shallow neural network:
#
# In this example, we create a new PyTorch module called `MLPRegressor`.
# The constructor takes three arguments: input_size, output_size, and hidden_size.
# These correspond to the number of input features, output classes, and the number of hidden units in the network.
#
# Inside the constructor, we create a new `nn.Sequential` module called `fc_layers`.
# This module contains three layers: a fully connected layer that maps the input to the hidden layer, a LeakyReLU activation function, and another fully connected layer that maps the hidden layer to the output.
#
# The forward method takes an input tensor x and passes it through the `fc_layers` module using the `self.fc_layers(x)` line.
# The output of this module is then returned.
#
# To use this module, we can create an instance of `MLPRegressor` and pass input tensors through it like this:
#
# This creates an instance of the ShallowMLP module with 10 input features, 5 output classes, and a hidden layer of size 20. We then create a random input tensor with 2 samples and 10 features, and pass it through the model using model(input_tensor). The output is a tensor with shape (2, 5), which corresponds to the 2 samples and 5 output classes.
#
# We can observe the exact same value of the loss function compared our version without `nn.Sequential`.
#
# By stacking layers together with `nn.Sequential`, we can create complex models that are easy to read and modify.
# For instance, here's an example implementation of `MLPRegressor` that allows for a variable number of hidden layers based on a tuple of neurons passed in as hidden_size argument, following the interface used in `sklearn.neural_network.MLPRegressor`:
#
# Here, we allow for a variable number of hidden layers by accepting a tuple of neurons in hidden_size argument. We use a loop to create a list of layers with nn.Linear and the specified activation function and dropout (if any), followed by setting the final output layer.
#
# Note that in the constructor we also added two new arguments: activation and dropout, which allow the user to specify the activation function for the hidden layers and the dropout probability respectively. We support three common activation functions: ReLU, sigmoid, and tanh, but you can add others if you like. We also allow the user to set the dropout probability, with a default of zero (i.e. no dropout).
#
# Here's an example usage:
#
# This can be deployed to get identical results as before:
#
# However, we can also easily modify the architecture to obtain better results:
#
# Here the performance is improved by about 4x by increasing the depth of the network.
# The `hidden_size` could be controlled by a hyperparameter tuning algorithm, whereas it would be difficult to build new `nn.Module` subclasses during the optimization loop.

# %%
class MLPRegressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLPRegressor, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x

input_size = 1
output_size = 1
hidden_size = 10

model = MLPRegressor(input_size, output_size, hidden_size)

import tqdm

def train_and_plot_mlp(model):
    torch.manual_seed(0)

    # make torch tensors from np arrays
    xlt = torch.Tensor(xls.reshape(-1, 1))
    ylt = torch.Tensor(yls.reshape(-1, 1))

    # define the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)

    # do the training
    pbar = tqdm.tqdm(np.arange(400))
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

    with torch.no_grad():
        model_in = torch.tensor(xl.reshape(-1, 1), dtype=torch.float)
        model_out = model(model_in).detach().numpy()

    fig, ax = plt.subplots()
    _ = ax.plot(xl, f(xl), label='Function')
    _ = ax.plot(xls, f(xls), '.', label='Data')
    _ = ax.plot(xl, model_out, label='NN')
    _ = ax.legend()

train_and_plot_mlp(model)

class MLPRegressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=(100, ), activation=nn.ReLU()):
        super(MLPRegressor, self).__init__()

        layers = []
        prev_layer_size = input_size
        for layer_size in hidden_size:
            layers.append(nn.Linear(prev_layer_size, layer_size))
            layers.append(activation)
            prev_layer_size = layer_size

        layers.append(nn.Linear(prev_layer_size, output_size))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        return x

input_size = 1
output_size = 1
hidden_size = (10, )
activation = nn.ReLU()

model = MLPRegressor(input_size, output_size, hidden_size, activation=activation)

train_and_plot_mlp(model)

input_size = 1
output_size = 1
hidden_size = (10, 10, 10, 10)
activation = nn.LeakyReLU()

model = MLPRegressor(input_size, output_size, hidden_size, activation=activation)

train_and_plot_mlp(model)
