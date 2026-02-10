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
# * streamlining `pytorch` code
# * saving and loading models
# * using `DataLoader` objects
# * `pytorch-lightning` workflows

# %% [markdown]
# Let's work with our visual example from last time:

# %%
from scipy.special import legendre
import numpy as np
from matplotlib import pyplot as plt

rng = np.random.default_rng(0)

npoly = 5
a = 2*(rng.random(npoly) - 0.5)
xl = np.linspace(-1, 1, 101)

def f(x):
    y = np.sum([a[i] * legendre(i)(x) for i in range(npoly)], axis=0)
    return y

ns = 200
xls = (rng.random(ns) - 0.5)*2  # draws samples uniformly on [-1, 1]
yls = f(xls)

fig, ax = plt.subplots()
_ = ax.plot(xl, f(xl), label='Function')
_ = ax.plot(xls, yls, '.', label='Data')
_ = ax.legend()

# %% [markdown]
# We implemented a shallow neural network by defining a `torch.nn.Module` like so:

# %%
import torch
from torch import nn

class MLPRegressor(torch.nn.Module):

    def __init__(self):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,1)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


# %% [markdown]
# It was trained like this:

# %%
import tqdm

torch.manual_seed(0)

# make torch tensors from np arrays
xlt = torch.Tensor(xls.reshape(-1, 1))
ylt = torch.Tensor(yls.reshape(-1, 1))

# define the model
model = MLPRegressor()
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

# %% [markdown]
# Finally, we evaluated it like this:

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
# We observed that `ReLU` leads to a piecewise linear function while `tanh` would produce smoother curves.

# %% [markdown]
# # Streamlining `pytorch` model code

# %% [markdown]
# ## `nn.Sequential`
#
# nn.Sequential is a PyTorch module that allows you to stack several layers and create a neural network. This is particularly useful for creating shallow networks, where all the layers have a similar structure and purpose.
#
# Here is an example of how to use `nn.Sequential` to create a fully connected shallow neural network:

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


# %% [markdown]
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

# %%
input_size = 1
output_size = 1
hidden_size = 10

model = MLPRegressor(input_size, output_size, hidden_size)

# %% [markdown]
# This creates an instance of the ShallowMLP module with 10 input features, 5 output classes, and a hidden layer of size 20. We then create a random input tensor with 2 samples and 10 features, and pass it through the model using model(input_tensor). The output is a tensor with shape (2, 5), which corresponds to the 2 samples and 5 output classes.

# %%
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


# %% [markdown]
# We can observe the exact same value of the loss function compared our version without `nn.Sequential`.

# %% [markdown]
# By stacking layers together with `nn.Sequential`, we can create complex models that are easy to read and modify.
# For instance, here's an example implementation of `MLPRegressor` that allows for a variable number of hidden layers based on a tuple of neurons passed in as hidden_size argument, following the interface used in `sklearn.neural_network.MLPRegressor`:

# %%
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


# %% [markdown]
# Here, we allow for a variable number of hidden layers by accepting a tuple of neurons in hidden_size argument. We use a loop to create a list of layers with nn.Linear and the specified activation function and dropout (if any), followed by setting the final output layer.
#
# Note that in the constructor we also added two new arguments: activation and dropout, which allow the user to specify the activation function for the hidden layers and the dropout probability respectively. We support three common activation functions: ReLU, sigmoid, and tanh, but you can add others if you like. We also allow the user to set the dropout probability, with a default of zero (i.e. no dropout).
#
# Here's an example usage:

# %%
input_size = 1
output_size = 1
hidden_size = (10, )
activation = nn.ReLU()

model = MLPRegressor(input_size, output_size, hidden_size, activation=activation)

# %% [markdown]
# This can be deployed to get identical results as before:

# %%
train_and_plot_mlp(model)

# %% [markdown]
# However, we can also easily modify the architecture to obtain better results:

# %%
input_size = 1
output_size = 1
hidden_size = (10, 10, 10, 10)
activation = nn.LeakyReLU()

model = MLPRegressor(input_size, output_size, hidden_size, activation=activation)

train_and_plot_mlp(model)


# %% [markdown]
# Here the performance is improved by about 4x by increasing the depth of the network.
# The `hidden_size` could be controlled by a hyperparameter tuning algorithm, whereas it would be difficult to build new `nn.Module` subclasses during the optimization loop.

# %% [markdown]
# ## Lazy layers
#
# In the previous implementation, we manually specified the input and output dimensions of the `nn.Linear` layers.
# There is now support for so-called "lazy" layers that infer the input dimensions based on what is passed from the preceding layer.
# In this module, the weight and bias are of `torch.nn.UninitializedParameter` class.
# They will be initialized after the first call to forward is done and the module will become a regular `torch.nn.Linear` module.
#
# The result is a slightly shorter model class definition with fewer repeated parameters:
#
#

# %%
class MLPRegressor(nn.Module):
    def __init__(self, output_size, hidden_size=(100, ), activation=nn.ReLU()):
        super(MLPRegressor, self).__init__()

        layers = []
        for layer_size in hidden_size:
            layers.append(nn.LazyLinear(layer_size))
            layers.append(activation)

        layers.append(nn.LazyLinear(output_size))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        return x


# %% [markdown]
# This operates exactly the same way as before:

# %%
output_size = 1
hidden_size = (10, 10, 10, 10)
activation = nn.LeakyReLU()

model = MLPRegressor(output_size, hidden_size, activation=activation)

train_and_plot_mlp(model)


# %% [markdown]
# At the moment this is a minor convenience but when we get into convolutional layers or other more sophisticated transforms it will be a major quality of life improvement!
# It is especially helpful for automated architecture tuning.

# %% [markdown]
# ## Submodules
#
# There is no reason why all of the features of your model needs to be implemented in a single class.
# Instead, we can build subclasses inside the main model object.
# This can help us build up more complex architectures with reusable components.
# Here's a contrived example to demonstrate:

# %%
class LinearPlusLeakyReLU(nn.Module):
    def __init__(self, output_size):
        super(LinearPlusLeakyReLU, self).__init__()

        self.fc = nn.LazyLinear(output_size)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x

class MLPRegressor(nn.Module):
    def __init__(self, output_size, hidden_size=(100, )):
        super(MLPRegressor, self).__init__()

        layers = []
        for layer_size in hidden_size:
            layers.append(LinearPlusLeakyReLU(layer_size))

        layers.append(nn.LazyLinear(output_size))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        return x


# %% [markdown]
# This won't change anything in terms of the performance of the model:

# %%
output_size = 1
hidden_size = (10, 10, 10, 10)

model = MLPRegressor(output_size, hidden_size)

train_and_plot_mlp(model)

# %% [markdown]
# However, it lets us build resuable "blocks" that can combine into bigger, more complex models like so:
#
# <img src="../lectures/assets/lecture12_resnet_architecture.jpg" alt="Architecture diagram of the ResNet-50 convolutional neural network" width=600>

# %% [markdown]
# # Batches and data I/O
#
# When training a neural network, it's often not feasible to pass the entire dataset through the network at once.
# This is because the dataset might be too large to fit into memory, or because passing the entire dataset through the network in one go might be computationally expensive.
#
# To overcome these issues, we can break the dataset into smaller **batches** and feed one batch at a time to the network.
# This is known as mini-batch training, and it has several benefits:
#
# 1. Memory efficiency: By processing data in small batches, we can work with larger datasets that we wouldn't be able to fit into memory otherwise.
#
# 2. Computation efficiency: Processing one batch at a time is often faster than processing the entire dataset at once, especially if we use a GPU to perform the computations.
#
# 3. Better generalization: Mini-batch training can help prevent overfitting by introducing noise into the optimization process, which can help the network generalize better to new data.
#
# To implement mini-batch training in PyTorch, we can use the `Dataset` and `DataLoader` classes.

# %% [markdown]
# ## `DataLoader`
#
# We'll start with `DataLoader`, an iterable that abstracts this complexity for us in an easy API:

# %%
from torch.utils.data import DataLoader

# create an interable with (x, y) pairs:
train_data = [(xlt[i], ylt[i]) for i in range(xlt.shape[0])]

# make the DataLoader object:
dl = DataLoader(train_data, batch_size=8, shuffle=True)

# %% [markdown]
# The `batch_size` argument specifies the number of samples to include in each batch, and the `shuffle` argument tells the DataLoader to shuffle the data before each epoch.
# We could also specify `num_workers` to use a specific number of subprocesses to load the data, which can speed up the loading process if it's expensive.
#
# To demonstrate the effect, here's how we would loop over the data during a training loop:

# %%
for x, y in dl:
    print(x)
    print(y)
    break  # avoid flooding the output

# %% [markdown]
# ## Dataset
#
# Sometimes we can't load all the data at once in order to make an iterable like `train_data` above.
# In this case, we need to use a `Dataset` that can load specific observations (and unload them!) during the training loop.
# Here's a contrived example that implements the necessary methods for tabular data.
# We'll start by loading a `DataFrame`:

# %%
import pandas as pd
import os

# Set the path to the data file
filename = 'concrete.csv'
local_path = f'../datasets/{filename}'
github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'

# Load the data: try local path first, fallback to GitHub for Colab
if os.path.exists(local_path):
    df = pd.read_csv(local_path)
else:
    df = pd.read_csv(github_url)
df.head()


# %% [markdown]
# Now we need to define a `Dataset` that implements the `__len__` and `__getitem__` methods:

# %%
class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values
        self.features = self.data[:, :-1]
        self.labels = self.data[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.features[index]).float()
        y = torch.tensor(self.labels[index]).float()
        return x, y


# %% [markdown]
# Finally, we can iterate over this `Dataset` object to retrieve `(x, y)` pairs:

# %%
dataset = TabularDataset(df)
for x, y in dataset:
    print(x)
    print(y)
    break

# %% [markdown]
# This would tpyically be implemented inside a `DataLoader`:

# %%
dl = DataLoader(dataset, batch_size=4, shuffle=True)

for x, y in dl:
    print(x)
    print(y)
    break

# %% [markdown]
# Now we can easily retrieve shuffled minibatches of a given `batch_size` without having to worry about reshaping or converting the `tensors` from `array` or `DataFrame` during the training loop.

# %% [markdown]
# ## [Check your understanding]
#
# * Create separate `Dataset` and `DataLoader` for the concrete compressive strength dataset **with train/test split**
# > You should split the data first, then make the separate objects for each set
# * Train a regression model in `pytorch`
# * Report the performance on the test set
# * *Bonus:* evaluate whether `batch_size` has any effect on the performance
# * *Bonus:* tune the other hyperparameters to improve the performance

# %%
from sklearn import model_selection

class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, index):
        self.data = dataframe.values
        self.features = self.data[index, :-1]
        self.labels = self.data[index, -1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = torch.tensor(self.features[index]).float()
        y = torch.tensor(self.labels[index]).float()
        return x, y

train_idx, test_idx = model_selection.train_test_split(np.arange(len(df)))

train_ds = TabularDataset(df, train_idx)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

test_ds = TabularDataset(df, test_idx)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)

model = MLPRegressor(1, hidden_size=(20, 20, 20))
for x, y in train_dl:
    out = model(x)

# %%
# define the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)

# do the training
pbar = tqdm.tqdm(np.arange(10))
model.train()
for epoch in pbar:
    for x, y in train_dl:
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x)
        # Compute Loss
        loss = criterion(y_pred, y)

        pbar.set_postfix_str(f'train loss: {loss.item():.3e}')
        # Backward pass
        loss.backward()
        optimizer.step()

with torch.no_grad():
    mse = 0
    for x, y in test_dl:
        model_out = model(x).detach().numpy()
        mse += np.sum((model_out - y.detach().numpy())**2) / len(x)
    print('rmse = ', np.sqrt(mse))

# %% [markdown]
# # Model I/O
#
# Saving and loading trained models is an essential step in the machine learning pipeline, as it enables us to reuse and deploy our models.
#
# There are different ways to save and load PyTorch models, and each method has its own benefits and drawbacks. Let's explore some of the common ways to save and load PyTorch models.

# %% [markdown]
# ## Saving and loading the entire model
# The simplest way to save a PyTorch model is to save the entire model, including its architecture and trained parameters, in a file. This method uses the `torch.save()` function to save the model and the `torch.load()` function to load the model.
# For example, to save a model to a file, we can use the following code:
#
#
#

# %%
torch.save(model, 'model.pth')

# %% [markdown]
# And to load the model from the file, we can use the following code:

# %%
model = torch.load('model.pth')

# %% [markdown]
# The benefit of this method is that it is easy to use and works well for small models. However, it can be slow for large models, and it requires loading the entire model into memory, which can be a problem for memory-constrained systems.

# %% [markdown]
# ## `state_dict`
# An alternative method is to save and load only the model's state_dict, which is a dictionary containing the model's parameters. This method uses the model.state_dict() function to save the state_dict and the model.load_state_dict() function to load the state_dict.
# For example, to save a model's state_dict to a file, we can use the following code:
#

# %%
torch.save(model.state_dict(), 'model_state_dict.pth')

# %% [markdown]
# And to load the state_dict from the file, we can use the following code:

# %%
model.load_state_dict(torch.load('model_state_dict.pth'))

# %% [markdown]
# The benefit of this method is that it is more memory-efficient than saving the entire model, as it only saves the model's parameters. However, it requires the model architecture to be defined before loading the state_dict, which can be a problem if the model architecture has changed since the model was saved.

# %% [markdown]
# ## Checkpoint files
#
# A third method is to save and load checkpoint files, which are files that contain not only the model's state_dict but also other information such as the optimizer state, the epoch, and the loss. This method uses the torch.save() function to save a dictionary containing the state_dict, the optimizer state, and other information, and the torch.load() function to load the dictionary.
# For example, to save a checkpoint file, we can use the following code:
#
#

# %%
checkpoint = {'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'epoch': epoch,
              'loss': loss}
torch.save(checkpoint, 'checkpoint.pth')

# %% [markdown]
# And to load the checkpoint file, we can use the following code:

# %%
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# %% [markdown]
# The benefit of this method is that it allows us to resume training from a saved checkpoint, as it contains the optimizer state and the epoch. However, it can be more complicated to use than the other methods.

# %% [markdown]
# ## TorchScript
#
# TorchScript is a way to convert PyTorch models into a format that can be executed outside of the Python runtime environment. This is useful when deploying models to production environments or other environments where Python is not available or not practical. TorchScript uses a tracing or scripting approach to create a serialized representation of the model's computation graph.
#
# To use TorchScript, you can create a ScriptModule from a PyTorch module by using the torch.jit.script() function. This function compiles the model's computation graph into a serialized TorchScript representation that can be saved to a file and executed in a C++ runtime. Here's an example:
#
#

# %%
traced_model = torch.jit.script(model)
traced_model.save("model.pt")

# %% [markdown]
# This creates an instance of MyModel, converts it to a TorchScript representation using `torch.jit.script()`, and saves it to a file named model.pt. This TorchScript representation can then be loaded and executed outside of the Python environment.
#
# The model can later be loaded using the following:

# %%
traced_model = torch.jit.load("model.pt")

# %% [markdown]
# Warning: you need to make sure that the model is loaded onto the right device!
# If the model parameters are stored on the GPU or another hardware accelerator, it won't be able to be loaded onto the CPU.
# I use the following commands when saving and loading models with TorchScript:

# %%
# saving
traced_model = torch.jit.script(model.cpu())
traced_model.save("model.pt")

# loading
model = torch.jit.load("model.pt", map_location='cpu')
model.eval()

# %% [markdown]
# ## ONNX
#
# ONNX is another tool that is used to deploy models in different environments. ONNX stands for Open Neural Network Exchange, and it is an open-source format that allows for the exchange of models between different frameworks. ONNX models can be used in environments such as mobile devices, web applications, and cloud services.
#
# To convert a PyTorch model to the ONNX format, you can use the `torch.onnx.export()` function. This function exports the model's computation graph to an ONNX file that can be loaded and executed in other environments. Here's an example:

# %%
input_tensor = torch.randn_like(x)

torch.onnx.export(model, input_tensor, "model.onnx", input_names=["input"], output_names=["output"])

# %% [markdown]
# This exports the `MLPRegressor` model's computation graph to an ONNX file named `model.onnx`, and specifies the input and output tensor names.
# Here's the actual content of the file:

# %%
with open('model.onnx', 'rb') as fid:
    lines = fid.readlines()

for l in lines:
    print(l)

# %% [markdown]
# The data stored in this byte string can be read by an ONNX parser to recreate the trained model.

# %% [markdown]
# # `pytorch-lightning`
#
# PyTorch Lightning is a lightweight wrapper around PyTorch that helps you organize your code and reduce boilerplate, making it easier to write and maintain complex deep learning projects.
# We'll go over how to use PyTorch Lightning to streamline your PyTorch code.
#
# The first step will be installation:

# %%
# !pip install pytorch-lightning

# %% [markdown]
# ### `LightningModule`
#
# Let's see how we can streamline `MLPRegressor` code from before using PyTorch Lightning.
# The first thing we need to do is create a PyTorch Lightning module that wraps our existing PyTorch model.
# Here's what that might look like:

# %%
import pytorch_lightning as pl
from torch import optim


# define the LightningModule
class LitMLPRegressor(pl.LightningModule):
    # the top part looks the same as torch.nn.Module:
    def __init__(self, output_size, hidden_size=(100, ), activation=nn.ReLU()):
        super(LitMLPRegressor, self).__init__()

        layers = []
        for layer_size in hidden_size:
            layers.append(nn.LazyLinear(layer_size))
            layers.append(activation)

        layers.append(nn.LazyLinear(output_size))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        return x

    # the bottom part is for Pytorch Lightning:
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        out = self.forward(x)
        loss = nn.functional.mse_loss(out, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the lightning model
pl_model = LitMLPRegressor(1, (100, ), activation=nn.Tanh())

# %% [markdown]
# As you can see, we've defined a new class `LitMLPRegressor` that extends `pl.LightningModule`.
# We've moved our model definition into this new class, but we've also defined two new methods: `training_step` and `configure_optimizers`.
# The `training_step` method defines what happens in each iteration of the training loop, while the `configure_optimizers` method defines the optimizer used during training.

# %% [markdown]
# ## `Trainer`
#
# One of the main benefits of PyTorch Lightning is that it provides many other features that can help streamline your code, such as automatic checkpointing, early stopping, and distributed training. To use these features, we need to create a Trainer object:

# %%
trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1)

# %% [markdown]
# Note that if hardware accelerators were available, the `trainer` could handle moving our data around between the CPU and CUDA device.
# To train the model we need to provide data and call `trainer.fit` to start the training loop:

# %%
train_data = [(xlt[i], ylt[i]) for i in range(xlt.shape[0])]
dl = DataLoader(train_data, batch_size=64, shuffle=True)

trainer.fit(pl_model, dl)

# %% [markdown]
# By default, PyTorch Lightning will log the training loss and other metrics to the terminal. You can also configure PyTorch Lightning to log to a file or to a remote server. For example, to log to a file, you can pass the logger argument to the `Trainer` constructor:

# %%
from pytorch_lightning.loggers import CSVLogger

logger = CSVLogger('logs', name='mlp-regressor')

trainer = pl.Trainer(max_epochs=10, logger=logger, log_every_n_steps=1)
trainer.fit(pl_model, dl)

# %% [markdown]
# This will create a new folder called logs in the current directory, and save the training logs to a CSV file inside that folder.
# We can access the data like this:
#

# %%
log_path = 'logs/mlp-regressor/version_0/metrics.csv'
if os.path.exists(log_path):
    metrics = pd.read_csv(log_path)
else:
    colab_path = '/content/logs/mlp-regressor/version_0/metrics.csv'
    if os.path.exists(colab_path):
        metrics = pd.read_csv(colab_path)
    else:
        print(f"Warning: Log file not found at {log_path} or {colab_path}")
        metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'validation_loss'])
metrics.head(6)

# %% [markdown]
# This can be used to create a plot of the train loss after the fact:

# %%
plt.plot( metrics['epoch'] * len(dl) + metrics['step'], metrics['train_loss'])


# %% [markdown]
# ## Validation

# %% [markdown]
# Another benefit of PyTorch Lightning is that it provides a simple interface for testing your trained model on a separate validation or test dataset.
# Here's an example of how you can define a validation_step method in your PyTorch Lightning module to test your model:

# %%
class LitMLPRegressor(pl.LightningModule):
    def __init__(self, output_size, hidden_size=(100, ), activation=nn.ReLU()):
        super(LitMLPRegressor, self).__init__()

        layers = []
        for layer_size in hidden_size:
            layers.append(nn.LazyLinear(layer_size))
            layers.append(activation)

        layers.append(nn.LazyLinear(output_size))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.mse_loss(out, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        return optimizer

    # new feature:
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.mse_loss(out, y)
        residual = out - y
        rmse = torch.sqrt( torch.mean( residual**2 ) )
        rsq = 1 - torch.var(residual) / torch.var(y)
        self.log('val_loss', loss)
        # additional metrics can be logged here:
        self.log('val_rmse', rmse)
        self.log('val_rsq', rsq)


# %% [markdown]
# In this example, we've defined a new method validation_step that computes the loss and accuracy on a validation batch. We're using the `self.log` method to log these metrics, which will be displayed in the PyTorch Lightning progress bar and can also be saved to a file.
#
# Let's retrain the updated model so we can take advantage of this feature:
#
#

# %%
pl_model = LitMLPRegressor(1, (100, ), activation=nn.Tanh())
trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1)
trainer.fit(pl_model, dl)

# %% [markdown]
# To test the trained model on a validation dataset, we can call the `trainer.validate` method.
# This will run the `validation_step` method for each batch in the validation dataset and log the results to the terminal or to a file.

# %%
# we'll just reuse the training data for this example:
val_dl = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
trainer.validate(pl_model, val_dl)

# %% [markdown]
# ## [Check your understanding]
#
# Implement a regression model for the concrete compressive strength data using Pytorch Lightning.

# %%

# %% [markdown]
# # Bonus: `skorch`
#
# [`skorch`](https://skorch.readthedocs.io/en/stable/) is a package that gives `pytorch` models an interface very similar to `sklearn` (thus the name **sk**learn-pyt**orch**).
# It ends up looking something like this:
#
# ```
# class MyModule(torch.nn.Module):
#     ...
#
# net = NeuralNet(
#     module=MyModule,
#     criterion=torch.nn.NLLLoss,
# )
# net.fit(X, y)
# y_pred = net.predict(X_valid)
# ```
#
# This may be helpful when you want to train deep learning models side by side with simpler models like linear regression or trees.

# %%
