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
# id: Lecture12_submodules
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## Submodules
#
# There is no reason why all of the features of your model needs to be implemented in a single class.
# Instead, we can build subclasses inside the main model object.
# This can help us build up more complex architectures with reusable components.
# Here's a contrived example to demonstrate:
#
# This won't change anything in terms of the performance of the model:
#
# However, it lets us build resuable "blocks" that can combine into bigger, more complex models like so:
#
# <img src="../lectures/assets/lecture12_resnet_architecture.jpg" alt="Architecture diagram of the ResNet-50 convolutional neural network" width=600>

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

output_size = 1
hidden_size = (10, 10, 10, 10)

model = MLPRegressor(output_size, hidden_size)

train_and_plot_mlp(model)
