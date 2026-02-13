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
# id: Lecture12_lazy_layers
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## Lazy layers
#
# In the previous implementation, we manually specified the input and output dimensions of the `nn.Linear` layers.
# There is now support for so-called "lazy" layers that infer the input dimensions based on what is passed from the preceding layer.
# In this module, the weight and bias are of `torch.nn.UninitializedParameter` class.
# They will be initialized after the first call to forward is done and the module will become a regular `torch.nn.Linear` module.
#
# The result is a slightly shorter model class definition with fewer repeated parameters:
#
# This operates exactly the same way as before:
#
# At the moment this is a minor convenience but when we get into convolutional layers or other more sophisticated transforms it will be a major quality of life improvement!
# It is especially helpful for automated architecture tuning.

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

output_size = 1
hidden_size = (10, 10, 10, 10)
activation = nn.LeakyReLU()

model = MLPRegressor(output_size, hidden_size, activation=activation)

train_and_plot_mlp(model)
