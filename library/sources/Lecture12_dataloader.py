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
# id: Lecture12_dataloader
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## `DataLoader`
#
# We'll start with `DataLoader`, an iterable that abstracts this complexity for us in an easy API:
#
# The `batch_size` argument specifies the number of samples to include in each batch, and the `shuffle` argument tells the DataLoader to shuffle the data before each epoch.
# We could also specify `num_workers` to use a specific number of subprocesses to load the data, which can speed up the loading process if it's expensive.
#
# To demonstrate the effect, here's how we would loop over the data during a training loop:

# %%
from torch.utils.data import DataLoader

# create an interable with (x, y) pairs:
train_data = [(xlt[i], ylt[i]) for i in range(xlt.shape[0])]

# make the DataLoader object:
dl = DataLoader(train_data, batch_size=8, shuffle=True)

for x, y in dl:
    print(x)
    print(y)
    break  # avoid flooding the output
