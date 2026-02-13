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
# id: Lecture12_dataset
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## Dataset
#
# Sometimes we can't load all the data at once in order to make an iterable like `train_data` above.
# In this case, we need to use a `Dataset` that can load specific observations (and unload them!) during the training loop.
# Here's a contrived example that implements the necessary methods for tabular data.
# We'll start by loading a `DataFrame`:
#
# Now we need to define a `Dataset` that implements the `__len__` and `__getitem__` methods:
#
# Finally, we can iterate over this `Dataset` object to retrieve `(x, y)` pairs:
#
# This would tpyically be implemented inside a `DataLoader`:
#
# Now we can easily retrieve shuffled minibatches of a given `batch_size` without having to worry about reshaping or converting the `tensors` from `array` or `DataFrame` during the training loop.

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

dataset = TabularDataset(df)
for x, y in dataset:
    print(x)
    print(y)
    break

dl = DataLoader(dataset, batch_size=4, shuffle=True)

for x, y in dl:
    print(x)
    print(y)
    break
