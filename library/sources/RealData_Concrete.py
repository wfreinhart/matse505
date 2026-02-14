# %% [markdown]
# ---
# id: RealData_Concrete
# type: Data
# ---
# # Concrete Compressive Strength Data
# Loading the concrete dataset from Kaggle.

# %%
import pandas as pd
import os

filename = 'concrete.csv'
local_path = '../datasets/' + filename

if os.path.exists(local_path):
    data = pd.read_csv(local_path)
else:
    # Fallback to local path relative to repo root if needed
    data = pd.read_csv('datasets/concrete.csv')

# Prepare features and target
x = data.drop('Concrete compressive strength(MPa, megapascals) ', axis=1)
y = data['Concrete compressive strength(MPa, megapascals) ']

print(f"Loaded concrete dataset: {x.shape}")
