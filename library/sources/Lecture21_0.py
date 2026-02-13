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
# id: Lecture21_0
# type: Foundational
# parent_lecture: Lecture21
# ---
#
#
#
# Today's topics:
# * Working with molecules
# * Machine learning with fingerprints
# * Geometric deep learning
#
# Let's install some dependencies up front so we don't have to restart our runtime later:

# %%
# Install rdkit
# !pip install rdkit

# Install pytorch-geometric
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# Install pytorch-lightning
# !pip install pytorch-lightning
