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
# id: Lecture12_saving_and_loading_the_entire_model
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## Saving and loading the entire model
# The simplest way to save a PyTorch model is to save the entire model, including its architecture and trained parameters, in a file. This method uses the `torch.save()` function to save the model and the `torch.load()` function to load the model.
# For example, to save a model to a file, we can use the following code:
#
# And to load the model from the file, we can use the following code:
#
# The benefit of this method is that it is easy to use and works well for small models. However, it can be slow for large models, and it requires loading the entire model into memory, which can be a problem for memory-constrained systems.

# %%
torch.save(model, 'model.pth')

model = torch.load('model.pth')
