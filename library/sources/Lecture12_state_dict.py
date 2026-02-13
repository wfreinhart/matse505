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
# id: Lecture12_state_dict
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## `state_dict`
# An alternative method is to save and load only the model's state_dict, which is a dictionary containing the model's parameters. This method uses the model.state_dict() function to save the state_dict and the model.load_state_dict() function to load the state_dict.
# For example, to save a model's state_dict to a file, we can use the following code:
#
# And to load the state_dict from the file, we can use the following code:
#
# The benefit of this method is that it is more memory-efficient than saving the entire model, as it only saves the model's parameters. However, it requires the model architecture to be defined before loading the state_dict, which can be a problem if the model architecture has changed since the model was saved.

# %%
torch.save(model.state_dict(), 'model_state_dict.pth')

model.load_state_dict(torch.load('model_state_dict.pth'))
