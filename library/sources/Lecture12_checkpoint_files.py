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
# id: Lecture12_checkpoint_files
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## Checkpoint files
#
# A third method is to save and load checkpoint files, which are files that contain not only the model's state_dict but also other information such as the optimizer state, the epoch, and the loss. This method uses the torch.save() function to save a dictionary containing the state_dict, the optimizer state, and other information, and the torch.load() function to load the dictionary.
# For example, to save a checkpoint file, we can use the following code:
#
# And to load the checkpoint file, we can use the following code:
#
# The benefit of this method is that it allows us to resume training from a saved checkpoint, as it contains the optimizer state and the epoch. However, it can be more complicated to use than the other methods.

# %%
checkpoint = {'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'epoch': epoch,
              'loss': loss}
torch.save(checkpoint, 'checkpoint.pth')

checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
