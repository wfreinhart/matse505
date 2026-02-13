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
# id: Lecture12_torchscript
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## TorchScript
#
# TorchScript is a way to convert PyTorch models into a format that can be executed outside of the Python runtime environment. This is useful when deploying models to production environments or other environments where Python is not available or not practical. TorchScript uses a tracing or scripting approach to create a serialized representation of the model's computation graph.
#
# To use TorchScript, you can create a ScriptModule from a PyTorch module by using the torch.jit.script() function. This function compiles the model's computation graph into a serialized TorchScript representation that can be saved to a file and executed in a C++ runtime. Here's an example:
#
# This creates an instance of MyModel, converts it to a TorchScript representation using `torch.jit.script()`, and saves it to a file named model.pt. This TorchScript representation can then be loaded and executed outside of the Python environment.
#
# The model can later be loaded using the following:
#
# Warning: you need to make sure that the model is loaded onto the right device!
# If the model parameters are stored on the GPU or another hardware accelerator, it won't be able to be loaded onto the CPU.
# I use the following commands when saving and loading models with TorchScript:

# %%
traced_model = torch.jit.script(model)
traced_model.save("model.pt")

traced_model = torch.jit.load("model.pt")

# saving
traced_model = torch.jit.script(model.cpu())
traced_model.save("model.pt")

# loading
model = torch.jit.load("model.pt", map_location='cpu')
model.eval()
