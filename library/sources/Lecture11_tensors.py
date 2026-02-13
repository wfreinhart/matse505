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
# id: Lecture11_tensors
# type: Foundational
# parent_lecture: Lecture11
# ---
#
# ## Tensors
#
# Pytorch `tensors` are similar to numpy `arrays`.
# However, they are not interoperable -- all calculations performed in pytorch need to be done on `tensors`.
#
# These can interact with each other and with the usual operators just like `arrays`:
#
# They also have methods for common operations just like `arrays`:
#
# It is very important to consider the shape and size of the `tensors`.
# Let's explore how these work:
#
# `tensor` plus `int` adds the value of `int` to all elements in the `tensor`.
# What about lower-dimension `tensor` plus higher-dimension `tensor`?
#
# What about a different shape?
# We can change the dimensions using `unsqueeze`:
#
# Now that we know how to change the dimensions, we can see how they interact with the `+` operator:
#
# Unlike `arrays`, `tensors` have an additional attribute called `device`:
#
# This allows for computing on hardware accelerators like Graphics Processing Units (GPUs):
#
# <img src="../lectures/assets/lecture11_computer_parts.jpg" alt="Diagram showing various computer components like CPU and GPU" width=600>
#
# <img src="../lectures/assets/lecture11_hardware_accel.jpg" alt="Comparison between CPU and GPU architectures" width=600>

# %%
import torch

xt = torch.from_numpy(x.values).float()
yt = torch.from_numpy(y.values).float()

print(xt.shape, yt.shape)  # shape works just like numpy
print()
print(xt[:5])              # indexing works just like numpy

print( xt[:1] * 2 )
print( xt[:1]**2 )
print( xt[:1] + yt[:1] )

print( xt.mean() )
print( xt.mean(dim=0) )  # using "dim" instead of "axis"

a = torch.zeros([2, 2, 2])
b = 1
print( a )
print()
print( a + b )

b = torch.tensor([1, 2])
print( b )
print()
print( a + b )

print( b.shape )
print( b )
print()
print( b.unsqueeze(0).shape )
print( b.unsqueeze(0) )
print()
print( b.unsqueeze(1).shape )
print( b.unsqueeze(1) )

c = b.unsqueeze(0)
print( c )
print()
print( a + c )

c = b.unsqueeze(1)
print( c )
print()
print( a + c )

c = b.unsqueeze(1).unsqueeze(2)
print( c )
print()
print( a + c )

xt.device
