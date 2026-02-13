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
# id: Lecture15_a_simple_example
# type: Foundational
# parent_lecture: Lecture15
# ---
#
# ## A simple example
#
# We will return to the polynomial example for simplicity.
# In future lectures we will implement autoencoders for images using the same principles but with CNNs.
# Let's create some sample polynomials with 5 basis functions (the Legendre polynomials).
#
# These are 101-dimensional data artifacts, but they really only represent a 5-dimensional space (i.e., coefficients of the basis set).
# I select this example because there is an analytical relationship between the high-dimensional objects and the low-dimensional coefficients.
#
# Now we'll use this as our dataset:

# %%
import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

npoly = 5
a = 2*(rng.rand(npoly) - 0.5)
x = np.linspace(-1, 1, 101)

def f(x):
    y = np.sum([a[i] * legendre(i)(x) for i in range(npoly)], axis=0)
    return y

fig, ax = plt.subplots()
ax.plot(x, f(x), '.')

import tqdm

rng = np.random.RandomState(0)

npoly = 5
x = np.linspace(-1, 1, 101)

def f(x, a):
    return np.sum([a[i] * legendre(i)(x) for i in range(npoly)], axis=0)

def make_function():
    a = 2*(rng.rand(npoly) - 0.5)
    return a, f(x, a)

# need to make a whole bunch of functions!
N_samples = 1000
a = np.zeros([N_samples, npoly])
y = np.zeros([N_samples, len(x)])
for i in tqdm.tqdm(np.arange(N_samples)):
    this_a, this_f = make_function()
    a[i] = this_a
    y[i] = this_f

print( y.shape )
