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
# id: Lecture20_a_simple_synthetic_dataset
# type: Foundational
# parent_lecture: Lecture20
# ---
#
# # A simple synthetic dataset
#
# Let's create a simpler dataset and see if we can achieve good results using any of these methods.
#
# This is a fictitious ternary system.
# We could think of it like a 3-component metal alloy.
#
# The above is a 2D representation in the phase diagram, but we need to be able to convert between this and the full 3D representation:
#
# Let's define some "property" of this system.
#
# The true minimum of this function is $f = 6/11$ at the point $(x, y, z) = (6/11, 3/11, 2/11)$, or approximately $(0.545, 0.273, 0.182)$.
# This corresponds to the point $(0.455, 0.472)$.
# Without the equation we wouldn't be able to determine this analytically.

# %%
def sample_tri(Ns):
    """Use rejection sampling to find `Ns` points inside an equilateral triangle."""
    # draw samples
    xy = np.random.rand(Ns*4, 2)
    xy[:, 1] *= np.sqrt(3)/2
    # constrain to the triangle
    c1 = xy[:, 1] < xy[:, 0]*np.sqrt(3)
    c2 = xy[:, 1] < np.sqrt(3)*(1-xy[:, 0])
    xy = xy[np.logical_and(c1,c2)]
    # return only the number requested
    xy = xy[:Ns]
    return xy

out = sample_tri(1000)
fig, ax = plt.subplots()
ax.plot(*out.T, '.')
ax.set_aspect('equal')

def xy_to_comp(xy):
    """Convert 2D coordinates to a ternary composition."""
    x, y = xy.T
    a = 2/np.sqrt(3)*y
    b = 1 - a/2 - x
    c = 2*x + b - 1
    return np.vstack([a,b,c]).T

def comp_to_xy(abc):
    """Convert ternary compositions to 2D coordinates."""
    a, b, c = abc.T
    x = ( 1 + c - b ) /2
    y = np.sqrt(3) * a / 2

    return np.vstack([x, y]).T

a, b, c = xy_to_comp(sample_tri(1000)).T

# draw histograms
fig, ax = plt.subplots()
_ = ax.hist(a, alpha=0.5)
_ = ax.hist(b, alpha=0.5)
_ = ax.hist(c, alpha=0.5)

def quadratic_fom(xy, noise=0):
    """Define a figure of merit to model."""
    abc = xy_to_comp( xy )
    lab = np.array([[1,2,3]])
    metric = np.sum(abc**2 * lab, axis=1)
    metric += noise * np.random.standard_normal(metric.shape)
    return metric

x = sample_tri(1000)
y = quadratic_fom(x, 0.10).reshape(-1, 1)
fig, ax = plt.subplots()
im = ax.scatter(*x.T, s=4, c=y)
ax.set_aspect('equal')
cb = plt.colorbar(im)
