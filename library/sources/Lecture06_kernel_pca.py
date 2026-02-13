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
# id: Lecture06_kernel_pca
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Kernel PCA
#
# We can apply the "kernel trick" to use PCA for nonlinear data.
# First we apply the nonlinear kernel and then we do the linear projection.
# This is very similar to the example of polynomial features in a linear regression.
#
# Here is a great visual example of the kernel trick at work:
#
# <img src="../lectures/assets/kernel_trick.jpg" width=600 alt="Diagram illustrating the kernel trick: projecting linearly inseparable data into a higher dimension where it becomes separable">
#
# After this nonlinear transformation, the decision boundary can be very easily defined by a linear model.
#
# Let's try the `poly` kernel on the nonlinear data above.
# Note that for manifold learning, `sklearn` uses `fit()` and `transform()` just like the PCA interface instead of the `predict()` for supervised learning:
#
# Unfortunately, this result doesn't look any "better" than the linear PCA.
# We might also try the `rbf` option, which stands for Radial Basis Function.
# This is a localized basis function that links nearby points together, which makes sense when trying to identify this spiral manifold.
#
# Nope, the RBF detects the wrong structure in the data.
# That's fine, there's no guarantee that more complex models will yield better results for any given problem.

# %%
# fit the model
pca = decomposition.KernelPCA(kernel='poly').fit(S)

# project X using PCA
St = pca.transform(S)

# plot result
fig, ax = plt.subplots()
ax.scatter(St[:, 0], St[:, 1], c=t)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# fit the model
pca = decomposition.KernelPCA(kernel='rbf').fit(S)

# project X using PCA
St = pca.transform(S)

# plot result
fig, ax = plt.subplots()
ax.scatter(St[:, 0], St[:, 1], c=t)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
