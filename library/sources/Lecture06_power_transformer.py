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
# id: Lecture06_power_transformer
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Power transformer
#
# The power transformer is a scheme to make a non-Gaussian distribution look more Gaussian.
# Here's a visual example:
#
# <img src="../lectures/assets/power_transformer.jpg" width=600 alt="Effect of PowerTransformer on skewed data distributions to make them more Gaussian-like">
#
# Why would we do this?
# This is an attempt to avoid the long tails dominating the reduced space.
#
# Here the power transformer provides a middle ground between the standard scaler and min-max scaler.
# Note that all three of these preprocessing methods make the point cloud more isotropic and portray additional features compared to the PCA on raw data.
#
# A quick note on the inverse transform:

# %%
scaler = preprocessing.PowerTransformer().fit(x)
xs = scaler.transform(x)

pca = decomposition.PCA().fit(xs)
P = pca.transform(xs)

fig, ax = plt.subplots()
im = ax.scatter(P[:, 0], P[:, 1], c=clean_data['V'])
plt.colorbar(im)

print( 'original', x.values[0] )
scaler = preprocessing.StandardScaler().fit(x)
xs = scaler.transform(x)
print( 'scaled', xs[0] )
print( 'reconstructed', scaler.inverse_transform(xs)[0] )
