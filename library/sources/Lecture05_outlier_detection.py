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
# id: Lecture05_outlier_detection
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## Outlier detection
#
# Another use of dimensionality reduction can be outlier detection. Let's switch our definition of $X$ from the compositions to the properties.
#
# Clearly something is strange with that one point off on its own. Let's check it out:
#
# Look at that `Tensile Strength` value! If we examine the `Tensile Strength` data, we will see that it is indeed anomalous:
#
# This value was almost certainly entered incorrectly (e.g., wrong decimal place). We can remove the outlier using `drop()`:
#
# Then we can look at our manifold again:
#
# Of course here we see some more strange behavior. We could continue to investigate these additional outliers using this method (though we may not always want to remove them, some might be "real" special cases).
#
# To be clear, we could easily have found this anomaly by analyzing the `Tensile Strength (MPa)` column individually.
# However, the upshot is that we see that data point as an anomaly in the first Principal Component, without considering any column-wise statistics.

# %%
x = data.loc[:, ' 0.2% Proof Stress (MPa)':' Reduction in Area (%)']

pca = decomposition.PCA().fit(x)
P = pca.transform(x)

fig, ax = plt.subplots()
ax.scatter(P[:, 0], P[:, 1])
ax.set_xlabel('$P_0$')
ax.set_ylabel('$P_1$')

outlier = np.argmax(P[:, 0])  # find the largest value in Z_0
print(x.loc[outlier])

fig, ax = plt.subplots()
_ = ax.hist(x[' Tensile Strength (MPa)'], bins=100)
ax.set_xlabel('Tensile Strength (MPa)')
ax.set_ylabel('Count')

clean_data = data.drop(index=outlier)
clean_y = encoder.transform(clean_data['Alloy family'])

clean_x = clean_data.loc[:, ' 0.2% Proof Stress (MPa)':' Reduction in Area (%)']

pca = decomposition.PCA().fit(clean_x)
P = pca.transform(clean_x)

fig, ax = plt.subplots()
ax.scatter(P[:, 0], P[:, 1])
ax.set_xlabel('$P_0$')
ax.set_ylabel('$P_1$')
