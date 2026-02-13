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
# id: Lecture06_concept_of_reconstruction
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Concept of reconstruction
#
# We can also take these embeddings and "un-project" them back to the originals.
# (Remember this embedding operation was basically a rotation).
# Let's evaluate these in the 2D plane that corresponds to the greatest variance in the data:

# %%
ascending = np.argsort(pca.components_[0])
descending = ascending[::-1]
top2 = descending[:2]
print(top2)
print(x.columns[top2])

x_recon = np.dot(z_manual, pca.components_) + np.mean(x.values, axis=0)

fig, ax = plt.subplots()
ax.scatter(*x_recon[:, top2].T, label='Reconstruction')
ax.scatter(*x.values[:, top2].T, label='Original',
           marker='s', edgecolor='tab:orange', facecolor='none')
ax.legend()
