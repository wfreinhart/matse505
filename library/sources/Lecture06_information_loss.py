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
# id: Lecture06_information_loss
# type: Foundational
# parent_lecture: Lecture06
# ---
#
# ## Information loss
#
# We can plot the cumulative explained variance to see how many of these are significant.
#
# While this is slightly subjective (or at least depends on the intended application), we can probably say that we won't be able to see a difference with more than 9 components (>99.99% explained variance).
# Let's investigate what happens when we use less or more components.
#
# So far, so good. Let's make a function and try using subsequently fewer components.
#
# Here we see that the fidelity of the reconstruction decays rapidly as we move to fewer components.
# However, these were the dominant elements in the original space.
# If we consider some other elements we'll see much worse reconstruction much earlier.
#
# Since the principal components did not consider the variations in these (Si, C), they are not captured perfectly here.

# %%
fig, ax = plt.subplots()
n = np.arange(pca.n_components_)+1
var_c = np.cumsum(pca.explained_variance_ratio_)
print(var_c)
ax.plot(n, var_c, '.-')
ax.set_xlabel('Components')
ax.set_ylabel('Cumulative explained variance')

nc = 9
z_trunc = z_manual[:, :nc]
x_recon = np.dot(z_trunc, pca.components_[:nc, :]) + np.mean(x.values, axis=0)

fig, ax = plt.subplots()
ax.scatter(*x_recon[:, top2].T, label='Reconstruction')
ax.scatter(*x.values[:, top2].T, label='Original',
           marker='s', edgecolor='tab:orange', facecolor='none')
ax.legend()

def plot_reconstruction(nc):
    z_trunc = z_manual[:, :nc]
    x_recon = np.dot(z_trunc, pca.components_[:nc, :]) + np.mean(x.values, axis=0)

    fig, ax = plt.subplots()
    ax.scatter(*x_recon[:, top2].T, label='Reconstruction')
    ax.scatter(*x.values[:, top2].T, label='Original',
            marker='s', edgecolor='tab:orange', facecolor='none')
    ax.legend()
    ax.text(0.95, 0.05, f'{nc} components; {var_c[nc]*100:.2f}% explained variance',
            ha='right', transform=ax.transAxes)

    ax.set_xlabel(x.columns[top2[0]])
    ax.set_ylabel(x.columns[top2[1]])

    return fig

fig = plot_reconstruction(8)
fig = plot_reconstruction(6)
fig = plot_reconstruction(4)
fig = plot_reconstruction(2)

nc = 8  # looked fine for [Cr, Mo]

z_trunc = z_manual[:, :nc]
x_recon = np.dot(z_trunc, pca.components_[:nc, :]) + np.mean(x.values, axis=0)

fig, ax = plt.subplots()
ax.scatter(*x_recon[:, :2].T, label='Reconstruction')
ax.scatter(*x.values[:, :2].T, label='Original',
        marker='s', edgecolor='tab:orange', facecolor='none')
ax.legend()
ax.text(0.95, 0.05, f'{nc} components; {var_c[nc]*100:.2f}% explained variance',
        ha='right', transform=ax.transAxes)

ax.set_xlabel(x.columns[0])
ax.set_ylabel(x.columns[1])
