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
# id: Lecture20_screening
# type: Foundational
# parent_lecture: Lecture20
# ---
#
# ## Screening
#
# And now we repeat our screening approach by KDE:
#
# We see that the value obtained by screening is close but not equal to the true minimum.

# %%
from scipy.stats import gaussian_kde

# fit kde model
kde = gaussian_kde(x[idx_train].T)

# perform sampling
x_fake = kde.resample(10000).T

# predict on these new samples
y_fake = model.predict(x_fake)

# make figure
fig, ax = plt.subplots()
im = ax.scatter(*x_fake.T, s=4, c=y_fake)
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
ax.set_aspect('equal')
cb = plt.colorbar(im)

# plot the true minimum
ax.plot([0.455], [0.472], 's', color='tab:green', label='Ground Truth')

# plot the minimum by screening
min_idx = np.argmin(y_fake)
ax.plot(*x_fake[min_idx], '^', color='tab:orange', label='Best Result')
print(f'min value is {y_fake[min_idx]} at {xy_to_comp( x_fake[min_idx] )}')

ax.legend()
