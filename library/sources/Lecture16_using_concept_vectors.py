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
# id: Lecture16_using_concept_vectors
# type: Foundational
# parent_lecture: Lecture16
# ---
#
# ## Using concept vectors
#
# We can also modify single curves using the concept vectors.
# Here's a sample curve that lies close to the origin:
#
# Let's compute the $a_0$ concept vector again:
#
# We can normalize these coefficients to obtain a unit vector (direction):
#
# Then we apply the vector to modify the selected latent code:
#
# If we increase the magnitude of the vector we will increase the effect:

# %%
close_to_origin = np.argmin(np.linalg.norm(z, axis=1))

fig, axes = plt.subplots(1, 2)

ax = axes[0]
_ = ax.scatter(*z[close_to_origin, :2], label='Real')
_ = ax.legend()

ax = axes[1]
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.legend()

from sklearn import linear_model

lr = linear_model.LinearRegression().fit(z, a[:, [0]])
print(f'R2 = {lr.score(z, a[:, 0]):.3f}; coef = {lr.coef_}')

vec_a0 = lr.coef_ / np.linalg.norm(lr.coef_)

# compute the latent codes and decode the functions
z_plus = z[close_to_origin] + vec_a0
out_plus = model.decoder(torch.tensor(z_plus).float())
gen_y_plus = out_plus[0].detach().numpy()

z_minus = z[close_to_origin] - vec_a0
out_minus = model.decoder(torch.tensor(z_minus).float())
gen_y_minus = out_minus[0].detach().numpy()

# make the plot
fig, axes = plt.subplots(1, 2)

ax = axes[0]
_ = ax.scatter(*z[:, :2].T, label='All data')
_ = ax.scatter(*z[close_to_origin, :2], label='Real')
_ = ax.scatter(*z_plus[0, :2], label='+a0')
_ = ax.scatter(*z_minus[0, :2], label='-a0')
_ = ax.legend()

ax = axes[1]
_ = ax.plot(x, np.mean(y, axis=0), label='All data')
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.plot(x, gen_y_plus, label='+a0')
_ = ax.plot(x, gen_y_minus, label='-a0')
_ = ax.legend()

z_plus = z[close_to_origin] + vec_a0 * 5
out_plus = model.decoder(torch.tensor(z_plus).float())
gen_y_plus = out_plus[0].detach().numpy()

z_minus = z[close_to_origin] - vec_a0 * 5
out_minus = model.decoder(torch.tensor(z_minus).float())
gen_y_minus = out_minus[0].detach().numpy()

fig, axes = plt.subplots(1, 2)

ax = axes[0]
_ = ax.scatter(*z[:, :2].T, label='All data')
_ = ax.scatter(*z[close_to_origin, :2], label='Real')
_ = ax.scatter(*z_plus[0, :2], label='+a0')
_ = ax.scatter(*z_minus[0, :2], label='-a0')
_ = ax.legend()

ax = axes[1]
_ = ax.plot(x, np.mean(y, axis=0), label='All data')
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.plot(x, gen_y_plus, label='+a0')
_ = ax.plot(x, gen_y_minus, label='-a0')
_ = ax.legend()
