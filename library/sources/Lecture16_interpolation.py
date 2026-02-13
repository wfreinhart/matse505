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
# id: Lecture16_interpolation
# type: Foundational
# parent_lecture: Lecture16
# ---
#
# ## Interpolation
#
# Linear interpolation between observations or groups of observations is a very common scheme for generating new samples.
# Let's try interpolating between the maximum and minimum $z_0$ samples:

# %%
top = np.argmax(z[:, 0])
bot = np.argmin(z[:, 0])

fig, ax = plt.subplots()
_ = ax.plot(x, y[top], label='Top')
_ = ax.plot(x, y[bot], label='Bottom')
_ = ax.legend()
