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
# id: Lecture20_conditional_rejection_sampling
# type: Foundational
# parent_lecture: Lecture20
# ---
#
# ## Conditional (rejection) sampling
#
# SDV uses rejection sampling to achieve conditional samples on the distributions.
#
# Are any of these realistic?
# To find out, we need to run the inputs through the predictive model.
#
# Unfortunately it seems like these designs give us an above-average TS but one that is within the 600-700 range rather than 800 as requested.
#
# What if we reduce our TS requirement but add a Elongation requirement as well?
#
# Now it seems that the results are all over the place.
# The additional condition made it even harder to approximate this.

# %%
from sdv.sampling import Condition

target = Condition(num_rows=100, column_values={' Tensile Strength (MPa)': 800})
designs = synthesizer.sample_from_conditions(conditions=[target])

designs

xs_fake = x_scaler.transform(designs.iloc[:, 1:-4])

ys_fake = model.predict(xs_fake)
y_fake = y_scaler.inverse_transform(ys_fake)

# make figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

ax.scatter(y_fake[:, 1], y_fake[:, 2], marker='d', color='tab:orange', label='Fake')

ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

ax.legend()

target = Condition(num_rows=100, column_values={' Tensile Strength (MPa)': 500, ' Elongation (%)': 40})
designs = synthesizer.sample_from_conditions(conditions=[target])

xs_fake = x_scaler.transform(designs.iloc[:, 1:-4])

ys_fake = model.predict(xs_fake)
y_fake = y_scaler.inverse_transform(ys_fake)

# make figure
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

ax.scatter(y_fake[:, 1], y_fake[:, 2], marker='d', color='tab:orange', label='Fake')

ax.set_xlabel(y.iloc[:, 1].name)
ax.set_ylabel(y.iloc[:, 2].name)

ax.legend()
