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
# id: Lecture20_rejection_sampling
# type: Foundational
# parent_lecture: Lecture20
# ---
#
# ## Rejection sampling
#
# We can also try the rejection sampling approach with SDV to get values close to some target:
#
# You will see here that the sampled points are all over the place.
# While they are generally biased towards the true minimum, they aren't any better than screening.

# %%
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

ternary_df = pd.DataFrame({'x': x[:, 0], 'y': x[:, 1], 'obj': y.flatten()})

df_meta = SingleTableMetadata()
df_meta.detect_from_dataframe(ternary_df)

# create a synthetic data generator using GaussianCopula model
synthesizer = GaussianCopulaSynthesizer(df_meta)
synthesizer.fit(ternary_df)

# generate synthetic data with no missing values
synthetic_data = synthesizer.sample(num_rows=10)
synthetic_data

from sdv.sampling import Condition

low_objective = Condition(num_rows=100, column_values={'obj': 0.45})
synthetic_data = synthesizer.sample_from_conditions(conditions=[low_objective])

synthetic_data

x_fake = synthetic_data.iloc[:, 0:2].values
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
