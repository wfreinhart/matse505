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
# id: Lecture20_data_generation_with_sdv
# type: Foundational
# parent_lecture: Lecture20
# ---
#
# ## Data generation with SDV
#
# Let's use Synthetic Data Vault to train a conditional probabilistic model on our dataset.
# This will be similar to the use of KDE above, except that SDV permits conditional generation using rejection sampling (we'll see that in a minute).
#
# We can repeat a similar process compared to what we did above by generating many thousands of samples and testing to see if any are like what we want:
#
# These results are no better than using the simpler KDE above (may actually be worse).

# %%
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

df_meta = SingleTableMetadata()
df_meta.detect_from_dataframe(data)

# create a synthetic data generator using GaussianCopula model
synthesizer = GaussianCopulaSynthesizer(df_meta)
synthesizer.fit(data)

# generate synthetic data with no missing values
synthetic_data = synthesizer.sample(num_rows=10)
synthetic_data

# perform sampling
x_fake = synthesizer.sample(10000).iloc[:, 1:-4]
xs_fake = x_scaler.transform(x_fake)

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
