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
# id: Lecture21_tuning_the_representation
# type: Foundational
# parent_lecture: Lecture21
# ---
#
# ## Tuning the representation
#
# Note that the resulting representation stores an integer *count* of the fragments rather than only the binary *bits*:
#
# The new model performs much better because it contains richer information about the fragments (i.e., counting fragments more than once).
#
# In this case, the function being modeled is linear, so the linear regression actually out-performs the non-linear models:

# %%
# Compute the Morgan fingerprints for each molecule and store them in a list
all_fp = []
for mol in molecules:
    fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=2).GetNonzeroElements()
    all_fp.append(fp)

# Identify keys (i.e., substructure identifiers) that appear in any of the molecules
all_keys = []
for fp in all_fp:
    all_keys += sorted(fp.keys())
all_keys = list(set(all_keys))

# Create a lookup table to map each key to an index in the full fingerprint matrix
key_lookup = {k: i for i, k in enumerate(all_keys)}

# Create a matrix to store the full fingerprint vectors for all molecules
full_mfp = np.zeros([len(all_fp), len(all_keys)])

# Populate the matrix with the fingerprint values for each molecule
for i, fp in enumerate(all_fp):
    for key, val in fp.items():
        full_mfp[i, key_lookup[key]] = val

# Identify the keys that appear in more than ten molecules
common = np.argwhere( (full_mfp > 0).astype(int).sum(axis=0) > 10 ).flatten()

# Select only the common keys for each molecule and store the result
x = full_mfp[:, common]

print(x.shape, x.max())
print(x[0, 600:700])

lr = linear_model.LinearRegression().fit(x[train_idx], y[train_idx])
print(f'Train R2 = {lr.score(x[train_idx], y[train_idx]):.3f}')
print(f'Test  R2 = {lr.score(x[test_idx], y[test_idx]):.3f}')

for model in [ensemble.RandomForestRegressor, neighbors.KNeighborsRegressor]:
    lr = model().fit(x[train_idx], y[train_idx])
    print(model)
    print(f'Train R2 = {lr.score(x[train_idx], y[train_idx]):.3f}')
    print(f'Test  R2 = {lr.score(x[test_idx], y[test_idx]):.3f}')
    print()
