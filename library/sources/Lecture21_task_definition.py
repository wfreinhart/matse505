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
# id: Lecture21_task_definition
# type: Foundational
# parent_lecture: Lecture21
# ---
#
# ## Task definition
#
# Let's try a simple example of regression using the molecular fingerprints.
# We'll try to predict the molecular weight from the fingerprint.
# It's trivial but will demonstrate some of the challenges.
# In principle any scalar quantity can be substituted for the molecular weight.
#
# It is critical to understand that this uses *bits* (which are binary) and so we only know whether or not a fragment is active in a molecule:

# %%
import numpy as np

# Generate molecules
molecules = [Chem.MolFromSmiles(smiles) for smiles in df['smiles']]

all_fp = []
all_mw = []
for mol in molecules:
    # Compute fingerprints
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    all_fp.append(fp)
    # Compute molecular weight
    mw = Descriptors.MolWt(mol)
    all_mw.append(mw)

# create dataset for regression task
x = np.array(all_fp)
y = np.array(all_mw)

print(x.shape, x.max())
print(x[0])
