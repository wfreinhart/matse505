# Expects: nothing
# Produces: data (DataFrame), ele_data (DataFrame indexed by Symbol)
# ---
# %% [markdown]
# ## Dataset: Elemental Properties
#
# This dataset contains DFT-computed and experimental properties of 52 chemical elements,
# including bulk static energy, cohesive energy, atomic mass, crystal structure,
# vacancy formation energy, and ionization energies.
#
# Columns include: `Symbol`, `Bulk Static Energy (eV)`, `Atomic Mass`, `Natural Crystal Structure`,
# `Vacancy Formation Energy (eV)`, `Ionization Energies (eV)`, `Atomic Number`, and more.
#
# Note that some entries contain `NaN` (Not a Number) values where data was unavailable.

# %%
import pandas as pd
import os

_local = '../datasets/elements.csv'
_github = 'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/elements.csv'

data = pd.read_csv(_local if os.path.exists(_local) else _github)
ele_data = data.set_index('Symbol')  # version with Symbol as row index

print(f'{len(data)} elements, {data.shape[1]} properties')
data
