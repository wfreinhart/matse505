# Expects: nothing
# Produces: data (DataFrame), x (DataFrame of features), y (Series of target), alloy_family (Series of simplified labels)
# ---
# %% [markdown]
# ## Dataset: Steel Alloy Compositions and Mechanical Properties
#
# This dataset contains elemental compositions and mechanical properties of 915 steel alloys.
#
# Composition columns (in wt%): `C`, `Si`, `Mn`, `P`, `S`, `Ni`, `Cr`, `Mo`, `Cu`, `V`, `Al`, `N`, `Ceq`, `Nb + Ta`
#
# Target columns: `0.2% Proof Stress (MPa)`, `Tensile Strength (MPa)`, `Elongation (%)`, `Reduction in Area (%)`
#
# The `Alloy code` column contains labels like `A1`, `B3`, etc. We derive `alloy_family` (first letter only)
# to get a small set of categorical labels useful for classification tasks.

# %%
import pandas as pd
import os

_local = '../datasets/steels.csv'
_github = 'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/steels.csv'

data = pd.read_csv(_local if os.path.exists(_local) else _github)

# derive alloy family label (first letter of alloy code)
data['Alloy family'] = [c[0] for c in data['Alloy code']]

# set up features (composition columns) and target
x = data.loc[:, ' C':'Nb + Ta']
y = data['Alloy family']
alloy_family = data['Alloy family']

print(f'{len(data)} samples, {x.shape[1]} composition features')
data.head()
