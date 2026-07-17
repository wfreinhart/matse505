# Expects: nothing
# Produces: data (DataFrame), x (DataFrame of features), y (Series of target)
# ---
# %% [markdown]
# ## Dataset: Concrete Compressive Strength
#
# We'll use a dataset of concrete compressive strengths for our examples.
#
# Each row represents a concrete mixture. The 8 input columns are component
# amounts (kg/m³) and curing age (days). The output is measured compressive
# strength in MPa.
#
# > Reuse of this database is unlimited with retention of copyright notice for
# > Prof. I-Cheng Yeh and the following published paper:
# > I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial
# > neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)

# %%
import pandas as pd
import numpy as np
import os

_local = '../datasets/concrete.csv'
_github = 'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/concrete.csv'
_sharepoint = 'https://pennstateoffice365-my.sharepoint.com/:x:/g/personal/wfr5091_psu_edu/EZaAQFZWkipHsG8UaXkQxN8BzEUvZOb-xtER99yFGwKxaQ?e=2SyAUk&download=1'

if os.path.exists(_local):
    data = pd.read_csv(_local)
else:
    try:
        data = pd.read_csv(_github)
    except Exception:
        import requests
        r = requests.get(_sharepoint)
        with open('data.csv', 'w') as fid:
            fid.write(r.text)
        data = pd.read_csv('data.csv')

# set up features (X) and target (y)
x = data.loc[:, 'Cement (component 1)(kg in a m^3 mixture)':'Age (day)']
y = data.loc[:, 'Concrete compressive strength(MPa, megapascals) ']

print(f'{len(data)} samples, {x.shape[1]} features')
data
