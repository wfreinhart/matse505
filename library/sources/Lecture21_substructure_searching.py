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
# id: Lecture21_substructure_searching
# type: Foundational
# parent_lecture: Lecture21
# ---
#
# ## Substructure searching
#
# Search for a specific substructure pattern in a molecule:
#
# Search for all molecules in a database that contain a specific substructure pattern:
# >  This will search through each molecule in the database and print the names of the molecules that contain the substructure pattern.
#
# Now parse all these molecules and search for substructures:
#
# Note that this is more sophisticated than searching for a match in the SMILES string:

# %%
from rdkit import Chem
from IPython.display import display

# Define a molecule
mol = Chem.MolFromSmiles('Nc1nc(O)c2nc(CNc3ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc3)cnc2n1')
print('entire molecule:')
display(mol)

# Define a substructure pattern
# we'll use Pyrazine
pattern = 'C1=CN=CC=N1'
substructure = Chem.MolFromSmiles(pattern)
print('target substructure:')
display(substructure)

# Perform substructure searching
matches = mol.GetSubstructMatches(substructure)

# Print the atom indices of the matches
print('matches:')
for match in matches:
    print(match)

import pandas as pd
import requests, io

# Load a database of molecules (originally from ZINC database, but stored on my OneDrive)
url = 'https://pennstateoffice365-my.sharepoint.com/:x:/g/personal/wfr5091_psu_edu/EczKvWTme_pLur6JRE8TSGcB-lVEDvsvt1ZOMQloSkLVrg?e=qwvOcx&download=1'
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))
display(df)

# Parse SMILES into molecules
molecules = []
for smiles in df['smiles']:
    mol = Chem.MolFromSmiles(smiles)
    molecules.append(mol)

# Perform substructure searching on each molecule
print('matching molecules:')
is_match = []
for i, mol in enumerate(molecules):
    if mol is None:
        continue
    if mol.HasSubstructMatch(substructure):
        # Do something with the matching molecule
        display(mol)
        is_match.append(i)

print(pattern)

for i in is_match:
    mol_smiles = df['smiles'].iloc[i]
    print(pattern in mol_smiles, pattern.lower() in mol_smiles.lower(), '->', mol_smiles)
