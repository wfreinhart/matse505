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
# id: Lecture21_intro_to_rdkit
# type: Foundational
# parent_lecture: Lecture21
# ---
#
# ## Intro to RDKit
#
# RDKit is an open-source cheminformatics toolkit written in C++ and Python. It provides a wide range of tools for molecular modeling and drug discovery, including molecule drawing and visualization, molecular similarity calculations, substructure searching, and property prediction.
#
# Some of the key features of RDKit include:
#
# * **Molecule manipulation:** RDKit can read and write molecule files in various formats, generate 2D and 3D coordinates, calculate molecular properties and descriptors, and perform chemical transformations (e.g., adding or removing atoms or bonds, changing atom types).
#
# * **Substructure searching:** RDKit can search for substructures or pharmacophores within molecules using SMARTS or SMILES patterns.
#
# * **Molecular fingerprints:** RDKit can generate molecular fingerprints, which are binary or count vectors that represent the presence or absence of certain chemical features in a molecule. Fingerprints can be used for molecular similarity calculations or machine learning applications.
#
# * **Machine learning:** RDKit provides a wide range of tools for machine learning on molecular data, including feature extraction, data preprocessing, and model building. RDKit can be used in conjunction with popular machine learning libraries such as scikit-learn and TensorFlow.
#
# RDKit is widely used in academic and industrial settings for drug discovery and molecular modeling. It is actively developed and maintained by a community of contributors and has a large user base. RDKit is available under the permissive BSD license and can be downloaded from the RDKit website or installed using pip.
#
# Let's start with a basic example of RDKit's usage: reading and writing SMILES:
#
# This doesn't do anything on its own, but we can use `IPython` to render a 2D version of the molecule:
#
# Calculate molecular descriptors for a molecule:
#
# Generate a 2D depiction of a molecule and save it to file:
#
# Generate a 3D conformer of a molecule and save it to file (as `pdb`):
#
# Evaluate Gasteiger partial charges:
# > Note this is based on empirical rules and does not perform any quantum mechanical calculations!

# %%
from rdkit import Chem

# Define a SMILES string
smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'

# Create a molecule from the SMILES string
mol = Chem.MolFromSmiles(smiles)

from IPython.display import display
display(mol)

from rdkit.Chem import Descriptors

# Calculate the molecular weight of the molecule
mw = Descriptors.MolWt(mol)
print(f'molecular weight of {smiles} is {mw:.3f}')

# Calculate the number of rotatable bonds in the molecule
num_rot_bonds = Descriptors.NumRotatableBonds(mol)
print(f'{smiles} has {num_rot_bonds} rotatable bonds')

from rdkit.Chem import Draw

# Generate a 2D depiction of the molecule
mol_img = Draw.MolToImage(mol)

# Save the image to a file
mol_img.save('mol.jpg')

from rdkit.Chem import AllChem

# Generate a 3D conformer of the molecule
AllChem.EmbedMolecule(mol)

# Optimize the conformer
AllChem.UFFOptimizeMolecule(mol)

# Write the molecule to a file in PDB format
Chem.MolToPDBFile(mol, 'mol.pdb')

from rdkit.Chem.Draw import SimilarityMaps

AllChem.ComputeGasteigerCharges(mol)
contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap='RdBu_r', contourLines=0)
