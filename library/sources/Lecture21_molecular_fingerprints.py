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
# id: Lecture21_molecular_fingerprints
# type: Foundational
# parent_lecture: Lecture21
# ---
#
# ## Molecular fingerprints
#
# Molecular fingerprints are a way of representing a molecule as a fixed-length vector of numbers or bits that capture information about its chemical structure and properties. The basic idea behind molecular fingerprints is to encode the presence or absence of certain structural features, such as functional groups or atom types, into a binary or integer vector.
#
# > [!NOTE]
# > The molecular fingerprint schematic from ResearchGate is currently unavailable due to access restrictions.
#
# There are many different types of molecular fingerprints, each with its own method of encoding molecular structure. For example, one common type of fingerprint is the Morgan fingerprint, which encodes the connectivity of atoms in a molecule within a certain radius. Another type is the Extended Connectivity Fingerprints (ECFP), which encode the local topology of a molecule.
#
# Molecular fingerprints are widely used in cheminformatics and drug discovery because they allow molecules to be compared and searched quickly and efficiently using computational algorithms. They can be used for tasks such as virtual screening, clustering, similarity searching, and machine learning.
#
# **However, it is important to note that molecular fingerprints are an approximation of the true molecular structure** and properties and may not always capture all relevant information. Therefore, they should be used with caution and in combination with other molecular descriptors and experimental data when possible.
#
# Here's an example of computing the Morgan fingerprints in `rdkit`:
#
# Here, we use the `GetMorganFingerprintAsBitVect()` function from the `rdMolDescriptors` module to generate a Morgan fingerprint for the molecule.
# The `radius` parameter specifies the radius of the fingerprint, and the `nBits` parameter specifies the length of the resulting fingerprint vector.
# The resulting fingerprint is returned as a `BitVect` object, which can be converted to a binary string using the `ToBitString()` method.
#
# Note that in this example, we are generating a fixed-length fingerprint with 1024 bits. You can adjust the parameters of the fingerprint generation function to change the length and type of fingerprint as needed for your specific application.

# %%
from rdkit.Chem import rdMolDescriptors

# Create an RDKit molecule from a SMILES string
smiles = 'c1ccccc1'
mol = Chem.MolFromSmiles(smiles)
display(mol)

# Generate a binary fingerprint for the molecule
fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

# Convert the fingerprint to a binary array and print the result
bits = fp.ToBitString()
print(bits)
