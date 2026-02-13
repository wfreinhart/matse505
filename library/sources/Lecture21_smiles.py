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
# id: Lecture21_smiles
# type: Foundational
# parent_lecture: Lecture21
# ---
#
# ## SMILES
#
# SMILES (Simplified Molecular Input Line Entry System) is a compact string representation of a molecule's structure. It is a line notation that describes the connectivity of atoms and bonds in a molecule using ASCII characters.
#
# In SMILES notation, each atom is represented by its atomic symbol, and each bond is represented by a special character, such as "-", "=", "#", or ":". The SMILES string starts with the atom that has the highest connectivity (i.e., the highest number of bonds), and the atoms are listed in a linear sequence according to their connectivity. Branches in the molecule are indicated by parentheses, and ring closures are denoted by numbers.
#
# For example, the SMILES notation for methane, CH4, is "C". The SMILES notation for ethanol, CH3CH2OH, is "CCO". The SMILES notation for benzene, C6H6, is "c1ccccc1".
#
# SMILES notation provides a standardized way to represent a molecule's connectivity without ambiguity, regardless of how it is drawn or its orientation in 3D space. This is important for molecular modeling, where a molecule's structure needs to be represented in a consistent way for calculations and analysis.
#
# SMILES notation is also compact and efficient, which makes it useful for storing and exchanging large numbers of molecular structures. Because SMILES notation uses ASCII characters, it can be easily transmitted over the internet or included in text files.
