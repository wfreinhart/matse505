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
# id: Lecture21_working_with_molecules_in_python
# type: Foundational
# parent_lecture: Lecture21
# ---
#
# # Working with molecules in Python
#
# Let's consider how we might work with molecules for machine learning.
# In chemistry class you would have seen molecules written or drawn in the following ways:
#
# <img src="../lectures/assets/lecture21_molecular_representations.jpg" alt="Illustration of various molecular representations" width=600>
#
# As shown in this illustration, molecules can have complex 3D structures with multiple atoms and bonds, and there are many possible ways to draw a molecule's structure.
# This introduces two related problems:
# 1. How would we encode any given representation in a way that PyTorch or scikit-learn can read it?
# 2. Are some representations better than others?
#
# Representing molecules for machine learning involves converting a molecule's structural information into a format that can be easily processed by a machine learning algorithm. The goal of molecule representation is to extract relevant features of the molecule's structure that can be used to make predictions about its properties or behavior.
# Here are some examples of how the molecule can be "featurized":
#
# <img src="../lectures/assets/lecture21_molecule_featurization.jpg" alt="Various ways to featurize molecules for ML" width=600>
#
# Today we will focus on only a few of these: SMILES, fragments, and graphs.
