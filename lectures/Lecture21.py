# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Lecture21

# %%
class Context(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault('data', None)
        self.setdefault('tensors', {})
        self.setdefault('model', None)
        self.setdefault('viz', None)

ctx = Context()

# %% [markdown]
# Today's topics:
# * Working with molecules
# * Machine learning with fingerprints
# * Geometric deep learning
#
# Let's install some dependencies up front so we don't have to restart our runtime later:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    # Install rdkit
    # !pip install rdkit

    # Install pytorch-geometric
    import os
    import torch
    os.environ['TORCH'] = torch.__version__
    print(torch.__version__)

    # !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
    # !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
    # !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

    # Install pytorch-lightning
    # !pip install pytorch-lightning
run_module(ctx)

# %% [markdown]
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

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
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

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    display = ctx.get('tensors', {}).get('display')
    i = ctx.get('tensors', {}).get('i')
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
    ctx['tensors']['contribs'] = contribs
    ctx['tensors']['mol'] = mol
    ctx['tensors']['mol_img'] = mol_img
    ctx['tensors']['num_rot_bonds'] = num_rot_bonds
    ctx['tensors']['smiles'] = smiles
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    display = ctx.get('tensors', {}).get('display')
    io = ctx.get('tensors', {}).get('io')
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
    ctx['data'] = df
    ctx['tensors']['is_match'] = is_match
    ctx['tensors']['matches'] = matches
    ctx['tensors']['mol'] = mol
    ctx['tensors']['mol_smiles'] = mol_smiles
    ctx['tensors']['molecules'] = molecules
    ctx['tensors']['pattern'] = pattern
    ctx['tensors']['substructure'] = substructure
    ctx['tensors']['url'] = url
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    display = ctx.get('tensors', {}).get('display')
    rdMolDescriptors = ctx.get('tensors', {}).get('rdMolDescriptors')
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
    ctx['tensors']['bits'] = bits
    ctx['tensors']['mol'] = mol
    ctx['tensors']['smiles'] = smiles
run_module(ctx)

# %% [markdown]
# # Machine learning with molecular fingerprints

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Task definition
#
# Let's try a simple example of regression using the molecular fingerprints.
# We'll try to predict the molecular weight from the fingerprint.
# It's trivial but will demonstrate some of the challenges.
# In principle any scalar quantity can be substituted for the molecular weight.
#
# It is critical to understand that this uses *bits* (which are binary) and so we only know whether or not a fragment is active in a molecule:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    df = ctx.get('data')
    rdMolDescriptors = ctx.get('tensors', {}).get('rdMolDescriptors')
    smiles = ctx.get('tensors', {}).get('smiles')
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
    ctx['tensors']['all_fp'] = all_fp
    ctx['tensors']['all_mw'] = all_mw
    ctx['tensors']['molecules'] = molecules
    ctx['tensors']['x'] = x
    ctx['tensors']['y'] = y
run_module(ctx)

# %% [markdown]
# ## Simple regression models
#
# We can also try the regularized linear regression models:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    x = ctx.get('tensors', {}).get('x')
    y = ctx.get('tensors', {}).get('y')
    from sklearn import linear_model, model_selection

    train_idx, test_idx = model_selection.train_test_split(np.arange(x.shape[0]), random_state=0)

    lr = linear_model.LinearRegression().fit(x[train_idx], y[train_idx])
    print(f'Train R2 = {lr.score(x[train_idx], y[train_idx]):.3f}')
    print(f'Test  R2 = {lr.score(x[test_idx], y[test_idx]):.3f}')

    for model in [linear_model.Lasso, linear_model.Ridge]:
        lr = model().fit(x[train_idx], y[train_idx])
        print(model)
        print(f'Train R2 = {lr.score(x[train_idx], y[train_idx]):.3f}')
        print(f'Test  R2 = {lr.score(x[test_idx], y[test_idx]):.3f}')
        print()

    from sklearn import ensemble, neighbors

    for model in [ensemble.RandomForestRegressor, neighbors.KNeighborsRegressor]:
        lr = model().fit(x[train_idx], y[train_idx])
        print(model)
        print(f'Train R2 = {lr.score(x[train_idx], y[train_idx]):.3f}')
        print(f'Test  R2 = {lr.score(x[test_idx], y[test_idx]):.3f}')
        print()
run_module(ctx)

# %% [markdown]
# ## Tuning the representation
#
# Note that the resulting representation stores an integer *count* of the fragments rather than only the binary *bits*:
#
# The new model performs much better because it contains richer information about the fragments (i.e., counting fragments more than once).
#
# In this case, the function being modeled is linear, so the linear regression actually out-performs the non-linear models:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    i = ctx.get('tensors', {}).get('i')
    k = ctx.get('tensors', {}).get('k')
    molecules = ctx.get('tensors', {}).get('molecules')
    rdMolDescriptors = ctx.get('tensors', {}).get('rdMolDescriptors')
    test_idx = ctx.get('tensors', {}).get('test_idx')
    train_idx = ctx.get('tensors', {}).get('train_idx')
    y = ctx.get('tensors', {}).get('y')
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
    ctx['tensors']['all_fp'] = all_fp
    ctx['tensors']['all_keys'] = all_keys
    ctx['tensors']['common'] = common
    ctx['tensors']['full_mfp'] = full_mfp
    ctx['tensors']['key_lookup'] = key_lookup
    ctx['tensors']['x'] = x
run_module(ctx)

# %% [markdown]
# # Geometric deep learning
#
# Geometric deep learning is a subfield of machine learning that focuses on the development of algorithms and models for processing and analyzing structured data, particularly data that can be represented as graphs or networks.
# Traditional machine learning models are designed to work with flat, feature-based representations of data, such as images or text, and struggle to effectively capture the inherent structural relationships and interactions present in graph-structured data.
#
# <img src="../lectures/assets/lecture21_mpnn.jpg" alt="Message Passing Neural Network (MPNN) diagram" width=400>
#
# Geometric deep learning seeks to address this limitation by developing algorithms and models that can directly process graph-structured data.
# This is achieved through the development of new types of neural network architectures, which are specifically designed to operate on graph-structured data.
# These models incorporate techniques such as graph convolutions, graph pooling, and attention mechanisms, which allow them to effectively process the complex relationships and interactions present in graph-structured data.
#
# Applications of geometric deep learning are broad and include areas such as drug discovery, social network analysis, recommender systems, and natural language processing. In the context of drug discovery, geometric deep learning has been used to develop models for predicting drug properties and identifying novel drug candidates by leveraging the graph structure of molecules.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Molecules as graphs
#
# Molecules can be naturally represented as graphs, where atoms correspond to nodes and chemical bonds correspond to edges. In this graph representation, each atom is a node in the graph, labeled with the atom type (such as carbon or oxygen) and additional properties such as the number of bonds it forms. Each bond is an edge in the graph, labeled with the type of chemical bond (such as single or double bonds) and potentially additional properties such as bond length or bond angle.
#
# The graph representation of a molecule can capture the structural relationships and interactions between atoms and bonds, which is important for many applications such as drug discovery or material science. For example, in drug discovery, the graph representation of a molecule can be used to predict its properties or to search for other molecules with similar structures that may have similar properties.
#
# In general, an attributed graph looks like this:
#
# > [!NOTE]
# > The attributed graph example from ResearchGate is currently unavailable due to access restrictions.
#
# In addition to the atom and bond information, which are captured by the graph topology, molecular graphs can also incorporate additional features such as atom charges or atomic coordinates. These additional features can provide important contextual information that can help improve the accuracy of predictions or analyses.
#
# > [!NOTE]
# > The dopamine molecular graph visualization from ResearchGate is currently unavailable due to access restrictions.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Graph data structure
#
# RDKit supports conversion from `mol` to graph format using `networkx`, a library for working with graphs in python:
#
# Let's inspect the node attributes:
#
# Edges are defined by $(i, j)$ pairs of nodes.
# Edges can have attributes just like nodes (e.g., bond type).
# Let's inspect the edge attributes:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    display = ctx.get('tensors', {}).get('display')
    mol_to_nx = ctx.get('tensors', {}).get('mol_to_nx')
    nx = ctx.get('tensors', {}).get('nx')
    # Create an RDKit molecule object
    mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')
    display(mol)

    print(f"Number of atoms: {mol.GetNumAtoms()}")
    print(f"Number of bonds: {mol.GetNumBonds()}")

    import networkx as nx

    # Define a function to convert an RDKit molecule to a NetworkX graph
    def mol_to_nx(mol):
        # Create an empty undirected NetworkX graph
        G = nx.Graph()

        # Add a node to the NetworkX graph for each atom in the RDKit molecule
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       atomic_num=atom.GetAtomicNum(),
                       is_aromatic=atom.GetIsAromatic(),
                       atom_symbol=atom.GetSymbol())

        # Add an edge to the NetworkX graph for each bond in the RDKit molecule
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       bond_type=bond.GetBondType())

        # Return the resulting NetworkX graph
        return G

    # Convert RDKit molecule object to NetworkX graph object
    G = mol_to_nx(mol)

    # Print the number of nodes and edges in the NetworkX graph
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    G.nodes(data=True)

    G.edges(data=True)
    ctx['tensors']['mol'] = mol
run_module(ctx)

# %% [markdown]
# ## Pytorch-Geometric
#
# `torch_geometric` is a package for working with graph-structured data in conjunction with `torch` for deep learning models.
#
# PyTorch Geometric (PyG) represents graphs using the Data class, which consists of the following attributes:
#
# * `x`: node feature matrix (shape: `[num_nodes, num_node_features]`).
# * `edge_index`: edge index matrix (shape: `[2, num_edges]`). Each column represents an edge (u, v) by its source node index u and destination node index v.
# * `edge_attr`: edge feature matrix (shape: `[num_edges, num_edge_features]`).
# * `y`: graph label (optional, shape: `[1]`).
# * `pos`: node position matrix (optional, shape: `[num_nodes, num_dimensions]`).
#
# The `Data` class also provides a variety of utility methods for manipulating graph data, such as `to()` and `from_dict()`, which allow conversion to and from other formats, and `num_nodes` and `num_edges`, which return the number of nodes and edges in the graph, respectively.
#
# PyG also provides a `Batch` class, which is used to batch multiple Data objects together.
# It includes a `batch` attribute, which indicates the index of the graph each node belongs to, and a `ptr` attribute, which points to the start and end indices of each graph in the batch.
# This allows for efficient batched processing of graphs with varying sizes.
#
# Let's define a function to process a SMILES string (a string representation of a molecule) and convert it into a PyG graph:
#
# This function creates a list of node features and edge features for each atom and bond, respectively, and then converts them to PyTorch tensors to create a Data object. The node features consist of a one-hot encoding of the atom type (C, N, O, F, Cl, Br, or I) and the total valence of the atom. The edge features consist of a one-hot encoding of the bond type (single, double, triple, or aromatic).
#
# > Note: ChatGPT wrote almost all of this function from the prompt, "write a python function to convert from rdkit molecule to pytorch-geometric Data object. include atom type and valence as node features and bond type as edge features."
#
# We can create a `Data` object for each molecule in the database:
#
# We can further investigate the features:
#
# We can use a PyG `DataLoader` to batch these graphs.
# Note that it is much more complicated to batch graphs than tabular data since they come in collections of nodes and edges!
#
# Here's what I mean about the batching being complicated:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    mol_to_pyg = ctx.get('tensors', {}).get('mol_to_pyg')
    molecules = ctx.get('tensors', {}).get('molecules')
    import torch
    from torch_geometric.data import Data

    def mol_to_pyg(mol, y=None):
        # Get number of atoms and bonds
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        # Initialize node and edge lists
        node_feats = []
        edge_index = []
        edge_feats = []

        # Iterate over atoms and add node features
        for atom in mol.GetAtoms():
            # Add atom symbol and valence as one-hot encoding
            symbol = atom.GetSymbol()
            valence = atom.GetTotalValence()
            node_feats.append([symbol == 'C', symbol == 'N', symbol == 'O',
                               symbol == 'F', symbol == 'Cl', symbol == 'Br',
                               symbol == 'I', valence])

        # Iterate over bonds and add edge features
        for bond in mol.GetBonds():
            # Add bond type as one-hot encoding
            bond_type = bond.GetBondTypeAsDouble()
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_feats.append([bond_type == 1.0, bond_type == 1.5, bond_type == 2.0, bond_type == 3.0])
            # Since the graph is undirected, add an edge in the opposite direction
            edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            edge_feats.append([bond_type == 1.0, bond_type == 1.5, bond_type == 2.0, bond_type == 3.0])

        # Convert lists to PyTorch tensors
        x = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

        # Return PyG Data object
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    mol = molecules[0]
    pyg_graph = mol_to_pyg(mol)
    print( pyg_graph )

    print('Node features:')
    print( pyg_graph.x )
    print('Edge features:')
    print( pyg_graph.edge_attr )

    from torch_geometric.loader import DataLoader

    graphs = []
    for mol in molecules:
        graphs.append( mol_to_pyg(mol, y=Descriptors.MolWt(mol)) )

    n_train = int(0.8 * len(graphs))
    n_val = len(graphs) - n_train
    train_data, val_data = torch.utils.data.random_split(graphs, [n_train, n_val])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    for batch in train_loader:
        print(batch)
        print(batch.x.shape, batch.y.shape)
        break
    ctx['tensors']['graphs'] = graphs
    ctx['tensors']['mol'] = mol
    ctx['tensors']['n_train'] = n_train
    ctx['tensors']['n_val'] = n_val
    ctx['tensors']['pyg_graph'] = pyg_graph
    ctx['tensors']['train_loader'] = train_loader
    ctx['tensors']['val_loader'] = val_loader
run_module(ctx)

# %% [markdown]
# ## Graph learning model
#
# Next, we can define our message passing network using PyG:
#
# Finally we train with `pytorch-lightning`:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    global_mean_pool = ctx.get('tensors', {}).get('global_mean_pool')
    pl = ctx.get('tensors', {}).get('pl')
    train_loader = ctx.get('tensors', {}).get('train_loader')
    val_loader = ctx.get('tensors', {}).get('val_loader')
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.utils import add_self_loops, degree
    import pytorch_lightning as pl


    class MPNRegressor(pl.LightningModule):
        def __init__(self, num_features, hidden_channels, out_channels, lr):
            super().__init__()

            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.lin = torch.nn.Linear(out_channels, 1)

            self.lr = lr

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            # Perform convolutions
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))

            # Pool over all nodes in the graph
            x = F.relu(global_mean_pool(x, data.batch))

            x = self.lin(x)
            return x

        def training_step(self, batch, batch_idx):
            output = self(batch)
            # print(output.shape, batch.y.shape)
            loss = F.mse_loss(output, batch.y.unsqueeze(1))
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            output = self(batch)
            loss = F.mse_loss(output, batch.y.unsqueeze(1))
            self.log('val_loss', loss)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

    model = MPNRegressor(num_features=8,
                         hidden_channels=16, out_channels=1,
                         lr=1e-2)

    for batch in train_loader:
        out = model(batch)
        break

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_loader, val_loader)

    results = []
    with torch.no_grad():
        for loader in [train_loader, val_loader]:
            loss = 0
            real = []
            pred = []
            for batch in loader:
                output = model(batch)
                loss += torch.sum((output - batch.y.unsqueeze(1))**2) / len(batch.y)
                real.append(batch.y.detach().numpy())
                pred.append(output.detach().squeeze(1).numpy())
            rmse = torch.sqrt(loss)
            results.append((np.hstack(real), np.hstack(pred)))
            print(str(loader), rmse.item())

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    for r in results:
        ax.scatter(r[0], r[1])
    ctx['model'] = model
    ctx['tensors']['loss'] = loss
    ctx['tensors']['out'] = out
    ctx['tensors']['output'] = output
    ctx['tensors']['pred'] = pred
    ctx['tensors']['real'] = real
    ctx['tensors']['results'] = results
    ctx['tensors']['rmse'] = rmse
    ctx['tensors']['trainer'] = trainer
run_module(ctx)
