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
# id: Lecture21_pytorch_geometric
# type: Foundational
# parent_lecture: Lecture21
# ---
#
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
