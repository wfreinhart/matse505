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
# id: Lecture21_graph_data_structure
# type: Foundational
# parent_lecture: Lecture21
# ---
#
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
