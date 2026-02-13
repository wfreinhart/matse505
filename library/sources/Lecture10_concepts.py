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
# id: Lecture10_concepts
# type: Foundational
# parent_lecture: Lecture10
# ---
#
# ## Concepts
#
# "Evolutionary algorithm," "genetic algorithm," or "evolutionary optimization" is a scheme that utilizes the idea of natural selection to perform numerical optimization:
#
# <img src="../lectures/assets/genetic_algorithm_concept.jpg" width=600 alt="Visual metaphor for Genetic Algorithms based on biological evolution: selection, crossover, and mutation">
#
# In each "generation," traits from the best individuals are combined in a process analagous to gene transfer between DNA of parents in biological organisms:
#
# <img src="../lectures/assets/genetic_algorithm_flow.jpg" width=600 alt="Step-by-step flowchart of the Genetic Algorithm iterative loop">
#
# We also include mutations to permit new traits to arise in the population:
#
# <img src="../lectures/assets/genetic_algorithm_operators.jpg" width=600 alt="Visual detail of Genetic Algorithm operators: Crossover (recombination) and Mutation">
#
# If the new traits lead to greater fitness, they persist and are passed on to future generations.
#
# Why use evolutionary algorithm (EA) over Gaussian Process (GP)?
# One answer is that GP stops working well in higher dimensions due to the ambiguity of distances in those high-dimensional spaces.
# Another is that GP is meant for continuous spaces, while we often have discrete choices in model tuning.
# EA has no problem with high-dimensional spaces as the crossover and mutation can occur in any dimension.
# In addition, GP scales like $\mathcal{O}(N^3)$, which can get out of hand quickly.
# EA has no fitting so it's simply $\mathcal{O}(N)$ -- although it may converge less quickly.
