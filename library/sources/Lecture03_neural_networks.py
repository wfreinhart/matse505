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
# id: Lecture03_neural_networks
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Neural Networks
#
# Neural Networks are perhaps the most famous ML models. They have achieved exceptional results in computer vision and natural language tasks, which have historically been incredibly challenging problems. The model name comes from its similarity to the architecture of a neuron in your brain.
# Here is a schematic of how it works mathematically:
#
# <img src="../lectures/assets/lecture03_neuron.jpg" alt="Mathematical schematic of a single artificial neuron" width=600>
#
# Basically the model alternates between a linear algebra block (which is fast and easy to compute) and a nonlinear function (the activation). By stacking multiple of these "layers," it is mathematically provable that the network can approximate any function. This has led to daring attempts by Deep Learning experts to learn functions with no conceivable alternative functional form, like human face generation (e.g., DeepFakes).
#
# Here we use the simplest version of a NN called a Multi-Layer Perceptron. It is basically a very shallow NN.
#
# > We call this **shallow** learning because there is only one layer (few parameters). Many layers of NN becomes a **deep** network. Deep networks are capable of expressing more complex relationships between variables.
#
# > **Practical Note:** Neural networks are sensitive to the scale of input features. In real-world applications, it is standard practice to scale features (e.g., using `StandardScaler`) before training. We will explore this in detail later.
#
# > We will spend a lot of time with deep neural networks later in the course, using the `pytorch` library.
#
# We see that it does a little better than the LinearRegression, depending on the random seed. It is fairly consistent between train and test performance.

# %%
from sklearn import neural_network

# we increase the max iterations to ensure convergence
model = neural_network.MLPRegressor(max_iter=1000).fit(xtrain, ytrain)

evaluate_model(model, xtrain, xtest, ytrain, ytest)

plot_model(model, xtrain, xtest, ytrain, ytest)
