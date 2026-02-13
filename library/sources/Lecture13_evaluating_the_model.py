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
# id: Lecture13_evaluating_the_model
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Evaluating the model
#
# Let's evaluate the model predictions on the training set.
#
# We can start by investigating this `y_prob` result, the raw model output:
#
# We can convert these raw outputs to class probabilities using the softmax function.
# From Wikipedia:
#
# The softmax function takes as input a vector $z$ of $K$ real numbers, and normalizes it into a probability distribution consisting of $K$ probabilities proportional to the exponentials of the input numbers. That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval (0,1), and the components will add up to 1, so that they can be interpreted as probabilities.
#
# $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$
#
# These prediction probabilities can be converted to integer class labels using the `torch.max` function to identify the most likely class:
#
# Let's try to turn this into a confusion matrix.

# %%
import torch
import numpy as np

y_outs = []
y_true = []

with torch.no_grad():
    for x, y in dataloader:
        y_outs += model(x).detach().numpy().tolist()
        y_true += y.detach().numpy().tolist()

y_outs = np.array(y_outs)
y_true = np.array(y_true)

from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes[0]
_ = ax.hist(y_outs[y_true==0], density=True)
ax.set_xlabel('Prediction')
ax.set_ylabel('Probability density')
ax.set_title('True Patches')
ax = axes[1]
_ = ax.hist(y_outs[y_true==1], density=True)
ax.set_xlabel('Prediction')
ax.set_ylabel('Probability density')
ax.set_title('True Scratches')

y_prob = nn.functional.softmax(torch.tensor(y_outs), dim=1).detach().numpy()

fig, ax = plt.subplots()
_ = ax.hist(y_prob[y_true==0][:, 0])
_ = ax.hist(y_prob[y_true==1][:, 1])

y_pred = []
y_true = []

with torch.no_grad():
    for x, y in dataloader:
        outputs = model(x)
        _, label = torch.max(outputs.data, 1)
        y_pred += label.detach().numpy().tolist()
        y_true += y.detach().numpy().tolist()

y_pred = np.array(y_pred)
y_true = np.array(y_true)

confusion = np.zeros([2, 2], dtype=int)
for i in range(len(y_pred)):
    row_idx = y_pred[i].round()
    col_idx = y_true[i].round()
    confusion[row_idx.astype(int), col_idx.astype(int)] += 1

fig, ax = plt.subplots()
im = ax.imshow(confusion, 'Blues')
cb = plt.colorbar(im)
cb.set_label('Frequency')
_ = ax.set_xlabel('True label')
_ = ax.set_ylabel('Predicted label')

for i in range(2):
    for j in range(2):
        if confusion[i, j] > 100:
            tc = 'w'
        else:
            tc = 'k'
        ax.text(i, j, confusion[i, j], ha='center', color=tc)
