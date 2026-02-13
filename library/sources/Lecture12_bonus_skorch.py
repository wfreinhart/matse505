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
# id: Lecture12_bonus_skorch
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# # Bonus: `skorch`
#
# [`skorch`](https://skorch.readthedocs.io/en/stable/) is a package that gives `pytorch` models an interface very similar to `sklearn` (thus the name **sk**learn-pyt**orch**).
# It ends up looking something like this:
#
# ```
# class MyModule(torch.nn.Module):
#     ...
#
# net = NeuralNet(
#     module=MyModule,
#     criterion=torch.nn.NLLLoss,
# )
# net.fit(X, y)
# y_pred = net.predict(X_valid)
# ```
#
# This may be helpful when you want to train deep learning models side by side with simpler models like linear regression or trees.
