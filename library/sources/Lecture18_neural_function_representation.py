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
# id: Lecture18_neural_function_representation
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## Neural function representation

# %%
from sklearn import neural_network

rng = np.random.RandomState(2)
training_indices = rng.choice(np.arange(y.size), size=18, replace=False)
x_train, y_train = x.values[training_indices].reshape(-1, 1), y.values[training_indices]

nn = neural_network.MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=4000, activation='tanh', random_state=0).fit(x_train, y_train)
nn.score(x.values.reshape(-1, 1), y)

mu = nn.predict(x.values.reshape(-1, 1))

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x, mu, color='tab:blue', label='GP Model')
