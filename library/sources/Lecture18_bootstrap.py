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
# id: Lecture18_bootstrap
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## Bootstrap
#
# **Bootstrap sampling** is a resampling technique used in machine learning to estimate the accuracy of a statistical model or to assess the stability of a model's predictions.
#
# The basic idea behind bootstrap sampling is to create multiple samples of the original dataset by randomly selecting observations with replacement. This means that some observations may be selected multiple times, while others may not be selected at all. The resulting samples, called bootstrap samples, have the same size as the original dataset but are created by sampling with replacement.
#
# Once the bootstrap samples have been created, a statistical model can be trained on each sample and the predictions can be combined to estimate the accuracy of the model or the stability of its predictions. For example, the average of the predictions across all bootstrap samples can be used as an estimate of the model's performance on new data.
#
# From the bootstrap procedure, we have 3 different models.
# We can query the discrepancy between the models as an estimate of uncertainty in the function:
#
# We can also explore the extrapolation behavior:

# %%
models = []
for i in range(3):
    # do bootstrap sampling
    rng = np.random.RandomState(i)
    training_indices = rng.choice(np.arange(y.size), size=18, replace=False)
    x_train, y_train = x.values[training_indices].reshape(-1, 1), y.values[training_indices]

    # train a model
    nn = neural_network.MLPRegressor(hidden_layer_sizes=(100, 100),
                                     max_iter=4000, activation='tanh',
                                     random_state=0).fit(x_train, y_train)
    print(f'model {i}', nn.score(x.values.reshape(-1, 1), y))
    models.append(nn)

y_hat = []
for nn in models[1:]:
    y_hat.append( nn.predict(x.values.reshape(-1, 1)) )

mu = np.mean(y_hat, axis=0)
sigma = np.std(y_hat, axis=0)

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x, mu, color='tab:blue', label='GP Model')
ax.fill_between(x, mu-1.96*sigma, mu+1.96*sigma, alpha=0.2, color='tab:blue')

x_extrap = np.linspace(0.6, 1, 11).reshape(-1, 1)
y_hat = []
for rf in models:
    y_hat.append( rf.predict(x_extrap) )

mu = np.mean(y_hat, axis=0)
sigma = np.std(y_hat, axis=0)

fig, ax = plt.subplots()
ax.plot(x, y, '-', color='tab:orange', label='Observation')
ax.plot(x_train, y_train, '.', color='tab:orange')
ax.plot(x_extrap[:, 0], mu, color='tab:blue', label='GP Model')
ax.fill_between(x_extrap[:, 0], mu-1.96*sigma, mu+1.96*sigma, alpha=0.2, color='tab:blue')
