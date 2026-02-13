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
# id: Lecture07_evaluating_a_bunch_of_models
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## Evaluating a bunch of models
#
# Let's try this on some additional regressors and see if we can find some common trends.
#
# Let's quickly check what the baseline performance looked like:
#
# We should keep this in mind going into the results that Random Forest was by far the most successful model.
# It is important to keep in mind this is probably due to using default hyperparameters!
# Feature importance of a bad (poorly tuned) model might not mean anything.
#
# We'll use a line plot for the delta RMSE this time since there will be so many series:
#
# It seems all the models depend heavily on Cement and Age, although the tree-based models experience the strongest influence of Age.
# Water and Superplasticizer also appear to have the strongest influence on the tree-based models.
# Again, this could be due to the fact that the other models are not tuned well.
# Things could change if they were optimized properly.

# %%
from sklearn import neighbors, tree, neural_network

# make a list of model constructors that can be called like constructor().fit(x, y)
constructors = [linear_model.LinearRegression,
                ensemble.RandomForestRegressor,
                neighbors.KNeighborsRegressor,
                tree.DecisionTreeRegressor,
                neural_network.MLPRegressor,
                ]

results = {}
for constructor in constructors:
    try:
        model = constructor(random_state=0).fit(xtrain, ytrain)
    except:
        model = constructor().fit(xtrain, ytrain)
    b, p = permutation_importance(model, xtest, ytest)
    # save the results to a dictionary for later:
    results[str(constructor)] = {'baseline': b, 'permuted': p}

baseline = []
names = []
for model_name, scores in results.items():
    baseline.append( scores['baseline'] )
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    names.append( short_model_name )

xticks = np.arange(len(constructors))

fig, ax = plt.subplots()
ax.bar(xticks, baseline)
ax.set_xticks(xticks)
ax.set_xticklabels(names, rotation=45, horizontalalignment='right')
ax.set_ylabel('Baseline RMSE')

fig, ax = plt.subplots()

xticks = np.arange(x.columns.shape[0])
for model_name, scores in results.items():
    delta_percent = 100 * (scores['permuted'] - scores['baseline']) / scores['baseline']
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    ax.plot(xticks, delta_percent, 's-', label=short_model_name)

ax.set_xticks(xticks)
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.set_ylabel('Delta Model RMSE (%)')
ax.legend()
