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
# id: Lecture07_modification_for_classifiers
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## Modification for classifiers
#
# We need to use new `constructors_cl` and specify `metric=metrics.accuracy_score` as a kwarg of `permutation_importance`:
#
# Again, let's check the baseline accuracy of our models:
#
# Here all the models perform at nearly 100% accuracy -- even on test data.
# Remember this was a trivial classification problem.
#
# And now we can visualize the change in performance with permuted columns:
#
# We can see that Mn, Ni, Cr, and Mo are the key elements that control the classification decisions for most of the models.
# Decision Tree picks up V and K-Neighbors picks up Mn, but the others don't.
#
# Let's repeat this with drop column importance:
#
# Plotting the drop-column feature importance:
#
# Overall this looks similar to the results from permutation importance, except much of the noise for elements other than Cr and Mo is removed.
# In other words, none of the models mistakenly attribute much weight to elements other than Cr and Mo.

# %%
from sklearn import metrics, preprocessing

x_cl = data_cl.loc[:, ' C':'Nb + Ta']
y_cl = preprocessing.LabelEncoder().fit_transform(data_cl['Alloy family'])

xtrain_cl, xtest_cl, ytrain_cl, ytest_cl = model_selection.train_test_split(x_cl, y_cl, test_size=0.20, shuffle=True, random_state=0)

# make a list of model constructors that can be called like constructor().fit(x, y)
constructors_cl = [ensemble.RandomForestClassifier,
                  neighbors.KNeighborsClassifier,
                  tree.DecisionTreeClassifier,
                  neural_network.MLPClassifier,
                  ]

results = {}
for constructor in constructors_cl:
    try:
        model = constructor(random_state=0).fit(xtrain_cl, ytrain_cl)
    except:
        model = constructor().fit(xtrain_cl, ytrain_cl)
    b, p = permutation_importance(model, xtest_cl, ytest_cl, metric=metrics.accuracy_score)
    # save the results to a dictionary for later:
    results[str(constructor)] = {'baseline': b, 'permuted': p}

baseline = []
names = []
for model_name, scores in results.items():
    baseline.append( scores['baseline'] )
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    names.append( short_model_name )

xticks = np.arange(len(constructors_cl))

fig, ax = plt.subplots()
ax.bar(xticks, baseline)
ax.set_xticks(xticks)
ax.set_xticklabels(names, rotation=45, horizontalalignment='right')
ax.set_ylabel('Baseline Accuracy')

fig, ax = plt.subplots()

xticks = np.arange(x_cl.columns.shape[0])
for model_name, scores in results.items():
    delta = scores['permuted'] - scores['baseline']
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    ax.plot(xticks, delta, 's-', label=short_model_name)

ax.set_xticks(xticks)
ax.set_xticklabels([it for it in x_cl.columns], rotation=90)
ax.set_ylabel('Delta Model Accuracy')
ax.legend()

results = {}
for constructor in constructors_cl:
    try:
        model = constructor(random_state=0)  # instantiate the model object from class name
    except:
        model = constructor()
    b, d = drop_column_importance(model, x_cl, y_cl, metric=metrics.accuracy_score)
    # save the results to a dictionary for later:
    results[str(constructor)] = {'baseline': b, 'dropped': d}

fig, ax = plt.subplots()

xticks = np.arange(x_cl.columns.shape[0])
for model_name, scores in results.items():
    delta = scores['dropped'] - scores['baseline']
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    ax.plot(xticks, delta, 's-', label=short_model_name)

ax.set_xticks(xticks)
ax.set_xticklabels([it for it in x_cl.columns], rotation=90)
ax.set_ylabel('Delta Model Accuracy')
ax.legend()
