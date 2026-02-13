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
# id: Lecture07_doing_the_same_with_drop_columns
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## Doing the same with drop columns
#
# Remember that the permutation feature importance only measures how much the model relies on those features.
# Dropping columns gives the model a chance to learn the same information somewhere else in the data.
# Let's try it with all those models above:
#
# This will take longer since we have to train the models many times:
#
# Finally we evaluate the result:
#
# Note how differently these models behave in the drop column test compared to permtuations!
#
# This result is a little crazy because it says that tree-based models perform **better** when dropping features and retraining -- for all except Age.
# This confirms the claim above that any other component can be inferred from the rest, while it also shows that the model gets less confused without additional variables to make decisions with.
#
# For K-Neighbors and Linear Regression, Cement and Age show up again as important.
#
# The MLP result is crazy and probably indicates poorly designed and poorly fitted models.
# We shouldn't read too much into this without additional tuning (neural networks need a lot of tuning!)

# %%
def drop_column_importance(model, x, y, metric=calc_rmse):
    """Compute the drop-column importance on a trained model."""
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)

    model.fit(xtrain, ytrain)
    baseline = metric(ytest, model.predict(xtest))

    dropped = np.zeros_like(xtest.columns)
    for i, col in enumerate(xtest.columns):
        x_dropped = xtest.copy().drop(columns=col)
        model.fit(x_dropped, ytest)
        dropped[i] = metric(ytest, model.predict(x_dropped))

    return baseline, dropped

results = {}
for constructor in constructors:
    try:
        model = constructor(random_state=0)  # instantiate the model object from class name
    except:
        model = constructor()
    b, d = drop_column_importance(model, x, y)
    # save the results to a dictionary for later:
    results[str(constructor)] = {'baseline': b, 'dropped': d}

fig, ax = plt.subplots()

xticks = np.arange(x.columns.shape[0])
for model_name, scores in results.items():
    delta_percent = 100 * (scores['dropped'] - scores['baseline']) / scores['baseline']
    short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
    ax.plot(xticks, delta_percent, 's-', label=short_model_name)

ax.set_xticks(xticks)
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.set_ylabel('Delta Model RMSE (%)')
ax.legend()
