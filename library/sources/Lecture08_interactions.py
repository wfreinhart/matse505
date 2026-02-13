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
# id: Lecture08_interactions
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# ## Interactions
#
# We have already introduced the idea of polynomial features when we first learned about nonlinear regression.
# These polynomial terms incorporate interactions between variables.
#
# Let's see how these interacting features compare to the original ones:
#
# Of course we're cheating here by not using a test set.
# Let's fix that:
#
# We can evaluate the importance of these features using functions we developed last time:
#
# And of course we should plot them to understand the result:

# %%
from sklearn import preprocessing

poly = preprocessing.PolynomialFeatures(degree=2).fit(x)
print(poly.get_feature_names_out())

xp = poly.transform(x)
model = linear_model.LinearRegression().fit(xp, y)
model.score(xp, y)

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)

model = linear_model.LinearRegression().fit(xtrain, ytrain)
print( model.score(xtest, ytest) )

model = linear_model.LinearRegression().fit(poly.transform(xtrain), ytrain)
print( model.score(poly.transform(xtest), ytest) )

from sklearn import model_selection


def calc_rmse(y, y_pred):
    residuals = y - y_pred
    return np.sqrt(np.mean(residuals**2))


def permutation_importance(model, x, y, metric=calc_rmse):
    """Compute the permutation importance on a trained model."""
    baseline = metric(y, model.predict(x))

    permuted = np.zeros_like(x.columns)
    for i, col in enumerate(x.columns):
        x_permuted = x.copy()
        x_permuted[col] = np.random.permutation(x[col])
        permuted[i] = metric(y, model.predict(x_permuted))

    return baseline, permuted


def drop_column_importance(model, x, y, metric=calc_rmse):
    """Compute the drop-column importance on a trained model."""
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)

    model.fit(xtrain, ytrain)
    baseline = metric(ytest, model.predict(xtest))

    dropped = np.zeros_like(xtest.columns)
    for i, col in enumerate(xtest.columns):
        x_dropped_train = xtrain.copy().drop(columns=col)
        x_dropped_test = xtest.copy().drop(columns=col)
        model.fit(x_dropped_train, ytrain)
        dropped[i] = metric(ytest, model.predict(x_dropped_test))

    return baseline, dropped


model = linear_model.LinearRegression().fit(xtrain, ytrain)
baseline, permuted = permutation_importance(model, xtest, ytest)

baseline, dropped = drop_column_importance(linear_model.LinearRegression(), x, y)

fig, ax = plt.subplots()
ax.bar(x.columns, permuted, label='Permuted')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(baseline, 0, len(x.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend(loc='lower center')

fig, ax = plt.subplots()
ax.bar(x.columns, dropped, label='Dropped')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(baseline, 0, len(x.columns)-1, linestyles='dashed', label='Baseline')
ax.set_ylabel('Model RMSE')
ax.legend(loc='lower center')
