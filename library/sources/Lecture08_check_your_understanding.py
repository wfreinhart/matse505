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
# id: Lecture08_check_your_understanding
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# ## [Check your understanding]
#
# Try utilizing the one-hot encodings in a regression model for one of the continuous features.
# Compare the performance to the baseline case of excluding that categorical feature and the (incorrect) case of `int` label encoding.

# %%
data.columns

from sklearn import linear_model, model_selection


# baseline without categorical columns

x = data.dropna().drop(columns=['Ionization Energies (eV)'])
y = data.dropna().loc[:, 'Ionization Energies (eV)']

x = x.drop(columns=['STP Phase', 'Natural Crystal Structure'])
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, random_state=0)

model = linear_model.LinearRegression().fit(xtrain, ytrain)
print('no cat columns', model.score(xtest, ytest))

# with one-hot encodings
x = pd.get_dummies(data.dropna()).drop(columns=['Ionization Energies (eV)'])
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, random_state=0)
model = linear_model.LinearRegression().fit(xtrain, ytrain)
print('one-hot', model.score(xtest, ytest))

# with int encodings
x = data.dropna().drop(columns=['Ionization Energies (eV)'])
y = data.dropna().loc[:, 'Ionization Energies (eV)']

x['STP Phase'] = x['STP Phase'].astype('category').cat.codes
x['Natural Crystal Structure'] = x['Natural Crystal Structure'].astype('category').cat.codes
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, random_state=0)
model = linear_model.LinearRegression().fit(xtrain, ytrain)
print('int', model.score(xtest, ytest))
