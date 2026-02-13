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
# id: Lecture03_train_test_split
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Train/Test split
#
# We'll use the `sklearn.model_selection.train_test_split` function to split our data into a train and test set:
#
# Now we can evaluate the performance on both the `train` and `test` sets separately:
#
# Typically we expect our test performance to be lower than our train performance since it is new data that was not fit to, but it doesn't have to be. The fit to the training data is basically telling you the best case for your model -- you should not expect the performance to be higher on the test set since the model parameters were already optimized for the training set!

# %%
from sklearn.model_selection import train_test_split

x = data.loc[:, 'Cement (component 1)(kg in a m^3 mixture)':'Age (day)']
y = data.loc[:, 'Concrete compressive strength(MPa, megapascals) ']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0)
# I use fixed random_state to avoid changing the answer when re-running the code
print(xtrain.shape, xtest.shape)

# train on only the (Xtrain, ytrain) data!
model = linear_model.LinearRegression().fit(xtrain, ytrain)

y_pred = model.predict(xtrain)
residuals = y_pred - ytrain

r2 = 1 - np.var(residuals) / np.var(ytrain - ytrain.mean())

rmse = np.sqrt(np.mean(residuals**2))

print(f'train: Rsq = {r2:.3f}, RMSE = {rmse:.3f}')

y_pred = model.predict(xtest)
residuals = y_pred - ytest

r2 = 1 - np.var(residuals) / np.var(ytest - ytest.mean())

rmse = np.sqrt(np.mean(residuals**2))

print(f'test:  Rsq = {r2:.3f}, RMSE = {rmse:.3f}')
