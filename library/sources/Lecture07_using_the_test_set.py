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
# id: Lecture07_using_the_test_set
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## Using the test set
#
# If you were paying attention last week, all these `model.fit(x, y)` calls should be bothering you.
# Instead, we should be using `train_test_split` to assess the performance on only data not seen during train time.
# Let's fix this now.
#
# Of course we'll visualize this result like we did above:
#
# The validation set doesn't change much compared to our analysis on the train set, which is good.
# In this case, the test performance is incidentally better than training.
# This doesn't usually happen, but the model is pretty poor ($R^2 \approx 0.6$) so there is plenty of room for some random fluctutations depending on the particular data in each set.

# %%
from sklearn import model_selection

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)

model = linear_model.LinearRegression().fit(xtrain, ytrain)  # train on train data
base_train = calc_rmse(ytrain, model.predict(xtrain))
base_test = calc_rmse(ytest, model.predict(xtest))  # evaluations will be on test data
print(f'baseline rmse is {base_train} / {base_test}')

permuted = np.zeros_like(x.columns)  # create empty array to store values
for i, col in enumerate(x.columns):
    x_permuted = xtest.copy()  # don't scramble the original dataframe!
    x_permuted[col] = np.random.permutation(xtest[col])
    permuted[i] = calc_rmse(ytest, model.predict(x_permuted))  # score on the permuted column
print('rmse on permuted columns is:', permuted)

fig, ax = plt.subplots()
ax.bar(x.columns, permuted, label='Permuted')
ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
ax.hlines(base_train, 0, len(x.columns)-1, linestyles='dashed', label='Base, Train')
ax.hlines(base_test, 0, len(x.columns)-1, linestyles='dotted', label='Base, Test')
ax.set_ylabel('Model RMSE')
ax.legend()
