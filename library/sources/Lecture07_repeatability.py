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
# id: Lecture07_repeatability
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# # Repeatability
#
# These methods include a significant amount of randomness.
# We should repeat them several times and average the result to get the best picture of what's going on.
#
# Now we'll make the same chart but with [`plt.errorbar`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html)
#
# Here we find that repeated trials average out any result from elements other than Cr and Mo and confirm that the drop in classification accuracy for these two are indeed significant.

# %%
n_repeats = 10

b_list = np.zeros([n_repeats, 1])
p_list = np.zeros([n_repeats, x_cl.columns.shape[0]])

for k in range(n_repeats):

    split_data = model_selection.train_test_split(x_cl, y_cl, test_size=0.20,
                                                shuffle=True, random_state=k)
    xtrain_cl, xtest_cl, ytrain_cl, ytest_cl = split_data

    model = ensemble.RandomForestClassifier(random_state=k)
    model.fit(xtrain_cl, ytrain_cl)

    b_list[k], p_list[k] = permutation_importance(model, xtest_cl, ytest_cl, metric=metrics.accuracy_score)

fig, ax = plt.subplots()

delta = (p_list - b_list).mean(axis=0)
sigma = (p_list - b_list).std(axis=0)

xticks = np.arange(x_cl.columns.shape[0])
short_model_name = model_name.split('.')[3][:-2]  # cleans up the names
ax.errorbar(xticks, delta, yerr=sigma, marker='s')

ax.set_xticks(xticks)
ax.set_xticklabels([it for it in x_cl.columns], rotation=90)
ax.set_ylabel('Delta Model Accuracy')
