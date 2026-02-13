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
# id: Lecture03_some_convenience_functions
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Some convenience functions
#
# Let's set up some convenience functions so we don't have to repeat the same code all the time...

# %%
def evaluate_model(model, xtrain, xtest, ytrain, ytest):
    "Evaluate model performance on train and test set, then print the results."

    for name, x, y in [('train', xtrain, ytrain), ('test', xtest, ytest)]:

        y_pred = model.predict(x)
        residuals = y_pred - y

        r2 = 1 - np.var(residuals) / np.var(y - y.mean())

        rmse = np.sqrt(np.mean(residuals**2))

        print(f'{name:5s}: Rsq = {r2:.3f}, RMSE = {rmse:.3f}')


def plot_model(model, xtrain, xtest, ytrain, ytest, title=None):
    "Create a parity plot using a trained model with train/test split."

    fig, ax = plt.subplots(figsize=(5, 5))

    # plot the results
    ax.plot(ytrain, model.predict(xtrain), '.', label='Training Data')
    ax.plot(ytest, model.predict(xtest), '.', label='Testing Data')

    # create reference line along y = x to show the desired behavior
    min_max = np.array([y.min(), y.max()])
    ax.plot(min_max, min_max, 'k--', label='Reference')
    ax.set_aspect('equal')  # very helpful to show y = x relationship

    # add labels and legend
    ax.set_xlabel('Observation')
    ax.set_ylabel('Prediction')
    ax.legend()

    if title is not None:
        ax.set_title(title)
