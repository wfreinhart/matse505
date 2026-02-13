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
# id: Lecture08_exercise_feature_elimination
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# ## [Exercise: feature elimination]
#
# Try iteratively (automatically) removing features using drop-column feature importance until a specific $k$ number of features are left.
# Evaluate the performance each time.
# Create a chart showing the performance with each number of features.
#
# Augment the features with polynomial interaction terms.
# Then repeat the exercise above.
# > If done correctly, you should be able to use the same code!
# Just change the input variable (features).

# %%
def choose_worst_feature(model, x, y):
    baseline, dropped = drop_column_importance(model, x, y)
    return baseline, dropped, np.argmin(dropped)

def make_plot(x, dropped):
    fig, ax = plt.subplots()
    ax.bar(x.columns, dropped, label='Dropped')
    ax.set_xticklabels([it.split('(')[0].strip() for it in x.columns], rotation=90)
    ax.hlines(baseline, 0, len(x.columns)-1, linestyles='dashed', label='Baseline')
    ax.set_ylabel('Model RMSE')
    ax.legend(loc='lower center')

    return fig

x_trim = x.copy()
rmse = []

model = linear_model.LinearRegression()

for k in range(x_trim.shape[1]-1):
    baseline, dropped, worst = choose_worst_feature(model, x_trim, y)
    fig = make_plot(x_trim, dropped)
    rmse.append(baseline)
    print(x_trim.columns[worst])
    x_trim = x_trim.drop(columns=[x_trim.columns[worst]])

# print( x_trim.columns )

fig, ax = plt.subplots()
_ = ax.plot(rmse)
