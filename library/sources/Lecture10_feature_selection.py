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
# id: Lecture10_feature_selection
# type: Foundational
# parent_lecture: Lecture10
# ---
#
# ## Feature selection
#
# Let's consider an example
#
# Let's make sure this does what we expect:
#
# Compared to the full feature set:
#
# Set up the `pygad` optimization problem (it has a lot of parameters!!!):
#
# Run it:
#
# View the best solution and its fitness:
#
# We can plot the result as a function of iteration:
#
# We can visualize which features were selected by looking back at the `solutions` attribute of the `ga_instance`:
#
# And like with `ax-platform`, we can also check test performance in this case:
#
# We got lucky here in that our test set actually performs better than our validation set!
#
# Let's also quickly compare to what we would get from recursive drop-column feature importance.
#
# If we compare that to the results from EA we may see something interesting:
#
# The features are different -- how do they compare in performance?

# %%
# !pip install pygad

from sklearn import neighbors

# set up train/validation/test sets
x_trv, x_test, y_trv, y_test = model_selection.train_test_split(x, y, test_size=0.20, shuffle=True, random_state=0)
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_trv, y_trv, train_size=0.75, shuffle=True, random_state=0)

# set up fitness function
def fitness(features, solution_index):
    """A fitness function that selects features based on the input array and trains a KNeighborsRegressor.
    It returns the model R-squared on validation data."""
    f = np.argwhere(features > 0.5).flatten()
    model = neighbors.KNeighborsRegressor().fit(x_train.iloc[:, f], y_train)
    r_squared = model.score(x_val.iloc[:, f], y_val)
    return r_squared

all_features = np.ones(x.shape[1])
print( fitness(all_features, None) )

model = neighbors.KNeighborsRegressor().fit(x_train, y_train)
print('validation: ', model.score(x_val, y_val))
print('test: ', model.score(x_test, y_test))

import pygad

features = np.zeros(x.shape[1], dtype=int)
fitness_function = fitness

num_generations = 16
num_parents_mating = 2

sol_per_pop = 8
num_genes = len(features)

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=0,
                       init_range_high=1,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       random_seed=0, save_solutions=True,
                       )

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : ", x.columns[solution>0.5])
print("Fitness value of the best solution = ", solution_fitness)

_ = ga_instance.plot_fitness()

from matplotlib import pyplot as plt

solutions = np.array(ga_instance.solutions) > 0.5

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

ax = axes[0]
_ = ax.imshow(np.array(ga_instance.solutions_fitness).reshape(1, -1))

ax = axes[1]
_ = ax.set_yticks(np.arange(x.shape[1]))
_ = ax.set_yticklabels(x.columns)
_ = ax.imshow(solutions.T)
_ = ax.set_xlabel('Solution index')
_ = ax.set_ylabel('Feature index')
_ = ax.set_aspect('auto')

plt.subplots_adjust(hspace=0)

f = np.argwhere(solution > 0.5).flatten()
model = neighbors.KNeighborsRegressor().fit(x_train.iloc[:, f], y_train)
print('validation: ', model.score(x_val.iloc[:, f], y_val))
print('test: ', model.score(x_test.iloc[:, f], y_test))

def drop_column_importance(model, xtrain, ytrain, xtest, ytest):
    """Compute the drop-column importance on a trained model."""

    model.fit(xtrain, ytrain)
    baseline = model.score(xtest, ytest)

    dropped = np.zeros_like(xtest.columns)
    for i, col in enumerate(xtest.columns):
        x_dropped_train = xtrain.copy().drop(columns=col)
        x_dropped_test = xtest.copy().drop(columns=col)
        model.fit(x_dropped_train, ytrain)
        dropped[i] = model.score(x_dropped_test, ytest)

    return baseline, dropped


def choose_worst_feature(model, x_train, y_train, x_val, y_val):
    baseline, dropped = drop_column_importance(model, x_train, y_train, x_val, y_val)
    return baseline, dropped, np.argmax(dropped)


model = neighbors.KNeighborsRegressor()

x_train_trim = x_train.copy()
x_val_trim = x_val.copy()

r2 = []
feature_order = []

for k in range(x_train_trim.shape[1]-1):
    baseline, dropped, worst = choose_worst_feature(model, x_train_trim, y_train, x_val_trim, y_val)
    r2.append(baseline)
    feature_order.append(x_train_trim.columns[worst])
    x_train_trim = x_train_trim.drop(columns=[x_train_trim.columns[worst]])
    x_val_trim = x_val_trim.drop(columns=[x_val_trim.columns[worst]])

# we have one feature left that we didn't drop:
last_feature = x_train_trim.columns[0]
feature_order.append( last_feature )

print(np.round(r2, 2))
print(feature_order)  # later is better (dropped last)

f = np.argwhere(solution > 0.5).flatten()

print( x.columns[f] )

model = neighbors.KNeighborsRegressor().fit(x_train.iloc[:, f], y_train)
print('validation: ', model.score(x_val.iloc[:, f], y_val))
print('test: ', model.score(x_test.iloc[:, f], y_test))

top_features_rfe = feature_order[-len(f):]
print(top_features_rfe)

model = model.fit(x_train.loc[:, top_features_rfe], y_train)

print( 'validation: ', model.score(x_val.loc[:, top_features_rfe], y_val) )
print( 'test: ', model.score(x_test.loc[:, top_features_rfe], y_test) )
