# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Lecture20

# %%
class Context(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault('data', None)
        self.setdefault('tensors', {})
        self.setdefault('model', None)
        self.setdefault('viz', None)

ctx = Context()

# %% [markdown]
# Today's topics:
# * Search and screening
# * Conditional data generation
# * Inverse function approximation

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# # Requirements
#
# Let's install the necessary packages first so we don't have to restart the runtime later!

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    # !pip install sdv==1.0.0 pygad ax-platform nflows
run_module(ctx)

# %% [markdown]
# # Dataset
#
# We'll use the alloys dataset again for much of the lesson, then switch to another dataset later on.
#
# There is an outlier in this dataset that needs to be corrected:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    import os

    # Set the path to the data file
    filename = 'steels.csv'
    local_path = f'../datasets/{filename}'
    github_url = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename}'

    # Load the data: try local path first, fallback to GitHub for Colab
    if os.path.exists(local_path):
        data = pd.read_csv(local_path)
    else:
        data = pd.read_csv(github_url)
    data                            # show a view of the data file

    import numpy as np

    bad_idx = np.argmax( data.loc[:, ' Tensile Strength (MPa)'] )
    data.loc[bad_idx, ' Tensile Strength (MPa)'] /= 10.0  # missed a decimal point
    ctx['data'] = data
    ctx['tensors']['bad_idx'] = bad_idx
    ctx['tensors']['filename'] = filename
    ctx['tensors']['github_url'] = github_url
    ctx['tensors']['local_path'] = local_path
run_module(ctx)

# %% [markdown]
# # Screening and search
#
# Let's consider how to identify and evaluate optimal designs using Machine Learning.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Design space
#
# Our design space will be the composition and temperature of the alloys.
# We can evaluate our existing database to see what the known mechanical properties are.
#
# Let's imagine that we want to maximize Tensile Strength and Elongation (some proxy for ductility) at the same time.
# From the chart above you can see that these two properties compete with each other and lead to tradeoffs.
#
# We should consider the "Pareto front" AKA "efficient frontier" like so:
#
# <img src="../lectures/assets/lecture20_pareto_frontier.jpg" alt="Pareto efficient frontier showing trade-off between two objectives" width=400>
#
# Here is a short code to compute the efficient frontier:
#
# If we want to maximize both properties, the best solution would be one of these.
# Which solution is best depends on the relative value of each of the two objectives, strength or ductility.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    data = ctx.get('data')
    is_pareto_efficient = ctx.get('tensors', {}).get('is_pareto_efficient')
    from matplotlib import pyplot as plt

    x = data.iloc[:, 1:-4]
    y = data.iloc[:, -4:]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i == j:
                ax.hist(y.iloc[:, i])
            else:
                ax.scatter(y.iloc[:, i], y.iloc[:, j])
            if i == 3:
                ax.set_xlabel(y.columns[j])
            if j == 0:
                ax.set_ylabel(y.columns[i])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(y.iloc[:, 1], y.iloc[:, 2])
    ax.set_xlabel(y.iloc[:, 1].name)
    ax.set_ylabel(y.iloc[:, 2].name)

    def is_pareto_efficient(costs, return_mask = True):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    # find the efficient frontier
    costs = y.iloc[:, [1, 2]].values
    on_frontier = is_pareto_efficient(-costs)

    # make figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
    ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier', markerfacecolor='none')

    ax.set_xlabel(y.iloc[:, 1].name)
    ax.set_ylabel(y.iloc[:, 2].name)

    ax.legend()
    ctx['tensors']['costs'] = costs
    ctx['tensors']['on_frontier'] = on_frontier
    ctx['tensors']['x'] = x
    ctx['tensors']['y'] = y
run_module(ctx)

# %% [markdown]
# ## Surrogate modeling
#
# Let's generate a surrogate model for this simplified system:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    x = ctx.get('tensors', {}).get('x')
    y = ctx.get('tensors', {}).get('y')
    # split data
    idx_train, idx_test = model_selection.train_test_split(np.arange(x.shape[0]))

    # fit model
    model = ensemble.RandomForestRegressor(random_state=0)
    _ = model.fit(x[idx_train], y[idx_train].flatten())

    # report performance
    print(f'Train {model.score(x[idx_train], y[idx_train]):.3f}')
    print(f'Test  {model.score(x[idx_test],  y[idx_test]):.3f}')
    ctx['model'] = model
run_module(ctx)

# %% [markdown]
# ## Screening
#
# And now we repeat our screening approach by KDE:
#
# We see that the value obtained by screening is close but not equal to the true minimum.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    gaussian_kde = ctx.get('tensors', {}).get('gaussian_kde')
    idx_train = ctx.get('tensors', {}).get('idx_train')
    model = ctx.get('model')
    x = ctx.get('tensors', {}).get('x')
    xy_to_comp = ctx.get('tensors', {}).get('xy_to_comp')
    from scipy.stats import gaussian_kde

    # fit kde model
    kde = gaussian_kde(x[idx_train].T)

    # perform sampling
    x_fake = kde.resample(10000).T

    # predict on these new samples
    y_fake = model.predict(x_fake)

    # make figure
    fig, ax = plt.subplots()
    im = ax.scatter(*x_fake.T, s=4, c=y_fake)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
    ax.set_aspect('equal')
    cb = plt.colorbar(im)

    # plot the true minimum
    ax.plot([0.455], [0.472], 's', color='tab:green', label='Ground Truth')

    # plot the minimum by screening
    min_idx = np.argmin(y_fake)
    ax.plot(*x_fake[min_idx], '^', color='tab:orange', label='Best Result')
    print(f'min value is {y_fake[min_idx]} at {xy_to_comp( x_fake[min_idx] )}')

    ax.legend()
    ctx['tensors']['kde'] = kde
    ctx['tensors']['min_idx'] = min_idx
    ctx['tensors']['x_fake'] = x_fake
    ctx['tensors']['y_fake'] = y_fake
run_module(ctx)

# %% [markdown]
# ## Search
#
# We can employ Bayesian Optimization to search for the minimum:
#
# Your result will vary here.
# In my trials I sometimes got very close and other times much farther away.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    ax_fitness = ctx.get('tensors', {}).get('ax_fitness')
    model = ctx.get('model')
    optimize = ctx.get('tensors', {}).get('optimize')
    p = ctx.get('tensors', {}).get('p')
    plot_contour = ctx.get('tensors', {}).get('plot_contour')
    render = ctx.get('tensors', {}).get('render')
    xy_to_comp = ctx.get('tensors', {}).get('xy_to_comp')
    from ax.service.managed_loop import optimize

    def ax_fitness(parameterization):
        xy = np.array([parameterization[p] for p in 'xy']).reshape(1, -1)
        return model.predict(xy)[0]

    best_parameters, values, experiment, ax_model = optimize(
        parameters=[
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
        ],
        evaluation_function=ax_fitness,
        objective_name='objective',
        minimize=True,
    )

    from ax.utils.notebook.plotting import render
    from ax.plot.contour import plot_contour

    render(plot_contour(model=ax_model, param_x='x', param_y='y', metric_name='objective'))

    # process the experiments
    x_out = []
    y_out = []
    for i, trial in experiment.trials.items():
        parameterization = trial.arm.parameters
        features = np.array([parameterization[p] for p in 'xy'])
        x_out.append(features)
        out = model.predict(features.reshape(1, -1))[0]
        y_out.append(out)

    # make figure
    fig, ax = plt.subplots()
    im = ax.scatter(*np.array(x_out).T, s=4, c=y_out)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
    ax.set_aspect('equal')
    cb = plt.colorbar(im)

    # plot the true minimum
    ax.plot([0.455], [0.472], 's', color='tab:green', label='Ground Truth')

    # plot the minimum by Bayesian Optimization
    best_features = np.array([best_parameters[p] for p in 'xy'])
    best_outcome = model.predict(best_features.reshape(1, -1))[0]
    ax.plot(*best_features, '^', color='tab:orange', label='Best Result')
    print(f'min value is {best_outcome} at {xy_to_comp( best_features )}')

    ax.legend()
    ctx['tensors']['best_features'] = best_features
    ctx['tensors']['best_outcome'] = best_outcome
    ctx['tensors']['features'] = features
    ctx['tensors']['out'] = out
    ctx['tensors']['parameterization'] = parameterization
    ctx['tensors']['x_out'] = x_out
    ctx['tensors']['y_out'] = y_out
run_module(ctx)

# %% [markdown]
# # Synthetic data generation

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Data generation with SDV
#
# Let's use Synthetic Data Vault to train a conditional probabilistic model on our dataset.
# This will be similar to the use of KDE above, except that SDV permits conditional generation using rejection sampling (we'll see that in a minute).
#
# We can repeat a similar process compared to what we did above by generating many thousands of samples and testing to see if any are like what we want:
#
# These results are no better than using the simpler KDE above (may actually be worse).

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    data = ctx.get('data')
    model = ctx.get('model')
    on_frontier = ctx.get('tensors', {}).get('on_frontier')
    x_scaler = ctx.get('tensors', {}).get('x_scaler')
    y = ctx.get('tensors', {}).get('y')
    y_scaler = ctx.get('tensors', {}).get('y_scaler')
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    df_meta = SingleTableMetadata()
    df_meta.detect_from_dataframe(data)

    # create a synthetic data generator using GaussianCopula model
    synthesizer = GaussianCopulaSynthesizer(df_meta)
    synthesizer.fit(data)

    # generate synthetic data with no missing values
    synthetic_data = synthesizer.sample(num_rows=10)
    synthetic_data

    # perform sampling
    x_fake = synthesizer.sample(10000).iloc[:, 1:-4]
    xs_fake = x_scaler.transform(x_fake)

    ys_fake = model.predict(xs_fake)
    y_fake = y_scaler.inverse_transform(ys_fake)

    # make figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
    ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

    ax.scatter(y_fake[:, 1], y_fake[:, 2], marker='d', color='tab:orange', label='Fake')

    ax.set_xlabel(y.iloc[:, 1].name)
    ax.set_ylabel(y.iloc[:, 2].name)

    ax.legend()
    ctx['tensors']['df_meta'] = df_meta
    ctx['tensors']['synthesizer'] = synthesizer
    ctx['tensors']['synthetic_data'] = synthetic_data
    ctx['tensors']['x_fake'] = x_fake
    ctx['tensors']['xs_fake'] = xs_fake
    ctx['tensors']['y_fake'] = y_fake
    ctx['tensors']['ys_fake'] = ys_fake
run_module(ctx)

# %% [markdown]
# ## Conditional (rejection) sampling
#
# SDV uses rejection sampling to achieve conditional samples on the distributions.
#
# Are any of these realistic?
# To find out, we need to run the inputs through the predictive model.
#
# Unfortunately it seems like these designs give us an above-average TS but one that is within the 600-700 range rather than 800 as requested.
#
# What if we reduce our TS requirement but add a Elongation requirement as well?
#
# Now it seems that the results are all over the place.
# The additional condition made it even harder to approximate this.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    model = ctx.get('model')
    on_frontier = ctx.get('tensors', {}).get('on_frontier')
    synthesizer = ctx.get('tensors', {}).get('synthesizer')
    x_scaler = ctx.get('tensors', {}).get('x_scaler')
    y = ctx.get('tensors', {}).get('y')
    y_scaler = ctx.get('tensors', {}).get('y_scaler')
    from sdv.sampling import Condition

    target = Condition(num_rows=100, column_values={' Tensile Strength (MPa)': 800})
    designs = synthesizer.sample_from_conditions(conditions=[target])

    designs

    xs_fake = x_scaler.transform(designs.iloc[:, 1:-4])

    ys_fake = model.predict(xs_fake)
    y_fake = y_scaler.inverse_transform(ys_fake)

    # make figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
    ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

    ax.scatter(y_fake[:, 1], y_fake[:, 2], marker='d', color='tab:orange', label='Fake')

    ax.set_xlabel(y.iloc[:, 1].name)
    ax.set_ylabel(y.iloc[:, 2].name)

    ax.legend()

    target = Condition(num_rows=100, column_values={' Tensile Strength (MPa)': 500, ' Elongation (%)': 40})
    designs = synthesizer.sample_from_conditions(conditions=[target])

    xs_fake = x_scaler.transform(designs.iloc[:, 1:-4])

    ys_fake = model.predict(xs_fake)
    y_fake = y_scaler.inverse_transform(ys_fake)

    # make figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(y.iloc[:, 1], y.iloc[:, 2], label='Data')
    ax.plot(y.iloc[on_frontier, 1], y.iloc[on_frontier, 2], 'rs', label='Frontier')

    ax.scatter(y_fake[:, 1], y_fake[:, 2], marker='d', color='tab:orange', label='Fake')

    ax.set_xlabel(y.iloc[:, 1].name)
    ax.set_ylabel(y.iloc[:, 2].name)

    ax.legend()
    ctx['tensors']['designs'] = designs
    ctx['tensors']['target'] = target
    ctx['tensors']['xs_fake'] = xs_fake
    ctx['tensors']['y_fake'] = y_fake
    ctx['tensors']['ys_fake'] = ys_fake
run_module(ctx)

# %% [markdown]
# # A simple synthetic dataset
#
# Let's create a simpler dataset and see if we can achieve good results using any of these methods.
#
# This is a fictitious ternary system.
# We could think of it like a 3-component metal alloy.
#
# The above is a 2D representation in the phase diagram, but we need to be able to convert between this and the full 3D representation:
#
# Let's define some "property" of this system.
#
# The true minimum of this function is $f = 6/11$ at the point $(x, y, z) = (6/11, 3/11, 2/11)$, or approximately $(0.545, 0.273, 0.182)$.
# This corresponds to the point $(0.455, 0.472)$.
# Without the equation we wouldn't be able to determine this analytically.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    quadratic_fom = ctx.get('tensors', {}).get('quadratic_fom')
    sample_tri = ctx.get('tensors', {}).get('sample_tri')
    xy_to_comp = ctx.get('tensors', {}).get('xy_to_comp')
    def sample_tri(Ns):
        """Use rejection sampling to find `Ns` points inside an equilateral triangle."""
        # draw samples
        xy = np.random.rand(Ns*4, 2)
        xy[:, 1] *= np.sqrt(3)/2
        # constrain to the triangle
        c1 = xy[:, 1] < xy[:, 0]*np.sqrt(3)
        c2 = xy[:, 1] < np.sqrt(3)*(1-xy[:, 0])
        xy = xy[np.logical_and(c1,c2)]
        # return only the number requested
        xy = xy[:Ns]
        return xy

    out = sample_tri(1000)
    fig, ax = plt.subplots()
    ax.plot(*out.T, '.')
    ax.set_aspect('equal')

    def xy_to_comp(xy):
        """Convert 2D coordinates to a ternary composition."""
        x, y = xy.T
        a = 2/np.sqrt(3)*y
        b = 1 - a/2 - x
        c = 2*x + b - 1
        return np.vstack([a,b,c]).T

    def comp_to_xy(abc):
        """Convert ternary compositions to 2D coordinates."""
        a, b, c = abc.T
        x = ( 1 + c - b ) /2
        y = np.sqrt(3) * a / 2

        return np.vstack([x, y]).T

    a, b, c = xy_to_comp(sample_tri(1000)).T

    # draw histograms
    fig, ax = plt.subplots()
    _ = ax.hist(a, alpha=0.5)
    _ = ax.hist(b, alpha=0.5)
    _ = ax.hist(c, alpha=0.5)

    def quadratic_fom(xy, noise=0):
        """Define a figure of merit to model."""
        abc = xy_to_comp( xy )
        lab = np.array([[1,2,3]])
        metric = np.sum(abc**2 * lab, axis=1)
        metric += noise * np.random.standard_normal(metric.shape)
        return metric

    x = sample_tri(1000)
    y = quadratic_fom(x, 0.10).reshape(-1, 1)
    fig, ax = plt.subplots()
    im = ax.scatter(*x.T, s=4, c=y)
    ax.set_aspect('equal')
    cb = plt.colorbar(im)
    ctx['tensors']['out'] = out
    ctx['tensors']['x'] = x
    ctx['tensors']['y'] = y
run_module(ctx)

# %% [markdown]
# ## Surrogate modeling
#
# Let's generate a surrogate model for this simplified system:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    x = ctx.get('tensors', {}).get('x')
    y = ctx.get('tensors', {}).get('y')
    # split data
    idx_train, idx_test = model_selection.train_test_split(np.arange(x.shape[0]))

    # fit model
    model = ensemble.RandomForestRegressor(random_state=0)
    _ = model.fit(x[idx_train], y[idx_train].flatten())

    # report performance
    print(f'Train {model.score(x[idx_train], y[idx_train]):.3f}')
    print(f'Test  {model.score(x[idx_test],  y[idx_test]):.3f}')
    ctx['model'] = model
run_module(ctx)

# %% [markdown]
# ## Screening
#
# And now we repeat our screening approach by KDE:
#
# We see that the value obtained by screening is close but not equal to the true minimum.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    gaussian_kde = ctx.get('tensors', {}).get('gaussian_kde')
    idx_train = ctx.get('tensors', {}).get('idx_train')
    model = ctx.get('model')
    x = ctx.get('tensors', {}).get('x')
    xy_to_comp = ctx.get('tensors', {}).get('xy_to_comp')
    from scipy.stats import gaussian_kde

    # fit kde model
    kde = gaussian_kde(x[idx_train].T)

    # perform sampling
    x_fake = kde.resample(10000).T

    # predict on these new samples
    y_fake = model.predict(x_fake)

    # make figure
    fig, ax = plt.subplots()
    im = ax.scatter(*x_fake.T, s=4, c=y_fake)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
    ax.set_aspect('equal')
    cb = plt.colorbar(im)

    # plot the true minimum
    ax.plot([0.455], [0.472], 's', color='tab:green', label='Ground Truth')

    # plot the minimum by screening
    min_idx = np.argmin(y_fake)
    ax.plot(*x_fake[min_idx], '^', color='tab:orange', label='Best Result')
    print(f'min value is {y_fake[min_idx]} at {xy_to_comp( x_fake[min_idx] )}')

    ax.legend()
    ctx['tensors']['kde'] = kde
    ctx['tensors']['min_idx'] = min_idx
    ctx['tensors']['x_fake'] = x_fake
    ctx['tensors']['y_fake'] = y_fake
run_module(ctx)

# %% [markdown]
# ## Search
#
# We can employ Bayesian Optimization to search for the minimum:
#
# Your result will vary here.
# In my trials I sometimes got very close and other times much farther away.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    ax_fitness = ctx.get('tensors', {}).get('ax_fitness')
    model = ctx.get('model')
    optimize = ctx.get('tensors', {}).get('optimize')
    p = ctx.get('tensors', {}).get('p')
    plot_contour = ctx.get('tensors', {}).get('plot_contour')
    render = ctx.get('tensors', {}).get('render')
    xy_to_comp = ctx.get('tensors', {}).get('xy_to_comp')
    from ax.service.managed_loop import optimize

    def ax_fitness(parameterization):
        xy = np.array([parameterization[p] for p in 'xy']).reshape(1, -1)
        return model.predict(xy)[0]

    best_parameters, values, experiment, ax_model = optimize(
        parameters=[
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
        ],
        evaluation_function=ax_fitness,
        objective_name='objective',
        minimize=True,
    )

    from ax.utils.notebook.plotting import render
    from ax.plot.contour import plot_contour

    render(plot_contour(model=ax_model, param_x='x', param_y='y', metric_name='objective'))

    # process the experiments
    x_out = []
    y_out = []
    for i, trial in experiment.trials.items():
        parameterization = trial.arm.parameters
        features = np.array([parameterization[p] for p in 'xy'])
        x_out.append(features)
        out = model.predict(features.reshape(1, -1))[0]
        y_out.append(out)

    # make figure
    fig, ax = plt.subplots()
    im = ax.scatter(*np.array(x_out).T, s=4, c=y_out)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
    ax.set_aspect('equal')
    cb = plt.colorbar(im)

    # plot the true minimum
    ax.plot([0.455], [0.472], 's', color='tab:green', label='Ground Truth')

    # plot the minimum by Bayesian Optimization
    best_features = np.array([best_parameters[p] for p in 'xy'])
    best_outcome = model.predict(best_features.reshape(1, -1))[0]
    ax.plot(*best_features, '^', color='tab:orange', label='Best Result')
    print(f'min value is {best_outcome} at {xy_to_comp( best_features )}')

    ax.legend()
    ctx['tensors']['best_features'] = best_features
    ctx['tensors']['best_outcome'] = best_outcome
    ctx['tensors']['features'] = features
    ctx['tensors']['out'] = out
    ctx['tensors']['parameterization'] = parameterization
    ctx['tensors']['x_out'] = x_out
    ctx['tensors']['y_out'] = y_out
run_module(ctx)

# %% [markdown]
# ## Rejection sampling
#
# We can also try the rejection sampling approach with SDV to get values close to some target:
#
# You will see here that the sampled points are all over the place.
# While they are generally biased towards the true minimum, they aren't any better than screening.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    model = ctx.get('model')
    x = ctx.get('tensors', {}).get('x')
    xy_to_comp = ctx.get('tensors', {}).get('xy_to_comp')
    y = ctx.get('tensors', {}).get('y')
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer

    ternary_df = pd.DataFrame({'x': x[:, 0], 'y': x[:, 1], 'obj': y.flatten()})

    df_meta = SingleTableMetadata()
    df_meta.detect_from_dataframe(ternary_df)

    # create a synthetic data generator using GaussianCopula model
    synthesizer = GaussianCopulaSynthesizer(df_meta)
    synthesizer.fit(ternary_df)

    # generate synthetic data with no missing values
    synthetic_data = synthesizer.sample(num_rows=10)
    synthetic_data

    from sdv.sampling import Condition

    low_objective = Condition(num_rows=100, column_values={'obj': 0.45})
    synthetic_data = synthesizer.sample_from_conditions(conditions=[low_objective])

    synthetic_data

    x_fake = synthetic_data.iloc[:, 0:2].values
    y_fake = model.predict(x_fake)

    # make figure
    fig, ax = plt.subplots()
    im = ax.scatter(*x_fake.T, s=4, c=y_fake)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
    ax.set_aspect('equal')
    cb = plt.colorbar(im)

    # plot the true minimum
    ax.plot([0.455], [0.472], 's', color='tab:green', label='Ground Truth')

    # plot the minimum by screening
    min_idx = np.argmin(y_fake)
    ax.plot(*x_fake[min_idx], '^', color='tab:orange', label='Best Result')
    print(f'min value is {y_fake[min_idx]} at {xy_to_comp( x_fake[min_idx] )}')

    ax.legend()
    ctx['tensors']['df_meta'] = df_meta
    ctx['tensors']['low_objective'] = low_objective
    ctx['tensors']['min_idx'] = min_idx
    ctx['tensors']['synthesizer'] = synthesizer
    ctx['tensors']['synthetic_data'] = synthetic_data
    ctx['tensors']['ternary_df'] = ternary_df
    ctx['tensors']['x_fake'] = x_fake
    ctx['tensors']['y_fake'] = y_fake
run_module(ctx)

# %% [markdown]
# # Inverse function approximation

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Normalizing flows
#
# <img src="../lectures/assets/lecture20_normalizing_flow.jpg" alt="Conceptual diagram of Normalizing Flows for density estimation" width=600>

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    optim = ctx.get('tensors', {}).get('optim')
    plot_flow = ctx.get('tensors', {}).get('plot_flow')
    sample_tri = ctx.get('tensors', {}).get('sample_tri')
    tqdm = ctx.get('tensors', {}).get('tqdm')
    import torch
    from torch import nn
    from torch import optim

    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
    from nflows.transforms.permutations import ReversePermutation

    num_layers = 5
    base_dist = StandardNormal(shape=[2])

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                              hidden_features=4))
    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    optimizer = optim.Adam(flow.parameters())

    def plot_flow(flow, x):
        xline = torch.linspace(0, 1, 100)
        yline = torch.linspace(0, np.sqrt(3)/2, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

        fig, ax = plt.subplots()
        ax.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        ax.plot(*x.T, 'w.')
        return fig

    out = sample_tri(128)
    fig = plot_flow(flow, out)

    import tqdm
    import numpy as np

    num_iter = 1000
    for i in tqdm.tqdm(np.arange(num_iter)):
        x_out = sample_tri(256)
        x_out = torch.tensor(x_out, dtype=torch.float32)
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=x_out).mean()
        loss.backward()
        optimizer.step()

        if (i + 1) % 500 == 0:
            fig = plot_flow(flow, x_out)
            fig.axes[0].set_title('iteration {}'.format(i + 1))
            plt.show()

    x_out, y_out = flow.sample(1000).detach().numpy().T
    fig, ax = plt.subplots()
    ax.plot(x_out, y_out, '.')
    ax.set_aspect('equal')
    ctx['tensors']['base_dist'] = base_dist
    ctx['tensors']['flow'] = flow
    ctx['tensors']['loss'] = loss
    ctx['tensors']['num_iter'] = num_iter
    ctx['tensors']['num_layers'] = num_layers
    ctx['tensors']['optimizer'] = optimizer
    ctx['tensors']['out'] = out
    ctx['tensors']['transform'] = transform
    ctx['tensors']['transforms'] = transforms
    ctx['tensors']['x_out'] = x_out
run_module(ctx)

# %% [markdown]
# ## Conditional Normalizing Flows

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    it = ctx.get('tensors', {}).get('it')
    model = ctx.get('model')
    nn = ctx.get('tensors', {}).get('nn')
    optim = ctx.get('tensors', {}).get('optim')
    plot_cond_flow = ctx.get('tensors', {}).get('plot_cond_flow')
    tqdm = ctx.get('tensors', {}).get('tqdm')
    x = ctx.get('tensors', {}).get('x')
    y = ctx.get('tensors', {}).get('y')
    import torch
    from torch import nn
    from torch import optim

    from nflows.flows.base import Flow
    from nflows.distributions.normal import ConditionalDiagonalNormal
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
    from nflows.transforms.permutations import ReversePermutation
    from nflows.nn.nets import ResidualNet


    num_layers = 5
    base_dist = ConditionalDiagonalNormal(shape=[2],
                                          context_encoder=nn.Linear(1, 4))

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                              hidden_features=4,
                                                              context_features=1))
    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    optimizer = optim.Adam(flow.parameters())

    def plot_cond_flow(flow, c=None):
        xline = torch.linspace(0, 1, 100)
        yline = torch.linspace(0, np.sqrt(3)/2, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        if c is None:
            c = torch.rand_like(xyinput[:, :1]) * 2 + 1

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput, c*torch.ones_like(xyinput[:, :1])).exp().reshape(100, 100)

        fig, ax = plt.subplots()
        ax.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
        return fig

    fig = plot_cond_flow(flow, 0.5)
    fig = plot_cond_flow(flow, 1.0)
    fig = plot_cond_flow(flow, 1.5)

    import tqdm
    import numpy as np

    num_iter = 2000

    x_out = torch.tensor(x, dtype=torch.float32)
    y_out = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    for i in tqdm.tqdm(np.arange(num_iter)):

        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=x_out, context=y_out).mean()
        loss.backward()
        optimizer.step()

        if (i + 1) % 500 == 0:
            fig = plot_cond_flow(flow, 1.0)
            fig.axes[0].set_title('iteration {}'.format(i + 1))
            plt.show()
            flow = flow

    fig = plot_cond_flow(flow, 0.5)
    fig = plot_cond_flow(flow, 1.0)
    fig = plot_cond_flow(flow, 1.5)

    # generate samples
    targets = np.arange(0.5, 3.0, 0.25)
    out = flow.sample(1024, context=torch.tensor(targets, dtype=torch.float32).reshape(-1, 1))
    xy = out.detach().numpy()

    # plot the samples
    fig, ax = plt.subplots()
    for i, xi in enumerate(xy):
        ax.plot(*xi.T, '.', label=f'c={targets[i]}', zorder=len(targets)-i, alpha=0.33)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', alpha=0.5)
    ax.set_aspect('equal')
    ax.legend()

    # evaluate the results
    metrics = np.array([model.predict(it) for it in xy])

    # create parity plot
    fig, ax = plt.subplots()
    ax.errorbar(targets, metrics.mean(axis=1), yerr=metrics.std(axis=1), ls='none', marker='o', label='Generated')
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', label='Reference')

    ax.set_aspect('equal')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Sampled')
    ax.legend()
    ctx['tensors']['base_dist'] = base_dist
    ctx['tensors']['flow'] = flow
    ctx['tensors']['loss'] = loss
    ctx['tensors']['metrics'] = metrics
    ctx['tensors']['num_iter'] = num_iter
    ctx['tensors']['num_layers'] = num_layers
    ctx['tensors']['optimizer'] = optimizer
    ctx['tensors']['out'] = out
    ctx['tensors']['targets'] = targets
    ctx['tensors']['transform'] = transform
    ctx['tensors']['transforms'] = transforms
    ctx['tensors']['x_out'] = x_out
    ctx['tensors']['y_out'] = y_out
run_module(ctx)

# %% [markdown]
# ## conditional Generative Adversarial Networks
#
# <img src="../lectures/assets/lecture20_generative_models.jpg" alt="Different types of generative models like VAE, GAN, Flow-based" width=600>
#
# Details for a future lecture...

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)
