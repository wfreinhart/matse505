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
# id: Lecture05_without_labels
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## Without labels
#
# In most clustering scenarios, ground truth labels are not known (this would typically lead to a classification problem).
# Thus we must make do with metrics that evaluate only the clusters themselves and not their accuracy relative to a target outcome.
#
# The "Silhouette Coefficient" evaluates how well the clusters are separated from each other.
# It is calculated as:
#
# $s = \frac{b-a}{\max(a,b)}$,
#
# where $a$ is the mean distance between samples in the same cluster and $b$ is the mean distance between samples in next-nearest clusters.
# Therefore, $s=1$ corresponds to very high separation ($b \gg a$) while $s=0$ corresponds to very low separation ($b \ll a$).
# An intermediate value of $s=0.5$ means samples in different clusters are about twice as far apart as samples within clusters.
#
# We can try calculating the performance of our `KMeans` model:
#
# Let's consider how to use this score to evaluate a suitable number of clusters.
# We can write a `for` loop to iterate over a large number of possible clusters:
#
# This shows that $s$ increases substantially from $k=3$ to around $k=5$, then stagnates, and has non-monotonic increases until $k=15$.
# Among these options we need to evaluate how many clusters will be meaningful for our application.
# I think $k=5$ would be the first choice, then maybe $k=9$, and then $k=14$ if this is still few enough to be useful.
#
# What if we zoom out and try many more clusters?
# After all, it looks like $s$ is increasing with $k$...
#
# This result shows that the Silhouette score generally increases as we move from 5 to 95 clusters.
# Now you have to ask yourself: is 95 clusters a useful result?
# In many cases, probably not.
# There is something unusual happening around $k=25$ clusters (a local dip in $s$ before it recovers again at $k=35$).
# Is this useful?
#
# This simply illustrates that metrics only provide guidance and typically no single metric can determine which model to use.
# The importance of different metrics will greatly depend on how you plan to deploy the models.
# There is often a general range of allowable hyperparameters (such as $k \le 10$) and then you can determine an optimal choice within this range.

# %%
model = cluster.KMeans(n_clusters=4).fit(x)
labels = model.predict(x)

metrics.silhouette_score(x, labels, metric='euclidean')

from matplotlib import pyplot as plt

k_list = np.arange(2, 16)
s = np.zeros(len(k_list))
for i, k in enumerate(k_list):

    model = cluster.KMeans(n_clusters=k, random_state=0).fit(x)
    labels = model.predict(x)
    s[i] = metrics.silhouette_score(x, labels, metric='euclidean')

fig, ax = plt.subplots()
ax.plot(k_list, s, 's')
ax.set_xlabel('$k$')
ax.set_ylabel('$s$')

model = cluster.KMeans(n_clusters=14, random_state=0).fit(x)
labels = model.predict(x)

px.scatter_3d(x=data[' Mo'], y=data[' Cr'], z=data['V'], color=labels)

from matplotlib import pyplot as plt
import tqdm  # a very useful package for progress bars

k_list = np.arange(5, 96, 5)
s = np.zeros(len(k_list))
for i, k in tqdm.tqdm(enumerate(k_list), total=len(k_list)):

    model = cluster.KMeans(n_clusters=k, random_state=0).fit(x)
    labels = model.predict(x)
    s[i] = metrics.silhouette_score(x, labels, metric='euclidean')

fig, ax = plt.subplots()
_ = ax.plot(k_list, s, 's')
_ = ax.set_xlabel('$k$')
_ = ax.set_ylabel('$s$')
