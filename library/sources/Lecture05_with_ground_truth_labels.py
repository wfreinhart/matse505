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
# id: Lecture05_with_ground_truth_labels
# type: Foundational
# parent_lecture: Lecture05
# ---
#
# ## With ground truth labels
#
# ...or can we?
#
# If we have labels (like in this case) we could check how well the chosen clustering method reflects these.
# The `adjusted_rand_score` evaluates something like accuracy while accounting for the fact that cluster indices can be in any order.
# Meanwhile the `normalized_mutual_info_score` is a measure of [mutual information](https://en.wikipedia.org/wiki/Mutual_information) (something kind of like a correlation) between the two labeling schemes.
# In either case, higher values are better (indicating greater correlation between the obtained labels and the ground truth).
#
# Here we see very similar performance between all 3 model classes, with a slight edge to GMM and KMeans (tied).

# %%
from sklearn import metrics

def report_cluster_scores(labels):
    "Compare labels predicted by a clustering algorithm to ground truth."
    ars = metrics.adjusted_rand_score(y, labels)
    amis = metrics.adjusted_mutual_info_score(y, labels)

    print(f'{str(model):40s}: ARS = {ars:.3f}, AMIS = {amis:.3f}')


# apply the function to several clustering models...

model = GaussianMixture(n_components=4).fit(x)
labels = model.predict(x)
report_cluster_scores(labels)

model = cluster.KMeans(n_clusters=4).fit(x)
labels = model.predict(x)
report_cluster_scores(labels)

model = cluster.AgglomerativeClustering(n_clusters=4)
labels = model.fit_predict(x)
report_cluster_scores(labels)
