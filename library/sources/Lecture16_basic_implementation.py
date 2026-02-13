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
# id: Lecture16_basic_implementation
# type: Foundational
# parent_lecture: Lecture16
# ---
#
# ## Basic implementation
#
# We don't have to only use real curves that were put through the encoder -- we can also generate "synthetic" curves by passing an arbitrary $z$ through the decoder.
# Here we'll use all zeros:
#
# We can plot this to see we recover a full (but synthetic) observation of $y$:
#
# We can also compare this to a real curve nearby to this point in latent space ( in this case, the origin):
#
# The two curves are not going to be exactly the same since their latent codes are slightly different:
#
# We can extend this to generate a series of synthetic curves spread across the latent space:
#
# Why are some of the curves overlapping?
#
# We could also do so over a regular grid:
#
# Now sample at those points:
#
# We can increase the number of samples to really see the fine details of this space:

# %%
from sklearn import cluster

km = cluster.KMeans(n_clusters=64).fit(z[:, :2])

cluster_ids = []
for c in km.cluster_centers_:
    cluster_ids.append( np.argmin( np.linalg.norm(z[:, :2] - c, axis=1) ) )

scale = 0.5

fig, ax = plt.subplots(figsize=(16, 16))

for i in cluster_ids:
    ax.plot(z[i, 0]+x*scale, z[i, 1]+y[i]*scale)

ax.set_aspect('equal')

model.decoder(torch.zeros([1, a.shape[1]]).float())

gen_y = model.decoder(torch.zeros([1, a.shape[1]]).float())

fig, ax = plt.subplots()
ax.plot(x, gen_y[0].detach().numpy())

close_to_origin = np.argmin(np.linalg.norm(z, axis=1))
print(z[close_to_origin])

fig, ax = plt.subplots()
_ = ax.plot(x, y[close_to_origin], label='Real')
_ = ax.plot(x, gen_y[0].detach().numpy(), label='Generated')
_ = ax.legend()

fig, ax = plt.subplots(figsize=(16, 16))

km = cluster.KMeans(n_clusters=64).fit(z)

for c in km.cluster_centers_:
    gen_y = model.decoder(torch.tensor(c).float().unsqueeze(0))
    this_y = gen_y.detach().numpy()[0]
    ax.plot(c[0]+x*scale, c[1]+this_y*scale)

ax.set_aspect('equal')

steps = 10

z0v = np.linspace(z[:, 0].min(), z[:, 0].max(), steps)
z1v = np.linspace(z[:, 1].min(), z[:, 1].max(), steps)
z0g, z1g = np.meshgrid(z0v, z1v)

fig, ax = plt.subplots()
ax.scatter(z0g.flatten(), z1g.flatten())
ax.set_aspect('equal')

zgrid = np.vstack([z0g.flatten(), z1g.flatten(),
                   [0]*steps**2, [0]*steps**2, [0]*steps**2]).T  # why is this here?

fig, ax = plt.subplots(figsize=(16, 16))

for c in zgrid:
    gen_y = model.decoder(torch.tensor(c).float().unsqueeze(0))
    this_y = gen_y.detach().numpy()[0]
    ax.plot(c[0]+x*scale, c[1]+this_y*scale)

ax.set_aspect('equal')

steps = 50

z0v = np.linspace(z[:, 0].min(), z[:, 0].max(), steps)
z1v = np.linspace(z[:, 1].min(), z[:, 1].max(), steps)
z0g, z1g = np.meshgrid(z0v, z1v)

zgrid = np.vstack([z0g.flatten(), z1g.flatten(),
                   [0]*steps**2, [0]*steps**2, [0]*steps**2]).T  # why is this here?

fig, ax = plt.subplots(figsize=(16, 16))

scale = 0.1

for c in zgrid:
    gen_y = model.decoder(torch.tensor(c).float().unsqueeze(0))
    this_y = gen_y.detach().numpy()[0]
    ax.plot(c[0]+x*scale, c[1]+this_y*scale)

ax.set_aspect('equal')
