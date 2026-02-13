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
# id: Lecture20_conditional_normalizing_flows
# type: Foundational
# parent_lecture: Lecture20
# ---
#
# ## Conditional Normalizing Flows

# %%
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
