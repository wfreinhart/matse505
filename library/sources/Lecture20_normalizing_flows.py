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
# id: Lecture20_normalizing_flows
# type: Foundational
# parent_lecture: Lecture20
# ---
#
# ## Normalizing flows
#
# <img src="../lectures/assets/lecture20_normalizing_flow.jpg" alt="Conceptual diagram of Normalizing Flows for density estimation" width=600>

# %%
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
