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
# id: Lecture21_graph_learning_model
# type: Foundational
# parent_lecture: Lecture21
# ---
#
# ## Graph learning model
#
# Next, we can define our message passing network using PyG:
#
# Finally we train with `pytorch-lightning`:

# %%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
import pytorch_lightning as pl


class MPNRegressor(pl.LightningModule):
    def __init__(self, num_features, hidden_channels, out_channels, lr):
        super().__init__()

        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.lin = torch.nn.Linear(out_channels, 1)

        self.lr = lr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Perform convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Pool over all nodes in the graph
        x = F.relu(global_mean_pool(x, data.batch))

        x = self.lin(x)
        return x

    def training_step(self, batch, batch_idx):
        output = self(batch)
        # print(output.shape, batch.y.shape)
        loss = F.mse_loss(output, batch.y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = F.mse_loss(output, batch.y.unsqueeze(1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

model = MPNRegressor(num_features=8,
                     hidden_channels=16, out_channels=1,
                     lr=1e-2)

for batch in train_loader:
    out = model(batch)
    break

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)

results = []
with torch.no_grad():
    for loader in [train_loader, val_loader]:
        loss = 0
        real = []
        pred = []
        for batch in loader:
            output = model(batch)
            loss += torch.sum((output - batch.y.unsqueeze(1))**2) / len(batch.y)
            real.append(batch.y.detach().numpy())
            pred.append(output.detach().squeeze(1).numpy())
        rmse = torch.sqrt(loss)
        results.append((np.hstack(real), np.hstack(pred)))
        print(str(loader), rmse.item())

from matplotlib import pyplot as plt

fig, ax = plt.subplots()
for r in results:
    ax.scatter(r[0], r[1])
