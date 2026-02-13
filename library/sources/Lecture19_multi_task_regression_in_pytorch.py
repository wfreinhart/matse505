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
# id: Lecture19_multi_task_regression_in_pytorch
# type: Foundational
# parent_lecture: Lecture19
# ---
#
# ## Multi-task regression in PyTorch
#
# Now we can implement a Single Input Multi Output (SIMO) scheme in PyTorch.
#
# As always, we start with PyTorch Lightning:
#
# We set up the datasets like so:
#
# ### Custom loss function
#
# We need a custom loss function to handle missing (`NaN`) values:
#
# ### Model architecture
#
# The model will have some extra tricks compared to our usual MLP:
#
# ### Implementation and evaluation
#
# Let's plot the performance:
#
# And if we evaluate the performance on each task:
#
# So far this looks worse than our train performance but similar to our test performance with the single-task models from `scikit-learn`.
#
# What about on the full dataset?
#
# If you recall, our performance on the Full dataset with our 4 individual models was:
#
# ```
# Task 0: 1.398
# Task 1: 1.405
# Task 2: 1.356
# Task 3: 1.412
# ```
#
# and our performance with the full 4-output model was:
#
# ```
# Task 0: 1.389
# Task 1: 1.362
# Task 2: 1.622
# Task 3: 1.492
# ```
#
# These terrible scores were clearly the result of overfitting a large model (10k parameters) to small datasets.
# With the Multi-Task Learning scheme, we get several advantages:
#
# 1. **More rows with $(x, y)$ pairs.** Because there are a small number of rows missing all 4 labels compared to the number missing 0-3 labels, more $x$ samples are included in training. Each of these helps train the backbone to learn something useful.
# 2. **More $(x, y)$ pairs per row.** Because there are 4 targets being predicted from one input $x$, the total number of target labels is much higher.
# 3. **Better generalization.** Because there is a common backbone for 4 different tasks, the model is explicitly forced to learn a representation that is most general. This predictably helps the model avoid overfitting.
#
# We can visualize this improved performance with a parity plot again:
#
# Much better! Recall that these are the scaled outputs.
# We can unscale the outputs with the `inverse_transform` method of the scaler:
#
# Note that if we were to regress on these raw labels we would heavily bias the MTL model to `Tensile Strength` since it is measured in 100's compared to `Elongation` which is measured in 10's.

# %%
try:
    import pytorch_lightning as pl
except:
    # !pip install pytorch_lightning
    import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader

idx_train, idx_test = model_selection.train_test_split(np.arange(missing.shape[0]), random_state=0)

x_arr = missing.iloc[:, 1:-4].values.astype(float)  # 15 features (skip categorical codes)
y_arr = missing.iloc[:, -4:].values.astype(float)    # 4 labels (the mechanical properties)

x_scaler = preprocessing.StandardScaler().fit(x_arr[idx_train])
y_scaler = preprocessing.StandardScaler().fit(y_arr[idx_train])

x = torch.tensor(x_scaler.transform(x_arr)).float()
y = torch.tensor(y_scaler.transform(y_arr)).float()

ds_train = [(x[i], y[i]) for i in idx_train]
ds_test = [(x[i], y[i]) for i in idx_test]

dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=256, shuffle=False)

from torch import nn
import torch.nn.functional as F

class MultiTaskMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Mask out the NaN values in the targets
        mask = ~torch.isnan(target)

        # Compute the mean squared error, ignoring NaN values
        mse = F.mse_loss(input[mask], target[mask], reduction='mean')

        return mse

import pytorch_lightning as pl

class MLPRegressor(nn.Module):
    def __init__(self, hidden_size=(100, )):
        super(MLPRegressor, self).__init__()

        layers = []
        for layer_size in hidden_size:
            layers.append(nn.LazyLinear(layer_size))
            layers.append(nn.LeakyReLU())

        self.fc_layers = nn.Sequential(*layers[:-1])

    def forward(self, x):
        x = self.fc_layers(x)
        return x

class MultiTaskMLP(pl.LightningModule):
    def __init__(self, hidden_size, n_tasks):
        super(MultiTaskMLP, self).__init__()

        self.n_tasks = n_tasks

        # make a shared backbone for all tasks
        self.backbone = MLPRegressor(hidden_size)

        # now make a head for each task
        self.heads = [nn.LazyLinear(1) for _ in range(n_tasks)]

        # self.criterion = nn.MSELoss(reduction='none')  # handle nan specially
        self.criterion = MultiTaskMSELoss()

    def forward(self, x):
        x = self.backbone(x)
        out = [h(x) for h in self.heads]  # task-specific layers
        return out

    def training_step(self, batch, batch_idx):

        x, y = batch
        out = self(x)

        # compute loss for each task separately
        loss = 0
        for i in range(self.n_tasks):
            loss += self.criterion(out[i], y[:, i].unsqueeze(1))

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch
        with torch.no_grad():
            out = self(x)

            loss = 0
            for i in range(self.n_tasks):
                task_loss = self.criterion(out[i], y[:, i].unsqueeze(1))
                loss += task_loss
                self.log(f"task_{i}_loss", task_loss)

        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

from pytorch_lightning.loggers import CSVLogger

logger = CSVLogger('logs', name='mtl-mlp')

torch.manual_seed(0)  # control random effects

model_ft = MultiTaskMLP([128, 64, ], 4)

# Initialize the weights before sending to pl to count trainable weights
model_ft(dl_train.dataset[0][0].unsqueeze(0))

# Use pl to train
trainer = pl.Trainer(max_epochs=40, logger=logger, log_every_n_steps=1)
trainer.fit(model=model_ft, train_dataloaders=dl_train, val_dataloaders=dl_test)

from matplotlib import pyplot as plt

log_path = 'logs/mtl-mlp/version_0/metrics.csv'
if os.path.exists(log_path):
    metrics = pd.read_csv(log_path)
else:
    colab_path = '/content/logs/mtl-mlp/version_0/metrics.csv'
    if os.path.exists(colab_path):
        metrics = pd.read_csv(colab_path)
    else:
        print(f"Warning: Log file not found at {log_path} or {colab_path}")
        metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'validation_loss'])

fig, ax = plt.subplots()
ax.plot( metrics['epoch'] * len(dl_train) + metrics['step'], np.sqrt(metrics['train_loss']), '.', label='Train')
ax.plot( metrics['epoch'] * len(dl_train) + metrics['step'], np.sqrt(metrics['validation_loss']), '.', label='Test')
ax.set_yscale('log')
ax.set_xlabel('Train Step')
ax.set_ylabel('RMSE')
ax.legend()

for name, dl in [('Train', dl_train), ('Test', dl_test)]:
    print(name)
    print('-------------')
    losses = [0] * model_ft.n_tasks

    for batch in dl:
        x, y = batch
        with torch.no_grad():
            out = model_ft(x)

            for i in range(model_ft.n_tasks):
                this_loss = model_ft.criterion(out[i], y[:, i].unsqueeze(1)).item()
                losses[i] += this_loss

    for i in range(model_ft.n_tasks):
        print(f'Task {i}: {np.sqrt(losses[i]):.3f}')
    # print(f'> Total: {np.sqrt(np.sum(losses)):.3f}')
    print()

x_full = data.iloc[:, 1:-4].values.astype(float)
y_full = data.iloc[:, -4:].values.astype(float)

xf_sc = torch.tensor(x_scaler.transform(x_full)).float()
yf_sc = torch.tensor(y_scaler.transform(y_full)).float()

with torch.no_grad():
    y_hat = model_ft(xf_sc)

print('Full')
print('-------------')

for i in range(y.shape[1]):
    residuals = y_hat[i] - yf_sc[:, i].unsqueeze(1)
    print(f'Task {i}: {np.sqrt(np.mean(residuals.detach().numpy()**2)):.3f}')

for batch in dl_test:
    x, y = batch
    with torch.no_grad():
        out = model_ft(x)

fig, ax = plt.subplots()
ax.plot([-3, 3], [-3, 3], 'k--')
for i in range(model_ft.n_tasks):
    ax.scatter(out[i].detach().numpy(), y[:, i].detach().numpy(), label=f'{missing.columns[i-4]}')

ax.set_xlabel('Predicted')
ax.set_ylabel('Observed')
ax.legend()

y_orig = y_scaler.inverse_transform( y.detach().numpy() )
y_hat = y_scaler.inverse_transform( np.hstack([it.detach().numpy() for it in out]) )

fig, ax = plt.subplots()
ax.plot([0, 800], [0, 800], 'k--')
for i in range(model_ft.n_tasks):
    ax.scatter(y_hat[:, i], y_orig[:, i], label=f'{missing.columns[i-4]}')

ax.set_xlabel('Predicted')
ax.set_ylabel('Observed')
ax.legend()
