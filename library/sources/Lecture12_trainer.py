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
# id: Lecture12_trainer
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## `Trainer`
#
# One of the main benefits of PyTorch Lightning is that it provides many other features that can help streamline your code, such as automatic checkpointing, early stopping, and distributed training. To use these features, we need to create a Trainer object:
#
# Note that if hardware accelerators were available, the `trainer` could handle moving our data around between the CPU and CUDA device.
# To train the model we need to provide data and call `trainer.fit` to start the training loop:
#
# By default, PyTorch Lightning will log the training loss and other metrics to the terminal. You can also configure PyTorch Lightning to log to a file or to a remote server. For example, to log to a file, you can pass the logger argument to the `Trainer` constructor:
#
# This will create a new folder called logs in the current directory, and save the training logs to a CSV file inside that folder.
# We can access the data like this:
#
# This can be used to create a plot of the train loss after the fact:

# %%
trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1)

train_data = [(xlt[i], ylt[i]) for i in range(xlt.shape[0])]
dl = DataLoader(train_data, batch_size=64, shuffle=True)

trainer.fit(pl_model, dl)

from pytorch_lightning.loggers import CSVLogger

logger = CSVLogger('logs', name='mlp-regressor')

trainer = pl.Trainer(max_epochs=10, logger=logger, log_every_n_steps=1)
trainer.fit(pl_model, dl)

log_path = 'logs/mlp-regressor/version_0/metrics.csv'
if os.path.exists(log_path):
    metrics = pd.read_csv(log_path)
else:
    colab_path = '/content/logs/mlp-regressor/version_0/metrics.csv'
    if os.path.exists(colab_path):
        metrics = pd.read_csv(colab_path)
    else:
        print(f"Warning: Log file not found at {log_path} or {colab_path}")
        metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'validation_loss'])
metrics.head(6)

plt.plot( metrics['epoch'] * len(dl) + metrics['step'], metrics['train_loss'])
