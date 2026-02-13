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
# id: Lecture12_validation
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## Validation
#
# Another benefit of PyTorch Lightning is that it provides a simple interface for testing your trained model on a separate validation or test dataset.
# Here's an example of how you can define a validation_step method in your PyTorch Lightning module to test your model:
#
# In this example, we've defined a new method validation_step that computes the loss and accuracy on a validation batch. We're using the `self.log` method to log these metrics, which will be displayed in the PyTorch Lightning progress bar and can also be saved to a file.
#
# Let's retrain the updated model so we can take advantage of this feature:
#
# To test the trained model on a validation dataset, we can call the `trainer.validate` method.
# This will run the `validation_step` method for each batch in the validation dataset and log the results to the terminal or to a file.

# %%
class LitMLPRegressor(pl.LightningModule):
    def __init__(self, output_size, hidden_size=(100, ), activation=nn.ReLU()):
        super(LitMLPRegressor, self).__init__()

        layers = []
        for layer_size in hidden_size:
            layers.append(nn.LazyLinear(layer_size))
            layers.append(activation)

        layers.append(nn.LazyLinear(output_size))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.mse_loss(out, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        return optimizer

    # new feature:
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.mse_loss(out, y)
        residual = out - y
        rmse = torch.sqrt( torch.mean( residual**2 ) )
        rsq = 1 - torch.var(residual) / torch.var(y)
        self.log('val_loss', loss)
        # additional metrics can be logged here:
        self.log('val_rmse', rmse)
        self.log('val_rsq', rsq)

pl_model = LitMLPRegressor(1, (100, ), activation=nn.Tanh())
trainer = pl.Trainer(max_epochs=100, log_every_n_steps=1)
trainer.fit(pl_model, dl)

# we'll just reuse the training data for this example:
val_dl = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
trainer.validate(pl_model, val_dl)
