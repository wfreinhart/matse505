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
# id: Lecture14_in_practice
# type: Foundational
# parent_lecture: Lecture14
# ---
#
# ## In practice
#
# Now let's try fine-tuning the ResNet18 model on our dataset.
# We can again use `pytorch_lightning` for convenience:
#
# We can simply "wrap" the pretrained model inside our `pl.LightningModule`:
#
# Now we can train using the `pl.Trainer`:
#
# Now we can evaluate the accuracy of the model (before we only compute the Cross Entropy Loss):
#
# If we want to understand the changes to the model during fine-tuning, we can repeat the feature extraction task.
# Note that we need to take the features from the pretrained model which is upstream from the final classification layer!

# %%
# !pip install pytorch_lightning

import torch.nn as nn
from torch import optim
import pytorch_lightning as pl


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()

        self.pretrained = resnet18(weights="IMAGENET1K_V1")
        self.classifier = nn.LazyLinear(num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pretrained(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

torch.manual_seed(0)  # control random effects
num_classes = len(ds_train.classes)
model_ft = ResNetClassifier(num_classes)

# Initialize the weights before sending to pl to count trainable weights
model_ft(dl_train.dataset[0][0].unsqueeze(0))

# Now include shuffle in the DataLoader!
dl_train_shuffle = DataLoader(ds_train, batch_size=16, shuffle=True)

# Use pl to train
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model=model_ft, train_dataloaders=dl_train_shuffle)

# Evaluate the model
model_ft.eval()
with torch.no_grad():
    correct = 0
    total = 0
    pbar = tqdm.tqdm(enumerate(dl_train), total=len(dl_train))
    for i, (images, labels) in pbar:

        # Forward pass
        outputs = model_ft(images)
        _, predicted = torch.max(outputs.data, 1)

        # Compute accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nTrain Accuracy: {accuracy:.2f}%')
