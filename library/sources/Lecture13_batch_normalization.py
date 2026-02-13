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
# id: Lecture13_batch_normalization
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## Batch normalization
#
# Batch normalization is a technique that helps to stabilize the training process of neural networks by normalizing the activations of each layer.
# It works by normalizing the activations across the batch dimension, which reduces the internal covariate shift and allows the network to learn more efficiently.
#
# <img src="../lectures/assets/lecture13_batch_norm_1.jpg" alt="Schematic showing the Batch Normalization process" width=600>
#
# > [!NOTE]
# > A second Batch Normalization diagram is currently unavailable due to access restrictions.
#
# The implementation in PyTorch is called `nn.BatchNorm2d`:
#
# In the forward method, we apply batch normalization after each convolutional layer and fully connected layer.
# Note that batch normalization should be applied before the activation function.
#
# Adding batch normalization can improve the convergence speed and generalization performance of the CNN.
# It reduces the internal covariate shift, which helps to stabilize the training process and allows the network to learn more efficiently.
# Additionally, it can act as a regularization technique and prevent overfitting.
#
# Let's try training the model again with batch normalization in place:
#
# We see right away that the loss is significantly lower than before
# Let's make the confusion matrix:

# %%
class ConvBlock(nn.Module):
    def __init__(self, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.LazyConv2d(out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# note: no need to redefine the ClassifierCNN model!

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ClassifierCNN([8, 8], 32, 2)

trainer = pl.Trainer(max_epochs=5)
trainer.fit(model=model, train_dataloaders=dataloader)

y_pred = []
y_true = []

with torch.no_grad():
    for x, y in dataloader:
        outputs = model(x)
        _, label = torch.max(outputs.data, 1)
        y_pred += label.detach().numpy().tolist()
        y_true += y.detach().numpy().tolist()

y_pred = np.array(y_pred)
y_true = np.array(y_true)

confusion = np.zeros([2, 2], dtype=int)
for i in range(len(y_pred)):
    row_idx = y_pred[i].round()
    col_idx = y_true[i].round()
    confusion[row_idx.astype(int), col_idx.astype(int)] += 1

fig, ax = plt.subplots()
im = ax.imshow(confusion, 'Blues')
cb = plt.colorbar(im)
cb.set_label('Frequency')
_ = ax.set_xlabel('True label')
_ = ax.set_ylabel('Predicted label')

for i in range(2):
    for j in range(2):
        if confusion[i, j] > 100:
            tc = 'w'
        else:
            tc = 'k'
        ax.text(i, j, confusion[i, j], ha='center', color=tc)
