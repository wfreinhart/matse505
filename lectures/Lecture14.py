# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Lecture14

# %%
class Context(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault('data', None)
        self.setdefault('tensors', {})
        self.setdefault('model', None)
        self.setdefault('viz', None)

ctx = Context()

# %% [markdown]
# Today's topics:
# * Pretrained models
# * Transfer learning
# * Fine-tuning
# * Data augmentation
#
# We'll again load the [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm), a publicly available dataset of images of various surfaces with different types of defects.
#
# This time we'll use a harder version of the problem, with Crazing, Inclusion, and Patches instead of just Patches and Scratches.
# Here are some samples of the three classes:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    zipfile = ctx.get('tensors', {}).get('zipfile')
    import zipfile, requests

    url = 'https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/wfr5091_psu_edu/EZwz7XK8nMVOkp_V0pXP3HsBYiC_1B8JhHXscCnJFli6yw?e=iFYoO3&download=1'
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open('data.zip', 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    zip_file = zipfile.ZipFile('data.zip')
    zip_file.extractall('/content/')
    zip_file.close()

    from PIL import Image
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for i, label in enumerate(['crazing', 'inclusion', 'patches']):
        ax = axes[i]
        img = Image.open(f'NEU-DET-CIP/train/{label}/{label}_1.jpg')
        ax.imshow(img)
        ax.set_title(label)
    ctx['tensors']['img'] = img
    ctx['tensors']['url'] = url
    ctx['tensors']['zip_file'] = zip_file
run_module(ctx)

# %% [markdown]
# # Pretrained models
#
# Pretrained vision models are neural network models that have been trained on large-scale image datasets such as ImageNet, and then made available to the research community for use in other applications.
# Training a large vision model is significantly different from a simple example in Colab or Jupyter notebook in several ways:
#
# * Computational requirements: Training a large vision model can require significant computational resources, including high-end GPUs or even specialized hardware such as TPUs. This is because the models are often very large, with many layers and millions of parameters, and require a large amount of data to be processed.
# * Training time: Training a large vision model can take several days or even weeks to complete, depending on the complexity of the model and the size of the dataset. This is in contrast to a simple example in Jupyter, which can be trained in a matter of minutes or seconds.
# * Data preparation: Large vision models require a large amount of data to be prepared and processed, often involving complex data augmentation techniques to increase the diversity of the training data. This can be time-consuming and require specialized tools and techniques.
# * Hyperparameter tuning: Training a large vision model requires careful selection and tuning of hyperparameters such as learning rate, batch size, and regularization strength. This process can be time-consuming and requires careful experimentation to find the optimal settings.
# * Model architecture: Large vision models often involve complex architectures with many layers and advanced techniques such as residual connections and attention mechanisms. This requires a deep understanding of neural networks and computer vision, and may involve reviewing and adapting research papers from the field.
#
# There are several advantages to using pretrained vision models:
#
# * Improved performance: Pretrained models are trained on large amounts of data and have learned to recognize a wide variety of image features. As a result, they often perform better than models that are trained from scratch on smaller datasets.
# * Reduced training time: Training deep neural networks from scratch can be computationally expensive and time-consuming. Pretrained models can be used as a starting point, with only the final layers of the network being fine-tuned on a new dataset. This can significantly reduce training time and computational resources needed.
# * Accessibility: Pretrained models are often made publicly available, making them accessible to researchers and developers who may not have the resources or expertise to train their own models from scratch.
# * Benchmarking: Pretrained models are often used as benchmarks in research, allowing for fair comparisons between different approaches.
# * Transfer learning: Pretrained models can be used as a form of transfer learning, where the learned features of the model are transferred to a new task. This is particularly useful when the new dataset is small, as it allows the model to learn from a larger, more diverse dataset without overfitting
# (*more on this in a minute!*).

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## ImageNet dataset
#
# The ImageNet dataset is a large-scale image database designed for visual object recognition research. It was created in 2009 and has been used as a benchmark dataset for image classification, object detection, and other computer vision tasks.
#
# The dataset contains over 14 million images of 1000 object categories, each with a varying number of images. The images are collected from the web and tagged with relevant metadata, including object category labels and bounding boxes for object localization.
#
# The images in the dataset vary in size and quality, and include a wide range of object categories such as animals, vehicles, and everyday objects. The dataset has been used to train and evaluate a variety of deep learning models, including CNNs, and has played a key role in advancing the field of computer vision.
#
# <img src="../lectures/assets/lecture14_imagenet.jpg" alt="Samples from the ImageNet dataset" width=600>

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## ResNet
#
# ResNet (short for "Residual Network") is a family of convolutional neural network (CNN) architectures that was introduced in 2015. ResNet was designed to address the problem of vanishing gradients in deep neural networks, which can make it difficult to train models with many layers.
#
# The key innovation of ResNet is the use of **"skip connections"** that allow information to flow more easily between layers. In a traditional CNN, each layer takes the output of the previous layer as its input. In ResNet, some layers take the output of an earlier layer as their input, effectively "skipping over" one or more layers. This allows information to flow directly from the input to the output of the network, bypassing intermediate layers and making it easier to train deep models.
#
# <img src="../lectures/assets/lecture14_skip_connection.jpg" alt="A skip connection in a Residual Network block" width=600>
#
# ResNet comes in several variations, including ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152. The number in each name corresponds to the number of layers in the network. All ResNet models share a similar structure, with several blocks of convolutional layers followed by a global average pooling layer and a fully connected layer. Each block consists of several convolutional layers with skip connections, and the last block is followed by a global average pooling layer and a fully connected layer that produces the final classification output.
#
# One notable feature of ResNet is its ability to achieve state-of-the-art performance on a variety of computer vision tasks with relatively few parameters, making it more efficient to train and deploy than other models. ResNet has been used for image classification, object detection, semantic segmentation, and other computer vision tasks, and has achieved top performance on several benchmark datasets.
#
# > [!NOTE]
# > The ResNet-18 architecture diagram from ResearchGate is currently unavailable due to access restrictions.
#
# We will specifically utilize ResNet18 for expediency in our in-class examples.
# ResNet18 is a convolutional neural network (CNN) architecture that was introduced in 2015 as part of the ResNet family of models. Like other ResNet models, ResNet18 is designed to address the problem of vanishing gradients in deep neural networks by using skip connections that allow information to flow more easily between layers.
#
# ResNet18 consists of 18 layers, including 16 convolutional layers and 2 fully connected layers. The first layer is a convolutional layer that takes as input a 224x224 RGB image, followed by a max pooling layer that reduces the spatial dimensions of the output. The remaining layers are grouped into four blocks, each consisting of several convolutional layers with skip connections. The last block is followed by a global average pooling layer that averages the output of each feature map across its spatial dimensions, and a fully connected layer that produces the final classification output.
#
# Let's try loading the ResNet18 model from `torchvision.models`:

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    resnet18 = ctx.get('tensors', {}).get('resnet18')
    from torchvision.models import resnet18

    model = resnet18(weights="IMAGENET1K_V1")
    model.eval()  # set to evaluation mode (freeze the trainable weights)
    ctx['model'] = model
run_module(ctx)

# %% [markdown]
# ## [Exercise]
#
# Look closely at the images.
# Can you determine any interpretation of the principal components?

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# # Feature extraction

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Concepts
#
# Fine-tuning refers to the process of taking a pre-trained neural network model, such as a vision model, and training it further on a new, related dataset. The aim is to adapt the pre-trained model's learned weights to better fit the new data and improve its performance on the specific task.
#
# <img src="../lectures/assets/lecture14_fine_tuning.jpg" alt="Illustration of fine-tuning by unfreezing layers" width=500>
#
# The process of fine-tuning typically involves several steps:
#
# 1. **Freezing:** The weights of the pre-trained layers are "frozen" and kept fixed during training. This is done to preserve the learned representations of the original model.
#
# 2. **Replacing the classifier:** The final classification layer(s) of the model are replaced with new ones, with the appropriate number of outputs for the new task.
#
# 3. **Training the new layers:** Only the newly added layers are trained on the new data, while the pre-trained layers are kept frozen.
#
# 4. **Unfreezing:** After a certain number of epochs or when the new layers have converged, the weights of the pre-trained layers are unfrozen and the entire model is fine-tuned on the new data. This can be done with a lower learning rate than the new layers to avoid overfitting.
#
# Fine-tuning is a common technique in transfer learning, where a pre-trained model is used as a starting point for a related task. It can save significant training time and computational resources, as the pre-trained model already has learned useful features that can be adapted to the new task. However, care must be taken when fine-tuning to avoid overfitting and preserve the original learned representations.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    dl_train = ctx.get('tensors', {}).get('dl_train')
    ds_train = ctx.get('tensors', {}).get('ds_train')
    nn = ctx.get('tensors', {}).get('nn')
    optim = ctx.get('tensors', {}).get('optim')
    pl = ctx.get('tensors', {}).get('pl')
    resnet18 = ctx.get('tensors', {}).get('resnet18')
    tqdm = ctx.get('tensors', {}).get('tqdm')
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
    ctx['tensors']['accuracy'] = accuracy
    ctx['tensors']['correct'] = correct
    ctx['tensors']['dl_train_shuffle'] = dl_train_shuffle
    ctx['tensors']['model_ft'] = model_ft
    ctx['tensors']['num_classes'] = num_classes
    ctx['tensors']['outputs'] = outputs
    ctx['tensors']['pbar'] = pbar
    ctx['tensors']['total'] = total
    ctx['tensors']['trainer'] = trainer
run_module(ctx)

# %% [markdown]
# ## [Exercise]
#
# Look closely at the images.
# Can you determine any interpretation of the principal components?

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# # Transfer learning

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Concepts
#
# Fine-tuning refers to the process of taking a pre-trained neural network model, such as a vision model, and training it further on a new, related dataset. The aim is to adapt the pre-trained model's learned weights to better fit the new data and improve its performance on the specific task.
#
# <img src="../lectures/assets/lecture14_fine_tuning.jpg" alt="Illustration of fine-tuning by unfreezing layers" width=500>
#
# The process of fine-tuning typically involves several steps:
#
# 1. **Freezing:** The weights of the pre-trained layers are "frozen" and kept fixed during training. This is done to preserve the learned representations of the original model.
#
# 2. **Replacing the classifier:** The final classification layer(s) of the model are replaced with new ones, with the appropriate number of outputs for the new task.
#
# 3. **Training the new layers:** Only the newly added layers are trained on the new data, while the pre-trained layers are kept frozen.
#
# 4. **Unfreezing:** After a certain number of epochs or when the new layers have converged, the weights of the pre-trained layers are unfrozen and the entire model is fine-tuned on the new data. This can be done with a lower learning rate than the new layers to avoid overfitting.
#
# Fine-tuning is a common technique in transfer learning, where a pre-trained model is used as a starting point for a related task. It can save significant training time and computational resources, as the pre-trained model already has learned useful features that can be adapted to the new task. However, care must be taken when fine-tuning to avoid overfitting and preserve the original learned representations.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    dl_train = ctx.get('tensors', {}).get('dl_train')
    ds_train = ctx.get('tensors', {}).get('ds_train')
    nn = ctx.get('tensors', {}).get('nn')
    optim = ctx.get('tensors', {}).get('optim')
    pl = ctx.get('tensors', {}).get('pl')
    resnet18 = ctx.get('tensors', {}).get('resnet18')
    tqdm = ctx.get('tensors', {}).get('tqdm')
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
    ctx['tensors']['accuracy'] = accuracy
    ctx['tensors']['correct'] = correct
    ctx['tensors']['dl_train_shuffle'] = dl_train_shuffle
    ctx['tensors']['model_ft'] = model_ft
    ctx['tensors']['num_classes'] = num_classes
    ctx['tensors']['outputs'] = outputs
    ctx['tensors']['pbar'] = pbar
    ctx['tensors']['total'] = total
    ctx['tensors']['trainer'] = trainer
run_module(ctx)

# %% [markdown]
# ## [Check your understanding]
#
# Perform feature extraction using the new pretrained model (wrapped inside our ResNetClassifier).
# Reduce their dimensions with PCA and plot the explained variance and class labels.
# How has the result changed after fine-tuning (compared to our "Feature Extraction" section at the beginning)?
#
# Finally we will use the `OffsetImage` and `AnnotationBbox` objects from `pyplot` to draw the images at their locations in the PCA space.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    dl_train = ctx.get('tensors', {}).get('dl_train')
    model_ft = ctx.get('tensors', {}).get('model_ft')
    tqdm = ctx.get('tensors', {}).get('tqdm')
    train_labels = ctx.get('tensors', {}).get('train_labels')
    # feature_model = nn.Sequential(*list(model_ft.children())[:-1])

    ft_features = []
    ft_labels = []

    with torch.no_grad():
        prog = tqdm.tqdm(dl_train, total=len(dl_train))
        for x, y in prog:
            out = model_ft.pretrained(x)  # only use the pretrained model! the last layer is our classifier
            ft_features.append(out.detach().numpy())
            ft_labels.append(y.detach().numpy())

    ft_features = np.vstack(ft_features)
    ft_features = ft_features.reshape(ft_features.shape[:2])
    ft_labels = np.hstack(ft_labels)

    # perform the PCA embedding on image features
    pca = decomposition.PCA()
    z_ft = pca.fit_transform(ft_features)

    # plot the explained variance of the embedding
    fig, ax = plt.subplots()
    _ = ax.plot(np.arange(1, 101), np.cumsum(pca.explained_variance_ratio_[:100]), '.-')
    _ = ax.set_xlabel('Components')
    _ = ax.set_ylabel('Explained Variance')
    ax.set_xscale('log')

    # plot the principal components of the image features with class labels
    fig, ax = plt.subplots()
    _ = ax.scatter(*z_ft[:, [0, 1]].T, c=ft_labels)
    _ = ax.set_xlabel('PC 1')
    _ = ax.set_ylabel('PC 2')
    ax.set_aspect('equal')

    from sklearn import cluster

    # perform the clustering
    km = cluster.KMeans(n_clusters=64, random_state=0).fit(z_ft[:, :2])

    # determine which sample IDs are closest to the cluster centers
    center_id = []
    for i, c in enumerate(km.cluster_centers_):
        dist = np.linalg.norm(c - z_ft[:, :2], axis=1)
        center_id.append( np.argmin(dist) )

    # plot the cluster centers
    _ = ax.plot(*km.cluster_centers_.T, 'rx')
    fig

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    fig, ax = plt.subplots(figsize=(10, 10))
    _ = ax.scatter(*z_ft[:, :2].T, c=train_labels)
    _ = ax.set_xlabel('PC 1')
    _ = ax.set_xlabel('PC 2')
    ax.set_aspect('equal')
    # add image thumbnails as annotations to the scatter plot
    thumbnail_size = (64, 64)
    for id in center_id:
        img_tensor, _ = dl_train.dataset[id]
        img = img_tensor.numpy().transpose()
        image_offset = OffsetImage(img, zoom=0.1)
        ab = AnnotationBbox(image_offset, z_ft[id, :2], xycoords='data')
        ax.add_artist(ab)
    ctx['tensors']['center_id'] = center_id
    ctx['tensors']['dist'] = dist
    ctx['tensors']['ft_features'] = ft_features
    ctx['tensors']['ft_labels'] = ft_labels
    ctx['tensors']['image_offset'] = image_offset
    ctx['tensors']['img'] = img
    ctx['tensors']['out'] = out
    ctx['tensors']['pca'] = pca
    ctx['tensors']['prog'] = prog
    ctx['tensors']['thumbnail_size'] = thumbnail_size
    ctx['tensors']['z_ft'] = z_ft
run_module(ctx)

# %% [markdown]
# # Fine-tuning

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
# ## Concepts
#
# Fine-tuning refers to the process of taking a pre-trained neural network model, such as a vision model, and training it further on a new, related dataset. The aim is to adapt the pre-trained model's learned weights to better fit the new data and improve its performance on the specific task.
#
# <img src="../lectures/assets/lecture14_fine_tuning.jpg" alt="Illustration of fine-tuning by unfreezing layers" width=500>
#
# The process of fine-tuning typically involves several steps:
#
# 1. **Freezing:** The weights of the pre-trained layers are "frozen" and kept fixed during training. This is done to preserve the learned representations of the original model.
#
# 2. **Replacing the classifier:** The final classification layer(s) of the model are replaced with new ones, with the appropriate number of outputs for the new task.
#
# 3. **Training the new layers:** Only the newly added layers are trained on the new data, while the pre-trained layers are kept frozen.
#
# 4. **Unfreezing:** After a certain number of epochs or when the new layers have converged, the weights of the pre-trained layers are unfrozen and the entire model is fine-tuned on the new data. This can be done with a lower learning rate than the new layers to avoid overfitting.
#
# Fine-tuning is a common technique in transfer learning, where a pre-trained model is used as a starting point for a related task. It can save significant training time and computational resources, as the pre-trained model already has learned useful features that can be adapted to the new task. However, care must be taken when fine-tuning to avoid overfitting and preserve the original learned representations.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)

# %% [markdown]
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
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    dl_train = ctx.get('tensors', {}).get('dl_train')
    ds_train = ctx.get('tensors', {}).get('ds_train')
    nn = ctx.get('tensors', {}).get('nn')
    optim = ctx.get('tensors', {}).get('optim')
    pl = ctx.get('tensors', {}).get('pl')
    resnet18 = ctx.get('tensors', {}).get('resnet18')
    tqdm = ctx.get('tensors', {}).get('tqdm')
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
    ctx['tensors']['accuracy'] = accuracy
    ctx['tensors']['correct'] = correct
    ctx['tensors']['dl_train_shuffle'] = dl_train_shuffle
    ctx['tensors']['model_ft'] = model_ft
    ctx['tensors']['num_classes'] = num_classes
    ctx['tensors']['outputs'] = outputs
    ctx['tensors']['pbar'] = pbar
    ctx['tensors']['total'] = total
    ctx['tensors']['trainer'] = trainer
run_module(ctx)

# %% [markdown]
# ## [Check your understanding]
#
# Perform feature extraction using the new pretrained model (wrapped inside our ResNetClassifier).
# Reduce their dimensions with PCA and plot the explained variance and class labels.
# How has the result changed after fine-tuning (compared to our "Feature Extraction" section at the beginning)?
#
# Finally we will use the `OffsetImage` and `AnnotationBbox` objects from `pyplot` to draw the images at their locations in the PCA space.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    dl_train = ctx.get('tensors', {}).get('dl_train')
    model_ft = ctx.get('tensors', {}).get('model_ft')
    tqdm = ctx.get('tensors', {}).get('tqdm')
    train_labels = ctx.get('tensors', {}).get('train_labels')
    # feature_model = nn.Sequential(*list(model_ft.children())[:-1])

    ft_features = []
    ft_labels = []

    with torch.no_grad():
        prog = tqdm.tqdm(dl_train, total=len(dl_train))
        for x, y in prog:
            out = model_ft.pretrained(x)  # only use the pretrained model! the last layer is our classifier
            ft_features.append(out.detach().numpy())
            ft_labels.append(y.detach().numpy())

    ft_features = np.vstack(ft_features)
    ft_features = ft_features.reshape(ft_features.shape[:2])
    ft_labels = np.hstack(ft_labels)

    # perform the PCA embedding on image features
    pca = decomposition.PCA()
    z_ft = pca.fit_transform(ft_features)

    # plot the explained variance of the embedding
    fig, ax = plt.subplots()
    _ = ax.plot(np.arange(1, 101), np.cumsum(pca.explained_variance_ratio_[:100]), '.-')
    _ = ax.set_xlabel('Components')
    _ = ax.set_ylabel('Explained Variance')
    ax.set_xscale('log')

    # plot the principal components of the image features with class labels
    fig, ax = plt.subplots()
    _ = ax.scatter(*z_ft[:, [0, 1]].T, c=ft_labels)
    _ = ax.set_xlabel('PC 1')
    _ = ax.set_ylabel('PC 2')
    ax.set_aspect('equal')

    from sklearn import cluster

    # perform the clustering
    km = cluster.KMeans(n_clusters=64, random_state=0).fit(z_ft[:, :2])

    # determine which sample IDs are closest to the cluster centers
    center_id = []
    for i, c in enumerate(km.cluster_centers_):
        dist = np.linalg.norm(c - z_ft[:, :2], axis=1)
        center_id.append( np.argmin(dist) )

    # plot the cluster centers
    _ = ax.plot(*km.cluster_centers_.T, 'rx')
    fig

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    fig, ax = plt.subplots(figsize=(10, 10))
    _ = ax.scatter(*z_ft[:, :2].T, c=train_labels)
    _ = ax.set_xlabel('PC 1')
    _ = ax.set_xlabel('PC 2')
    ax.set_aspect('equal')
    # add image thumbnails as annotations to the scatter plot
    thumbnail_size = (64, 64)
    for id in center_id:
        img_tensor, _ = dl_train.dataset[id]
        img = img_tensor.numpy().transpose()
        image_offset = OffsetImage(img, zoom=0.1)
        ab = AnnotationBbox(image_offset, z_ft[id, :2], xycoords='data')
        ax.add_artist(ab)
    ctx['tensors']['center_id'] = center_id
    ctx['tensors']['dist'] = dist
    ctx['tensors']['ft_features'] = ft_features
    ctx['tensors']['ft_labels'] = ft_labels
    ctx['tensors']['image_offset'] = image_offset
    ctx['tensors']['img'] = img
    ctx['tensors']['out'] = out
    ctx['tensors']['pca'] = pca
    ctx['tensors']['prog'] = prog
    ctx['tensors']['thumbnail_size'] = thumbnail_size
    ctx['tensors']['z_ft'] = z_ft
run_module(ctx)

# %% [markdown]
# # Data augmentation
#
# Data augmentation is a technique used to artificially increase the size of a dataset by creating additional, slightly modified versions of the original data. This is typically done by applying various image transformations such as rotating, flipping, cropping, zooming, or changing the brightness and contrast of the images.
#
# The purpose of data augmentation is to increase the diversity of the training data and to reduce overfitting. By creating more variations of the original data, the model is exposed to a wider range of examples, making it more robust and better able to generalize to new, unseen data.
#
# Data augmentation is commonly used in computer vision tasks, such as image classification or object detection, where the number of available training examples may be limited. By generating new data, it is possible to improve the performance of the model without the need for additional labeled data.
#
# It is important to note that the choice of data augmentation techniques depends on the specific task and the characteristics of the dataset. Some transformations may be more appropriate for certain types of images, while others may be less effective or even detrimental to performance. It is also important to be careful not to introduce too much noise or artifacts into the data, as this can have a negative impact on the model's performance.
#
# <img src="../lectures/assets/lecture14_data_augmentation.jpg" alt="Examples of data augmentation techniques applied to an image" width=600>
#
# Here are some specific techniques for data augmentation in image processing:
#
# 1. Flipping: horizontally or vertically flipping the image. This can be useful for tasks such as object detection or face recognition where the orientation of the object may vary.
#
# 2. Rotation: rotating the image by a certain angle. This can be useful for tasks such as document recognition where the text may be oriented in different directions.
#
# 3. Zooming: zooming in or out of the image. This can be useful for tasks such as object recognition or image classification where the size of the object in the image may vary.
#
# 4. Cropping: cropping the image to a smaller size. This can be useful for tasks such as image segmentation where the object of interest may be smaller than the full image.
#
# 5. Brightness and contrast adjustments: changing the brightness and contrast of the image. This can be useful for tasks such as face recognition where lighting conditions may vary.
#
# 6. Adding noise: adding random noise to the image. This can be useful for tasks such as denoising or image restoration where the input image may be noisy.
#
# 7. Color space transformations: transforming the image from one color space to another, such as converting from RGB to grayscale. This can be useful for tasks such as object detection or image segmentation where color is not important.
#
# These techniques can be applied in different combinations and with different parameters to create a diverse set of training data for the model. It is important to strike a balance between creating diverse data and not introducing too much noise or artifacts into the images.

# %%
def run_module(ctx):
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy import stats
    import sklearn
    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture
    pass
run_module(ctx)
