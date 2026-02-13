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
# id: Lecture14_pretrained_models
# type: Foundational
# parent_lecture: Lecture14
# ---
#
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
