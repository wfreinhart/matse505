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
# id: Lecture14_concepts
# type: Foundational
# parent_lecture: Lecture14
# ---
#
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
