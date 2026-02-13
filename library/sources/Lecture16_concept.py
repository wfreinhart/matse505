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
# id: Lecture16_concept
# type: Foundational
# parent_lecture: Lecture16
# ---
#
# ## Concept
#
# The autoencoder's latent space can be very poorly behaved.
# For instance, there is no guarantee that points very near to each other have a meaningful interpolation between them; the only thing that the model seeks is the faithful reconstruction of input samples.
# This limits the autoencoder's utility in generative modeling.
#
# <img src="../lectures/assets/lecture16_vae_architecture.jpg" alt="Architecture of a Variational Autoencoder" width=600>
#
# The Variational Autoencoder (VAE) utilizes the notion of distributions inside the bottleneck:
#
# <img src="../lectures/assets/lecture16_reparameterization.jpg" alt="Illustration of the reparameterization trick in VAEs" width=600>
#
# Thus, nearby points blend together and the latent space is forced to have some notion of smoothness during training in order to achieve a low loss.
#
#
# <img src="../lectures/assets/lecture16_vae_latent.jpg" alt="Latent space visualization of a trained VAE" width=600>
#
#
# In practice this is achieved by fitting a mean $\mu$ and standard deviation $\sigma$ using neural networks:
#
# <img src="../lectures/assets/lecture16_adversarial_training.jpg" alt="Conceptual diagram of adversarial training in GANs" width=600>
#
# > There is an amazing article describing this in great detail in [Understanding Variational Autoencoders](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
#
# This will require some careful programming because the behavior is different during training and inference!
