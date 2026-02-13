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
# id: Lecture15_concept_vectors
# type: Foundational
# parent_lecture: Lecture15
# ---
#
# ## Concept vectors
#
# Concept vectors, also known as concept embeddings or distributed representations, are a type of vector space model used in natural language processing and machine learning. The basic idea behind concept vectors is to represent words or concepts as vectors in a high-dimensional vector space, such that words or concepts that are semantically similar are close together in the vector space.
#
# The concept vector model is based on the distributional hypothesis, which states that words that appear in similar contexts tend to have similar meanings. To create concept vectors, a large corpus of text is analyzed to identify patterns in the co-occurrence of words. These patterns are used to construct a high-dimensional vector space, where each word is represented as a vector in the space.
#
# In this vector space, each dimension represents a different feature or context of the words. For example, one dimension may represent the frequency of the word in the corpus, while another dimension may represent the frequency of the word in a particular context.
#
# <img src="../lectures/assets/lecture15_latent_arithmetic.jpg" alt="Example of vector arithmetic in latent space for generating faces" width=600>
#
# Once the concept vectors are constructed, they can be used for various natural language processing tasks, such as word similarity, document classification, and sentiment analysis. In word similarity tasks, the similarity between two words is measured by the cosine similarity between their corresponding concept vectors. In document classification, the concept vectors of the words in a document are averaged to create a document vector, which is then used to classify the document.
#
# Concept vectors are a powerful tool for natural language processing and machine learning, as they provide a way to represent the meaning of words and concepts in a way that is both computationally efficient and semantically meaningful. They have been used in many successful applications, such as language translation, speech recognition, and information retrieval.
#
# <img src="../lectures/assets/lecture15_gan_architecture.jpg" alt="Architecture of a Generative Adversarial Network" width=600>
#
# With the benefit of known regression labels, we can actually fit the concept vectors directly using linear regression:
#
# Once we have this vector, we can project the latent code onto the concept using a dot product:

# %%
from sklearn import linear_model

lr = linear_model.LinearRegression().fit(z, a[:, [0]])
print(f'R2 = {lr.score(z, a[:, 0]):.3f}; coef = {lr.coef_}')

z_proj = np.dot(z, lr.coef_.T)

fig, ax = plt.subplots()
ax.plot(z_proj, a[:, 0], '.')
ax.set_xlabel('$z \cdot v$')
ax.set_ylabel('$A_0$')
