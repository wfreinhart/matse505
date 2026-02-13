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
# id: Lecture03_statistical_modeling_versus_supervised_ml
# type: Foundational
# parent_lecture: Lecture03
# ---
#
# ## Statistical modeling versus Supervised ML
#
# *Wait, isn't this just curve fitting? I can do that in Excel*
#
# You can probably recall many times in which you've fitted (or *learned*) functional relationships of the form $f(x) = y$ in your engineering classes.
# Let's talk about how this is different from ML.
#
# “The major difference between machine learning and statistics is their purpose. Machine learning models are designed to make the most accurate predictions possible. Statistical models are designed for inference about the relationships between variables.”
#
# -Matthew Stewart, *The Actual Difference Between Statistics and Machine Learning*
#
# "Machine learning can be defined as the process of solving a practical problem by 1) gathering a dataset, and 2) algorithmically building a statistical model based on that dataset. That statistical model is assumed to be used somehow to solve the practical problem."
#
# -Andriy Burkov, *The Hundred-Page Machine Learning Book*
#
# We can see this in the Venn diagram that we used above. Scroll back up and notice how Machine Learning is the at intersection of Mathematics and Computer Science while Statistical Research is at the intersection of Mathematics and Domain Expertise.
#
# ML typically serves to handle generalizing models beyond the Domain or to extend models which cannot be made sufficiently predictive using Domain knowledge alone. However, Data Scientists require both ML and conventional Statistical Research to function effectively! ML is known to be somewhat brittle and can very easily fail when exposed to unforseen challenges. Domain Expertise is critical in identifying these failure modes and correcting them.
