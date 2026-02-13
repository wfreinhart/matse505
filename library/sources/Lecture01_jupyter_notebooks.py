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
# id: Lecture01_jupyter_notebooks
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# ## Jupyter notebooks
#
# *What's a Notebook?*
#
# We're inside a Jupyter notebook right now!
# A Jupyter Notebook is a more flexible and user-friendly way to write and read Python code.
# [Jupyter](https://en.wikipedia.org/wiki/Project_Jupyter) provides a sort of software wrapper around the Python ["kernel"](https://en.wikipedia.org/wiki/Kernel_(operating_system)) and acts as a go-between to provide some quality of life enhancements for the user (you!).
#
# Jupyter lets us include **formatted text** and $E_{Q}u^at_io^ns$ alongside [hyperlinks](https://en.wikipedia.org/wiki/Project_Jupyter) and images:
#
# <img src="../lectures/assets/python_logo.jpg" alt="The official Python logo, featuring two interlocking snakes in blue and yellow." height=200/>
#
# <img src="../lectures/assets/jupyter_logo.jpg" alt="The Project Jupyter logo, showing an orange planet-like circle with orbiting moons." height=200/>
#
# The notebook format lets us seamlessly integrate code and non-code blocks so we can easily create tutorials.
# This has actually become an industry standard for data science researchers, and software distributions are basically expected to include `.ipynb` walkthroughs in addition to the source code.
#
# A great example is Google's very own [Colab tutorial](https://colab.research.google.com/notebooks/intro.ipynb).
# Let's take a look...
#
# ...you may see there are plenty of things we won't cover such as the machine learning package Tensorflow and accelerated hardware like Graphics Processing Units (GPUs) and Tensor Processing Units (TPUs).
# However, we will cover basic Python syntax, working with data, and making charts as shown.
#
# When you look at plain Python source code, you will see a distinct difference.
# Source code does usually include comments like the green text above (after the `#` sign), but not any other elements like rich text, images, or actual code output.
# The picture below illustrates the difference:
#
# ![A comparison diagram showing a Jupyter Notebook with mixed text and code vs a plain Python script with only code.](../lectures/assets/lecture01_notebooks_vs_scripts.jpg)
#
# (image credit [Krishna Subramanian](https://krishna-subramanian.net/index.php/portfolio/notebooksvscripts/))

# %%
# jupyter also lets us embed python code right alongside all that
# this is a live python code cell!
