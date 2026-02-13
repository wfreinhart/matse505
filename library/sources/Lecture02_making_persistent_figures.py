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
# id: Lecture02_making_persistent_figures
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Making persistent `figures`
#
# *What happened to our line?*
#
# You may have noticed we did not create any variables so far, and have only been calling `pyplot` **functions**.
# This is because `pyplot` uses a concept called a "state machine" to store the chart behind the scenes rather than exposing it to the user.
# Executing a new code cell clears the state and begins a new plot.
#
# *What if we want to access our chart again later?*
#
# There is a simple way to get around the default behavior of `pyplot`, which is to explicitly create variables to store the chart.
# Matplotlib calls the elements of a chart the `Figure` and `Axes`, as illustrated in this diagram:
#
# <img src="../lectures/assets/lecture02_matplotlib_hierarchy.jpg" alt="Diagram showing the hierarchy of Matplotlib Figure elements" width=600>
#
# (image credit [ajaytech.co](https://ajaytech.co/matplotlib))
