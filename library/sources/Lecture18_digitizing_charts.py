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
# id: Lecture18_digitizing_charts
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# # Digitizing charts
#
# A common source of data is images published in journal articles, textbooks, or on the web. We've been relying on prepared `CSV` files to do our data analysis with Python so far, but this isn't representative of the real world. Let's try using [WebPlotDigitizer](https://apps.automeris.io/wpd/) to obtain raw data from the following chart of mechanical performance of an architected material from [mechanical performance of an architected material](https://doi.org/10.1088/1757-899X/433/1/012078):
#
# <img src="../lectures/assets/lecture18_chart_to_digitize.jpg" alt="Stress-strain chart of an architected material for digitization exercise" width=540>
