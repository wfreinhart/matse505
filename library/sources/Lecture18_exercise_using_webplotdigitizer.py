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
# id: Lecture18_exercise_using_webplotdigitizer
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## [Exercise: using WebPlotDigitizer]
#
# Here are the steps:
# * Load image -> select this image from disk
# * 2D (X-Y) Plot -> Align Axes
#   * Select low X, high X, low Y, high Y on the chart
#   * You can use arrow keys to shift the point slightly after each click
#   * Complete -> type in the values (i.e., 0.0, 0.8, 0.0, 80) -> OK
# * Foreground Color (click the colored box on the right)
#   * Color Picker -> Click on the blue line -> Done
# * Automatic Extraction: Box -> drag around the blue line
#   * You will see a yellow box appear
#   * Distance -> 80 (this is a color range) -> Filter Colors
#   * You will see the line highlighted in yellow
# * Averaging Window -> 6 Px, 6 Px -> Run
#   * You will see red dots appear on top of the blue line
# * View Data
#   * You will see the X, Y values in a popup window
