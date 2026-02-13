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
# id: Lecture18_loading_the_result
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# ## Loading the result
#
# * Save as CSV
#   * Click the "folder" icon on the left side of Colab
#   * Drag and drop the CSV into the Files pane
#   * OR use the "upload" button (file with an up arrow on it)
#
# You will get a message warning you that Colab often resets and you will lose files stored on this instance when that happens. This just means you can't store files permanently on Colab (you can link to Google Drive or OneDrive if you need to do this).
#
# Now load `pandas` and import with `read_csv` as usual:
#
# Note that `pandas` thinks the columns are *named* **0.005129...** and **-0.027810...** because there is no header in the file. We can add proper names by using the `header=None` and `names=` keyword arguments:

# %%
import pandas as pd

pd.read_csv('data.csv')

if os.path.exists(local_path):
    data = pd.read_csv(local_path, header=None, names=['Strain', 'Stress (MPa)'])
else:
    try:
        data = pd.read_csv(github_url, header=None, names=['Strain', 'Stress (MPa)'])
    except:
        data = pd.DataFrame(columns=['Strain', 'Stress (MPa)'])
data
