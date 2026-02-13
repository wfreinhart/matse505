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
# id: Lecture07_a_new_dataset
# type: Foundational
# parent_lecture: Lecture07
# ---
#
# ## A new dataset

# %%
# Set the path to the second data file
filename_cl = 'steels.csv'
local_path_cl = f'../datasets/{filename_cl}'
github_url_cl = f'https://raw.githubusercontent.com/wfreinhart/matse505/main/datasets/{filename_cl}'

# Load the data: try local path first, fallback to GitHub for Colab
if os.path.exists(local_path_cl):
    data_cl = pd.read_csv(local_path_cl)
else:
    data_cl = pd.read_csv(github_url_cl)
data_cl['Alloy family'] = [x[0] for x in data_cl['Alloy code']]
data_cl
