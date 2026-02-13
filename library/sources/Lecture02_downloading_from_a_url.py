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
# id: Lecture02_downloading_from_a_url
# type: Foundational
# parent_lecture: Lecture02
# ---
#
# ## Downloading from a URL
#
# We first have to acquire a data file.
# This short code downloads the data stored at a URL (hosted on OneDrive via Sharepoint, as you can see in the URL).
# It then writes the data to a `csv` file that we can later read.
#
# > If you already had a `csv` file this step would not be needed. I include it here to avoid having to distribute the file separately from the notebook.

# %%
import requests

# # define the url
# url = 'https://pennstateoffice365-my.sharepoint.com/:x:/g/personal/wfr5091_psu_edu/EU5JYKhddWRLhNaq_frzFS0BJOz9cXZTtxx-zKGJQEhVnw?e=jOmqFq&download=1'
#
# # fetch the data stored at the url
# r = requests.get(url)
#
# # write the data to the local file system
# with open('data.csv', 'w') as fid:
#     fid.write(r.text)
