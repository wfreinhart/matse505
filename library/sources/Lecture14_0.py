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
# id: Lecture14_0
# type: Foundational
# parent_lecture: Lecture14
# ---
#
#
#
# Today's topics:
# * Pretrained models
# * Transfer learning
# * Fine-tuning
# * Data augmentation
#
# We'll again load the [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm), a publicly available dataset of images of various surfaces with different types of defects.
#
# This time we'll use a harder version of the problem, with Crazing, Inclusion, and Patches instead of just Patches and Scratches.
# Here are some samples of the three classes:

# %%
import zipfile, requests

url = 'https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/wfr5091_psu_edu/EZwz7XK8nMVOkp_V0pXP3HsBYiC_1B8JhHXscCnJFli6yw?e=iFYoO3&download=1'
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open('data.zip', 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

zip_file = zipfile.ZipFile('data.zip')
zip_file.extractall('/content/')
zip_file.close()

from PIL import Image
from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for i, label in enumerate(['crazing', 'inclusion', 'patches']):
    ax = axes[i]
    img = Image.open(f'NEU-DET-CIP/train/{label}/{label}_1.jpg')
    ax.imshow(img)
    ax.set_title(label)
