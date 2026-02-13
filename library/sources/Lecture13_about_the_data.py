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
# id: Lecture13_about_the_data
# type: Foundational
# parent_lecture: Lecture13
# ---
#
# ## About the data
#
# The [NEU Surface Defect Database](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm) is a publicly available dataset of images of various surfaces with different types of defects. It was created by the researchers at Northeastern University, China. The database was released in 2018 and has been widely used in machine learning research.
#
# The database contains 1,800 grayscale images, each of size 200x200 pixels. The images show six types of surface defects on steel plates:
#
# * Rolled-in Scale (RS)
# * Scratches (Scr)
# * Pitted Surface (Pitted)
# * Rolled-in Dirt (RD)
# * Inclusion (In)
# * Crazing (Cr)
#
# Each type of defect has 300 images. The images are labeled with their corresponding defect type, and the labels are provided in a separate file.
#
# The NEU Surface Defect Database is useful for developing and testing image processing and machine learning algorithms for defect detection and classification. It can be used for tasks such as defect detection, classification, segmentation, and recognition.
#
# You need to download the `zip` file first:
#
# Then extract it to individual files with `zipfile`:
#
# We can load an image using `PIL`:

# %%
import zipfile, requests

url = 'https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/wfr5091_psu_edu/ERXYsfbOP4dGm7_M4oIh-0gBV3Ix19fKuSndDu4Ui6zHrQ?e=HOCHNN&download=1'
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open('data.zip', 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

zip_file = zipfile.ZipFile('data.zip')
zip_file.extractall('/content/')
zip_file.close()

from PIL import Image

filepath = 'NEU-DET-SP/train/scratches/scratches_1.jpg'
Image.open(filepath)

filepath = 'NEU-DET-SP/train/patches/patches_1.jpg'
Image.open(filepath)
