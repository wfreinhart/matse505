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
# id: Lecture17_dataset
# type: Foundational
# parent_lecture: Lecture17
# ---
#
# ## Dataset

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

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def rgb2grey(img):
    return img[0].unsqueeze(0)

transform = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),   # Convert the image to a PyTorch tensor
    transforms.Lambda(lambda x: rgb2grey(x))  # Convert the RGB image to greyscale
])

ds_train = datasets.ImageFolder('NEU-DET-SP/train', transform=transform)
dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
print(f'Number of training images (all classes): {len(ds_train)}')

ds_valid = datasets.ImageFolder('NEU-DET-SP/validation', transform=transform)
dl_valid = DataLoader(ds_valid, batch_size=64, shuffle=False)
print(f'Number of validation images (all classes): {len(ds_valid)}')
