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
# id: Lecture12_onnx
# type: Foundational
# parent_lecture: Lecture12
# ---
#
# ## ONNX
#
# ONNX is another tool that is used to deploy models in different environments. ONNX stands for Open Neural Network Exchange, and it is an open-source format that allows for the exchange of models between different frameworks. ONNX models can be used in environments such as mobile devices, web applications, and cloud services.
#
# To convert a PyTorch model to the ONNX format, you can use the `torch.onnx.export()` function. This function exports the model's computation graph to an ONNX file that can be loaded and executed in other environments. Here's an example:
#
# This exports the `MLPRegressor` model's computation graph to an ONNX file named `model.onnx`, and specifies the input and output tensor names.
# Here's the actual content of the file:
#
# The data stored in this byte string can be read by an ONNX parser to recreate the trained model.

# %%
input_tensor = torch.randn_like(x)

torch.onnx.export(model, input_tensor, "model.onnx", input_names=["input"], output_names=["output"])

with open('model.onnx', 'rb') as fid:
    lines = fid.readlines()

for l in lines:
    print(l)
