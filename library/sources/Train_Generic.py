# %% [markdown]
# ---
# id: Train_Generic
# type: Training
# ---
# # Generic Training
# Training the model on the generated tensors.

# %%
# Uses variables from global state: model, X, y
model.fit(X, y)
score = model.score(X, y)
print(f"R2 Score: {score:.3f}")
