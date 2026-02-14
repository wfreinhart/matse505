# %% [markdown]
# ---
# id: RealFeat_Poly
# type: FeatureEngineering
# ---
# # Polynomial Features
# Generating degree=2 polynomial features.

# %%
from sklearn.preprocessing import PolynomialFeatures

# We update x in the global scope
poly = PolynomialFeatures(degree=2).fit(x)
x = poly.transform(x)

print(f"Generated polynomial features: {x.shape}")
