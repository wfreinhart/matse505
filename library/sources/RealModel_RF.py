# %% [markdown]
# ---
# id: RealModel_RF
# type: Model
# ---
# # Random Forest Regressor
# Random Forest with max_depth=5 as suggested in Lecture 03.

# %%
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=5, random_state=0)
