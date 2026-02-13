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
# id: Lecture08_exercise_expert_knowledge
# type: Foundational
# parent_lecture: Lecture08
# ---
#
# ## [Exercise: Expert knowledge]
#
# In the above case, we are using binning to approximate something like the group and period information in the periodic table.
# There are often cases where "expert knowledge" can be infused to greatly improve the result.
# Try introducing group and period information from [this periodic table dataset](https://gist.github.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee) to improve the performance of the Linear Regression model.
#
# Follow these steps:
# * Either download the file or fetch it with `requests`
# * Load it with `pandas`
# * Create new columns in our `data` DataFrame that include Period and Group
# * Try fitting a Linear Regression model with this new data

# %%
elements = pd.read_csv('../datasets/elements.csv', index_col='Symbol')
elements

data['Period'] = [elements.loc[it, 'Period'] for it in data.index]
data['Group'] = [elements.loc[it, 'Group'] for it in data.index]

clean_data = data.dropna(subset=['Atomic Mass', 'Period', 'Group', 'Ionization Energies (eV)'])
x = clean_data.loc[:, ['Atomic Mass', 'Period', 'Group']]
y = clean_data.dropna(subset=['Atomic Mass', 'Period', 'Group', 'Ionization Energies (eV)']).loc[:, 'Ionization Energies (eV)']

fig, ax = plt.subplots()
_ = ax.plot(x['Atomic Mass'], y, '.')

model = linear_model.LinearRegression().fit(x, y)
y_hat = model.predict(x)
print( 'R2 = ', model.score(x, y) )

_ = ax.plot(x['Atomic Mass'], y_hat, '.')
