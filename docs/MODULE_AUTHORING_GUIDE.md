# Module Authoring Guide

## Philosophy
The goal of the Composable Module system is to create a library of reusable, mix-and-match components that can be assembled into various lectures, workshops, or tutorials. Instead of writing monolithic notebooks, we write small, focused "source modules" that perform specific tasks (e.g., "Load Data", "Train Model", "Visualize Results").

## Global State Workflow
The modules are designed to run in a standard Jupyter Notebook environment where variables defined in one cell (module) are available to subsequent cells.
*   **Decoupling:** Modules don't need to know *where* a variable came from, only that it is defined in the global namespace when the module runs.
*   **Validation:** Even though we use global state, the `Refactorer` tool statically analyzes your code during compilation to ensure that if Module B needs a variable `X`, some earlier Module A actually provides it.

## Anatomy of a Module
Modules are Python files in `library/sources/` with YAML frontmatter.

### Example: `library/sources/ToyData.py`
```python
# %% [markdown]
# ---
# id: toy_data_loader
# type: Data
# ---
# # Load Toy Data
# We will generate a simple synthetic dataset.

# %%
import numpy as np
import pandas as pd

# Generate data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.normal(0, 1, 100)

data = pd.DataFrame({'x': X.flatten(), 'y': y})
```

## Component Library (Toy Examples)

To demonstrate mix-and-match capabilities, let's define a few standardized components.

### 1. Data Components
**`ToyData_Linear.py`**
```python
# ... frontmatter ...
# Generates linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.flatten() + 1 + np.random.normal(0, 1, 100)
```

**`ToyData_Quadratic.py`**
```python
# ... frontmatter ...
# Generates quadratic data
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 0.5 * X.flatten()**2 + np.random.normal(0, 1, 100)
```

### 2. Model Components
**`Model_LinearRegression.py`**
```python
# ... frontmatter ...
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

**`Model_DecisionTree.py`**
```python
# ... frontmatter ...
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=3)
```

### 3. Training Component
**`Train_Generic.py`**
```python
# ... frontmatter ...
# Uses variables from global scope: model, X, y
model.fit(X, y)
score = model.score(X, y)
print(f"R2 Score: {score:.3f}")
```

### 4. Visualization Component
**`Viz_Regression.py`**
```python
# ... frontmatter ...
import matplotlib.pyplot as plt
# Uses global scope: model, X, y
plt.scatter(X, y, color='black', label='Data')
plt.plot(X, model.predict(X), color='red', label='Prediction')
plt.legend()
plt.show()
```

## Mix-and-Match Scenarios

By standardizing variable names (e.g., `X`, `y`, `model`), we can assemble completely different lectures from these components without changing their code.

### Scenario A: Linear Regression on Linear Data
**`lecture_defs/ScenarioA.yaml`**
```yaml
id: ScenarioA
title: Linear Regression Intro
modules:
  - id: ToyData_Linear        # Provides X, y (linear)
  - id: Model_LinearRegression # Provides model (Linear)
  - id: Train_Generic         # Trains Linear model on Linear data
  - id: Viz_Regression        # Plots result
```

### Scenario B: Decision Tree on Quadratic Data
**`lecture_defs/ScenarioB.yaml`**
```yaml
id: ScenarioB
title: Modeling Non-Linearity
modules:
  - id: ToyData_Quadratic     # Provides X, y (quadratic)
  - id: Model_DecisionTree    # Provides model (Tree)
  - id: Train_Generic         # Trains Tree model on Quadratic data
  - id: Viz_Regression        # Plots result
```

### Scenario C: Failure Case (Linear Model on Quadratic Data)
**`lecture_defs/ScenarioC.yaml`**
```yaml
id: ScenarioC
title: Underfitting Example
modules:
  - id: ToyData_Quadratic     # Provides X, y (quadratic)
  - id: Model_LinearRegression # Provides model (Linear)
  - id: Train_Generic         # Trains Linear model on Quadratic data
  - id: Viz_Regression        # Plots result (Will show underfitting!)
```

## Best Practices

1.  **Standardize Variable Names:** Use clear, consistent names for your domain (e.g., `X_train`, `y_train`, `model`, `history`).
2.  **Atomic Responsibilities:**
    *   One module should do one thing (e.g., "Load Data", not "Load Data and Train").
    *   This maximizes reuse. If "Load" and "Train" are separate, you can swap the "Train" step for a different algorithm easily.
3.  **Idempotency:** Modules should be re-runnable. Avoid complex state changes that break if a cell is run twice.
## Hierarchical Groups (Macros)

Sometimes, defining every single module (e.g., `Model_RF`, `Train`, `Evaluate`) in every lecture is tedious. You can group modules into a "Macro" or "Group".

1.  Create a YAML file in `library/groups/` (e.g., `Concrete_RF_Group.yaml`).
2.  List the modules it contains:
    ```yaml
    modules:
      - id: RealModel_RF
      - id: RealWorkflow_Verify
    ```
3.  Reference the group by ID in your lecture definition:
    ```yaml
    modules:
      - id: RealData_Concrete
      - id: Concrete_RF_Group
    ```

The build system will automatically expand `Concrete_RF_Group` into its components.

## Real-World Case Study: Concrete Compressive Strength

To see how this works with actual course material, check out the following components in `library/sources/` (derived from Lecture 03):

*   **`RealData_Concrete.py`**: Loads the concrete dataset and prepares `x`, `y`.
*   **`RealFeat_Poly.py`**: Applies `PolynomialFeatures` transformation to `x`.
*   **`RealModel_Linear.py`**: Standard Linear Regression.
*   **`RealModel_RF.py`**: Random Forest with `max_depth=5`.
*   **`RealWorkflow_Verify.py`**: Standardized verification (Train/Test split, Scoring, Parity Plot).

You can mix-and-match these to compare the impact of feature engineering and model selection. 
**We also use Groups here to simplify the lecture definitions:**

```yaml
# Linear model with Polynomial Features
id: RealScenario_Linear
modules: 
  - id: RealData_Concrete
  - id: Concrete_Linear_Group # Expands to [RealFeat_Poly, RealModel_Linear, RealWorkflow_Verify]

# Random Forest on raw features
id: RealScenario_RF
modules: 
  - id: RealData_Concrete
  - id: Concrete_RF_Group     # Expands to [RealModel_RF, RealWorkflow_Verify]
```
