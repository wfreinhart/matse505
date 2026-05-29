# MATSE 505: Applied Machine Learning for Materials Science

This repository contains lecture materials, assignments, and resources for the course MATSE 505: Applied Machine Learning for Materials Science.

## Repository Structure

- `notebooks/`: Compiled `.ipynb` files for viewing on GitHub or Google Colab. (Diffs are hidden).
- `lectures/`: Paired `.py` files (percent format) for clean code review and editing.
- `assignments/`: Homework assignments and lab exercises.
- `lectures/assets/`: External resources and images used in lectures.
- `datasets/`: Course datasets (CSV files) - committed to Git for easy access.
- `.local/`: Local artifacts and temporary files (gitignored).

## Setup Instructions

### Environment
We use a hybrid Conda + `pip-tools` approach to keep local packages synced with Google Colab.

#### 1. Initial Setup
```bash
# Create the environment
conda create -n matse505 python=3.11
conda activate matse505

# Install pip-tools
conda install -c conda-forge pip-tools
```

#### 2. Syncing with Colab
To ensure your local versions match Colab's pre-installed packages:
1. Ensure `colab-constraints.txt` is updated (see `.agent/workflows/sync-env.md`).
2. Compile the locked requirements:
   ```bash
   pip-compile requirements.in --constraints colab-constraints.txt --output-file requirements.txt
   ```
3. Install/Sync your environment:
   ```bash
   pip-sync requirements.txt
   ```

### Jupytext
To keep the repository size manageable and version-control friendly, we use [Jupytext](https://jupytext.readthedocs.io/en/latest/). This allows us to store notebooks as paired Python scripts, avoiding the overhead of large binary notebook files in Git history.

#### How to work with Jupytext
1. **Automatic Syncing**: The `jupytext.toml` file is configured to pair `.ipynb` files in `notebooks/` with `.py` scripts in `lectures/`.
2. **Opening a Notebook**: When you open a file in `notebooks/`, Jupytext will automatically sync changes to the corresponding script in `lectures/`.
3. **Saving**: Every time you save the notebook in the Jupyter interface, Jupytext updates the corresponding `.py` script.
4. **Version Control**: Both the `.py` and `.ipynb` files are committed. Use the `.py` files in `lectures/` for reviewing code changes (diffs), while the `.ipynb` files in `notebooks/` provide ready-to-run environments for students.
5. **Manual Sync & CLI Usage**:
   If you are working in an editor like VS Code and want to force a sync or rebuild the `.ipynb` from the `.py` script, use the CLI:
   
   - **Sync both files**: `jupytext --sync lectures/LectureXX.py`
   - **Rebuild notebook from script**: `jupytext --to ipynb lectures/LectureXX.py`
- **Extract script from notebook**: `jupytext --to py:percent lectures/LectureXX.ipynb`

## License

This project is dual-licensed:
- **Lecture Materials & Diagrams**: Licensed under [CC-BY-4.0](LICENSE).
- **Software & Code Snippets**: Licensed under the [MIT License](LICENSE).

## Module Development

The courseware is built using a "Composable Module" system. Content is authored in small, reusable source files, compiled into JSON modules, and then assembled into full lectures.

## Separate Packaging

To package CAMEL, telemetry, and composable courseware independently from MATSE505 lecture materials, use:

```bash
python scripts/package_tracks.py --dry-run camel telemetry composable
python scripts/package_tracks.py camel telemetry composable
```

This stages plain directories under `dist/separated/` that you can move into another repo.
Use `--zip` only if you need archives.
See `docs/separate-packaging.md` for exact bundle contents and custom output options.

### 1. Workflow Overview
1.  **Source** (`library/sources/*.py`): Author content in Python scripts with markdown cells.
2.  **Compile** (`scripts/compile_modules.py`): Converts sources into JSON modules (`library/modules/*.json`), handling variable context and imports.
3.  **Assemble** (`scripts/build_lecture.py`): Combines modules defined in `lecture_defs/*.yaml` into a single lecture script (`lectures/LectureXX.py`).
4.  **Sync**: Jupytext automatically syncs the lecture script to `notebooks/LectureXX.ipynb`.

### 2. Creating a Module
Create a Python file in `library/sources/`. Use the `#%%` syntax to denote cells.

**Format:**
```python
# %% [markdown]
# ---
# id: my_unique_module_id
# type: Foundational
# parent_lecture: LectureXX
# ---
# # Module Title
# Markdown content here...

# %%
# Python code here...
x = 10
y = 20
```

**Key Rules:**
-   **Context (`ctx`)**: The system automatically wraps your code in a function `run_module(ctx)`.
-   **Inputs**: Variables used but not defined in your module are automatically retrieved from `ctx` (e.g., `data`, `model`, `tensors`).
-   **Outputs**: Top-level variable assignments are automatically saved to `ctx` for subsequent modules to use.

### 3. Compiling Modules
Run the compiler to generate/update JSON modules:
```bash
python scripts/compile_modules.py
# Or watch for changes:
python scripts/compile_modules.py --watch
```

### 4. Building a Lecture
Define the lecture structure in `lecture_defs/LectureXX.yaml`:
```yaml
id: LectureXX
title: My Lecture
modules:
  - id: module_id_1
  - id: module_id_2
```

Then build the lecture script:
```bash
python scripts/build_lecture.py --lecture LectureXX
# Or build all:
python scripts/build_lecture.py --all
```