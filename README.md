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
We use the `matse505` conda environment for this course. 

```bash
conda activate matse505
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

### Image Optimization
To keep the repository size manageable, we use an agent skill to resize and compress images in the `assets/` folder.

If you are an AI agent, use the `image-optimization` skill. For manual runs:
```bash
python .agent/skills/images/optimize_images.py
```
This will resize images to a maximum width of 600px and convert them to compressed JPGs.

## License

This project is dual-licensed:
- **Lecture Materials & Diagrams**: Licensed under [CC-BY-4.0](LICENSE).
- **Software & Code Snippets**: Licensed under the [MIT License](LICENSE).