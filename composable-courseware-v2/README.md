# composable-courseware-v2 (MATSE 505 port)

Jupytext `.py` authoring source for MATSE 505 lectures, ported over from the
`matse219-instructor` repo where it had been imported by mistake (that repo's
scope is 219 only). See `build.py` for how `# %% include: <module>` directives
get expanded and converted to `.ipynb`.

- `courses/505/Lecture01.py` .. `Lecture21.py` — the 21 lecture sources
- `modules/dataset_concrete.py`, `dataset_elements_505.py`, `dataset_steels.py`
  — the shared-module dependencies these lectures include
- `datasets/` — local mirrors of the CSVs those modules load (also already
  present at repo-root `datasets/`; each module tries the local path first,
  then falls back to `raw.githubusercontent.com/wfreinhart/matse505/main/datasets/...`)

Not yet reconciled with the JSON-based courseware on `feature/composable-modules`
-- this branch is a straight port for review, not a merge.

Build with `python composable-courseware-v2/build.py --ipynb`.
