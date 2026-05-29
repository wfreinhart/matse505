#!/usr/bin/env python
"""
Composability Demo — proves that dataset and task modules are interchangeable.

For each (dataset, task) pair, this script:
  1. Reads the raw_code from both compiled JSON modules
  2. Concatenates them into a single script
  3. Executes via exec() in a fresh namespace (plain Python, no ctx)
  4. Reports whether execution succeeded and what was produced

Result: a 2×2 matrix showing all 4 combinations work.
"""

import json
import os
import sys
import traceback

import matplotlib
matplotlib.use('Agg')

MODULES_DIR = os.path.join(os.path.dirname(__file__), '..', 'library', 'modules')

DATASETS = [
    'HSDemo_dataset_steels',
    'HSDemo_dataset_concrete',
]

TASKS = [
    'HSDemo_linear_regression',
    'HSDemo_descriptive_stats',
]


def load_raw_code(module_id):
    path = os.path.join(MODULES_DIR, f'{module_id}.json')
    with open(path) as f:
        mod = json.load(f)
    return mod['raw_code']


def run_combo(dataset_id, task_id):
    """Assemble and execute a (dataset, task) pair. Returns (ok, detail)."""
    dataset_code = load_raw_code(dataset_id)
    task_code = load_raw_code(task_id)

    combined = dataset_code + '\n\n' + task_code

    ns = {'__name__': '__combo__'}
    try:
        exec(compile(combined, f'<{dataset_id} + {task_id}>', 'exec'), ns)
    except Exception:
        return False, traceback.format_exc()

    details = []
    if 'result' in ns:
        r = ns['result']
        details.append(f'y = {r.slope:.2f}x + {r.intercept:.2f}  (R²={r.rvalue**2:.3f})')
    if 'fig' in ns:
        details.append('figure produced')

    return True, '; '.join(details) if details else 'ok'


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    print('=' * 70)
    print('  COMPOSABILITY DEMO — 2 datasets × 2 tasks = 4 combinations')
    print('=' * 70)
    print()

    results = {}
    for ds in DATASETS:
        for task in TASKS:
            ds_short = ds.replace('HSDemo_dataset_', '')
            task_short = task.replace('HSDemo_', '')
            label = f'{ds_short} + {task_short}'

            print(f'▸ {label}')
            ok, detail = run_combo(ds, task)
            results[(ds_short, task_short)] = ok

            status = 'PASS' if ok else 'FAIL'
            print(f'  [{status}] {detail}')
            print()

    import matplotlib.pyplot as plt
    plt.close('all')

    print('=' * 70)
    print('  RESULTS MATRIX')
    print('=' * 70)

    ds_names = [d.replace('HSDemo_dataset_', '') for d in DATASETS]
    task_names = [t.replace('HSDemo_', '') for t in TASKS]

    header = f'{"":>20s}  ' + '  '.join(f'{t:>22s}' for t in task_names)
    print(header)
    print('-' * len(header))
    for ds in ds_names:
        row = f'{ds:>20s}  '
        cells = []
        for task in task_names:
            mark = 'PASS' if results[(ds, task)] else 'FAIL'
            cells.append(f'{mark:>22s}')
        row += '  '.join(cells)
        print(row)

    print()
    total = len(results)
    passed = sum(results.values())
    print(f'{passed}/{total} combinations succeeded.')

    if passed < total:
        sys.exit(1)


if __name__ == '__main__':
    main()
