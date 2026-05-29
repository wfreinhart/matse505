#!/usr/bin/env python
"""
Append an LLM-authored interpretation cell to an executed lecture notebook.

Usage:
    python scripts/interpret_lecture.py --lecture HSDemo_Regression
    python scripts/interpret_lecture.py --all
    python scripts/interpret_lecture.py --lecture HSDemo_Regression --force

Requires:
    GEMINI_API_KEY in environment or .env file
    An already-executed notebook (notebooks/<LectureID>.ipynb with cell outputs)

What it does:
    1. Reads cell outputs from the executed notebook
    2. Calls the LLM with those outputs + module context
    3. Writes a short "Key Takeaways" markdown cell at the END of the notebook
    4. Caches a hash of the outputs in the transitions sidecar so re-runs are free
       unless the notebook outputs actually changed

The interpretation cell is marked with a sentinel comment so subsequent runs
can find and replace it rather than appending a duplicate.
"""

import os
import sys
import json
import hashlib
import argparse
import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import yaml
from dotenv import load_dotenv
from litellm import completion

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DEFS_DIR = 'lecture_defs'
NOTEBOOKS_DIR = 'notebooks'
DEFAULT_MODEL = 'gemini/gemini-3-flash-preview'

INTERPRETATION_SENTINEL = '<!-- hsdemo-interpretation -->'

SYSTEM_PROMPT = """\
You are a curriculum author writing a brief interpretation section for a high school math or science lesson.

The student has just run a Python notebook that loaded a real dataset and applied a mathematical technique.
You will be given the actual text outputs produced by the notebook cells.

Write a short section (3-6 bullet points) titled "## Key Takeaways" that:
- Refers to the specific numbers and results from the outputs (not generic statements)
- Explains what those numbers mean in plain language a high school student would understand
- Connects the result back to the original question posed by the lesson
- Notes anything surprising or worth thinking about

Output only valid markdown. Start with "## Key Takeaways".
Do not include code, raw numbers without context, or generic advice.
"""


def api_key():
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise EnvironmentError(
            'GEMINI_API_KEY is not set.\n'
            'Add it to .env or: export GEMINI_API_KEY=your_key_here'
        )
    return key


def extract_outputs(nb):
    """Pull all text outputs from executed code cells in the notebook."""
    lines = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        for output in cell.get('outputs', []):
            if output.get('output_type') in ('stream', 'execute_result'):
                text = output.get('text', output.get('data', {}).get('text/plain', ''))
                if isinstance(text, list):
                    text = ''.join(text)
                if text.strip():
                    lines.append(text.strip())
    return '\n\n'.join(lines)


def extract_lesson_context(nb):
    """Pull the first markdown cell text to give the LLM context about the lesson goal."""
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            src = ''.join(cell.get('source', []))
            if src.strip():
                return src[:800]
    return ''


def outputs_hash(outputs_text):
    return hashlib.md5(outputs_text.encode()).hexdigest()[:12]


def load_sidecar(lecture_id):
    path = os.path.join(DEFS_DIR, f'{lecture_id}.transitions.yaml')
    if os.path.exists(path):
        return yaml.safe_load(open(path)) or {}
    return {}


def save_sidecar(lecture_id, data):
    path = os.path.join(DEFS_DIR, f'{lecture_id}.transitions.yaml')
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def generate_interpretation(lesson_context, outputs_text, model=DEFAULT_MODEL):
    user_msg = (
        f'Lesson context (from the notebook introduction):\n{lesson_context}\n\n'
        f'Actual outputs produced by the notebook:\n{outputs_text}\n\n'
        'Write the Key Takeaways section.'
    )
    response = completion(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_msg},
        ],
        api_key=api_key(),
    )
    return response.choices[0].message.content.strip()


def inject_interpretation(nb, markdown_text):
    """Append (or replace) the interpretation markdown cell in the notebook."""
    sentinel_line = INTERPRETATION_SENTINEL

    # Find and remove any existing interpretation cell
    nb['cells'] = [
        c for c in nb['cells']
        if not (
            c.get('cell_type') == 'markdown' and
            sentinel_line in ''.join(c.get('source', []))
        )
    ]

    # Build the new cell
    cell_source = f'{sentinel_line}\n{markdown_text}'
    new_cell = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': cell_source,
    }
    nb['cells'].append(new_cell)
    return nb


def process_lecture(lecture_id, force=False, model=DEFAULT_MODEL):
    print(f'\n=== {lecture_id} ===')

    nb_path = os.path.join(NOTEBOOKS_DIR, f'{lecture_id}.ipynb')
    if not os.path.exists(nb_path):
        print(f'  Notebook not found: {nb_path}')
        print(f'  Run: jupyter nbconvert --execute --inplace notebooks/{lecture_id}.ipynb')
        return

    with open(nb_path) as f:
        nb = json.load(f)

    outputs_text = extract_outputs(nb)
    if not outputs_text.strip():
        print(f'  No cell outputs found -- has the notebook been executed?')
        return

    h = outputs_hash(outputs_text)

    # Check sidecar cache
    sidecar = load_sidecar(lecture_id)
    cached_interp = sidecar.get('interpretation', {})
    if cached_interp.get('hash') == h and not force:
        print(f'  Cached (hash {h}) -- injecting existing interpretation.')
        markdown_text = cached_interp['markdown']
    else:
        print(f'  Generating interpretation...', end='', flush=True)
        lesson_context = extract_lesson_context(nb)
        markdown_text = generate_interpretation(lesson_context, outputs_text, model=model)
        print(' done')

        sidecar['interpretation'] = {
            'hash': h,
            'generated_at': datetime.datetime.utcnow().isoformat(),
            'markdown': markdown_text,
        }
        save_sidecar(lecture_id, sidecar)

    nb = inject_interpretation(nb, markdown_text)

    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f'  Written -> {nb_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Append LLM interpretation cell to executed lecture notebooks.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--lecture', help='Lecture ID (e.g. HSDemo_Regression)')
    group.add_argument('--all', action='store_true', help='Process all lectures in lecture_defs/')
    parser.add_argument('--force', action='store_true', help='Regenerate even if cached')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'LiteLLM model slug (default: {DEFAULT_MODEL})')
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    if args.lecture:
        process_lecture(args.lecture, force=args.force, model=args.model)
    else:
        defs = sorted(Path(DEFS_DIR).glob('*.yaml'))
        for d in defs:
            if '.transitions' not in d.stem:
                process_lecture(d.stem, force=args.force, model=args.model)


if __name__ == '__main__':
    main()
